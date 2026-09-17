import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
from collections.abc import Mapping
from loguru import logger
from torch.utils.data import DataLoader

from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils.depth_ops import disp2depth, depth2normal
from utils.gpu_augment import GPUPostAugment


class ModelInterface(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))
        self.config = config
        self.maxdisp = config.maxdisp
        self.mindisp = config.mindisp
        self.viewspos = config.viewspos
        self.cropsize_hw = config.cropsize
        # Initialize datasets in setup(), after DDP workers start.
        self.train_dataset = None
        self.val_datasets = None
        self._initial_model()
        self._initial_loss()
        self._initial_train_metrics()
        self._initial_val_metrics()
        self.gpu_augment = GPUPostAugment(config.train_dataset)

    def _initial_model(self):
        self.model = instantiate(self.config.model)

    def _initial_loss(self):
        loss_cfg = self.config.loss
        self.loss_fns = nn.ModuleDict()
        self.active_losses = {}

        for field, field_loss_cfg in loss_cfg.items():
            for loss_name, loss_item_cfg in field_loss_cfg.items():
                weight = loss_item_cfg.get('weight', 1.0)
                if weight <= 0.0:
                    continue
                item_cfg = OmegaConf.to_container(loss_item_cfg, resolve=True)
                item_cfg.pop('weight', None)
                fn = instantiate(item_cfg)
                key = f"{field}/{loss_name}"
                self.loss_fns[key.replace("/", "_")] = fn
                self.active_losses[key] = (weight, fn)

        logger.info({k: w for k, (w, _) in self.active_losses.items()})

    def _build_metric_collection(self, metric_cfg):
        return torchmetrics.MetricCollection({
            name: instantiate(cfg) for name, cfg in metric_cfg.items()
        })

    def _initial_train_metrics(self):
        self.metric_fns = nn.ModuleDict({
            out_name: self._build_metric_collection(metric_cfg)
            for out_name, metric_cfg in self.config.metric.items()
        })

    def _initial_val_metrics(self):
        self.val_metric_fns = nn.ModuleDict()
        val_cfg = OmegaConf.select(self.config, "val_dataset")
        if val_cfg is None:
            return

        if '_target_' in val_cfg:
            dataset_names = [val_cfg.get('name', 'val_0')]
        else:
            dataset_names = [name for name in val_cfg if not name.startswith('_') and name != 'defaults']

        for ds_name in dataset_names:
            self.val_metric_fns[ds_name] = nn.ModuleDict({
                out_name: self._build_metric_collection(metric_cfg)
                for out_name, metric_cfg in self.config.metric.items()
            })

    def _post_process(self, data_dict):
        coef = data_dict.get('coef')
        keys = [k for k in data_dict.keys() if 'disp' in k and 'preds' not in k]

        for k in keys:
            disp = data_dict[k]
            depth_key = k.replace('disp', 'depth')
            if depth_key not in data_dict and coef is not None:
                data_dict[depth_key] = disp2depth(disp, coef)
            normal_key = k.replace('disp', 'normal')
            if normal_key not in data_dict:
                data_dict[normal_key] = depth2normal(disp, K=None, auto_limit=True)
        return data_dict

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.training:
            batch = self.gpu_augment(batch)
        batch.pop('views_disp', None)
        return batch

    def _shared_step(self, batch):
        viewimgs, viewspos = batch['viewimgs'], batch['viewspos']
        output = self.model(viewimgs, viewspos)
        if 'coef' in batch:
            output['coef'] = batch['coef']
        output = self._post_process(output)
        batch = self._post_process(batch)
        loss, loss_dict = self.compute_loss(batch, output)
        return loss, loss_dict, batch, output

    def training_step(self, batch, batch_idx):
        loss, loss_dict, batch, output = self._shared_step(batch)
        loss_dict = {f'{k}/train': v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=False)
        self._log_misc_stats(batch, output, 'train')
        return self._build_step_output(output, loss)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        dataset_name = None
        if isinstance(batch, dict) and 'dataset' in batch:
            dataset_field = batch['dataset']
            dataset_name = dataset_field[0] if isinstance(dataset_field, (list, tuple)) else dataset_field

        if dataset_name is None:
            metric_dataset_names = list(self.val_metric_fns.keys())
            if 0 <= dataloader_idx < len(metric_dataset_names):
                dataset_name = metric_dataset_names[dataloader_idx]
            else:
                dataset_name = f"val_{dataloader_idx}"

        loss, loss_dict, batch, output = self._shared_step(batch)
        loss_dict = {f'{k}/val/{dataset_name}': v for k, v in loss_dict.items()}
        self.compute_metrics(batch, output, dataset_name=dataset_name)
        self.log_dict(loss_dict, sync_dist=True, on_step=False, on_epoch=True, add_dataloader_idx=False)

        self._log_misc_stats(batch, output, f'val/{dataset_name}')
        return self._build_step_output(output, loss, dataset_name=dataset_name)

    def _build_step_output(self, output, loss, dataset_name=None):
        result = {
            'loss': loss,
            'disp': output['disp'],
            'disp_raw': output['disp_raw'],
            'normal': output['normal'],
        }
        if dataset_name is not None:
            result['dataset_name'] = dataset_name
        return result

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def compute_loss(self, batch, output):
        gt_disp = batch['disp']
        mask = batch['masks']
        loss_dict = {}
        total_loss = 0.0
        for key, (weight, fn) in self.active_losses.items():
            field = key.split('/')[0]
            pred = output[field]
            loss = fn(pred, gt_disp, mask)
            loss_dict[f'loss_detail/{key}'] = loss
            total_loss += loss * weight

        loss_dict['loss'] = total_loss
        return total_loss, loss_dict

    def compute_metrics(self, batch, output, dataset_name=None):
        mask = batch['masks']

        if self.training:
            metric_fns_to_use = self.metric_fns
        elif dataset_name and dataset_name in self.val_metric_fns:
            metric_fns_to_use = self.val_metric_fns[dataset_name]
        else:
            assert False, f"Dataset name {dataset_name} not found in val_metric_fns"

        for out_name, metric_collection in metric_fns_to_use.items():
            if out_name not in output:
                logger.error(f"Output {out_name} not found in model outputs")
                continue

            if 'disp' in out_name:
                gt = batch['disp']
            elif 'depth' in out_name:
                gt = batch['depth']
            elif 'normal' in out_name:
                gt = batch['normal']
            else:
                assert False, f"Unknown output name: {out_name}"
            pred = output[out_name]
            # (B,C,H,W) -> valid pixels across the batch.
            pred_masked = pred.permute(0,2,3,1)[mask.squeeze(1)]
            gt_masked = gt.permute(0,2,3,1)[mask.squeeze(1)]

            metrics = metric_collection(pred_masked, gt_masked)

        return metrics

    def _log_misc_stats(self, batch, output, suffix):
        misc_dict = {}

        def _add_stats(data_dict, prefix):
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor) and ('disp' in key.lower() or 'depth' in key.lower()):
                    misc_dict[f'misc/{prefix}_{key}_max_{suffix}'] = value.max()
                    misc_dict[f'misc/{prefix}_{key}_min_{suffix}'] = value.min()

        _add_stats(batch, 'batch')
        _add_stats(output, 'output')

        self.log_dict(misc_dict, sync_dist=False, on_step=True, on_epoch=False)

    def _reset_all_metrics(self):
        for metric in self.metric_fns.values():
            metric.reset()
        for val_metric_fn in self.val_metric_fns.values():
            for metric in val_metric_fn.values():
                metric.reset()

    def on_train_epoch_start(self):
        self._reset_all_metrics()

    def on_validation_epoch_start(self):
        self._reset_all_metrics()

    def on_test_epoch_start(self):
        self._reset_all_metrics()

    def on_validation_epoch_end(self):
        msg = ''
        monitor_key = self.config.checkpoint.monitor
        monitor_values = []
        for dataset_name, val_metric_fn in self.val_metric_fns.items():
            msg += f'\n[{self.global_step}/{self.config.training.max_steps}]  {dataset_name}: '
            for out_name, metric_collection in val_metric_fn.items():
                metrics = metric_collection.compute()
                for k, v in metrics.items():
                    metric_key = f'metric/{dataset_name}/{out_name}/{k}'
                    self.log(metric_key, v, sync_dist=True, on_epoch=True, on_step=False, prog_bar=False)
                    if f'{out_name}_{k}' == monitor_key:
                        monitor_values.append(v)
                    msg += f'{out_name}/{k}: {v:.2f} '
        if monitor_values:
            monitor_value = sum(monitor_values) / len(monitor_values)
            self.log(monitor_key, monitor_value, sync_dist=True, on_epoch=True, on_step=False, prog_bar=False, add_dataloader_idx=False)

        logger.info(msg)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def configure_optimizers(self):
        cfg = self.config.optimizer
        params_group = self.model.get_group_parameters()
        params_group[0]['lr'] = cfg.finetuned_lr if not cfg.same_lr else cfg.scratch_lr
        params_group[1]['lr'] = cfg.scratch_lr
        optimizer = torch.optim.AdamW(
            params=params_group,
            weight_decay=cfg.weight_decay,
        )
        total_steps = self.trainer.max_steps
        warmup_steps = cfg.warmup_ratio * total_steps
        power = cfg.power
        logger.info(optimizer)

        lambda0 = lambda cur_iter: cur_iter / warmup_steps if cur_iter < warmup_steps else \
            1/cfg.final_div_factor if cur_iter >= total_steps else \
            max((1 - (cur_iter - warmup_steps) / (total_steps - warmup_steps)) ** power, 1/cfg.final_div_factor)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        logger.info(scheduler)
        logger.info(f'total_steps: {total_steps}')
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def setup(self, stage: str):
        if stage in ('fit', 'validate', 'test'):
            if stage == 'fit' and self.train_dataset is None:
                self.train_dataset = instantiate(self.config.train_dataset)

            if self.val_datasets is None and OmegaConf.select(self.config, "val_dataset") is not None:
                val_dataset_result = instantiate(self.config.val_dataset)
                if isinstance(val_dataset_result, Mapping):
                    self.val_datasets = list(val_dataset_result.values())
                else:
                    self.val_datasets = [val_dataset_result]

    def train_dataloader(self):
        cfg = self.config.training
        return DataLoader(
            self.train_dataset,
            batch_size=cfg.batchsize,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        if not self.val_datasets:
            return []

        return [
            DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self):
        return self.val_dataloader()
