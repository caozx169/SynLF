from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import make_grid
from utils.depth_ops import colorize, depth2normal

class ImageLogger(Callback):
    def __init__(
        self,
        log_every_n_steps: int = 1000,
        max_images: int = 4,
        log_val_first_batch: bool = True,
    ):
        """
        Args:
            log_every_n_steps: 训练步数间隔
            max_images: 每次画多少张图
            log_val_first_batch: 是否只在验证集每个epoch的第一个batch画图
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.max_images = max_images
        self.log_val_first_batch = log_val_first_batch

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_every_n_steps == 0:
            self._log_images(trainer, pl_module, batch, outputs, split="train")

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.log_val_first_batch and batch_idx != 0:
            return

        dataset_name = outputs['dataset_name']
        self._log_images(trainer, pl_module, batch, outputs, split=f"val/{dataset_name}")

    def _log_images(self, trainer, pl_module, batch, outputs, split):
        logger = trainer.logger
        if logger is None: return
        centerview_index = pl_module.model.centerview_index
        N = min(batch['viewimgs'].shape[0], self.max_images)

        viewimgs = batch['viewimgs'][:N].detach().cpu()
        masks = batch['masks'][:N].detach().cpu()
        gt_disp = batch['disp'][:N].detach().cpu()

        pt_disp = outputs['disp']
        pt_disp_raw = outputs['disp_raw']
        pt_normal = outputs['normal']
        pt_disp = pt_disp[:N].detach().cpu()
        pt_disp_raw = pt_disp_raw[:N].detach().cpu()

        if 'normal' not in batch:
            gt_normal = (depth2normal(gt_disp, K=None, auto_limit=True) + 1) / 2
        else:
            gt_normal = (batch['normal'][:N].detach().cpu() + 1) / 2

        pt_normal_vis = None
        if pt_normal is not None:
             pt_normal_vis = (pt_normal[:N].detach().cpu() + 1) / 2

        # 原始视差的法线用于观察局部细节。
        pt_normal_from_disp = (depth2normal(pt_disp_raw, K=None, auto_limit=True) + 1) / 2

        current_step = trainer.global_step

        vmin_disp = gt_disp.min().item()
        vmax_disp = gt_disp.max().item()

        combined_center = [
            make_grid(viewimgs[:, centerview_index], nrow=N, padding=2, pad_value=1),
            self._colorize_grid(gt_disp, vmin=vmin_disp, vmax=vmax_disp),
            self._colorize_grid(pt_disp_raw, vmin=vmin_disp, vmax=vmax_disp),
            self._colorize_grid(pt_disp, vmin=vmin_disp, vmax=vmax_disp)
        ]
        grid_center = make_grid(combined_center, nrow=2, padding=5, pad_value=1)
        logger.experiment.add_image(f'{split}/view_disp', grid_center, current_step)

        if gt_normal is not None:
            combined_normal = [
                make_grid(gt_normal, nrow=N, padding=2, pad_value=1),
                make_grid(pt_normal_from_disp, nrow=N, padding=2, pad_value=1)
            ]
            if pt_normal_vis is not None:
                combined_normal.append(make_grid(pt_normal_vis, nrow=N, padding=2, pad_value=1))
            grid_normal = make_grid(combined_normal, nrow=1, padding=5, pad_value=1)
            logger.experiment.add_image(f'{split}/normals', grid_normal, current_step)

        diff_disp = (pt_disp - gt_disp)
        vmin_diff = -float(diff_disp.abs().max())
        vmax_diff = -vmin_diff
        combined_diff = [
            make_grid(viewimgs[:, 0], nrow=N, padding=2, pad_value=1),
            make_grid(viewimgs[:, -1], nrow=N, padding=2, pad_value=1),
            make_grid(masks, nrow=N, padding=2, pad_value=1),
            self._colorize_grid(diff_disp, vmin=vmin_diff, vmax=vmax_diff, cmap='bwr')
        ]
        grid_diff = make_grid(combined_diff, nrow=2, padding=5, pad_value=1)
        logger.experiment.add_image(f'{split}/diff_views', grid_diff, current_step)

    def _colorize_grid(self, tensor, vmin=None, vmax=None, cmap='Spectral_r'):
        N = tensor.shape[0]
        grid = make_grid(tensor, nrow=N, padding=2, pad_value=0, normalize=False)

        colored = colorize(grid[0], vmin=vmin, vmax=vmax, cmap=cmap)

        return colored / 255.0
