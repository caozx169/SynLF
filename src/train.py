import sys
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from callbacks.image_logger import ImageLogger
from model_interface import ModelInterface


def configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(log_dir / "train.log")


def train(cfg: DictConfig) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("SynLF training requires a CUDA GPU.")
    pl.seed_everything(int(cfg.seed), workers=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tensorboard = TensorBoardLogger(save_dir=cfg.logdir, name=cfg.name)
    configure_logging(Path(tensorboard.log_dir))
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    checkpoint_cfg = OmegaConf.to_container(cfg.checkpoint, resolve=True)
    checkpoint = ModelCheckpoint(**checkpoint_cfg)
    image_logger = ImageLogger(
        log_every_n_steps=max(int(cfg.training.max_steps) // 10, 1),
        max_images=2,
        log_val_first_batch=True,
    )
    callbacks = [
        checkpoint,
        LearningRateMonitor(logging_interval="step"),
        ModelSummary(max_depth=3),
        image_logger,
    ]

    devices = list(cfg.training.devices)
    model = ModelInterface(config=cfg)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=devices,
        precision=cfg.training.precision,
        max_steps=int(cfg.training.max_steps),
        max_epochs=-1,
        logger=tensorboard,
        callbacks=callbacks,
        log_every_n_steps=int(cfg.training.log_every_n_steps),
        gradient_clip_val=float(cfg.training.gradient_clip_val),
        accumulate_grad_batches=int(cfg.training.accumulate_grad_batches),
        num_sanity_val_steps=int(cfg.training.num_sanity_val_steps),
        strategy="ddp_find_unused_parameters_true" if len(devices) > 1 else "auto",
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
        reload_dataloaders_every_n_epochs=1,
    )
    trainer.fit(model, ckpt_path=cfg.resume)

    best_path = checkpoint.best_model_path or checkpoint.last_model_path
    if not best_path:
        raise RuntimeError("Training completed without writing a checkpoint")
    logger.info(f"Checkpoint: {best_path}")
    return best_path


@hydra.main(version_base=None, config_path="../configs", config_name="config_base")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
