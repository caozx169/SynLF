import os
from pathlib import Path

import numpy as np
import tifffile as tiff
import torch
from hydra.utils import instantiate

from config_utils import REPO_ROOT, load_run_config


DEFAULT_VIEW_POSITIONS = (
    (-2, 0), (-1, 0), (0, -2), (0, -1), (0, 0),
    (0, 1), (0, 2), (1, 0), (2, 0),
)


def default_weights_root() -> Path:
    return Path(os.environ.get("SYNLF_WEIGHTS_ROOT", REPO_ROOT / "checkpoints")).expanduser()


def default_data_root() -> Path:
    return Path(os.environ.get("SYNLF_DATA_ROOT", REPO_ROOT / "data")).expanduser()


def default_checkpoint_path() -> Path:
    return default_weights_root() / "visdepth.ckpt"


def default_config_path() -> Path:
    return REPO_ROOT / "configs" / "inference.yaml"


def load_visdepth(checkpoint: str | Path, config: str | None, device: torch.device):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    config_path = Path(config).expanduser().resolve() if config else default_config_path()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = load_run_config(config_path)
    model = instantiate(cfg.model, load_pretrained=False)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = payload.get("state_dict", payload)
    if "state_dict" in payload:
        state_dict = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
    model.load_state_dict(state_dict, strict=True)
    model = model.eval().to(device)
    viewspos = torch.tensor(cfg.viewspos, dtype=torch.float32, device=device)
    return model, cfg, viewspos, config_path


def load_lf_tiff(path: str | Path, viewspos=DEFAULT_VIEW_POSITIONS) -> torch.Tensor:
    """Read 9 selected views or select them from a row-major (v, u) 9x9 grid.

    uint8 images are scaled by 255; floating-point images are already normalized.
    A 9-view stack must already follow the model's configured view order.
    """
    path = Path(path)
    array = np.squeeze(np.asarray(tiff.imread(path)))
    if array.ndim != 3:
        raise ValueError(f"Expected a 9- or 81-view TIFF stack, got shape {array.shape}: {path}")
    if array.shape[0] in (9, 81):
        pass
    elif array.shape[-1] in (9, 81):
        array = array.transpose(2, 0, 1)
    else:
        raise ValueError(f"Expected 9 or 81 views, got shape {array.shape}: {path}")

    if array.shape[0] == 81:
        positions = torch.as_tensor(viewspos).detach().cpu().numpy()
        if positions.shape != (9, 2):
            raise ValueError(f"Expected 9 (v, u) view positions, got shape {positions.shape}")
        if (not np.isfinite(positions).all()
                or np.any(positions != np.rint(positions))
                or np.any(np.abs(positions) > 4)):
            raise ValueError("81-view TIFF positions must be integer (v, u) offsets in [-4, 4]")
        grid_positions = positions.astype(np.int64) + 4
        array = array[grid_positions[:, 0] * 9 + grid_positions[:, 1]]

    is_uint8 = array.dtype == np.uint8
    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if is_uint8:
        array /= 255.0
    return torch.from_numpy(array[:, None]).contiguous()


def crop_to_divisible(viewimgs: torch.Tensor, divisor: int = 8) -> torch.Tensor:
    if divisor <= 1:
        return viewimgs
    height, width = viewimgs.shape[-2:]
    crop_h = height - height % divisor
    crop_w = width - width % divisor
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(f"Input is too small for divisor {divisor}: {(height, width)}")
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return viewimgs[..., top : top + crop_h, left : left + crop_w]


def output_disparity(output: dict) -> torch.Tensor:
    if "disp" not in output:
        raise KeyError("VisDepth output does not contain 'disp'")
    disp = output["disp"]
    if disp.ndim == 4:
        disp = disp[0, 0]
    elif disp.ndim == 3:
        disp = disp[0]
    elif disp.ndim != 2:
        raise ValueError(f"Unexpected disparity shape: {tuple(disp.shape)}")
    return disp.detach().float().cpu()


@torch.inference_mode()
def predict_disparity(model, viewimgs: torch.Tensor, viewspos: torch.Tensor, device: torch.device):
    if viewimgs.shape[0] != viewspos.shape[0]:
        raise ValueError(
            f"View count mismatch: input has {viewimgs.shape[0]}, config has {viewspos.shape[0]}"
        )
    return output_disparity(model(viewimgs.unsqueeze(0).to(device), viewspos.unsqueeze(0)))
