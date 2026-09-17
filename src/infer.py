import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from loguru import logger
import tifffile as tiff
import torch
from tqdm import tqdm

from config_utils import load_calibration
from model_runtime import (
    crop_to_divisible,
    default_checkpoint_path,
    default_data_root,
    load_lf_tiff,
    load_visdepth,
    predict_disparity,
)
from utils.depth_ops import colorize, disp2depth


HCI_VIEW_IDS = (22, 31, 38, 39, 40, 41, 42, 49, 58)


@dataclass(frozen=True)
class Sample:
    name: str
    relative_dir: Path
    source: Path
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VisDepth inference.")
    parser.add_argument(
        "--checkpoint",
        default=str(default_checkpoint_path()),
        help="VisDepth model weights.",
    )
    parser.add_argument("--config", default=None, help="Model YAML; defaults to configs/inference.yaml.")
    parser.add_argument("--calibration", default=None, help="Calibration YAML; defaults to configs/data/synlf.yaml.")
    parser.add_argument(
        "--input",
        default=str(default_data_root()),
        help="Dataset root, scene directory, or 9-/81-view TIFF stack.",
    )
    parser.add_argument(
        "--format",
        choices=("synlf", "hci", "tiff"),
        default="synlf",
        help="Input layout. The default scans SynLF *_interp.tif files recursively.",
    )
    parser.add_argument("--output", default="outputs/infer", help="Output directory.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--crop-divisor", type=int, default=8)
    parser.add_argument("--coef", nargs=3, type=float, default=None, metavar=("A", "B", "C"))
    parser.add_argument("--disp-vmin", type=float, default=None)
    parser.add_argument("--disp-vmax", type=float, default=None)
    parser.add_argument("--depth-vmin", type=float, default=None)
    parser.add_argument("--depth-vmax", type=float, default=None)
    return parser.parse_args()


def discover_samples(input_path: Path, layout: str) -> list[Sample]:
    if layout == "synlf":
        if not input_path.is_dir():
            raise NotADirectoryError(input_path)
        paths = sorted(input_path.rglob("*_interp.tif"))
        samples = [
            Sample(path.name.removesuffix("_interp.tif"), path.parent.relative_to(input_path), path, layout)
            for path in paths
        ]
    elif layout == "hci":
        if not input_path.is_dir():
            raise NotADirectoryError(input_path)
        scene_dirs = [input_path] if _has_hci_views(input_path) else sorted(
            path for path in input_path.iterdir() if path.is_dir() and _has_hci_views(path)
        )
        samples = [Sample(path.name, Path(), path, layout) for path in scene_dirs]
    else:
        if input_path.is_file():
            paths = [input_path]
            base = input_path.parent
        else:
            paths = sorted(
                path
                for path in input_path.glob("*.tif*")
                if "_z_proj" not in path.stem and "_pred_" not in path.stem
            )
            base = input_path
        samples = [Sample(path.stem, path.parent.relative_to(base), path, layout) for path in paths]

    if not samples:
        raise FileNotFoundError(f"No {layout} samples found under {input_path}")
    return samples


def _find_hci_view(scene_dir: Path, index: int) -> Path | None:
    for suffix in ("png", "jpg", "jpeg"):
        path = scene_dir / f"input_Cam{index:03d}.{suffix}"
        if path.is_file():
            return path
    return None


def _has_hci_views(scene_dir: Path) -> bool:
    return all(_find_hci_view(scene_dir, index) is not None for index in HCI_VIEW_IDS)


def load_hci_scene(scene_dir: Path) -> torch.Tensor:
    views = []
    for index in HCI_VIEW_IDS:
        path = _find_hci_view(scene_dir, index)
        if path is None:
            raise FileNotFoundError(f"Missing input_Cam{index:03d} in {scene_dir}")
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        views.append(torch.from_numpy(image)[None])
    return torch.stack(views)


def save_color(path: Path, value: torch.Tensor, vmin, vmax, cmap="Spectral_r") -> None:
    rgb = colorize(value, vmin=vmin, vmax=vmax, cmap=cmap)
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.permute(1, 2, 0).numpy()
    else:
        rgb = np.asarray(rgb).transpose(1, 2, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.asarray(rgb, dtype=np.uint8))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    samples = discover_samples(input_path, args.format)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    coef = tuple(args.coef) if args.coef else (
        load_calibration(args.calibration) if args.format == "synlf" or args.calibration else None
    )

    if not torch.cuda.is_available():
        raise RuntimeError("VisDepth inference requires a CUDA GPU.")
    device = torch.device(f"cuda:{args.device}")
    model, _, viewspos, config_path = load_visdepth(args.checkpoint, args.config, device)
    logger.info(f"Checkpoint: {Path(args.checkpoint).resolve()}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Input: {input_path} ({len(samples)} samples, format={args.format})")

    for sample in tqdm(samples, desc="Infer"):
        viewimgs = (
            load_hci_scene(sample.source)
            if sample.kind == "hci"
            else load_lf_tiff(sample.source, viewspos)
        )
        viewimgs = crop_to_divisible(viewimgs, args.crop_divisor)
        disp = predict_disparity(model, viewimgs, viewspos, device)

        sample_dir = output_root / sample.relative_dir
        sample_dir.mkdir(parents=True, exist_ok=True)
        tiff.imwrite(sample_dir / f"{sample.name}_pred_disp.tif", disp.numpy().astype(np.float32))
        save_color(sample_dir / f"{sample.name}_pred_disp.png", disp, args.disp_vmin, args.disp_vmax)

        if coef is not None:
            depth = torch.nan_to_num(disp2depth(disp, coef), nan=0.0, posinf=0.0, neginf=0.0)
            tiff.imwrite(sample_dir / f"{sample.name}_pred_depth.tif", depth.numpy().astype(np.float32))
            save_color(
                sample_dir / f"{sample.name}_pred_depth.png",
                depth,
                args.depth_vmin,
                args.depth_vmax,
                cmap="plasma",
            )

    logger.info(f"Saved results to {output_root}")


if __name__ == "__main__":
    main()
