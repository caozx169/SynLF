import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter
import tifffile as tiff
from tqdm import tqdm

from config_utils import REPO_ROOT, load_calibration

DISP_MIN = -4.0
DISP_MAX = 4.0
EROSION_RADIUS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SynLF on the 142-sample test split."
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Folder containing the numbered input/GT pairs. Defaults to data/.",
    )
    parser.add_argument(
        "--split",
        default=str(REPO_ROOT / "assets" / "splits" / "test.txt"),
        help="Text file containing one sample ID per line.",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Evaluate existing predictions instead of running the model.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(Path(os.environ.get("SYNLF_WEIGHTS_ROOT", REPO_ROOT / "checkpoints")) / "visdepth.ckpt"),
        help="VisDepth model weights.",
    )
    parser.add_argument("--config", default=None, help="Model YAML; defaults to configs/inference.yaml.")
    parser.add_argument("--calibration", default=None, help="Calibration YAML; defaults to configs/data/synlf.yaml.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="outputs/test/metrics.json")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def read_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_prediction(prediction_root: Path, sample_id: str) -> np.ndarray:
    path = prediction_root / f"{sample_id}_pred_disp.tif"
    return np.squeeze(tiff.imread(path)).astype(np.float64)


def build_mask(gt_depth: np.ndarray, pred_depth: np.ndarray, coef) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        gt_disp = coef[0] / (gt_depth + coef[1]) + coef[2]
    raw = np.isfinite(gt_disp) & (gt_disp >= DISP_MIN) & (gt_disp <= DISP_MAX)
    eroded = minimum_filter(
        raw.astype(np.uint8),
        size=2 * EROSION_RADIUS + 1,
        mode="constant",
        cval=1,
    ).astype(bool)
    return eroded & np.isfinite(gt_depth) & np.isfinite(pred_depth) & (gt_depth > 1e-6)


def run_model(args):
    if not args.checkpoint:
        raise ValueError("Pass either --predictions or --checkpoint.")
    import torch
    from model_runtime import crop_to_divisible, load_lf_tiff, load_visdepth, predict_disparity

    if not torch.cuda.is_available():
        raise RuntimeError("End-to-end testing requires a CUDA GPU.")
    device = torch.device(f"cuda:{args.device}")
    model, _, viewspos, _ = load_visdepth(args.checkpoint, args.config, device)

    def predict(path: Path) -> np.ndarray:
        views = crop_to_divisible(load_lf_tiff(path, viewspos))
        return predict_disparity(model, views, viewspos, device).numpy().astype(np.float64)

    return predict


def main() -> None:
    args = parse_args()
    if args.data_root:
        data_root = Path(args.data_root).resolve()
    else:
        data_root = Path(os.environ.get("SYNLF_DATA_ROOT", REPO_ROOT / "data")).resolve()
    names = read_split(Path(args.split))
    coef = load_calibration(args.calibration)
    prediction_root = Path(args.predictions).resolve() if args.predictions else None
    predict = run_model(args) if prediction_root is None else None
    save_root = Path(args.output).resolve().parent / "predictions"

    sums = np.zeros(4, dtype=np.float64)
    valid_pixels = 0
    sample_count = 0

    for sample_id in tqdm(names, desc="Test"):
        input_path = data_root / f"{sample_id}_interp.tif"
        gt_path = data_root / f"{sample_id}_z_proj.tif"
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)

        pred_disp = (
            read_prediction(prediction_root, sample_id)
            if prediction_root is not None
            else predict(input_path)
        )
        gt_depth = np.squeeze(tiff.imread(gt_path)).astype(np.float64)
        if pred_disp.shape != gt_depth.shape:
            raise ValueError(
                f"Shape mismatch for {sample_id}: prediction {pred_disp.shape}, GT {gt_depth.shape}"
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            pred_depth = coef[0] / (pred_disp - coef[2]) - coef[1]
        mask = build_mask(gt_depth, pred_depth, coef)
        gt = gt_depth[mask]
        pred = pred_depth[mask]
        error = np.abs(pred - gt)
        ratio = np.maximum(pred / np.maximum(gt, 1e-6), gt / np.maximum(pred, 1e-6))
        sums += (
            error.sum(),
            np.square(error).sum(),
            (error / np.maximum(gt, 1e-6)).sum(),
            (ratio < 1.25).sum(),
        )
        valid_pixels += int(mask.sum())
        sample_count += 1

        if args.save_predictions and prediction_root is None:
            output_path = save_root / f"{sample_id}_pred_disp.tif"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tiff.imwrite(output_path, pred_disp.astype(np.float32))

    if sample_count != 142:
        raise RuntimeError(f"Expected 142 test samples, got {sample_count}")
    if valid_pixels == 0:
        raise RuntimeError("No valid pixels found")

    metrics = {
        "protocol": "SynLF 142-sample evaluation",
        "samples": sample_count,
        "valid_pixels": valid_pixels,
        "disp_range": [DISP_MIN, DISP_MAX],
        "erosion_radius": EROSION_RADIUS,
        "aggregation": "pixel-wise",
        "MAE_mm": float(sums[0] / valid_pixels),
        "RMSE_mm": float(math.sqrt(sums[1] / valid_pixels)),
        "AbsRel": float(sums[2] / valid_pixels),
        "delta1": float(sums[3] / valid_pixels),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
