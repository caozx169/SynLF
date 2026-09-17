import argparse
import json
from pathlib import Path

import numpy as np
import tifffile as tiff
import torch
from tqdm import tqdm

from dataset.hci import HCI_LFDepthDataset
from dataset.inria import InriaLFDataset
from config_utils import REPO_ROOT
from model_runtime import (
    default_checkpoint_path,
    default_config_path,
    default_data_root,
    load_visdepth,
    output_disparity,
)


PAPER_SCENES = {
    "hci": ("boxes", "cotton", "dino", "sideboard"),
    "inria": ("Flying_dice_dense", "Furniture_dense", "Pinenuts_blue_dense", "Toy_friends_dense"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VisDepth on public light-field benchmarks.")
    parser.add_argument("--dataset", choices=("hci", "inria"), required=True)
    parser.add_argument("--data-root", default=None, help="Dataset root; inferred from SYNLF_DATA_ROOT by default.")
    parser.add_argument(
        "--checkpoint",
        default=str(default_checkpoint_path()),
    )
    parser.add_argument("--config", default=str(default_config_path()), help="Model YAML; defaults to configs/inference.yaml.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default=str(REPO_ROOT / "outputs" / "eval"))
    parser.add_argument("--all-scenes", action="store_true", help="Evaluate every discovered scene.")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def build_dataset(name: str, root: Path, cfg, scenes):
    common = dict(
        rootpath=str(root),
        mindisp=float(cfg.mindisp),
        maxdisp=float(cfg.maxdisp),
        viewspos=cfg.viewspos,
        cropsize=cfg.cropsize,
        ispreload=False,
        randomrotaug=False,
        randomcrop=False,
        randomdisp=False,
        randomview=False,
        randomresize_aug=False,
        randomnoise=False,
        randomblur=False,
        randomgray=False,
        image_mode="L",
        istrain=False,
        scenes=scenes,
    )
    return HCI_LFDepthDataset(**common) if name == "hci" else InriaLFDataset(**common)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("VisDepth evaluation requires a CUDA GPU.")
    device = torch.device(f"cuda:{args.device}")
    model, cfg, _, config_path = load_visdepth(args.checkpoint, args.config, device)

    if args.data_root:
        data_root = Path(args.data_root).resolve()
    else:
        base = default_data_root().resolve()
        data_root = base / ("hci" if args.dataset == "hci" else "inria/DLFD")
    scenes = None if args.all_scenes else PAPER_SCENES[args.dataset]
    dataset = build_dataset(args.dataset, data_root, cfg, scenes)
    output_root = Path(args.output).resolve() / args.dataset

    per_scene = []
    with torch.inference_mode():
        for sample in tqdm(dataset, desc=f"Evaluate {args.dataset}"):
            viewimgs = sample["viewimgs"].unsqueeze(0).to(device)
            viewspos = sample["viewspos"].unsqueeze(0).to(device)
            pred = output_disparity(model(viewimgs, viewspos)).numpy()
            gt = sample["disp"].squeeze().numpy()
            mask = sample["masks"].squeeze().numpy().astype(bool)
            mask &= np.isfinite(gt) & np.isfinite(pred)
            if not mask.any():
                raise RuntimeError(f"No valid pixels for scene {sample['name']}")

            error = np.abs(pred[mask] - gt[mask])
            metrics = {
                "name": sample["name"],
                "MSE_x100": float(np.mean(np.square(error)) * 100.0),
                "BadPix_0.07_percent": float(np.mean(error > 0.07) * 100.0),
                "valid_pixels": int(mask.sum()),
            }
            per_scene.append(metrics)

            if args.save_predictions:
                output_root.mkdir(parents=True, exist_ok=True)
                tiff.imwrite(output_root / f"{sample['name']}_pred_disp.tif", pred.astype(np.float32))

    summary = {
        "dataset": args.dataset,
        "protocol": "all-scenes" if args.all_scenes else "paper-four-scene",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(config_path),
        "scenes": len(per_scene),
        "MSE_x100": float(np.mean([item["MSE_x100"] for item in per_scene])),
        "BadPix_0.07_percent": float(np.mean([item["BadPix_0.07_percent"] for item in per_scene])),
        "per_scene": per_scene,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
