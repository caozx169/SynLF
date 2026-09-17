# SynLF

**SynLF: Zero-Shot Metric Depth from Light Field Cameras via Physics-Grounded
Synthesis** — ECCV 2026.

[Project page](https://caozx169.github.io/SynLF/) ·
[Paper](static/pdfs/SynLF_ECCV2026.pdf)

PG-LF synthesizes light-field training data from Hypersim RGB-D images.
VisDepth estimates depth from nine light-field views.

## Install

Linux with an NVIDIA GPU is required.

```bash
conda create -n synlf python=3.10 -y
conda activate synlf
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## Data and weights

Download [data and weights from Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/b345f47c4c454c4bbb32/).
Place the shared `data/` and `checkpoints/` folders in the repository root:

```text
data/
  00001_interp.tif
  00001_z_proj.tif
  ...
  00173_interp.tif
  00173_z_proj.tif
checkpoints/
  visdepth.ckpt
configs/
  inference.yaml
  data/synlf.yaml
assets/splits/
  test.txt
```

Each `*_interp.tif` contains `(81,768,1024)` grayscale `uint8` images in row-major
`(v,u)` angular order. The reader selects indices
`[22,31,38,39,40,41,42,49,58]` and divides by 255. Paired `*_z_proj.tif` files
contain depth in millimeters. All captures use the shared calibration in
`configs/data/synlf.yaml`: `disparity = a/(depth+b)+c`.

The checkpoint includes the monocular prior. YAML configs and the test split live
in the repository; data and weights are ignored by Git.

## Run

```bash
# Infer all released real captures; save disparity, depth and PNG previews.
python src/infer.py

# Evaluate the 142-sample test split.
python src/test.py

# Evaluate the four HCI or Inria scenes used in the paper.
python src/eval.py --dataset hci
python src/eval.py --dataset inria

# Train with Hypersim: eight GPUs, batch size 16, 8,000 optimizer steps.
python src/train.py

# One GPU with the same effective batch size.
python src/train.py 'training.devices=[0]' training.accumulate_grad_batches=8
```

`infer.py --format tiff --input INPUT --coef A B C` also accepts an individual
81-view or normalized nine-view TIFF. Use `--help` for each entry point.

For training, place Hypersim in `data/hypersim/hypersim/` and the
[Depth Anything V2 Small initialization](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth)
in `checkpoints/depth_anything_v2/depth_anything_v2_vits.pth`.
Optional benchmark roots are `data/hci/` and `data/inria/DLFD/`.

The training configuration uses 69,420 Hypersim images, 768x768 crops and a
disparity search range of `[-9,20]`. The sample exclusion list is included in
`assets/hypersim_exclusions.txt`.

## Evaluation

The dataset contains 173 captures. Quantitative evaluation uses the 142-sample
split in `assets/splits/test.txt`. Metrics are computed over valid
GT pixels with disparity in `[-4,4]`, using 7x7 mask erosion and pixel-wise
aggregation.

The release includes all 81 views, updated preprocessing and shared calibration.
Both rows below use the same model weights.

| Data version | MAE mm | RMSE mm | AbsRel % | delta1 % |
| --- | ---: | ---: | ---: | ---: |
| Release (81 views) | 31.7143 | 147.5047 | 1.4980 | 98.8482 |
| Paper | 30.0512 | 144.3356 | 1.3829 | 98.7714 |

VisDepth is trained on Hypersim, with checkpoint selection based on real-data
validation. The released data use photometric calibration refined with reference
depths from a subset of the captures. Field-offset correction covers the 37
central views, including all nine model inputs; the remaining views retain the
extracted images.

## Third-party code

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `LICENSES/`.
FoundationStereo-derived code is restricted to non-commercial research;
Depth Anything V2 Small uses Apache 2.0.

Project-page maintenance is documented in [docs/project-page.md](docs/project-page.md).
