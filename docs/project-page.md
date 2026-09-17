# SynLF Project Page

Project page for:

> **SynLF: Zero-Shot Metric Depth from Light Field Cameras via Physics-Grounded Synthesis**
> Zhexuan Cao, Yuduo Guo, Peisheng Ding, Zhan Shi, Hui Qiao
> ECCV 2026

## Local preview

Run a static server from the repository root:

```bash
python -m http.server 8000
```

Then open `http://127.0.0.1:8000/`.

## GitHub Pages

Publish from the `main` branch and repository root in **Settings -> Pages**. The expected project URL is:

`https://caozx169.github.io/SynLF/`

The repository also contains the release code. See the root README for data,
weights, training and evaluation commands.

## Interactive reconstruction

The page includes synchronized Three.js mesh comparisons for four captures from the paper:

| Scene | Source group | Browser asset key |
| --- | --- | --- |
| 00007 | 20251222/real | 00007 |
| 00009 | 20251222/real | 00009 |
| 00079 | 20251222/real | 00079 |
| 00016 | 20251031/real2 | 20251031-real2-00016 |

Assets are loaded when a scene is selected. Meshes and scene metadata live under
`static/meshes/<asset-key>/`.

To regenerate a mesh pair, use the paper's nine-view data, calibration and
predictions. The exporter requires Python, NumPy, SciPy and tifffile:

```powershell
python tools\export_paper_mesh.py `
  --data-root data\20251222 `
  --prediction-root predictions\Results3 `
  --sample 00007 `
  --set-name real `
  --output-dir static\meshes\00007 `
  --stride 3
```

The exporter preserves GT holes and removes triangles across depth discontinuities.
Both meshes use center-view colors. Three.js is included in `static/vendor/three/`.

The `data-root` contains `iniCamPose.mat`, `coef.mat`, and the capture set.
For scene 00016, use the `20251031` root, `real2` set and `Results2` predictions.

## Attribution

The page structure is informed by the community [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template), which in turn draws from the [Nerfies project page](https://nerfies.github.io/). SynLF-specific implementation, styling, text, and media are maintained in this repository.
