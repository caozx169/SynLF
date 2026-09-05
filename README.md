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

The repository currently contains the project page and paper assets only. The code and model release are still in preparation and are labeled accordingly on the page.

## Interactive reconstruction

The page includes synchronized Three.js mesh comparisons for four selected captures from the paper-time evaluation data:

| Scene | Source group | Browser asset key |
| --- | --- | --- |
| 00007 | 20251222/real | 00007 |
| 00009 | 20251222/real | 00009 |
| 00079 | 20251222/real | 00079 |
| 00016 | 20251031/real2 | 20251031-real2-00016 |

Assets are loaded only when a scene is selected. Browser assets live under `static/meshes/<asset-key>/`; each manifest records its source group, calibration values, source hashes, triangulation policy, and per-scene metrics. These are selected qualitative examples, not a new benchmark or a complete dataset release. Predictions in missing-GT regions are not quantitatively validated.

Regenerate a mesh pair with Python, NumPy, SciPy, and tifffile installed. The paths below are placeholders for the matching archived data and predictions:

```powershell
python tools\export_paper_mesh.py `
  --data-root data\20251222 `
  --prediction-root predictions\Results3 `
  --sample 00007 `
  --set-name real `
  --output-dir static\meshes\00007 `
  --stride 3
```

The export preserves structured-light holes, applies no GT validity mask to the SynLF prediction, and rejects triangles that cross depth discontinuities. Metric depth comes from the paper-time `coef.mat`; the shared center-view appearance is robustly normalized for browser display. The Three.js runtime is pinned and vendored under `static/vendor/three/` so the viewer does not depend on a third-party CDN.

The `data-root` must contain `iniCamPose.mat`, `coef.mat`, and the requested capture set. Use the calibration from the same capture group, not from another date. For scene 00016, use the `20251031` data root, `real2` set, and its corresponding `Results2` predictions. The viewer contains display meshes only; raw LF stacks, full prediction archives, and local working files are not included.

## Attribution

The page structure is informed by the community [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template), which in turn draws from the [Nerfies project page](https://nerfies.github.io/). SynLF-specific implementation, styling, text, and media are maintained in this repository.
