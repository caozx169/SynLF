from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile as tif
from scipy.io import loadmat
from scipy.ndimage import minimum_filter


PAPER_DISPARITY_RANGE = (-4.0, 4.0)
PAPER_MASK_EROSION_RADIUS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_calibration(data_root: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    pose = loadmat(data_root / "iniCamPose.mat")
    coef_mat = loadmat(data_root / "coef.mat")
    intrinsic = np.asarray(pose["K"], dtype=np.float64).reshape(3, 3)
    coef = tuple(float(np.asarray(coef_mat[key]).squeeze()) for key in ("a", "b", "c"))
    return intrinsic, coef


def disparity_to_depth(disparity: np.ndarray, coef: tuple[float, float, float]) -> np.ndarray:
    a, b, c = coef
    return a / (disparity - c) - b


def depth_to_disparity(depth: np.ndarray, coef: tuple[float, float, float]) -> np.ndarray:
    a, b, c = coef
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / (depth + b) + c


def paper_metrics(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    coef: tuple[float, float, float],
) -> dict[str, float | int]:
    gt_disparity = depth_to_disparity(gt_depth, coef)
    raw_mask = (
        np.isfinite(gt_disparity)
        & (gt_disparity >= PAPER_DISPARITY_RANGE[0])
        & (gt_disparity <= PAPER_DISPARITY_RANGE[1])
    )
    kernel_size = 2 * PAPER_MASK_EROSION_RADIUS + 1
    eroded = minimum_filter(
        raw_mask.astype(np.uint8),
        size=kernel_size,
        mode="constant",
        cval=1,
    ).astype(bool)
    valid = eroded & np.isfinite(gt_depth) & np.isfinite(pred_depth) & (gt_depth > 1e-6)
    gt = gt_depth[valid].astype(np.float64)
    pred = pred_depth[valid].astype(np.float64)
    absolute_error = np.abs(pred - gt)
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "valid_pixels": int(valid.sum()),
        "mae_mm": float(absolute_error.mean()),
        "rmse_mm": float(np.sqrt(np.mean(absolute_error**2))),
        "abs_rel_percent": float(100.0 * np.mean(absolute_error / gt)),
        "delta1_percent": float(100.0 * np.mean(ratio < 1.25)),
    }


def make_mesh(
    depth_mm: np.ndarray,
    center_view: np.ndarray,
    intrinsic: np.ndarray,
    stride: int,
    depth_range_m: tuple[float, float],
    max_depth_edge_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth_m = depth_mm[::stride, ::stride].astype(np.float64) / 1000.0
    texture = center_view[::stride, ::stride]
    rows, cols = np.mgrid[0 : depth_mm.shape[0] : stride, 0 : depth_mm.shape[1] : stride]
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= depth_range_m[0])
        & (depth_m <= depth_range_m[1])
    )

    z = depth_m[valid]
    x = (cols[valid] - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = -(rows[valid] - intrinsic[1, 2]) * z / intrinsic[1, 1]
    intensity = np.clip(texture[valid], 0.0, 1.0)
    rgb = np.repeat(np.rint(intensity[:, None] * 255.0).astype(np.uint8), 3, axis=1)

    vertices = np.empty(
        z.size,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"] = x
    vertices["y"] = y
    vertices["z"] = -z
    vertices["red"] = rgb[:, 0]
    vertices["green"] = rgb[:, 1]
    vertices["blue"] = rgb[:, 2]

    vertex_index = np.full(valid.shape, -1, dtype=np.int32)
    vertex_index[valid] = np.arange(vertices.size, dtype=np.int32)
    tl = vertex_index[:-1, :-1]
    tr = vertex_index[:-1, 1:]
    bl = vertex_index[1:, :-1]
    br = vertex_index[1:, 1:]
    z_tl = depth_m[:-1, :-1]
    z_tr = depth_m[:-1, 1:]
    z_bl = depth_m[1:, :-1]
    z_br = depth_m[1:, 1:]

    def triangle(a: np.ndarray, b: np.ndarray, c: np.ndarray, za: np.ndarray, zb: np.ndarray, zc: np.ndarray) -> np.ndarray:
        connected = (a >= 0) & (b >= 0) & (c >= 0)
        local_range = np.maximum(np.maximum(za, zb), zc) - np.minimum(np.minimum(za, zb), zc)
        keep = connected & (local_range <= max_depth_edge_m)
        return np.column_stack((a[keep], b[keep], c[keep])).astype(np.int32, copy=False)

    faces = np.concatenate(
        (
            triangle(tl, bl, tr, z_tl, z_bl, z_tr),
            triangle(tr, bl, br, z_tr, z_bl, z_br),
        ),
        axis=0,
    )
    return vertices, faces


def write_binary_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    comment: str,
) -> None:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"comment {comment}",
            f"element vertex {vertices.size}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            f"element face {faces.shape[0]}",
            "property list uchar int vertex_indices",
            "end_header",
            "",
        ]
    ).encode("ascii")
    face_records = np.empty(
        faces.shape[0],
        dtype=[("count", "u1"), ("i0", "<i4"), ("i1", "<i4"), ("i2", "<i4")],
    )
    face_records["count"] = 3
    face_records["i0"] = faces[:, 0]
    face_records["i1"] = faces[:, 1]
    face_records["i2"] = faces[:, 2]
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)
        face_records.tofile(stream)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a frozen ECCV 2026 SynLF result as browser-ready metric meshes."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--sample", default="00007")
    parser.add_argument("--set-name", default="real")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--depth-min-m", type=float, default=0.5)
    parser.add_argument("--depth-max-m", type=float, default=2.5)
    parser.add_argument("--max-depth-edge-mm", type=float, default=50.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = args.data_root.resolve()
    prediction_root = args.prediction_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_root = data_root / args.set_name / args.sample
    input_path = Path(f"{sample_root}_interp.tif")
    gt_path = Path(f"{sample_root}_z_proj.tif")
    prediction_path = prediction_root / f"{args.sample}.tif"
    calibration_path = data_root / "iniCamPose.mat"
    coefficient_path = data_root / "coef.mat"
    required = [input_path, gt_path, prediction_path, calibration_path, coefficient_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required source assets: {missing}")

    view_stack = tif.imread(input_path)
    gt_depth = np.asarray(tif.imread(gt_path), dtype=np.float32).squeeze()
    prediction_disparity = np.asarray(tif.imread(prediction_path), dtype=np.float32).squeeze()
    if view_stack.shape != (9, 768, 1024):
        raise ValueError(f"Expected 9x768x1024 LF stack, got {view_stack.shape}")
    if gt_depth.shape != (768, 1024) or prediction_disparity.shape != (768, 1024):
        raise ValueError(
            f"Expected 768x1024 depth/disparity, got {gt_depth.shape} and {prediction_disparity.shape}"
        )

    intrinsic, coef = load_calibration(data_root)
    prediction_depth = disparity_to_depth(prediction_disparity, coef)
    center_view = np.asarray(view_stack[4], dtype=np.float32)
    finite_intensity = center_view[np.isfinite(center_view)]
    appearance_min, appearance_max = np.percentile(finite_intensity, (0.5, 99.5))
    center_view_display = np.clip(
        (center_view - appearance_min) / max(appearance_max - appearance_min, 1e-6),
        0.0,
        1.0,
    )
    depth_range_m = (args.depth_min_m, args.depth_max_m)
    max_depth_edge_m = args.max_depth_edge_mm / 1000.0

    gt_vertices, gt_faces = make_mesh(
        gt_depth,
        center_view_display,
        intrinsic,
        args.stride,
        depth_range_m,
        max_depth_edge_m,
    )
    prediction_vertices, prediction_faces = make_mesh(
        prediction_depth,
        center_view_display,
        intrinsic,
        args.stride,
        depth_range_m,
        max_depth_edge_m,
    )
    gt_output = output_dir / "structured-light-gt.ply"
    prediction_output = output_dir / "synlf.ply"
    write_binary_ply(
        gt_output,
        gt_vertices,
        gt_faces,
        "Structured-light GT metric mesh in the LF center-view frame",
    )
    write_binary_ply(
        prediction_output,
        prediction_vertices,
        prediction_faces,
        "SynLF zero-shot metric mesh in the LF center-view frame",
    )

    gt_disparity = depth_to_disparity(gt_depth, coef)
    gt_range_mask = (
        np.isfinite(gt_disparity)
        & (gt_disparity >= PAPER_DISPARITY_RANGE[0])
        & (gt_disparity <= PAPER_DISPARITY_RANGE[1])
    )
    manifest = {
        "schema_version": 2,
        "scene": args.sample,
        "dataset_snapshot": f"ECCV 2026 paper evaluation / {data_root.name}/{args.set_name}",
        "source_group": {
            "data_root": data_root.name,
            "set_name": args.set_name,
        },
        "selection": {
            "gt_range_coverage_percent": float(100.0 * gt_range_mask.mean()),
            "rationale": "Selected qualitative example from the paper-time evaluation data.",
        },
        "coordinate_frame": "LF center view; x right, y up, camera looks along -z; units meters",
        "center_view_index": 4,
        "calibration": {
            "intrinsic_matrix": intrinsic.tolist(),
            "disparity_to_depth_mm": {"a": coef[0], "b": coef[1], "c": coef[2]},
        },
        "display_processing": {
            "stride": args.stride,
            "depth_range_m": list(depth_range_m),
            "max_depth_edge_mm": args.max_depth_edge_mm,
            "gt_invalid_pixels": "omitted; structured-light holes are preserved",
            "prediction_mask": "independent finite/range mask; no GT validity mask applied",
            "triangulation": "image-grid triangles; faces crossing the depth-edge threshold are removed",
            "color": "LF center-view grayscale, shared 0.5-99.5 percentile display normalization",
            "appearance_input_range": [float(appearance_min), float(appearance_max)],
        },
        "published_protocol_scene_metrics": paper_metrics(gt_depth, prediction_depth, coef),
        "meshes": {
            "structured_light_gt": {
                "file": gt_output.name,
                "vertices": int(gt_vertices.size),
                "triangles": int(gt_faces.shape[0]),
                "sha256": sha256(gt_output),
            },
            "synlf": {
                "file": prediction_output.name,
                "vertices": int(prediction_vertices.size),
                "triangles": int(prediction_faces.shape[0]),
                "sha256": sha256(prediction_output),
            },
        },
        "source_hashes": {
            "input_lf_stack": sha256(input_path),
            "projected_structured_light_gt": sha256(gt_path),
            "synlf_v11_disparity": sha256(prediction_path),
            "disparity_depth_coefficients": sha256(coefficient_path),
            "center_view_calibration": sha256(calibration_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest["meshes"]}, indent=2))


if __name__ == "__main__":
    main()
