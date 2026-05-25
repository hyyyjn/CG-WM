from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-fast audit for using a Stage 1 PLY as a Stage 2 physical object."
    )
    parser.add_argument("--stage1_ply", required=True, type=Path)
    parser.add_argument("--object_asset", required=True, type=Path)
    parser.add_argument("--foreground_threshold", default=0.55, type=float)
    parser.add_argument("--opacity_threshold", default=0.0, type=float)
    parser.add_argument("--max_extent_ratio", default=3.0, type=float)
    parser.add_argument("--min_foreground_primitives", default=64, type=int)
    parser.add_argument("--output_json", default=None, type=Path)
    return parser.parse_args()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_vertices(path: Path):
    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plyfile is required. Install it in the runtime used for the demo.") from exc

    ply = PlyData.read(str(path))
    vertices = ply["vertex"].data
    names = vertices.dtype.names or ()
    for name in ("x", "y", "z"):
        if name not in names:
            raise ValueError(f"{path} is missing required field '{name}'.")
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1).astype(np.float32)
    return vertices, names, xyz


def extent_stats(points: np.ndarray) -> dict:
    if points.shape[0] == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "extent": None,
            "max_extent": None,
        }
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extent = maxs - mins
    return {
        "count": int(points.shape[0]),
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "mean": points.mean(axis=0).tolist(),
        "extent": extent.tolist(),
        "max_extent": float(extent.max()),
    }


def main() -> None:
    args = parse_args()
    asset = read_json(args.object_asset)
    vertices, names, xyz = load_vertices(args.stage1_ply)

    normalization = asset.get("normalization", {})
    bbox_min = np.asarray(normalization.get("bbox_min"), dtype=np.float32)
    bbox_max = np.asarray(normalization.get("bbox_max"), dtype=np.float32)
    if bbox_min.shape != (3,) or bbox_max.shape != (3,):
        raise ValueError("--object_asset must define normalization.bbox_min/bbox_max 3-vectors.")
    expected_extent = np.maximum((bbox_max - bbox_min) * float(normalization.get("scale", 1.0)), 1e-8)

    if "foreground_logit" not in names:
        raise ValueError("Stage 1 PLY has no foreground_logit field; cannot audit object-only geometry.")
    foreground = 1.0 / (1.0 + np.exp(-np.asarray(vertices["foreground_logit"]).astype(np.float32)))
    mask = foreground >= float(args.foreground_threshold)

    opacity = None
    if "opacity" in names:
        opacity = 1.0 / (1.0 + np.exp(-np.asarray(vertices["opacity"]).astype(np.float32)))
        if float(args.opacity_threshold) > 0.0:
            mask &= opacity >= float(args.opacity_threshold)

    all_stats = extent_stats(xyz)
    fg_stats = extent_stats(xyz[mask])
    ratio = None
    if fg_stats["extent"] is not None:
        ratio = (np.asarray(fg_stats["extent"], dtype=np.float32) / expected_extent).tolist()

    summary = {
        "stage1_ply": str(args.stage1_ply.resolve()),
        "object_asset": str(args.object_asset.resolve()),
        "foreground_threshold": float(args.foreground_threshold),
        "opacity_threshold": float(args.opacity_threshold),
        "expected_extent": expected_extent.tolist(),
        "all_gaussians": all_stats,
        "foreground_gaussians": fg_stats,
        "foreground_extent_ratio": ratio,
        "foreground_score": {
            "min": float(foreground.min()),
            "median": float(np.median(foreground)),
            "max": float(foreground.max()),
        },
    }
    if opacity is not None:
        summary["opacity"] = {
            "min": float(opacity.min()),
            "median": float(np.median(opacity)),
            "max": float(opacity.max()),
        }

    failures = []
    if fg_stats["count"] < int(args.min_foreground_primitives):
        failures.append(
            f"only {fg_stats['count']} foreground primitives pass the threshold; "
            f"need at least {int(args.min_foreground_primitives)}"
        )
    if ratio is not None and max(ratio) > float(args.max_extent_ratio):
        failures.append(
            "foreground bbox is too large for the metric object: "
            f"ratio={ratio}, max allowed={float(args.max_extent_ratio)}"
        )
    summary["passed"] = not failures
    summary["failures"] = failures

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
