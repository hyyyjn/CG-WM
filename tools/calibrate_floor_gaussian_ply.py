"""Convert a learned visual floor Gaussian field into a thin contact surface."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_ply", required=True, type=Path)
    parser.add_argument("--output_ply", required=True, type=Path)
    parser.add_argument("--max_primitives", type=int, default=1024)
    parser.add_argument("--xy_extent", type=float, default=1.0)
    parser.add_argument("--gaussian_scale", type=float, default=0.01,
                        help="Gaussian s; paper collision radius is r=2s.")
    parser.add_argument("--surface_z", type=float, default=0.0)
    parser.add_argument("--regular_grid_size", type=int, default=0,
                        help="If positive, resample the learned floor appearance onto an N x N Gaussian grid.")
    parser.add_argument("--plane_sphere_radius", type=float, default=0.0,
                        help="If positive, emit one large sphere tangent to the floor; useful for stable pairwise normals.")
    args = parser.parse_args()

    vertex = PlyData.read(args.input_ply)["vertex"].data
    names = vertex.dtype.names or ()
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"}
    if not required.issubset(names):
        raise ValueError(f"Missing PLY fields: {sorted(required - set(names))}")

    opacity = sigmoid(np.asarray(vertex["opacity"], dtype=np.float64))
    foreground = (
        sigmoid(np.asarray(vertex["foreground_logit"], dtype=np.float64))
        if "foreground_logit" in names else np.ones_like(opacity)
    )
    xy_ok = (np.abs(vertex["x"]) <= args.xy_extent) & (np.abs(vertex["y"]) <= args.xy_extent)
    score = opacity * foreground
    candidates = np.flatnonzero(xy_ok)
    if candidates.size == 0:
        raise ValueError("No learned floor Gaussians inside requested XY extent.")
    order = candidates[np.argsort(score[candidates])[::-1]]
    selected = order[: min(args.max_primitives, order.size)]
    out = vertex[selected].copy()

    if args.regular_grid_size > 0:
        grid_size = int(args.regular_grid_size)
        if grid_size < 2:
            raise ValueError("regular_grid_size must be at least 2")
        template = out[np.argmax(score[selected]) : np.argmax(score[selected]) + 1]
        out = np.repeat(template, grid_size * grid_size).copy()
        coords = np.linspace(-float(args.xy_extent), float(args.xy_extent), grid_size, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(coords, coords, indexing="xy")
        out["x"] = grid_x.reshape(-1)
        out["y"] = grid_y.reshape(-1)
        # A textureless floor should share its robust learned DC appearance.
        for field in ("f_dc_0", "f_dc_1", "f_dc_2"):
            if field in names:
                out[field] = np.median(vertex[field][selected])
        for field in (name for name in names if name.startswith("f_rest_")):
            out[field] = 0.0

    if args.plane_sphere_radius > 0.0:
        out = out[:1].copy()
        out["x"], out["y"] = 0.0, 0.0
        out["z"] = float(args.surface_z) - float(args.plane_sphere_radius)
        # PLY stores s while the paper collision radius is r=2s.
        args.gaussian_scale = float(args.plane_sphere_radius) * 0.5

    collision_radius = 2.0 * float(args.gaussian_scale)
    if args.plane_sphere_radius <= 0.0:
        out["z"] = float(args.surface_z) - collision_radius
    log_s = np.log(float(args.gaussian_scale))
    for field in ("scale_0", "scale_1", "scale_2"):
        out[field] = log_s
    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(names):
        out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = 1.0, 0.0, 0.0, 0.0
    if "foreground_logit" in names:
        out["foreground_logit"] = 10.0
    out["opacity"] = 10.0

    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(args.output_ply)
    metadata = args.output_ply.with_suffix(".spherical.json")
    metadata.write_text(
        '{\n  "representation": "isotropic_spherical_gaussian",\n'
        '  "ply_scale_semantics": "gaussian_standard_deviation_s",\n'
        '  "collision_radius_formula": "r=2s",\n'
        '  "scale_channels_equal": true,\n  "rotation": "identity_frozen",\n'
        '  "source": "learned_floor_xy_projected_to_calibrated_contact_plane",\n  "version": 1\n}\n'
    )
    print(f"wrote {len(out)} calibrated floor Gaussians to {args.output_ply}")


if __name__ == "__main__":
    main()
