from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .differentiable_complementarity_free_contact_dynamics import RigidState


SH_C0 = 0.28209479177387814


def make_cube_gaussian_cloud(
    half_extents: Iterable[float],
    *,
    points_per_axis: int = 8,
    dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a tiny colored cube-shaped Gaussian cloud for viewer smoke tests."""

    if points_per_axis < 2:
        raise ValueError("points_per_axis must be at least 2.")
    hx, hy, hz = [float(v) for v in half_extents]
    xs = np.linspace(-hx, hx, points_per_axis, dtype=dtype)
    ys = np.linspace(-hy, hy, points_per_axis, dtype=dtype)
    zs = np.linspace(-hz, hz, points_per_axis, dtype=dtype)
    points = []
    colors = []
    face_specs = [
        (0, hx, (0.96, 0.28, 0.22)),
        (0, -hx, (0.22, 0.72, 0.32)),
        (1, hy, (0.22, 0.52, 0.96)),
        (1, -hy, (0.98, 0.82, 0.20)),
        (2, hz, (0.68, 0.34, 0.92)),
        (2, -hz, (0.20, 0.84, 0.88)),
    ]
    grids = (xs, ys, zs)
    for axis, value, color in face_specs:
        other_axes = [idx for idx in range(3) if idx != axis]
        for a in grids[other_axes[0]]:
            for b in grids[other_axes[1]]:
                point = np.zeros(3, dtype=dtype)
                point[axis] = value
                point[other_axes[0]] = a
                point[other_axes[1]] = b
                points.append(point)
                colors.append(color)
    return np.asarray(points, dtype=dtype), np.asarray(colors, dtype=dtype)


def make_sphere_gaussian_cloud(
    radius: float,
    *,
    num_points: int = 512,
    dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a small colored sphere Gaussian cloud for SIBR sequence checks."""

    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if num_points < 16:
        raise ValueError("num_points must be at least 16.")

    indices = np.arange(num_points, dtype=dtype)
    golden_angle = dtype(2.399963229728653)
    z = 1.0 - (2.0 * indices + 1.0) / float(num_points)
    radial = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    theta = golden_angle * indices
    normals = np.stack((radial * np.cos(theta), radial * np.sin(theta), z), axis=-1).astype(dtype)
    points = normals * dtype(radius)
    colors = np.clip(0.5 + 0.5 * normals, 0.0, 1.0).astype(dtype)
    return points, colors


def _rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    return (rgb - 0.5) / SH_C0


def write_ascii_3dgs_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    opacity: float = 0.92,
    scale: float = 0.035,
) -> None:
    """Write a minimal SIBR/3DGS-compatible ASCII PLY."""

    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3).")
    if rgb.shape != xyz.shape:
        raise ValueError("rgb must have shape (N, 3).")

    f_dc = _rgb_to_sh_dc(np.clip(rgb, 0.0, 1.0))
    opacity_logit = float(np.log(opacity / max(1.0 - opacity, 1e-8)))
    log_scale = float(np.log(scale))
    f_rest = [0.0] * 45

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {xyz.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    header.extend(f"property float f_rest_{idx}" for idx in range(45))
    header.extend(
        [
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
        ]
    )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(header))
        handle.write("\n")
        for point, dc in zip(xyz, f_dc):
            row = [
                point[0],
                point[1],
                point[2],
                0.0,
                0.0,
                0.0,
                dc[0],
                dc[1],
                dc[2],
                *f_rest,
                opacity_logit,
                log_scale,
                log_scale,
                log_scale,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
            handle.write(" ".join(f"{float(value):.8f}" for value in row))
            handle.write("\n")


def _state_position(state: RigidState) -> np.ndarray:
    position = state.position
    if isinstance(position, torch.Tensor):
        return position.detach().cpu().numpy().astype(np.float32)
    return np.asarray(position, dtype=np.float32)


def export_sibr_sequence(
    output_dir: Path,
    states: Sequence[RigidState],
    local_xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    fps: int = 60,
    iteration: int = 0,
) -> Path:
    """Export frame-local Gaussian PLY files for a modified SIBR viewer path."""

    output_dir = Path(output_dir)
    frames_dir = output_dir / "sibr_sequence" / "frames"
    frame_entries = []
    for frame_idx, state in enumerate(states):
        frame_root = frames_dir / f"frame_{frame_idx:06d}"
        ply_path = frame_root / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        world_xyz = np.asarray(local_xyz, dtype=np.float32) + _state_position(state)[None, :]
        write_ascii_3dgs_ply(ply_path, world_xyz, rgb)
        frame_entries.append(
            {
                "frame_index": frame_idx,
                "time": frame_idx / float(fps),
                "model_path": str(frame_root),
                "ply_path": str(ply_path),
            }
        )

    manifest = {
        "format": "contactwm_stage2_sibr_sequence_v1",
        "fps": int(fps),
        "iteration": int(iteration),
        "num_frames": len(frame_entries),
        "viewer_note": (
            "Each frame is laid out like a tiny 3DGS model path. A SIBR viewer "
            "adapter can switch model_path per frame or load the listed PLYs "
            "through a time slider."
        ),
        "frames": frame_entries,
    }
    manifest_path = output_dir / "sibr_sequence" / "sequence_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path
