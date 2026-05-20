"""Smoke test: cube falling toward a floor with externally supplied query points.

This test is intentionally focused on collision detection, not on learning or
full rigid-body dynamics. It builds a cube-shaped spherical-Gaussian collision
proxy, synthesizes a falling cube trajectory, creates world-space floor query
points outside the collision module, and evaluates the general
evaluate_gaussian_union_sdf -> aggregate_gaussian_union_contacts path.

Optional:
  Pass --trajectory_json from tools/generate_mujoco_fall_dataset.py to evaluate
  the same collision code on a MuJoCo-generated cube trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_initiailization.stage2.differentiable_collision_detection import (
    aggregate_gaussian_union_contacts,
    evaluate_gaussian_union_sdf,
)

OUT_DIR = REPO_ROOT / "gaussian_initiailization" / "stage2" / "_outputs" / "cube_floor_collision"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate general Gaussian-union collision detection on a cube/floor scenario."
    )
    parser.add_argument("--trajectory_json", default=None, type=Path)
    parser.add_argument("--output_dir", default=OUT_DIR, type=Path)
    parser.add_argument("--frames", default=120, type=int)
    parser.add_argument("--fps", default=60, type=float)
    parser.add_argument("--half_extent", default=0.10, type=float)
    parser.add_argument("--drop_height", default=0.65, type=float)
    parser.add_argument("--gravity", default=-9.81, type=float)
    parser.add_argument("--tilt_deg", default="18 -11 23", type=str)
    parser.add_argument("--proxy_resolution", default=5, type=int)
    parser.add_argument("--query_resolution", default=17, type=int)
    parser.add_argument("--query_scale", default=1.45, type=float)
    parser.add_argument("--smooth_min_temperature", default=1.5e-2, type=float)
    parser.add_argument("--contact_softness", default=1.5e-3, type=float)
    parser.add_argument("--inside_penalty", default=0.03, type=float)
    parser.add_argument("--inside_sharpness", default=80.0, type=float)
    parser.add_argument("--num_contact_patches", default=4, type=int)
    return parser.parse_args()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def euler_xyz_to_quat_wxyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


def quat_wxyz_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
    w, x, y, z = quat.tolist()
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_cube_proxy_local(half_extent: float, resolution: int) -> tuple[torch.Tensor, torch.Tensor]:
    if half_extent <= 0.0:
        raise ValueError("half_extent must be positive.")
    if resolution < 2:
        raise ValueError("proxy_resolution must be at least 2.")

    coords = torch.linspace(-half_extent, half_extent, int(resolution), dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1).reshape(-1, 3)
    spacing = float(2.0 * half_extent / float(resolution - 1))
    radii = torch.full((grid.shape[0],), spacing * 0.62, dtype=torch.float32)
    return grid, radii


def make_floor_query_points_world(
    center_xy: np.ndarray,
    *,
    half_extent: float,
    resolution: int,
    query_scale: float,
) -> torch.Tensor:
    if resolution < 2:
        raise ValueError("query_resolution must be at least 2.")
    radius = float(half_extent) * math.sqrt(3.0) * float(query_scale)
    xs = torch.linspace(float(center_xy[0]) - radius, float(center_xy[0]) + radius, int(resolution))
    ys = torch.linspace(float(center_xy[1]) - radius, float(center_xy[1]) + radius, int(resolution))
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    zz = torch.zeros_like(xx)
    return torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)


def transform_points(local_points: torch.Tensor, position: np.ndarray, quat_wxyz: np.ndarray) -> torch.Tensor:
    rot = torch.as_tensor(quat_wxyz_to_matrix_np(quat_wxyz), dtype=local_points.dtype, device=local_points.device)
    pos = torch.as_tensor(position, dtype=local_points.dtype, device=local_points.device)
    return local_points @ rot.T + pos


def exact_box_sdf_world(
    query_points_world: torch.Tensor,
    position: np.ndarray,
    quat_wxyz: np.ndarray,
    half_extent: float,
) -> torch.Tensor:
    rot = torch.as_tensor(quat_wxyz_to_matrix_np(quat_wxyz), dtype=query_points_world.dtype, device=query_points_world.device)
    pos = torch.as_tensor(position, dtype=query_points_world.dtype, device=query_points_world.device)
    local = (query_points_world - pos) @ rot
    q = torch.abs(local) - float(half_extent)
    outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
    inside = torch.clamp(torch.max(q, dim=-1).values, max=0.0)
    return outside + inside


def lowest_corner_z_offset(half_extent: float, quat_wxyz: np.ndarray) -> float:
    corners = np.array(
        [
            [sx * half_extent, sy * half_extent, sz * half_extent]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    rot = quat_wxyz_to_matrix_np(quat_wxyz)
    return float((corners @ rot.T)[:, 2].min())


def synthesize_falling_cube_trajectory(args: argparse.Namespace) -> list[dict]:
    tilt = [float(value) for value in str(args.tilt_deg).split()]
    if len(tilt) != 3:
        raise ValueError("--tilt_deg expects three Euler XYZ values in degrees.")
    quat = euler_xyz_to_quat_wxyz(*(math.radians(value) for value in tilt))
    rest_z = -lowest_corner_z_offset(float(args.half_extent), quat)
    dt = 1.0 / float(args.fps)

    states = []
    for frame_idx in range(int(args.frames)):
        time = frame_idx * dt
        z_free = float(args.drop_height) + 0.5 * float(args.gravity) * time * time
        hit_floor = z_free <= rest_z
        z = max(z_free, rest_z)
        vz = float(args.gravity) * time if not hit_floor else 0.0
        states.append(
            {
                "frame_index": frame_idx,
                "time": time,
                "position": [0.0, 0.0, z],
                "quaternion_wxyz": quat.tolist(),
                "linear_velocity": [0.0, 0.0, vz],
                "source": "analytic_drop_clamped_at_first_contact",
            }
        )
    return states


def load_trajectory(path: Path) -> list[dict]:
    payload = read_json(path)
    states = payload["states"] if isinstance(payload, dict) and "states" in payload else payload
    if not states:
        raise ValueError(f"No states found in {path}")
    return states


def evaluate_collision_over_trajectory(states: list[dict], args: argparse.Namespace) -> tuple[list[dict], dict]:
    local_centers, radii = make_cube_proxy_local(float(args.half_extent), int(args.proxy_resolution))
    floor_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    rows = []
    for state in states:
        position = np.asarray(state["position"], dtype=np.float64)
        quat = np.asarray(state.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        query_points = make_floor_query_points_world(
            position[:2],
            half_extent=float(args.half_extent),
            resolution=int(args.query_resolution),
            query_scale=float(args.query_scale),
        )
        centers_world = transform_points(local_centers, position, quat)
        sdf = evaluate_gaussian_union_sdf(
            query_points,
            centers_world,
            radii,
            smooth_min_temperature=float(args.smooth_min_temperature),
            inside_penalty=float(args.inside_penalty),
            inside_sharpness=float(args.inside_sharpness),
        )
        contacts = aggregate_gaussian_union_contacts(
            sdf,
            softness=float(args.contact_softness),
            normal_hint=floor_normal,
            num_contact_patches=int(args.num_contact_patches),
        )
        exact_sdf = exact_box_sdf_world(query_points, position, quat, float(args.half_extent))
        gaussian_min = float(contacts.min_signed_distance.detach().cpu().item())
        exact_min = float(exact_sdf.min().detach().cpu().item())
        rows.append(
            {
                "frame_index": int(state.get("frame_index", len(rows))),
                "time": float(state.get("time", len(rows) / float(args.fps))),
                "position": position.tolist(),
                "quaternion_wxyz": quat.tolist(),
                "gaussian_min_signed_distance": gaussian_min,
                "exact_box_min_signed_distance": exact_min,
                "sdf_min_abs_error": abs(gaussian_min - exact_min),
                "max_penetration": float(contacts.max_penetration.detach().cpu().item()),
                "mean_contact_weight": float(contacts.contact_weights.mean().detach().cpu().item()),
                "max_contact_weight": float(contacts.contact_weights.max().detach().cpu().item()),
                "contact_point": contacts.contact_point.detach().cpu().tolist(),
                "contact_normal": contacts.contact_normal.detach().cpu().tolist(),
                "patch_points": contacts.patch_points.detach().cpu().tolist(),
                "patch_normals": contacts.patch_normals.detach().cpu().tolist(),
                "patch_weights": contacts.patch_weights.detach().cpu().tolist(),
            }
        )

    errors = np.asarray([row["sdf_min_abs_error"] for row in rows], dtype=np.float64)
    max_weights = np.asarray([row["max_contact_weight"] for row in rows], dtype=np.float64)
    contact_frames = [row["frame_index"] for row in rows if row["max_contact_weight"] > 0.5]
    summary = {
        "num_frames": len(rows),
        "num_query_points_per_frame": int(args.query_resolution) ** 2,
        "num_gaussian_primitives": int(local_centers.shape[0]),
        "half_extent": float(args.half_extent),
        "proxy_resolution": int(args.proxy_resolution),
        "query_resolution": int(args.query_resolution),
        "num_contact_patches": int(args.num_contact_patches),
        "first_contact_frame": int(contact_frames[0]) if contact_frames else None,
        "max_contact_weight": float(max_weights.max()) if max_weights.size else 0.0,
        "mean_min_sdf_abs_error": float(errors.mean()) if errors.size else 0.0,
        "max_min_sdf_abs_error": float(errors.max()) if errors.size else 0.0,
    }
    return rows, summary


def draw_summary_plot(rows: list[dict], path: Path) -> None:
    width, height = 900, 540
    margin_l, margin_r = 86, 24
    margin_t, margin_b = 62, 42
    panel_gap = 28
    panel_h = (height - margin_t - margin_b - panel_gap) // 2
    img = Image.new("RGB", (width, height), (250, 249, 246))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width, 44), fill=(28, 28, 30))
    draw.text((14, 14), "Cube/floor collision smoke test", fill=(255, 255, 245))

    times = np.asarray([row["time"] for row in rows], dtype=np.float64)
    z = np.asarray([row["position"][2] for row in rows], dtype=np.float64)
    exact = np.asarray([row["exact_box_min_signed_distance"] for row in rows], dtype=np.float64)
    gauss = np.asarray([row["gaussian_min_signed_distance"] for row in rows], dtype=np.float64)
    weights = np.asarray([row["max_contact_weight"] for row in rows], dtype=np.float64)

    def draw_panel(idx: int, title: str, series: list[tuple[np.ndarray, tuple[int, int, int]]], y_label: str):
        y_top = margin_t + idx * (panel_h + panel_gap)
        y_bot = y_top + panel_h
        x_left = margin_l
        x_right = width - margin_r
        draw.rectangle((x_left, y_top, x_right, y_bot), fill=(255, 255, 255), outline=(155, 155, 155))
        draw.text((x_left + 6, y_top + 5), title, fill=(30, 30, 30))
        draw.text((10, y_top + panel_h // 2 - 6), y_label, fill=(70, 70, 70))
        values = np.concatenate([vals for vals, _ in series])
        ymin = float(values.min())
        ymax = float(values.max())
        if ymin <= 0.0 <= ymax:
            pass
        pad = max(1e-4, 0.08 * (ymax - ymin + 1e-6))
        ymin -= pad
        ymax += pad
        span = max(ymax - ymin, 1e-6)
        t0, t1 = float(times[0]), float(times[-1])
        for k in range(5):
            x = x_left + int(k / 4 * (x_right - x_left))
            draw.line((x, y_bot, x, y_bot + 4), fill=(120, 120, 120))
            draw.text((x - 12, y_bot + 7), f"{t0 + (t1 - t0) * k / 4:.2f}", fill=(80, 80, 80))
        if ymin <= 0.0 <= ymax:
            zy = int(y_bot - (0.0 - ymin) / span * (y_bot - y_top))
            draw.line((x_left, zy, x_right, zy), fill=(220, 220, 220), width=1)
        for vals, color in series:
            pts = []
            for i, value in enumerate(vals):
                x = x_left + int(i / max(len(vals) - 1, 1) * (x_right - x_left))
                y = int(y_bot - (float(value) - ymin) / span * (y_bot - y_top))
                pts.append((x, y))
            if len(pts) > 1:
                draw.line(pts, fill=color, width=2)

    draw_panel(0, "center z (blue) and max contact weight (red)", [(z, (34, 87, 184)), (weights, (210, 55, 45))], "z/w")
    draw_panel(1, "min signed distance: exact box (green), gaussian union (purple)", [(exact, (35, 130, 70)), (gauss, (126, 70, 185))], "sdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    args = parse_args()
    states = load_trajectory(args.trajectory_json) if args.trajectory_json else synthesize_falling_cube_trajectory(args)
    rows, summary = evaluate_collision_over_trajectory(states, args)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "cube_floor_collision_summary.json", summary)
    write_json(output_dir / "cube_floor_collision_frames.json", {"frames": rows})
    draw_summary_plot(rows, output_dir / "cube_floor_collision_plot.png")

    summary["output_dir"] = str(output_dir)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
