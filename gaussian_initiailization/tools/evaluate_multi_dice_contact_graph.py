from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_initiailization.stage2.differentiable_collision_detection import (  # noqa: E402
    CollisionEngineConfig,
    GaussianCollisionBody,
    load_gaussian_collision_primitives_from_ply,
)
from gaussian_initiailization.stage2.differentiable_complementarity_free_contact_dynamics import (  # noqa: E402
    RigidBodyState,
)
from gaussian_initiailization.stage2.differentiable_contact_graph import (  # noqa: E402
    build_pairwise_contact_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a MuJoCo multi-dice trajectory through the Stage2 pairwise "
            "contact graph using one Stage1 dice Gaussian asset as N instances."
        )
    )
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--stage1_ply", required=True, type=Path)
    parser.add_argument("--rgb_dir", default=None, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_frames", default=120, type=int)
    parser.add_argument("--frame_stride", default=1, type=int)
    parser.add_argument("--max_primitives", default=512, type=int)
    parser.add_argument("--dice_half_extent", default=None, type=float)
    parser.add_argument("--radius_scale", default=1.0, type=float)
    parser.add_argument("--contact_threshold", default=0.25, type=float)
    parser.add_argument("--broad_phase_margin", default=0.025, type=float)
    parser.add_argument("--spatial_hash_cell_size", default=0.16, type=float)
    parser.add_argument("--contact_softness", default=2e-3, type=float)
    parser.add_argument("--num_contact_patches", default=4, type=int)
    parser.add_argument("--montage_frames", default=12, type=int)
    parser.add_argument("--gif_fps", default=12, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--gap_tolerance",
        default=0.012,
        type=float,
        help=(
            "Gap-aware ground truth: a dice pair counts as a positive when the exact "
            "box-box separation is at most this many meters. This matches the spatial "
            "resolution of the Stage1 sphere-proxy surface (~1 sphere radius)."
        ),
    )
    parser.add_argument(
        "--temporal_tolerance_frames",
        default=2,
        type=int,
        help=(
            "MuJoCo contacts are sampled only at frame boundaries while the simulation "
            "runs at a much finer timestep, so brief mid-air contacts can fall between "
            "frames. A predicted edge counts as a tolerant true positive when MuJoCo "
            "reports that pair within +/- this many frames."
        ),
    )
    return parser.parse_args()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_states(payload: dict, max_frames: int, frame_stride: int) -> list[dict]:
    states = payload.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("trajectory JSON must contain a non-empty states list.")
    stride = max(1, int(frame_stride))
    states = states[::stride]
    if max_frames > 0:
        states = states[: int(max_frames)]
    if not states:
        raise ValueError("No trajectory states remain after max_frames/frame_stride filtering.")
    return states


def build_scaled_body(
    stage1_ply: Path,
    *,
    dice_half_extent: float,
    max_primitives: int,
    radius_scale: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[GaussianCollisionBody, dict]:
    centers, radii = load_gaussian_collision_primitives_from_ply(
        stage1_ply,
        radius_scale=radius_scale,
        recenter=True,
        dtype=dtype,
        device=device,
    )
    if max_primitives > 0 and centers.shape[0] > max_primitives:
        # Deterministic subsampling keeps this CPU-friendly and reproducible.
        indices = torch.linspace(0, centers.shape[0] - 1, steps=max_primitives, device=device).round().long()
        centers = centers[indices]
        radii = radii[indices]

    bbox_min = torch.min(centers - radii.unsqueeze(-1), dim=0).values
    bbox_max = torch.max(centers + radii.unsqueeze(-1), dim=0).values
    max_axis = torch.max(bbox_max - bbox_min)
    target_axis = float(dice_half_extent) * 2.0
    scale = target_axis / float(max(max_axis.detach().cpu().item(), 1e-8))
    centers = centers * scale
    radii = torch.clamp(radii * scale, min=2e-4)
    body = GaussianCollisionBody(centers, radii, centers)
    scaled_min = torch.min(centers - radii.unsqueeze(-1), dim=0).values
    scaled_max = torch.max(centers + radii.unsqueeze(-1), dim=0).values
    metadata = {
        "source_ply": str(stage1_ply.resolve()),
        "primitive_count": int(centers.shape[0]),
        "stage1_to_mujoco_scale": float(scale),
        "scaled_bbox_min": scaled_min.detach().cpu().tolist(),
        "scaled_bbox_max": scaled_max.detach().cpu().tolist(),
        "target_axis": target_axis,
    }
    return body, metadata


def make_state(die_payload: dict, *, dtype: torch.dtype, device: torch.device) -> RigidBodyState:
    return RigidBodyState(
        position=torch.tensor(die_payload["position"], dtype=dtype, device=device),
        quaternion_wxyz=torch.tensor(die_payload["quaternion_wxyz"], dtype=dtype, device=device),
        linear_velocity=torch.tensor(die_payload.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=dtype, device=device),
        angular_velocity=torch.tensor(die_payload.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=dtype, device=device),
    )


def graph_for_frame(
    bodies: list[GaussianCollisionBody],
    states: list[RigidBodyState],
    *,
    names: list[str],
    args: argparse.Namespace,
) -> dict:
    config = CollisionEngineConfig(
        softness=float(args.contact_softness),
        num_contact_patches=int(args.num_contact_patches),
        broad_phase_margin=float(args.broad_phase_margin),
        broad_phase_mode="aabb",
        patch_selection="spatial",
    )
    graph = build_pairwise_contact_graph(
        bodies,
        states,
        names=names,
        candidate_pair_mode="spatial_hash",
        spatial_hash_cell_size=float(args.spatial_hash_cell_size),
        collision_config=config,
        include_inactive=True,
        contact_threshold=float(args.contact_threshold),
    )
    serial = graph.to_serializable(contact_threshold=float(args.contact_threshold))
    serial["active_edge_count"] = sum(1 for edge in serial["edges"] if edge["active"])
    serial["broad_phase_edge_count"] = sum(1 for edge in serial["edges"] if edge["broad_phase_overlap"])
    return serial


def quat_wxyz_to_rotmat_np(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def box_pair_gap(die_i: dict, die_j: dict, half_extent: float, iterations: int = 80) -> float:
    """Exact separation distance between two oriented cubes via alternating projection.

    Both cubes are convex, and projecting a point onto an oriented box is closed
    form, so alternating projections converge to the closest pair of points.
    Returns 0.0 when the boxes overlap.
    """
    pos_i = np.asarray(die_i["position"], dtype=np.float64)
    pos_j = np.asarray(die_j["position"], dtype=np.float64)
    rot_i = quat_wxyz_to_rotmat_np(np.asarray(die_i["quaternion_wxyz"], dtype=np.float64))
    rot_j = quat_wxyz_to_rotmat_np(np.asarray(die_j["quaternion_wxyz"], dtype=np.float64))

    def project(point: np.ndarray, pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
        local = rot.T @ (point - pos)
        return pos + rot @ np.clip(local, -half_extent, half_extent)

    x = project(pos_j, pos_i, rot_i)
    y = project(x, pos_j, rot_j)
    for _ in range(iterations):
        x = project(y, pos_i, rot_i)
        y = project(x, pos_j, rot_j)
    return float(np.linalg.norm(x - y))


def compare_graph_to_mujoco(
    frame_payload: dict,
    graph_serial: dict,
    *,
    half_extent: float | None = None,
    gap_tolerance: float | None = None,
) -> dict:
    graph_pairs = {
        (int(edge["body_i"]), int(edge["body_j"]))
        for edge in graph_serial["edges"]
        if edge["active"]
    }
    mujoco_pairs = {
        (int(contact["body_i"]), int(contact["body_j"]))
        for contact in frame_payload.get("mujoco_contacts", [])
    }
    true_positive = graph_pairs & mujoco_pairs
    false_positive = graph_pairs - mujoco_pairs
    false_negative = mujoco_pairs - graph_pairs
    precision = None if not graph_pairs else len(true_positive) / len(graph_pairs)
    recall = None if not mujoco_pairs else len(true_positive) / len(mujoco_pairs)
    result = {
        "graph_pairs": [list(pair) for pair in sorted(graph_pairs)],
        "mujoco_pairs": [list(pair) for pair in sorted(mujoco_pairs)],
        "true_positive": [list(pair) for pair in sorted(true_positive)],
        "false_positive": [list(pair) for pair in sorted(false_positive)],
        "false_negative": [list(pair) for pair in sorted(false_negative)],
        "precision": precision,
        "recall": recall,
    }
    if half_extent is not None and gap_tolerance is not None:
        dice = frame_payload.get("dice", [])
        gap_pairs = set()
        pair_gaps = {}
        for i in range(len(dice)):
            for j in range(i + 1, len(dice)):
                gap = box_pair_gap(dice[i], dice[j], half_extent)
                pair_gaps[f"{i}-{j}"] = gap
                if gap <= float(gap_tolerance):
                    gap_pairs.add((i, j))
        gap_tp = graph_pairs & gap_pairs
        gap_fp = graph_pairs - gap_pairs
        gap_fn = gap_pairs - graph_pairs
        result["gap_pairs"] = [list(pair) for pair in sorted(gap_pairs)]
        result["pair_gaps"] = pair_gaps
        result["gap_true_positive"] = [list(pair) for pair in sorted(gap_tp)]
        result["gap_false_positive"] = [list(pair) for pair in sorted(gap_fp)]
        result["gap_false_negative"] = [list(pair) for pair in sorted(gap_fn)]
    return result


def nearest_pairs(dice: list[dict], half_extent: float, max_pairs: int = 4) -> list[dict]:
    rows = []
    for i in range(len(dice)):
        for j in range(i + 1, len(dice)):
            pi = np.asarray(dice[i]["position"], dtype=np.float64)
            pj = np.asarray(dice[j]["position"], dtype=np.float64)
            center_distance = float(np.linalg.norm(pi - pj))
            approx_gap = center_distance - math.sqrt(3.0) * float(half_extent) * 2.0
            rows.append(
                {
                    "body_i": i,
                    "body_j": j,
                    "center_distance": center_distance,
                    "approx_sphere_gap": approx_gap,
                }
            )
    return sorted(rows, key=lambda row: row["approx_sphere_gap"])[:max_pairs]


def topdown_bounds(frame_reports: list[dict]) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for report in frame_reports:
        for die in report["dice"]:
            xs.append(float(die["position"][0]))
            ys.append(float(die["position"][1]))
    if not xs:
        return -1.0, 1.0, -1.0, 1.0
    pad = 0.18
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def draw_topdown_overlay(image: Image.Image, report: dict, bounds: tuple[float, float, float, float]) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    box_w, box_h = 260, 180
    left, top = width - box_w - 18, 18
    draw.rounded_rectangle((left, top, left + box_w, top + box_h), radius=8, fill=(255, 255, 255, 215), outline=(20, 20, 20, 80))
    min_x, max_x, min_y, max_y = bounds

    def project(pos):
        x, y = float(pos[0]), float(pos[1])
        px = left + 18 + (x - min_x) / max(max_x - min_x, 1e-6) * (box_w - 36)
        py = top + box_h - 24 - (y - min_y) / max(max_y - min_y, 1e-6) * (box_h - 46)
        return px, py

    active_edges = [(edge["body_i"], edge["body_j"]) for edge in report["graph"]["edges"] if edge["active"]]
    broad_edges = [
        (edge["body_i"], edge["body_j"])
        for edge in report["graph"]["edges"]
        if edge["broad_phase_overlap"] and not edge["active"]
    ]
    positions = [die["position"] for die in report["dice"]]
    for i, j in broad_edges:
        draw.line((*project(positions[i]), *project(positions[j])), fill=(70, 110, 200, 120), width=2)
    for i, j in active_edges:
        draw.line((*project(positions[i]), *project(positions[j])), fill=(220, 60, 55, 230), width=4)
    colors = [(220, 40, 44), (32, 95, 210), (35, 150, 80), (230, 165, 35), (120, 75, 190), (20, 160, 170)]
    for die in report["dice"]:
        idx = int(die["die"])
        px, py = project(die["position"])
        color = colors[idx % len(colors)]
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(*color, 235), outline=(0, 0, 0, 180), width=1)
        draw.text((px + 9, py - 7), str(idx), fill=(25, 25, 25, 255))
    draw.text((left + 14, top + 10), f"frame {report['frame_index']:03d}", fill=(20, 20, 20, 255))
    draw.text(
        (left + 14, top + 29),
        f"active {len(active_edges)} / broad {report['graph']['broad_phase_edge_count']}",
        fill=(20, 20, 20, 255),
    )
    if active_edges:
        label = " ".join(f"{i}-{j}" for i, j in active_edges[:4])
        draw.text((left + 14, top + 48), f"contacts {label}", fill=(160, 20, 20, 255))
    return canvas


def load_rgb_frame(rgb_dir: Path | None, frame_index: int, fallback_size=(960, 540)) -> Image.Image:
    if rgb_dir is not None:
        path = rgb_dir / f"{frame_index:06d}.png"
        if path.exists():
            return Image.open(path).convert("RGB")
    return Image.new("RGB", fallback_size, (245, 245, 242))


def build_visuals(output_dir: Path, reports: list[dict], rgb_dir: Path | None, montage_frames: int, gif_fps: int) -> None:
    if not reports:
        return
    bounds = topdown_bounds(reports)
    visual_frames = [
        draw_topdown_overlay(load_rgb_frame(rgb_dir, report["frame_index"]), report, bounds)
        for report in reports
    ]
    gif_path = output_dir / "multi_dice_contact_graph_overlay.gif"
    imageio.mimsave(gif_path, [np.asarray(frame) for frame in visual_frames], duration=1.0 / max(1, gif_fps))

    count = max(1, min(int(montage_frames), len(visual_frames)))
    indices = np.linspace(0, len(visual_frames) - 1, count).round().astype(int).tolist()
    thumbs = [visual_frames[idx].resize((320, 180), Image.Resampling.LANCZOS) for idx in indices]
    cols = min(4, count)
    rows = int(math.ceil(count / cols))
    montage = Image.new("RGB", (cols * 320, rows * 180), (245, 245, 242))
    for tile_idx, thumb in enumerate(thumbs):
        x = (tile_idx % cols) * 320
        y = (tile_idx // cols) * 180
        montage.paste(thumb, (x, y))
    montage.save(output_dir / "multi_dice_contact_graph_montage.png")

    active_reports = [report for report in reports if report["graph"]["active_edge_count"] > 0]
    if active_reports:
        active_count = min(6, len(active_reports))
        active_indices = np.linspace(0, len(active_reports) - 1, active_count).round().astype(int).tolist()
        active_frames = [
            draw_topdown_overlay(load_rgb_frame(rgb_dir, active_reports[idx]["frame_index"]), active_reports[idx], bounds)
            for idx in active_indices
        ]
        active_thumbs = [frame.resize((480, 270), Image.Resampling.LANCZOS) for frame in active_frames]
        active_cols = min(3, active_count)
        active_rows = int(math.ceil(active_count / active_cols))
        active_montage = Image.new("RGB", (active_cols * 480, active_rows * 270), (245, 245, 242))
        for tile_idx, thumb in enumerate(active_thumbs):
            x = (tile_idx % active_cols) * 480
            y = (tile_idx // active_cols) * 270
            active_montage.paste(thumb, (x, y))
        active_montage.save(output_dir / "multi_dice_active_contacts_montage.png")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = read_json(args.trajectory.resolve())
    dice_half_extent = float(args.dice_half_extent or payload.get("half_extent", 0.055))
    states_payload = load_states(payload, int(args.max_frames), int(args.frame_stride))
    dtype = torch.float32
    device = torch.device(args.device)
    body, body_metadata = build_scaled_body(
        args.stage1_ply.resolve(),
        dice_half_extent=dice_half_extent,
        max_primitives=int(args.max_primitives),
        radius_scale=float(args.radius_scale),
        dtype=dtype,
        device=device,
    )
    dice_count = int(payload.get("dice_count", len(states_payload[0].get("dice", []))))
    bodies = [body for _ in range(dice_count)]
    names = [f"die_{idx:02d}" for idx in range(dice_count)]

    reports = []
    for sampled_idx, frame_payload in enumerate(states_payload):
        dice = frame_payload["dice"]
        rb_states = [make_state(die, dtype=dtype, device=device) for die in dice]
        graph = graph_for_frame(bodies, rb_states, names=names, args=args)
        contact_comparison = compare_graph_to_mujoco(
            frame_payload,
            graph,
            half_extent=dice_half_extent,
            gap_tolerance=float(args.gap_tolerance),
        )
        reports.append(
            {
                "sample_index": sampled_idx,
                "frame_index": int(frame_payload.get("frame", sampled_idx * int(args.frame_stride))),
                "time": float(frame_payload.get("time", sampled_idx)),
                "dice": dice,
                "mujoco_contacts": frame_payload.get("mujoco_contacts", []),
                "graph": graph,
                "contact_comparison": contact_comparison,
                "nearest_pairs": nearest_pairs(dice, dice_half_extent),
            }
        )

    active_frames = [report for report in reports if report["graph"]["active_edge_count"] > 0]
    broad_frames = [report for report in reports if report["graph"]["broad_phase_edge_count"] > 0]
    edge_histogram: dict[str, int] = {}
    comparison_frames = [report for report in reports if report["contact_comparison"]["mujoco_pairs"] or report["contact_comparison"]["graph_pairs"]]
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for report in reports:
        for edge in report["graph"]["edges"]:
            if edge["active"]:
                key = f"{edge['name_i']}--{edge['name_j']}"
                edge_histogram[key] = edge_histogram.get(key, 0) + 1
        comparison = report["contact_comparison"]
        total_tp += len(comparison["true_positive"])
        total_fp += len(comparison["false_positive"])
        total_fn += len(comparison["false_negative"])
    gap_tp = sum(len(r["contact_comparison"].get("gap_true_positive", [])) for r in reports)
    gap_fp = sum(len(r["contact_comparison"].get("gap_false_positive", [])) for r in reports)
    gap_fn = sum(len(r["contact_comparison"].get("gap_false_negative", [])) for r in reports)
    gap_precision = None if gap_tp + gap_fp == 0 else gap_tp / (gap_tp + gap_fp)
    gap_recall = None if gap_tp + gap_fn == 0 else gap_tp / (gap_tp + gap_fn)
    aggregate_precision = None if total_tp + total_fp == 0 else total_tp / (total_tp + total_fp)
    aggregate_recall = None if total_tp + total_fn == 0 else total_tp / (total_tp + total_fn)

    # Temporally tolerant comparison: MuJoCo contact labels are instantaneous frame
    # samples of a much finer simulation, so credit a predicted edge when the pair is
    # in contact within +/- temporal_tolerance_frames.
    tolerance = max(0, int(args.temporal_tolerance_frames))
    graph_pairs_per_frame = [
        {tuple(pair) for pair in report["contact_comparison"]["graph_pairs"]} for report in reports
    ]
    mujoco_pairs_per_frame = [
        {tuple(pair) for pair in report["contact_comparison"]["mujoco_pairs"]} for report in reports
    ]
    tolerant_tp = 0
    tolerant_fp = 0
    tolerant_fn = 0
    for frame_idx in range(len(reports)):
        window = range(max(0, frame_idx - tolerance), min(len(reports), frame_idx + tolerance + 1))
        mujoco_window = set().union(*(mujoco_pairs_per_frame[w] for w in window)) if reports else set()
        graph_window = set().union(*(graph_pairs_per_frame[w] for w in window)) if reports else set()
        for pair in graph_pairs_per_frame[frame_idx]:
            if pair in mujoco_window:
                tolerant_tp += 1
            else:
                tolerant_fp += 1
        for pair in mujoco_pairs_per_frame[frame_idx]:
            if pair not in graph_window:
                tolerant_fn += 1
    tolerant_precision = (
        None if tolerant_tp + tolerant_fp == 0 else tolerant_tp / (tolerant_tp + tolerant_fp)
    )
    tolerant_recall = (
        None if tolerant_tp + tolerant_fn == 0 else tolerant_tp / (tolerant_tp + tolerant_fn)
    )

    summary = {
        "trajectory": str(args.trajectory.resolve()),
        "rgb_dir": None if args.rgb_dir is None else str(args.rgb_dir.resolve()),
        "stage1_body": body_metadata,
        "dice_count": dice_count,
        "sampled_frames": len(reports),
        "frame_stride": int(args.frame_stride),
        "dice_half_extent": dice_half_extent,
        "contact_threshold": float(args.contact_threshold),
        "broad_phase_margin": float(args.broad_phase_margin),
        "spatial_hash_cell_size": float(args.spatial_hash_cell_size),
        "frames_with_active_edges": len(active_frames),
        "frames_with_broad_phase_edges": len(broad_frames),
        "edge_active_frame_histogram": edge_histogram,
        "mujoco_contact_comparison": {
            "frames_with_any_compared_contact": len(comparison_frames),
            "true_positive_edges": total_tp,
            "false_positive_edges": total_fp,
            "false_negative_edges": total_fn,
            "precision": aggregate_precision,
            "recall": aggregate_recall,
            "temporal_tolerance_frames": tolerance,
            "tolerant_true_positive_edges": tolerant_tp,
            "tolerant_false_positive_edges": tolerant_fp,
            "tolerant_false_negative_edges": tolerant_fn,
            "tolerant_precision": tolerant_precision,
            "tolerant_recall": tolerant_recall,
            "gap_tolerance_m": float(args.gap_tolerance),
            "gap_true_positive_edges": gap_tp,
            "gap_false_positive_edges": gap_fp,
            "gap_false_negative_edges": gap_fn,
            "gap_precision": gap_precision,
            "gap_recall": gap_recall,
        },
        "first_active_frame": None
        if not active_frames
        else {
            "frame_index": active_frames[0]["frame_index"],
            "time": active_frames[0]["time"],
            "active_edges": [
                edge
                for edge in active_frames[0]["graph"]["edges"]
                if edge["active"]
            ],
        },
    }
    write_json(output_dir / "contact_graph_summary.json", summary)
    write_json(output_dir / "contact_graph_frames.json", reports)
    build_visuals(output_dir, reports, args.rgb_dir, int(args.montage_frames), int(args.gif_fps))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
