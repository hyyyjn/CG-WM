from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

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
from gaussian_initiailization.tools.generate_mujoco_multi_dice_rollout import build_mjcf  # noqa: E402


@dataclass
class SimState:
    position: torch.Tensor
    quaternion_wxyz: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor

    def rigid(self) -> RigidBodyState:
        return RigidBodyState(
            self.position,
            self.quaternion_wxyz,
            self.linear_velocity,
            self.angular_velocity,
        )

    def to_serializable(self, die: int) -> dict:
        return {
            "die": int(die),
            "position": self.position.detach().cpu().tolist(),
            "quaternion_wxyz": self.quaternion_wxyz.detach().cpu().tolist(),
            "linear_velocity": self.linear_velocity.detach().cpu().tolist(),
            "angular_velocity": self.angular_velocity.detach().cpu().tolist(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a lightweight Stage1-asset -> Stage2 N-body rollout for multi-dice "
            "and render a side-by-side comparison against MuJoCo GT."
        )
    )
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--gt_rgb_dir", required=True, type=Path)
    parser.add_argument("--stage1_ply", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_frames", default=120, type=int)
    parser.add_argument("--max_primitives", default=512, type=int)
    parser.add_argument("--mujoco_gl", default="glfw", choices=("glfw", "egl", "osmesa"))
    parser.add_argument("--width", default=960, type=int)
    parser.add_argument("--height", default=540, type=int)
    parser.add_argument("--fps", default=12, type=int)
    parser.add_argument("--substeps", default=6, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--contact_threshold", default=0.25, type=float)
    parser.add_argument("--broad_phase_margin", default=0.018, type=float)
    parser.add_argument("--spatial_hash_cell_size", default=0.15, type=float)
    parser.add_argument("--position_correction", default=0.85, type=float)
    parser.add_argument("--restitution", default=0.20, type=float)
    parser.add_argument("--floor_restitution", default=0.15, type=float)
    parser.add_argument("--floor_friction", default=0.50, type=float)
    parser.add_argument("--pair_friction", default=0.45, type=float)
    parser.add_argument("--contact_angular_damping", default=0.03, type=float)
    parser.add_argument("--linear_damping", default=0.018, type=float)
    parser.add_argument("--angular_damping", default=0.05, type=float)
    parser.add_argument("--radius_scale", default=1.0, type=float)
    parser.add_argument("--fit_iters", default=0, type=int)
    parser.add_argument("--fit_lr", default=0.04, type=float)
    parser.add_argument(
        "--fit_horizon_frames",
        default=0,
        type=int,
        help=(
            "Fit the initial velocities against only the first N frames. Dice rollouts "
            "are chaotic, so gradients through the full horizon are dominated by noise; "
            "a short horizon covering the flight and first bounces is far more stable. "
            "0 means use max_frames."
        ),
    )
    parser.add_argument("--fit_initial_velocity_l2", default=1e-3, type=float)
    parser.add_argument("--fit_position_weight", default=1.0, type=float)
    return parser.parse_args()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.linalg.norm(q), min=1e-12)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind()
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y))),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x))),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))),
        ),
        dim=0,
    )


# Unit cube corner signs; scaled by half_extent at use site.
CUBE_CORNER_SIGNS = [
    (sx, sy, sz)
    for sx in (-1.0, 1.0)
    for sy in (-1.0, 1.0)
    for sz in (-1.0, 1.0)
]


def cube_inertia_inverse(half_extent: float) -> float:
    # Solid cube, unit mass: I = m * (2h)^2 / 6 = (2/3) m h^2 (isotropic).
    return 1.0 / ((2.0 / 3.0) * float(half_extent) * float(half_extent))


def quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = lhs.unbind()
    w2, x2, y2, z2 = rhs.unbind()
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def integrate_quat(q: torch.Tensor, angular_velocity: torch.Tensor, dt: float) -> torch.Tensor:
    omega = torch.cat((torch.zeros(1, dtype=q.dtype, device=q.device), angular_velocity))
    q_dot = 0.5 * quat_mul(omega, q)
    return normalize_quat(q + float(dt) * q_dot)


def infer_dt(states: list[dict]) -> float:
    times = [float(state.get("time", idx)) for idx, state in enumerate(states)]
    diffs = np.diff(times)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0 / 30.0
    return float(np.median(diffs))


def build_scaled_body(
    stage1_ply: Path,
    *,
    half_extent: float,
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
        indices = torch.linspace(0, centers.shape[0] - 1, steps=max_primitives, device=device).round().long()
        centers = centers[indices]
        radii = radii[indices]
    bbox_min = torch.min(centers - radii.unsqueeze(-1), dim=0).values
    bbox_max = torch.max(centers + radii.unsqueeze(-1), dim=0).values
    scale = (float(half_extent) * 2.0) / float(torch.max(bbox_max - bbox_min).detach().cpu().item())
    centers = centers * scale
    radii = torch.clamp(radii * scale, min=2e-4)
    body = GaussianCollisionBody(centers, radii, centers)
    return body, {
        "source_ply": str(stage1_ply.resolve()),
        "primitive_count": int(centers.shape[0]),
        "stage1_to_mujoco_scale": float(scale),
    }


def initial_states(
    frame: dict,
    *,
    dtype: torch.dtype,
    device: torch.device,
    linear_velocity_delta: torch.Tensor | None = None,
    angular_velocity_delta: torch.Tensor | None = None,
) -> list[SimState]:
    states = []
    for die_idx, die in enumerate(frame["dice"]):
        linear_velocity = torch.tensor(die.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=dtype, device=device)
        angular_velocity = torch.tensor(die.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=dtype, device=device)
        if linear_velocity_delta is not None:
            linear_velocity = linear_velocity + linear_velocity_delta[die_idx]
        if angular_velocity_delta is not None:
            angular_velocity = angular_velocity + angular_velocity_delta[die_idx]
        states.append(
            SimState(
                position=torch.tensor(die["position"], dtype=dtype, device=device),
                quaternion_wxyz=normalize_quat(torch.tensor(die["quaternion_wxyz"], dtype=dtype, device=device)),
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
            )
        )
    return states


def graph_edges(
    bodies: list[GaussianCollisionBody],
    states: list[SimState],
    args: argparse.Namespace,
) -> list:
    config = CollisionEngineConfig(
        softness=2e-3,
        num_contact_patches=4,
        broad_phase_margin=float(args.broad_phase_margin),
        broad_phase_mode="aabb",
        patch_selection="spatial",
    )
    graph = build_pairwise_contact_graph(
        bodies,
        [state.rigid() for state in states],
        names=[f"die_{idx:02d}" for idx in range(len(states))],
        candidate_pair_mode="spatial_hash",
        spatial_hash_cell_size=float(args.spatial_hash_cell_size),
        collision_config=config,
        include_inactive=False,
        contact_threshold=float(args.contact_threshold),
    )
    return list(graph.active_edges(contact_threshold=float(args.contact_threshold)))


def apply_floor(states: list[SimState], *, half_extent: float, args: argparse.Namespace, dt: float) -> None:
    inv_inertia = cube_inertia_inverse(half_extent)
    restitution = float(args.floor_restitution)
    friction = float(args.floor_friction)
    for state in states:
        dtype = state.position.dtype
        device = state.position.device
        rotation = quat_to_rotmat(state.quaternion_wxyz)
        corner_signs = torch.tensor(CUBE_CORNER_SIGNS, dtype=dtype, device=device) * float(half_extent)
        corner_offsets = corner_signs @ rotation.T
        corners = state.position.unsqueeze(0) + corner_offsets
        penetrations = -corners[:, 2]
        max_penetration = torch.max(penetrations)
        if float(max_penetration.item()) <= 0.0:
            continue
        lift = float(args.position_correction) * max_penetration
        state.position = state.position + torch.stack(
            (torch.zeros((), dtype=dtype, device=device), torch.zeros((), dtype=dtype, device=device), lift)
        )
        normal = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        # Sequential impulses at the penetrating corners, deepest first.
        order = torch.argsort(penetrations, descending=True)
        for corner_idx in order.tolist():
            if float(penetrations[corner_idx].item()) <= -1e-4:
                break
            r = corner_offsets[corner_idx]
            corner_velocity = state.linear_velocity + torch.linalg.cross(state.angular_velocity, r)
            normal_speed = torch.sum(corner_velocity * normal)
            if float(normal_speed.item()) >= 0.0:
                continue
            r_cross_n = torch.linalg.cross(r, normal)
            effective_mass = 1.0 + inv_inertia * torch.sum(r_cross_n * r_cross_n)
            impulse_n = -(1.0 + restitution) * normal_speed / torch.clamp(effective_mass, min=1e-9)
            state.linear_velocity = state.linear_velocity + impulse_n * normal
            state.angular_velocity = state.angular_velocity + inv_inertia * torch.linalg.cross(r, impulse_n * normal)
            # Coulomb friction at the same corner.
            corner_velocity = state.linear_velocity + torch.linalg.cross(state.angular_velocity, r)
            tangent_velocity = corner_velocity - torch.sum(corner_velocity * normal) * normal
            tangent_speed = torch.linalg.norm(tangent_velocity)
            if float(tangent_speed.item()) > 1e-7:
                tangent = tangent_velocity / tangent_speed
                r_cross_t = torch.linalg.cross(r, tangent)
                tangent_mass = 1.0 + inv_inertia * torch.sum(r_cross_t * r_cross_t)
                impulse_t = torch.clamp(
                    tangent_speed / torch.clamp(tangent_mass, min=1e-9),
                    max=friction * torch.clamp(impulse_n, min=0.0),
                )
                state.linear_velocity = state.linear_velocity - impulse_t * tangent
                state.angular_velocity = state.angular_velocity - inv_inertia * torch.linalg.cross(r, impulse_t * tangent)
        # Mild rolling resistance while touching the floor helps dice settle flat.
        state.angular_velocity = state.angular_velocity * max(0.0, 1.0 - float(args.contact_angular_damping))


def apply_pair_contacts(
    states: list[SimState],
    edges: list,
    *,
    args: argparse.Namespace,
    half_extent: float,
    dt: float,
) -> int:
    inv_inertia = cube_inertia_inverse(half_extent)
    active_count = 0
    for edge in edges:
        weights = edge.contacts.patch_weights.detach()
        if weights.numel() == 0 or float(torch.max(weights).item()) <= float(args.contact_threshold):
            continue
        active_count += 1
        best = int(torch.argmax(weights).item())
        normal = edge.contacts.patch_normals[best].detach()
        normal = normal / torch.clamp(torch.linalg.norm(normal), min=1e-12)
        penetration = float(edge.contacts.patch_penetrations[best].detach().cpu().item())
        if penetration <= 0.0:
            continue
        point = edge.contacts.patch_points[best].detach()
        i, j = int(edge.body_i), int(edge.body_j)
        correction = float(args.position_correction) * penetration * normal * 0.5
        states[i].position = states[i].position + correction
        states[j].position = states[j].position - correction

        r_i = point - states[i].position
        r_j = point - states[j].position
        velocity_i = states[i].linear_velocity + torch.linalg.cross(states[i].angular_velocity, r_i)
        velocity_j = states[j].linear_velocity + torch.linalg.cross(states[j].angular_velocity, r_j)
        rel_velocity = velocity_i - velocity_j
        normal_velocity = torch.sum(rel_velocity * normal)
        impulse_n = None
        if float(normal_velocity.item()) < 0.0:
            ri_cross_n = torch.linalg.cross(r_i, normal)
            rj_cross_n = torch.linalg.cross(r_j, normal)
            effective_mass = 2.0 + inv_inertia * (
                torch.sum(ri_cross_n * ri_cross_n) + torch.sum(rj_cross_n * rj_cross_n)
            )
            impulse_n = -(1.0 + float(args.restitution)) * normal_velocity / torch.clamp(effective_mass, min=1e-9)
            states[i].linear_velocity = states[i].linear_velocity + impulse_n * normal
            states[j].linear_velocity = states[j].linear_velocity - impulse_n * normal
            states[i].angular_velocity = states[i].angular_velocity + inv_inertia * torch.linalg.cross(
                r_i, impulse_n * normal
            )
            states[j].angular_velocity = states[j].angular_velocity - inv_inertia * torch.linalg.cross(
                r_j, impulse_n * normal
            )
        # Coulomb friction between the dice.
        if impulse_n is not None:
            velocity_i = states[i].linear_velocity + torch.linalg.cross(states[i].angular_velocity, r_i)
            velocity_j = states[j].linear_velocity + torch.linalg.cross(states[j].angular_velocity, r_j)
            rel_velocity = velocity_i - velocity_j
            tangent_velocity = rel_velocity - torch.sum(rel_velocity * normal) * normal
            tangent_speed = torch.linalg.norm(tangent_velocity)
            if float(tangent_speed.item()) > 1e-7:
                tangent = tangent_velocity / tangent_speed
                ri_cross_t = torch.linalg.cross(r_i, tangent)
                rj_cross_t = torch.linalg.cross(r_j, tangent)
                tangent_mass = 2.0 + inv_inertia * (
                    torch.sum(ri_cross_t * ri_cross_t) + torch.sum(rj_cross_t * rj_cross_t)
                )
                impulse_t = torch.clamp(
                    tangent_speed / torch.clamp(tangent_mass, min=1e-9),
                    max=float(args.pair_friction) * torch.clamp(impulse_n, min=0.0),
                )
                states[i].linear_velocity = states[i].linear_velocity - impulse_t * tangent
                states[j].linear_velocity = states[j].linear_velocity + impulse_t * tangent
                states[i].angular_velocity = states[i].angular_velocity - inv_inertia * torch.linalg.cross(
                    r_i, impulse_t * tangent
                )
                states[j].angular_velocity = states[j].angular_velocity + inv_inertia * torch.linalg.cross(
                    r_j, impulse_t * tangent
                )
    return active_count


def step_states(
    states: list[SimState],
    bodies: list[GaussianCollisionBody],
    *,
    args: argparse.Namespace,
    half_extent: float,
    dt: float,
) -> int:
    gravity = torch.tensor([0.0, 0.0, -9.81], dtype=states[0].position.dtype, device=states[0].position.device)
    linear_decay = max(0.0, 1.0 - float(args.linear_damping) * dt)
    angular_decay = max(0.0, 1.0 - float(args.angular_damping) * dt)
    for state in states:
        state.linear_velocity = (state.linear_velocity + gravity * dt) * linear_decay
        state.angular_velocity = state.angular_velocity * angular_decay
        state.position = state.position + state.linear_velocity * dt
        state.quaternion_wxyz = integrate_quat(state.quaternion_wxyz, state.angular_velocity, dt)
    apply_floor(states, half_extent=half_extent, args=args, dt=dt)
    active_count = apply_pair_contacts(
        states, graph_edges(bodies, states, args), args=args, half_extent=half_extent, dt=dt
    )
    apply_floor(states, half_extent=half_extent, args=args, dt=dt)
    return active_count


def rollout(
    payload: dict,
    body: GaussianCollisionBody,
    args: argparse.Namespace,
    *,
    linear_velocity_delta: torch.Tensor | None = None,
    angular_velocity_delta: torch.Tensor | None = None,
    return_position_tensor: bool = False,
    frames_limit: int | None = None,
) -> tuple[list[dict], dict] | tuple[list[dict], dict, torch.Tensor]:
    source_states = payload["states"]
    if args.max_frames > 0:
        source_states = source_states[: int(args.max_frames)]
    if frames_limit is not None and frames_limit > 0:
        source_states = source_states[: int(frames_limit)]
    dtype = torch.float32
    device = torch.device(args.device)
    dice_count = int(payload.get("dice_count", len(source_states[0]["dice"])))
    half_extent = float(payload.get("half_extent", 0.055))
    states = initial_states(
        source_states[0],
        dtype=dtype,
        device=device,
        linear_velocity_delta=linear_velocity_delta,
        angular_velocity_delta=angular_velocity_delta,
    )
    bodies = [body for _ in range(dice_count)]
    frame_dt = infer_dt(source_states)
    substeps = max(1, int(args.substeps))
    dt = frame_dt / substeps
    predicted = []
    predicted_position_tensors = []
    active_counts = []
    for frame_idx, source_frame in enumerate(source_states):
        predicted_position_tensors.append(torch.stack([state.position for state in states], dim=0))
        predicted.append(
            {
                "frame_index": int(source_frame.get("frame_index", frame_idx)),
                "time": float(source_frame.get("time", frame_idx * frame_dt)),
                "dice": [state.to_serializable(idx) for idx, state in enumerate(states)],
            }
        )
        step_active = 0
        for _ in range(substeps):
            step_active += step_states(states, bodies, args=args, half_extent=half_extent, dt=dt)
        active_counts.append(step_active)
    pred_position_tensor = torch.stack(predicted_position_tensors, dim=0)
    gt_positions = np.asarray(
        [[die["position"] for die in frame["dice"]] for frame in source_states],
        dtype=np.float64,
    )
    pred_positions = np.asarray(
        [[die["position"] for die in frame["dice"]] for frame in predicted],
        dtype=np.float64,
    )
    errors = np.linalg.norm(pred_positions - gt_positions, axis=-1)
    gt_quats = np.asarray(
        [[die["quaternion_wxyz"] for die in frame["dice"]] for frame in source_states],
        dtype=np.float64,
    )
    pred_quats = np.asarray(
        [[die["quaternion_wxyz"] for die in frame["dice"]] for frame in predicted],
        dtype=np.float64,
    )
    quat_dots = np.abs(np.sum(gt_quats * pred_quats, axis=-1))
    quat_dots = quat_dots / np.maximum(
        np.linalg.norm(gt_quats, axis=-1) * np.linalg.norm(pred_quats, axis=-1), 1e-12
    )
    rotation_errors_deg = np.degrees(2.0 * np.arccos(np.clip(quat_dots, -1.0, 1.0)))
    metrics = {
        "position_rmse": float(np.sqrt(np.mean((pred_positions - gt_positions) ** 2))),
        "mean_center_error": float(np.mean(errors)),
        "final_mean_center_error": float(np.mean(errors[-1])),
        "max_center_error": float(np.max(errors)),
        "mean_rotation_error_deg": float(np.mean(rotation_errors_deg)),
        "final_mean_rotation_error_deg": float(np.mean(rotation_errors_deg[-1])),
        "stage2_active_pair_substeps": int(sum(active_counts)),
        "frame_dt": float(frame_dt),
        "substeps": substeps,
    }
    if return_position_tensor:
        return predicted, metrics, pred_position_tensor
    return predicted, metrics


def rollout_tensor(
    payload: dict,
    body: GaussianCollisionBody,
    args: argparse.Namespace,
    *,
    linear_velocity_delta: torch.Tensor,
    angular_velocity_delta: torch.Tensor,
    frames_limit: int | None = None,
) -> torch.Tensor:
    result = rollout(
        payload,
        body,
        args,
        linear_velocity_delta=linear_velocity_delta,
        angular_velocity_delta=angular_velocity_delta,
        return_position_tensor=True,
        frames_limit=frames_limit,
    )
    return result[2]


def fit_stage2_initial_state(
    payload: dict,
    body: GaussianCollisionBody,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    source_states = payload["states"]
    if args.max_frames > 0:
        source_states = source_states[: int(args.max_frames)]
    horizon = int(args.fit_horizon_frames)
    if horizon > 0:
        source_states = source_states[:horizon]
    device = torch.device(args.device)
    dtype = torch.float32
    dice_count = int(payload.get("dice_count", len(source_states[0]["dice"])))
    target_positions = torch.tensor(
        [[die["position"] for die in frame["dice"]] for frame in source_states],
        dtype=dtype,
        device=device,
    )
    linear_velocity_delta = torch.zeros((dice_count, 3), dtype=dtype, device=device, requires_grad=True)
    angular_velocity_delta = torch.zeros((dice_count, 3), dtype=dtype, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([linear_velocity_delta, angular_velocity_delta], lr=float(args.fit_lr))
    history = []
    best = {
        "loss": float("inf"),
        "linear_velocity_delta": None,
        "angular_velocity_delta": None,
    }
    for fit_iter in range(int(args.fit_iters)):
        optimizer.zero_grad(set_to_none=True)
        pred_positions = rollout_tensor(
            payload,
            body,
            args,
            linear_velocity_delta=linear_velocity_delta,
            angular_velocity_delta=angular_velocity_delta,
            frames_limit=horizon if horizon > 0 else None,
        )
        position_loss = torch.mean((pred_positions - target_positions) ** 2)
        regularizer = float(args.fit_initial_velocity_l2) * (
            torch.mean(linear_velocity_delta**2) + 0.01 * torch.mean(angular_velocity_delta**2)
        )
        loss = float(args.fit_position_weight) * position_loss + regularizer
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            linear_velocity_delta.clamp_(-4.0, 4.0)
            angular_velocity_delta.clamp_(-20.0, 20.0)
        loss_value = float(loss.detach().cpu().item())
        history.append(
            {
                "iter": fit_iter,
                "loss": loss_value,
                "position_mse": float(position_loss.detach().cpu().item()),
                "linear_delta_norm": float(torch.linalg.norm(linear_velocity_delta.detach()).cpu().item()),
                "angular_delta_norm": float(torch.linalg.norm(angular_velocity_delta.detach()).cpu().item()),
            }
        )
        if loss_value < best["loss"]:
            best["loss"] = loss_value
            best["linear_velocity_delta"] = linear_velocity_delta.detach().clone()
            best["angular_velocity_delta"] = angular_velocity_delta.detach().clone()
        if fit_iter == 0 or (fit_iter + 1) % 10 == 0 or fit_iter == int(args.fit_iters) - 1:
            print(json.dumps(history[-1]))
    return (
        best["linear_velocity_delta"],
        best["angular_velocity_delta"],
        {"history": history, "best_loss": best["loss"]},
    )


def set_qpos(model, data, frame: dict) -> None:
    import mujoco

    for die in frame["dice"]:
        idx = int(die["die"])
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"die_{idx:02d}_free")
        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(die["position"], dtype=np.float64)
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.asarray(die["quaternion_wxyz"], dtype=np.float64)
    mujoco.mj_forward(model, data)


def render_predicted_frames(payload: dict, predicted: list[dict], args: argparse.Namespace, output_dir: Path) -> Path:
    os.environ["MUJOCO_GL"] = args.mujoco_gl
    import mujoco

    dice_count = int(payload.get("dice_count", len(predicted[0]["dice"])))
    half_extent = float(payload.get("half_extent", 0.055))
    model = mujoco.MjModel.from_xml_string(
        build_mjcf(dice_count, half_extent, 0.002, int(args.width), int(args.height))
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=int(args.height), width=int(args.width))
    pred_rgb = output_dir / "stage2_predicted_rgb"
    pred_rgb.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame in predicted:
        set_qpos(model, data, frame)
        renderer.update_scene(data, camera="cam0")
        rgb = np.asarray(renderer.render(), dtype=np.uint8)[..., :3]
        rgb[(rgb.sum(axis=-1) == 0)] = np.array([248, 248, 244], dtype=np.uint8)
        path = pred_rgb / f"{int(frame['frame_index']):06d}.png"
        Image.fromarray(rgb).save(path)
        frames.append(rgb)
    imageio.mimsave(output_dir / "stage2_predicted_rollout.gif", frames, fps=max(1, int(args.fps)))
    return pred_rgb


def make_comparison_gif(gt_rgb_dir: Path, pred_rgb_dir: Path, predicted: list[dict], metrics: dict, output: Path, fps: int) -> None:
    frames = []
    canvases = []
    for frame in predicted:
        frame_idx = int(frame["frame_index"])
        gt = Image.open(gt_rgb_dir / f"{frame_idx:06d}.png").convert("RGB").resize((480, 270), Image.Resampling.LANCZOS)
        pred = Image.open(pred_rgb_dir / f"{frame_idx:06d}.png").convert("RGB").resize((480, 270), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (960, 326), (248, 248, 244))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 960, 56), fill=(28, 28, 28))
        draw.text((18, 10), "MuJoCo GT", fill=(255, 255, 255))
        draw.text((498, 10), "Stage1 asset -> Stage2 predicted rollout", fill=(255, 255, 255))
        draw.text(
            (498, 32),
            (
                f"pos RMSE {metrics['position_rmse']:.3f}m | final err {metrics['final_mean_center_error']:.3f}m"
                f" | final rot err {metrics.get('final_mean_rotation_error_deg', float('nan')):.1f}deg"
            ),
            fill=(220, 220, 220),
        )
        canvas.paste(gt, (0, 56))
        canvas.paste(pred, (480, 56))
        canvases.append(canvas)
        frames.append(np.asarray(canvas))
    imageio.mimsave(output, frames, duration=1.0 / max(1, int(fps)))
    montage_count = min(12, len(canvases))
    montage_indices = np.linspace(0, len(canvases) - 1, montage_count).round().astype(int).tolist()
    thumbs = [
        canvases[idx].resize((480, 163), Image.Resampling.LANCZOS)
        for idx in montage_indices
    ]
    cols = 2
    rows = int(math.ceil(montage_count / cols))
    montage = Image.new("RGB", (cols * 480, rows * 163), (248, 248, 244))
    for tile_idx, thumb in enumerate(thumbs):
        montage.paste(thumb, ((tile_idx % cols) * 480, (tile_idx // cols) * 163))
    montage.save(output.with_name(output.stem + "_montage.png"))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = read_json(args.trajectory.resolve())
    dtype = torch.float32
    device = torch.device(args.device)
    half_extent = float(payload.get("half_extent", 0.055))
    body, body_metadata = build_scaled_body(
        args.stage1_ply.resolve(),
        half_extent=half_extent,
        max_primitives=int(args.max_primitives),
        radius_scale=float(args.radius_scale),
        dtype=dtype,
        device=device,
    )
    fit_summary = None
    linear_velocity_delta = None
    angular_velocity_delta = None
    if int(args.fit_iters) > 0:
        linear_velocity_delta, angular_velocity_delta, fit_summary = fit_stage2_initial_state(payload, body, args)
    predicted, metrics = rollout(
        payload,
        body,
        args,
        linear_velocity_delta=linear_velocity_delta,
        angular_velocity_delta=angular_velocity_delta,
    )
    pred_rgb_dir = render_predicted_frames(payload, predicted, args, output_dir)
    comparison_path = output_dir / "gt_vs_stage2_predicted_rollout.gif"
    make_comparison_gif(args.gt_rgb_dir.resolve(), pred_rgb_dir, predicted, metrics, comparison_path, int(args.fps))
    write_json(output_dir / "stage2_predicted_trajectory.json", {"states": predicted})
    summary = {
        "trajectory": str(args.trajectory.resolve()),
        "gt_rgb_dir": str(args.gt_rgb_dir.resolve()),
        "stage1_body": body_metadata,
        "metrics": metrics,
        "fit": fit_summary,
        "comparison_gif": str(comparison_path),
        "predicted_rollout_gif": str(output_dir / "stage2_predicted_rollout.gif"),
        "note": (
            "This is an end-to-end Stage1 collision-proxy to Stage2 N-body rollout result. "
            "When fit is present, Stage2 initial linear/angular velocities were optimized against the GT trajectory; "
            "contact/damping parameters are still fixed."
        ),
    }
    write_json(output_dir / "stage2_rollout_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
