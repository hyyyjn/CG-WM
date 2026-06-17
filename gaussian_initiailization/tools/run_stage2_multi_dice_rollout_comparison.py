from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_initiailization.stage2.differentiable_collision_detection import (  # noqa: E402
    CollisionEngineConfig,
    GaussianCollisionBody,
    load_gaussian_collision_primitives_from_ply,
    make_box_surface_query_points,
)
from gaussian_initiailization.stage2.differentiable_complementarity_free_contact_dynamics import (  # noqa: E402
    DYNAMICS_BACKEND_IMPULSE_BASELINE,
    DYNAMICS_BACKEND_LEGACY_IMPULSE,
    DYNAMICS_BACKEND_STAGE2_IMPEDANCE,
    FRICTION_MODEL_DUAL_CONE,
    FRICTION_MODEL_SOFT_PROJECTION,
    MultiBodyGaussianImpedanceDynamics,
    MultiBodyImpedanceDynamicsConfig,
    RigidBodyState,
    normalize_dynamics_backend,
)
from gaussian_initiailization.stage2.differentiable_contact_graph import (  # noqa: E402
    build_pairwise_contact_graph,
)

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
    parser.add_argument("--gt_rgb_dir", default=None, type=Path)
    parser.add_argument("--stage1_ply", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_frames", default=120, type=int)
    parser.add_argument("--max_primitives", default=512, type=int)
    parser.add_argument("--mujoco_gl", default="glfw", choices=("glfw", "egl", "osmesa"))
    parser.add_argument("--width", default=960, type=int)
    parser.add_argument("--height", default=540, type=int)
    parser.add_argument("--fps", default=12, type=int)
    parser.add_argument("--skip_render", action="store_true")
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
    parser.add_argument("--fit_physics_iters", default=0, type=int)
    parser.add_argument("--fit_physics_lr", default=0.025, type=float)
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
    parser.add_argument("--gt_mask_dir", default=None, type=Path)
    parser.add_argument("--mask_loss_weight", default=0.0, type=float)
    parser.add_argument("--mask_loss_resolution", default=64, type=int)
    parser.add_argument("--mask_loss_softness", default=0.015, type=float)
    parser.add_argument("--gaussian_rgb_dir", default=None, type=Path)
    parser.add_argument("--gaussian_rgb_loss_weight", default=0.0, type=float)
    parser.add_argument("--gaussian_render_width", default=160, type=int)
    parser.add_argument("--gaussian_render_height", default=120, type=int)
    parser.add_argument("--gaussian_render_stride", default=4, type=int)
    parser.add_argument("--gaussian_render_loss", default="l1", choices=("l1", "mse"))
    parser.add_argument("--gaussian_render_white_background", action="store_true")
    parser.add_argument(
        "--dynamics_backend",
        default=DYNAMICS_BACKEND_STAGE2_IMPEDANCE,
        choices=(
            DYNAMICS_BACKEND_STAGE2_IMPEDANCE,
            DYNAMICS_BACKEND_IMPULSE_BASELINE,
            DYNAMICS_BACKEND_LEGACY_IMPULSE,
        ),
        help=(
            "stage2_impedance uses the paper-aligned Gaussian N-body impedance dynamics; "
            "impulse_baseline keeps the old demo solver. impulse is a deprecated alias."
        ),
    )
    parser.add_argument("--stage2_stiffness", default=72.0, type=float)
    parser.add_argument("--stage2_damping", default=4.0, type=float)
    parser.add_argument("--stage2_tangential_damping", default=1.2, type=float)
    parser.add_argument("--stage2_static_friction", default=0.0, type=float)
    parser.add_argument("--stage2_friction_transition_velocity", default=1e-3, type=float)
    parser.add_argument(
        "--stage2_friction_model",
        default=FRICTION_MODEL_SOFT_PROJECTION,
        choices=(FRICTION_MODEL_SOFT_PROJECTION, FRICTION_MODEL_DUAL_CONE),
        help="Tangential contact model. dual_cone uses a differentiable tangent-facet cone approximation.",
    )
    parser.add_argument("--stage2_friction_num_directions", default=8, type=int)
    parser.add_argument(
        "--stage2_patch_selection",
        default="spatial",
        choices=("spatial", "topk", "soft"),
        help="Contact patch construction for Stage 2 impedance dynamics. soft avoids discrete top-k/argmax patch identity.",
    )
    parser.add_argument(
        "--stage2_normal_mode",
        default="phi_soft",
        choices=("phi_soft", "signed_distance", "autograd"),
        help="Surface normal source for Gaussian SDF contacts. autograd is intended for validation and is slower.",
    )
    parser.add_argument("--floor_half_extent", default=1.8, type=float)
    parser.add_argument("--floor_thickness", default=0.025, type=float)
    parser.add_argument("--floor_resolution", default=17, type=int)
    parser.add_argument("--fit_physics_l2", default=1e-4, type=float)
    parser.add_argument("--save_refined_params", default=None, type=Path)
    parser.add_argument("--load_refined_params", default=None, type=Path)
    parser.add_argument("--fit_geometry_radii", action="store_true")
    parser.add_argument("--fit_geometry_centers", action="store_true")
    parser.add_argument("--fit_geometry_radius_l2", default=1e-3, type=float)
    parser.add_argument("--fit_geometry_center_l2", default=1e-2, type=float)
    parser.add_argument("--fit_geometry_max_log_radius_offset", default=0.7, type=float)
    parser.add_argument("--fit_geometry_max_center_offset", default=0.015, type=float)
    args = parser.parse_args()
    args.dynamics_backend = normalize_dynamics_backend(str(args.dynamics_backend))
    return args


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


def cube_inertia_diag(half_extent: float, mass: float = 1.0) -> tuple[float, float, float]:
    inertia = (2.0 / 3.0) * float(mass) * float(half_extent) * float(half_extent)
    return (inertia, inertia, inertia)


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


def scale_body_radii(body: GaussianCollisionBody, radius_multiplier: torch.Tensor | float) -> GaussianCollisionBody:
    multiplier = torch.as_tensor(radius_multiplier, dtype=body.radii.dtype, device=body.radii.device)
    radii = torch.clamp(body.radii * multiplier, min=2e-4)
    return GaussianCollisionBody(body.local_centers, radii, body.local_query_points)


def refine_body_geometry(
    body: GaussianCollisionBody,
    *,
    physics_params: dict[str, torch.Tensor] | None = None,
    geometry_params: dict[str, torch.Tensor] | None = None,
    max_log_radius_offset: float = 0.7,
    max_center_offset: float = 0.015,
) -> GaussianCollisionBody:
    centers = body.local_centers
    radii = body.radii
    if physics_params is not None and "radius_multiplier" in physics_params:
        radii = torch.clamp(radii * F.softplus(physics_params["radius_multiplier"]), min=2e-4)
    if geometry_params is not None and "log_radius_offsets" in geometry_params:
        log_offsets = torch.clamp(
            geometry_params["log_radius_offsets"].to(dtype=radii.dtype, device=radii.device),
            min=-abs(float(max_log_radius_offset)),
            max=abs(float(max_log_radius_offset)),
        )
        radii = torch.clamp(radii * torch.exp(log_offsets), min=2e-4)
    if geometry_params is not None and "center_offsets" in geometry_params:
        offsets = torch.clamp(
            geometry_params["center_offsets"].to(dtype=centers.dtype, device=centers.device),
            min=-abs(float(max_center_offset)),
            max=abs(float(max_center_offset)),
        )
        centers = centers + offsets
    query_points = body.local_query_points
    if query_points is not None and query_points.shape == body.local_centers.shape:
        query_points = centers
    return GaussianCollisionBody(centers, radii, query_points)


def build_floor_body(
    *,
    half_extent: float,
    thickness: float,
    resolution: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[GaussianCollisionBody, RigidBodyState]:
    floor_half_z = max(float(thickness), 1e-4)
    grid_resolution = max(3, int(resolution))
    query_points = make_box_surface_query_points(
        [float(half_extent), float(half_extent), floor_half_z],
        grid_resolution=grid_resolution,
        dtype=dtype,
        device=device,
    )
    floor_spacing = (2.0 * float(half_extent)) / float(max(grid_resolution - 1, 1))
    radii = torch.full(
        (query_points.shape[0],),
        max(floor_half_z * 0.75, floor_spacing * 0.55),
        dtype=dtype,
        device=device,
    )
    floor_body = GaussianCollisionBody(query_points, radii, query_points)
    floor_state = RigidBodyState(
        position=torch.tensor([0.0, 0.0, -floor_half_z], dtype=dtype, device=device),
        quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device),
        linear_velocity=torch.zeros(3, dtype=dtype, device=device),
        angular_velocity=torch.zeros(3, dtype=dtype, device=device),
    )
    return floor_body, floor_state


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
        patch_selection=str(args.stage2_patch_selection),
        normal_mode=str(args.stage2_normal_mode),
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


def make_stage2_dynamics(
    body: GaussianCollisionBody,
    *,
    dice_count: int,
    half_extent: float,
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
    dt: float,
    stiffness: torch.Tensor | None = None,
    damping: torch.Tensor | None = None,
    friction_coefficient: torch.Tensor | None = None,
    tangential_damping: torch.Tensor | None = None,
    mass_multiplier: torch.Tensor | None = None,
) -> tuple[MultiBodyGaussianImpedanceDynamics, RigidBodyState]:
    floor_body, floor_state = build_floor_body(
        half_extent=float(args.floor_half_extent),
        thickness=float(args.floor_thickness),
        resolution=int(args.floor_resolution),
        dtype=dtype,
        device=device,
    )
    bodies = tuple(body for _ in range(dice_count)) + (floor_body,)
    mass = 1.0
    inertia = cube_inertia_diag(half_extent, mass=mass)
    config = MultiBodyImpedanceDynamicsConfig(
        dt=float(dt),
        masses=tuple(mass for _ in range(dice_count)) + (1.0,),
        inertia_diags=tuple(inertia for _ in range(dice_count)) + (inertia,),
        dynamic_flags=tuple(True for _ in range(dice_count)) + (False,),
        gravity=(0.0, 0.0, -9.81),
        contact_softness=2e-3,
        smooth_min_temperature=1e-2,
        inside_penalty=0.02,
        inside_sharpness=50.0,
        normal_mode=str(args.stage2_normal_mode),
        num_contact_patches=4,
        broad_phase_margin=float(args.broad_phase_margin),
        broad_phase_mode="aabb",
        patch_selection=str(args.stage2_patch_selection),
        candidate_pair_mode="spatial_hash",
        spatial_hash_cell_size=float(args.spatial_hash_cell_size),
        contact_threshold=float(args.contact_threshold),
        linear_damping=float(args.linear_damping),
        angular_damping=float(args.angular_damping),
        friction_coefficient=float(args.pair_friction),
        static_friction_coefficient=(
            None if float(args.stage2_static_friction) <= 0.0 else float(args.stage2_static_friction)
        ),
        tangential_damping=float(args.stage2_tangential_damping),
        friction_softness=1e-6,
        friction_transition_velocity=float(args.stage2_friction_transition_velocity),
        friction_model=str(args.stage2_friction_model),
        friction_num_directions=int(args.stage2_friction_num_directions),
    )
    dynamics = MultiBodyGaussianImpedanceDynamics(
        bodies,
        stiffness=(
            torch.tensor(float(args.stage2_stiffness), dtype=dtype, device=device)
            if stiffness is None
            else stiffness
        ),
        damping=(
            torch.tensor(float(args.stage2_damping), dtype=dtype, device=device)
            if damping is None
            else damping
        ),
        friction_coefficient=friction_coefficient,
        tangential_damping=tangential_damping,
        mass_multiplier=mass_multiplier,
        names=tuple(f"die_{idx:02d}" for idx in range(dice_count)) + ("floor",),
        config=config,
    )
    return dynamics, floor_state


def _tensor_stat(value, *, op: str) -> float | None:
    if value is None or not torch.is_tensor(value) or value.numel() == 0:
        return None
    tensor = value.detach()
    if op == "max":
        return float(torch.max(tensor).cpu().item())
    if op == "mean":
        return float(torch.mean(tensor.float()).cpu().item())
    if op == "sum":
        return float(torch.sum(tensor).cpu().item())
    raise ValueError(f"Unsupported tensor stat op: {op}")


def summarize_stage2_step_diagnostics(diagnostics: dict) -> dict:
    """Compress one Stage2 substep's graph/friction diagnostics into JSON scalars."""

    friction_terms = list(diagnostics.get("friction", []))
    edge_gates = diagnostics.get("edge_gates", [])
    lambda_terms = diagnostics.get("lambda", [])
    contact_edges = 0
    max_edge_gate = 0.0
    max_lambda = 0.0
    max_friction_force = 0.0
    max_raw_friction = 0.0
    max_cone_violation = 0.0
    max_cone_ratio = 0.0
    max_facet_budget = 0.0
    max_facet_reconstruction_error = 0.0
    mean_slip_terms = []
    models = set()

    for edge_gate in edge_gates:
        gate_value = _tensor_stat(edge_gate, op="max")
        if gate_value is not None:
            max_edge_gate = max(max_edge_gate, gate_value)

    for lambdas in lambda_terms:
        lambda_value = _tensor_stat(lambdas, op="max")
        if lambda_value is not None:
            max_lambda = max(max_lambda, lambda_value)

    for term in friction_terms:
        models.add(str(term.get("friction_model", "")))
        edge_gate_value = _tensor_stat(term.get("edge_gate"), op="max")
        if edge_gate_value is not None and edge_gate_value > 0.0:
            contact_edges += 1
        for key, target in (
            ("friction_force_norm", "max_friction_force"),
            ("raw_friction_norm", "max_raw_friction"),
            ("friction_cone_violation", "max_cone_violation"),
            ("friction_force_to_cone_radius_ratio", "max_cone_ratio"),
            ("friction_facet_budget", "max_facet_budget"),
            ("friction_facet_reconstruction_error", "max_facet_reconstruction_error"),
        ):
            value = _tensor_stat(term.get(key), op="max")
            if value is None:
                continue
            if target == "max_friction_force":
                max_friction_force = max(max_friction_force, value)
            elif target == "max_raw_friction":
                max_raw_friction = max(max_raw_friction, value)
            elif target == "max_cone_violation":
                max_cone_violation = max(max_cone_violation, value)
            elif target == "max_cone_ratio":
                max_cone_ratio = max(max_cone_ratio, value)
            elif target == "max_facet_budget":
                max_facet_budget = max(max_facet_budget, value)
            elif target == "max_facet_reconstruction_error":
                max_facet_reconstruction_error = max(max_facet_reconstruction_error, value)
        slip_mean = _tensor_stat(term.get("slip_speed"), op="mean")
        if slip_mean is not None:
            mean_slip_terms.append(slip_mean)

    return {
        "candidate_edges": int(len(diagnostics.get("candidate_edges", []))),
        "active_edges": int(len(diagnostics.get("active_edges", []))),
        "contact_edges_with_gate": int(contact_edges),
        "max_edge_gate": float(max_edge_gate),
        "max_lambda": float(max_lambda),
        "max_friction_force": float(max_friction_force),
        "max_raw_friction": float(max_raw_friction),
        "max_friction_cone_violation": float(max_cone_violation),
        "max_friction_force_to_cone_radius_ratio": float(max_cone_ratio),
        "mean_slip_speed": float(np.mean(mean_slip_terms)) if mean_slip_terms else 0.0,
        "max_friction_facet_budget": float(max_facet_budget),
        "max_friction_facet_reconstruction_error": float(max_facet_reconstruction_error),
        "friction_models": sorted(model for model in models if model),
    }


def aggregate_stage2_frame_diagnostics(substep_rows: list[dict]) -> dict:
    if not substep_rows:
        return {
            "candidate_edges": 0,
            "active_edges": 0,
            "contact_edges_with_gate": 0,
            "max_edge_gate": 0.0,
            "max_lambda": 0.0,
            "max_friction_force": 0.0,
            "max_raw_friction": 0.0,
            "max_friction_cone_violation": 0.0,
            "max_friction_force_to_cone_radius_ratio": 0.0,
            "mean_slip_speed": 0.0,
            "max_friction_facet_budget": 0.0,
            "max_friction_facet_reconstruction_error": 0.0,
            "friction_models": [],
        }
    models = sorted({model for row in substep_rows for model in row.get("friction_models", [])})
    return {
        "candidate_edges": int(sum(row["candidate_edges"] for row in substep_rows)),
        "active_edges": int(sum(row["active_edges"] for row in substep_rows)),
        "contact_edges_with_gate": int(sum(row["contact_edges_with_gate"] for row in substep_rows)),
        "max_edge_gate": float(max(row["max_edge_gate"] for row in substep_rows)),
        "max_lambda": float(max(row["max_lambda"] for row in substep_rows)),
        "max_friction_force": float(max(row["max_friction_force"] for row in substep_rows)),
        "max_raw_friction": float(max(row["max_raw_friction"] for row in substep_rows)),
        "max_friction_cone_violation": float(max(row["max_friction_cone_violation"] for row in substep_rows)),
        "max_friction_force_to_cone_radius_ratio": float(
            max(row["max_friction_force_to_cone_radius_ratio"] for row in substep_rows)
        ),
        "mean_slip_speed": float(np.mean([row["mean_slip_speed"] for row in substep_rows])),
        "max_friction_facet_budget": float(max(row["max_friction_facet_budget"] for row in substep_rows)),
        "max_friction_facet_reconstruction_error": float(
            max(row["max_friction_facet_reconstruction_error"] for row in substep_rows)
        ),
        "friction_models": models,
    }


def aggregate_stage2_rollout_diagnostics(frame_rows: list[dict]) -> dict:
    if not frame_rows:
        return {}
    models = sorted({model for row in frame_rows for model in row.get("friction_models", [])})
    return {
        "friction_models": models,
        "frame_count": int(len(frame_rows)),
        "total_candidate_edges": int(sum(row["candidate_edges"] for row in frame_rows)),
        "total_active_edges": int(sum(row["active_edges"] for row in frame_rows)),
        "total_contact_edges_with_gate": int(sum(row["contact_edges_with_gate"] for row in frame_rows)),
        "max_edge_gate": float(max(row["max_edge_gate"] for row in frame_rows)),
        "max_lambda": float(max(row["max_lambda"] for row in frame_rows)),
        "max_friction_force": float(max(row["max_friction_force"] for row in frame_rows)),
        "max_raw_friction": float(max(row["max_raw_friction"] for row in frame_rows)),
        "max_friction_cone_violation": float(max(row["max_friction_cone_violation"] for row in frame_rows)),
        "max_friction_force_to_cone_radius_ratio": float(
            max(row["max_friction_force_to_cone_radius_ratio"] for row in frame_rows)
        ),
        "mean_slip_speed": float(np.mean([row["mean_slip_speed"] for row in frame_rows])),
        "max_friction_facet_budget": float(max(row["max_friction_facet_budget"] for row in frame_rows)),
        "max_friction_facet_reconstruction_error": float(
            max(row["max_friction_facet_reconstruction_error"] for row in frame_rows)
        ),
        "frames": frame_rows,
    }


def step_states_stage2_impedance(
    states: list[SimState],
    dynamics: MultiBodyGaussianImpedanceDynamics,
    floor_state: RigidBodyState,
) -> tuple[int, dict]:
    rigid_states = [state.rigid() for state in states] + [floor_state]
    next_states, diagnostics = dynamics.step(rigid_states)
    for idx, next_state in enumerate(next_states[: len(states)]):
        states[idx] = SimState(
            position=next_state.position,
            quaternion_wxyz=next_state.quaternion_wxyz,
            linear_velocity=next_state.linear_velocity,
            angular_velocity=next_state.angular_velocity,
        )
    return len(diagnostics["active_edges"]), summarize_stage2_step_diagnostics(diagnostics)


def rollout(
    payload: dict,
    body: GaussianCollisionBody,
    args: argparse.Namespace,
    *,
    linear_velocity_delta: torch.Tensor | None = None,
    angular_velocity_delta: torch.Tensor | None = None,
    physics_params: dict[str, torch.Tensor] | None = None,
    geometry_params: dict[str, torch.Tensor] | None = None,
    return_position_tensor: bool = False,
    return_state_tensors: bool = False,
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
    rollout_body = refine_body_geometry(
        body,
        physics_params=physics_params,
        geometry_params=geometry_params,
        max_log_radius_offset=float(args.fit_geometry_max_log_radius_offset),
        max_center_offset=float(args.fit_geometry_max_center_offset),
    )
    stage2_dynamics = None
    floor_state = None
    if args.dynamics_backend == DYNAMICS_BACKEND_STAGE2_IMPEDANCE:
        stage2_dynamics, floor_state = make_stage2_dynamics(
            rollout_body,
            dice_count=dice_count,
            half_extent=half_extent,
            args=args,
            dtype=dtype,
            device=device,
            dt=dt,
            stiffness=None if physics_params is None else physics_params.get("stiffness"),
            damping=None if physics_params is None else physics_params.get("damping"),
            friction_coefficient=None if physics_params is None else physics_params.get("friction_coefficient"),
            tangential_damping=None if physics_params is None else physics_params.get("tangential_damping"),
            mass_multiplier=None if physics_params is None else physics_params.get("mass_multiplier"),
        )
    predicted = []
    predicted_position_tensors = []
    predicted_quaternion_tensors = []
    active_counts = []
    stage2_frame_diagnostics = []
    for frame_idx, source_frame in enumerate(source_states):
        predicted_position_tensors.append(torch.stack([state.position for state in states], dim=0))
        predicted_quaternion_tensors.append(torch.stack([state.quaternion_wxyz for state in states], dim=0))
        predicted.append(
            {
                "frame_index": int(source_frame.get("frame_index", frame_idx)),
                "time": float(source_frame.get("time", frame_idx * frame_dt)),
                "dice": [state.to_serializable(idx) for idx, state in enumerate(states)],
            }
        )
        step_active = 0
        stage2_substep_rows = []
        for _ in range(substeps):
            if args.dynamics_backend == DYNAMICS_BACKEND_STAGE2_IMPEDANCE:
                active_edges, substep_diagnostics = step_states_stage2_impedance(states, stage2_dynamics, floor_state)
                step_active += active_edges
                stage2_substep_rows.append(substep_diagnostics)
            else:
                step_active += step_states(states, [rollout_body for _ in range(dice_count)], args=args, half_extent=half_extent, dt=dt)
        active_counts.append(step_active)
        if args.dynamics_backend == DYNAMICS_BACKEND_STAGE2_IMPEDANCE:
            frame_diagnostics = aggregate_stage2_frame_diagnostics(stage2_substep_rows)
            frame_diagnostics["frame_index"] = int(source_frame.get("frame_index", frame_idx))
            frame_diagnostics["time"] = float(source_frame.get("time", frame_idx * frame_dt))
            stage2_frame_diagnostics.append(frame_diagnostics)
    pred_position_tensor = torch.stack(predicted_position_tensors, dim=0)
    pred_quaternion_tensor = torch.stack(predicted_quaternion_tensors, dim=0)
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
        "dynamics_backend": str(args.dynamics_backend),
    }
    if stage2_frame_diagnostics:
        metrics["stage2_contact_diagnostics"] = aggregate_stage2_rollout_diagnostics(stage2_frame_diagnostics)
    if return_state_tensors:
        return predicted, metrics, {"positions": pred_position_tensor, "quaternions": pred_quaternion_tensor}
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
    physics_params: dict[str, torch.Tensor] | None = None,
    geometry_params: dict[str, torch.Tensor] | None = None,
    frames_limit: int | None = None,
) -> torch.Tensor:
    result = rollout(
        payload,
        body,
        args,
        linear_velocity_delta=linear_velocity_delta,
        angular_velocity_delta=angular_velocity_delta,
        physics_params=physics_params,
        geometry_params=geometry_params,
        return_position_tensor=True,
        frames_limit=frames_limit,
    )
    return result[2]


def rollout_state_tensors(
    payload: dict,
    body: GaussianCollisionBody,
    args: argparse.Namespace,
    *,
    linear_velocity_delta: torch.Tensor,
    angular_velocity_delta: torch.Tensor,
    physics_params: dict[str, torch.Tensor] | None = None,
    geometry_params: dict[str, torch.Tensor] | None = None,
    frames_limit: int | None = None,
) -> dict[str, torch.Tensor]:
    result = rollout(
        payload,
        body,
        args,
        linear_velocity_delta=linear_velocity_delta,
        angular_velocity_delta=angular_velocity_delta,
        physics_params=physics_params,
        geometry_params=geometry_params,
        return_state_tensors=True,
        frames_limit=frames_limit,
    )
    return result[2]


def load_mask_sequence(
    mask_dir: Path,
    states: list[dict],
    *,
    resolution: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    masks = []
    for frame_idx, frame in enumerate(states):
        index = int(frame.get("frame_index", frame_idx))
        path = mask_dir / f"{index:06d}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing mask frame: {path}")
        mask = Image.open(path).convert("L").resize((resolution, resolution), Image.Resampling.BILINEAR)
        array = np.asarray(mask, dtype=np.float32) / 255.0
        masks.append(torch.as_tensor(array, dtype=dtype, device=device))
    return torch.stack(masks, dim=0)


def topdown_soft_silhouette(
    positions: torch.Tensor,
    quaternions_wxyz: torch.Tensor,
    *,
    half_extent: float,
    resolution: int,
    softness: float,
) -> torch.Tensor:
    """Differentiable top-down dice silhouette used as a lightweight image loss."""

    dtype = positions.dtype
    device = positions.device
    xs = torch.linspace(-0.78, 0.78, resolution, dtype=dtype, device=device)
    ys = torch.linspace(0.72, -0.72, resolution, dtype=dtype, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(1, 1, resolution, resolution, 2)
    pos_xy = positions[..., :2].reshape(positions.shape[0], positions.shape[1], 1, 1, 2)
    w, x, y, z = quaternions_wxyz.unbind(dim=-1)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    delta = grid - pos_xy
    local_x = r00.reshape(*r00.shape, 1, 1) * delta[..., 0] + r10.reshape(*r10.shape, 1, 1) * delta[..., 1]
    local_y = r01.reshape(*r01.shape, 1, 1) * delta[..., 0] + r11.reshape(*r11.shape, 1, 1) * delta[..., 1]
    signed_distance = torch.maximum(torch.abs(local_x), torch.abs(local_y)) - float(half_extent)
    occupancy_per_die = torch.sigmoid(-signed_distance / max(float(softness), 1e-6))
    union = 1.0 - torch.prod(1.0 - occupancy_per_die, dim=1)
    return union


def mask_sequence_for_fit(
    payload: dict,
    states: list[dict],
    args: argparse.Namespace,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if float(args.mask_loss_weight) <= 0.0 or args.gt_mask_dir is None:
        return None
    return load_mask_sequence(
        args.gt_mask_dir.resolve(),
        states,
        resolution=max(8, int(args.mask_loss_resolution)),
        dtype=dtype,
        device=device,
    )


def inverse_softplus(value: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(float(value), dtype=dtype, device=device)
    return torch.log(torch.expm1(torch.clamp(tensor, min=1e-6)))


PHYSICS_PARAM_NAMES = (
    "stiffness",
    "damping",
    "friction_coefficient",
    "tangential_damping",
    "mass_multiplier",
    "radius_multiplier",
)


GEOMETRY_PARAM_NAMES = ("log_radius_offsets", "center_offsets")


def physics_params_to_serializable(params: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        "stiffness": float(F.softplus(params["stiffness"].detach()).cpu().item()),
        "damping": float(F.softplus(params["damping"].detach()).cpu().item()),
        "friction_coefficient": float(F.softplus(params["friction_coefficient"].detach()).cpu().item()),
        "tangential_damping": float(F.softplus(params["tangential_damping"].detach()).cpu().item()),
        "mass_multiplier": float(F.softplus(params["mass_multiplier"].detach()).cpu().item()),
        "radius_multiplier": float(F.softplus(params["radius_multiplier"].detach()).cpu().item()),
    }


def physics_values_to_raw_params(
    values: dict[str, float],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    missing = [name for name in PHYSICS_PARAM_NAMES if name not in values]
    if missing:
        raise ValueError(f"Refined physics params are missing keys: {missing}")
    return {
        name: inverse_softplus(float(values[name]), dtype=dtype, device=device)
        for name in PHYSICS_PARAM_NAMES
    }


def make_learnable_physics_params(args: argparse.Namespace, *, dtype: torch.dtype, device: torch.device) -> dict[str, torch.nn.Parameter]:
    return {
        "stiffness": torch.nn.Parameter(inverse_softplus(float(args.stage2_stiffness), dtype=dtype, device=device)),
        "damping": torch.nn.Parameter(inverse_softplus(float(args.stage2_damping), dtype=dtype, device=device)),
        "friction_coefficient": torch.nn.Parameter(inverse_softplus(float(args.pair_friction), dtype=dtype, device=device)),
        "tangential_damping": torch.nn.Parameter(
            inverse_softplus(float(args.stage2_tangential_damping), dtype=dtype, device=device)
        ),
        "mass_multiplier": torch.nn.Parameter(inverse_softplus(1.0, dtype=dtype, device=device)),
        "radius_multiplier": torch.nn.Parameter(inverse_softplus(1.0, dtype=dtype, device=device)),
    }


def geometry_params_to_serializable(
    params: dict[str, torch.Tensor] | None,
    *,
    max_log_radius_offset: float,
    max_center_offset: float,
) -> dict | None:
    if not params:
        return None
    payload = {
        "max_log_radius_offset": float(max_log_radius_offset),
        "max_center_offset": float(max_center_offset),
    }
    if "log_radius_offsets" in params:
        offsets = torch.clamp(
            params["log_radius_offsets"].detach(),
            min=-abs(float(max_log_radius_offset)),
            max=abs(float(max_log_radius_offset)),
        )
        payload["log_radius_offsets"] = offsets.cpu().tolist()
        payload["radius_multipliers"] = torch.exp(offsets).cpu().tolist()
    if "center_offsets" in params:
        offsets = torch.clamp(
            params["center_offsets"].detach(),
            min=-abs(float(max_center_offset)),
            max=abs(float(max_center_offset)),
        )
        payload["center_offsets"] = offsets.cpu().tolist()
    return payload


def make_learnable_geometry_params(
    body: GaussianCollisionBody,
    args: argparse.Namespace,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.nn.Parameter]:
    params: dict[str, torch.nn.Parameter] = {}
    if bool(args.fit_geometry_radii):
        params["log_radius_offsets"] = torch.nn.Parameter(
            torch.zeros_like(body.radii, dtype=dtype, device=device)
        )
    if bool(args.fit_geometry_centers):
        params["center_offsets"] = torch.nn.Parameter(
            torch.zeros_like(body.local_centers, dtype=dtype, device=device)
        )
    return params


def geometry_values_to_raw_params(
    values: dict | None,
    *,
    body: GaussianCollisionBody,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if not values:
        return None
    params: dict[str, torch.Tensor] = {}
    if values.get("log_radius_offsets") is not None:
        tensor = torch.as_tensor(values["log_radius_offsets"], dtype=dtype, device=device)
        if tuple(tensor.shape) != tuple(body.radii.shape):
            raise ValueError(f"log_radius_offsets shape must be {tuple(body.radii.shape)}, got {tuple(tensor.shape)}.")
        params["log_radius_offsets"] = tensor
    elif values.get("radius_multipliers") is not None:
        tensor = torch.as_tensor(values["radius_multipliers"], dtype=dtype, device=device)
        if tuple(tensor.shape) != tuple(body.radii.shape):
            raise ValueError(f"radius_multipliers shape must be {tuple(body.radii.shape)}, got {tuple(tensor.shape)}.")
        params["log_radius_offsets"] = torch.log(torch.clamp(tensor, min=1e-6))
    if values.get("center_offsets") is not None:
        tensor = torch.as_tensor(values["center_offsets"], dtype=dtype, device=device)
        if tuple(tensor.shape) != tuple(body.local_centers.shape):
            raise ValueError(
                f"center_offsets shape must be {tuple(body.local_centers.shape)}, got {tuple(tensor.shape)}."
            )
        params["center_offsets"] = tensor
    return params or None


def read_refined_params(
    path: Path,
    *,
    body: GaussianCollisionBody,
    dice_count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    dict[str, torch.Tensor] | None,
    dict[str, torch.Tensor] | None,
    dict,
]:
    payload = read_json(path)
    physics_values = payload.get("physics")
    if physics_values is None:
        physics_values = (payload.get("physics_fit") or {}).get("learned_physics")
    if physics_values is None:
        physics_values = payload.get("learned_physics")
    physics_params = None
    if physics_values is not None:
        physics_params = physics_values_to_raw_params(physics_values, dtype=dtype, device=device)
    geometry_params = geometry_values_to_raw_params(
        payload.get("geometry"),
        body=body,
        dtype=dtype,
        device=device,
    )

    linear_velocity_delta = payload.get("linear_velocity_delta")
    angular_velocity_delta = payload.get("angular_velocity_delta")
    linear_tensor = None
    angular_tensor = None
    if linear_velocity_delta is not None:
        linear_tensor = torch.as_tensor(linear_velocity_delta, dtype=dtype, device=device)
    if angular_velocity_delta is not None:
        angular_tensor = torch.as_tensor(angular_velocity_delta, dtype=dtype, device=device)
    for label, tensor in (("linear_velocity_delta", linear_tensor), ("angular_velocity_delta", angular_tensor)):
        if tensor is not None and tuple(tensor.shape) != (int(dice_count), 3):
            raise ValueError(f"{label} shape must be {(int(dice_count), 3)}, got {tuple(tensor.shape)}.")
    return linear_tensor, angular_tensor, physics_params, geometry_params, payload


def write_refined_params(
    path: Path,
    *,
    trajectory: Path,
    stage1_ply: Path,
    stage1_body: dict,
    linear_velocity_delta: torch.Tensor | None,
    angular_velocity_delta: torch.Tensor | None,
    physics_params: dict[str, torch.Tensor] | None,
    geometry_params: dict[str, torch.Tensor] | None,
    fit_summary: dict | None,
    source: str,
    args: argparse.Namespace,
) -> None:
    physics_values = None if physics_params is None else physics_params_to_serializable(physics_params)
    payload = {
        "format": "cgwm_stage2_refined_params_v1",
        "source": source,
        "trajectory": str(trajectory.resolve()),
        "stage1_ply": str(stage1_ply.resolve()),
        "stage1_body": stage1_body,
        "dynamics_backend": str(args.dynamics_backend),
        "contact_settings": {
            "friction_model": str(args.stage2_friction_model),
            "friction_num_directions": int(args.stage2_friction_num_directions),
            "patch_selection": str(args.stage2_patch_selection),
            "normal_mode": str(args.stage2_normal_mode),
            "dynamic_friction": float(args.pair_friction),
            "static_friction": None if float(args.stage2_static_friction) <= 0.0 else float(args.stage2_static_friction),
            "tangential_damping": float(args.stage2_tangential_damping),
            "transition_velocity": float(args.stage2_friction_transition_velocity),
        },
        "linear_velocity_delta": (
            None if linear_velocity_delta is None else linear_velocity_delta.detach().cpu().tolist()
        ),
        "angular_velocity_delta": (
            None if angular_velocity_delta is None else angular_velocity_delta.detach().cpu().tolist()
        ),
        "physics": physics_values,
        "geometry": geometry_params_to_serializable(
            geometry_params,
            max_log_radius_offset=float(args.fit_geometry_max_log_radius_offset),
            max_center_offset=float(args.fit_geometry_max_center_offset),
        ),
        "fit_summary": fit_summary,
    }
    write_json(path, payload)


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
    half_extent = float(payload.get("half_extent", 0.055))
    target_positions = torch.tensor(
        [[die["position"] for die in frame["dice"]] for frame in source_states],
        dtype=dtype,
        device=device,
    )
    target_masks = mask_sequence_for_fit(payload, source_states, args, dtype=dtype, device=device)
    half_extent = float(payload.get("half_extent", 0.055))
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


def fit_stage2_physics(
    payload: dict,
    body: GaussianCollisionBody,
    args: argparse.Namespace,
    *,
    stage1_to_mujoco_scale: float = 1.0,
    initial_linear_velocity_delta: torch.Tensor | None = None,
    initial_angular_velocity_delta: torch.Tensor | None = None,
    initial_physics_params: dict[str, torch.Tensor] | None = None,
    initial_geometry_params: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor] | None, dict]:
    if args.dynamics_backend != DYNAMICS_BACKEND_STAGE2_IMPEDANCE:
        raise ValueError("--fit_physics_iters requires --dynamics_backend stage2_impedance.")
    source_states = payload["states"]
    if args.max_frames > 0:
        source_states = source_states[: int(args.max_frames)]
    horizon = int(args.fit_horizon_frames)
    if horizon > 0:
        source_states = source_states[:horizon]
    device = torch.device(args.device)
    dtype = torch.float32
    dice_count = int(payload.get("dice_count", len(source_states[0]["dice"])))
    half_extent = float(payload.get("half_extent", 0.055))
    target_positions = torch.tensor(
        [[die["position"] for die in frame["dice"]] for frame in source_states],
        dtype=dtype,
        device=device,
    )
    target_masks = mask_sequence_for_fit(payload, source_states, args, dtype=dtype, device=device)
    gaussian_render_loss = None
    gaussian_render_indices = []
    if float(args.gaussian_rgb_loss_weight) > 0.0:
        if device.type != "cuda":
            raise ValueError("--gaussian_rgb_loss_weight requires --device cuda.")
        rgb_dir = args.gaussian_rgb_dir if args.gaussian_rgb_dir is not None else args.gt_rgb_dir
        if rgb_dir is None:
            raise ValueError("--gaussian_rgb_loss_weight requires --gaussian_rgb_dir or --gt_rgb_dir.")
        from gaussian_initiailization.stage2.differentiable_gaussian_render_loss import (
            GaussianRenderLossConfig,
            Stage2GaussianRenderLoss,
        )

        render_stride = max(1, int(args.gaussian_render_stride))
        gaussian_render_indices = list(range(0, len(source_states), render_stride))
        frame_indices = [
            int(source_states[state_idx].get("frame_index", state_idx))
            for state_idx in gaussian_render_indices
        ]
        gaussian_render_loss = Stage2GaussianRenderLoss(
            stage1_ply=args.stage1_ply.resolve(),
            gt_rgb_dir=rgb_dir.resolve(),
            frame_indices=frame_indices,
            config=GaussianRenderLossConfig(
                image_width=max(16, int(args.gaussian_render_width)),
                image_height=max(16, int(args.gaussian_render_height)),
                white_background=bool(args.gaussian_render_white_background),
                scale_multiplier=float(stage1_to_mujoco_scale),
                loss=str(args.gaussian_render_loss),
            ),
            dtype=dtype,
            device=device,
        )
    if initial_linear_velocity_delta is None:
        linear_velocity_delta = torch.zeros((dice_count, 3), dtype=dtype, device=device)
    else:
        linear_velocity_delta = initial_linear_velocity_delta.detach().to(dtype=dtype, device=device)
    if initial_angular_velocity_delta is None:
        angular_velocity_delta = torch.zeros((dice_count, 3), dtype=dtype, device=device)
    else:
        angular_velocity_delta = initial_angular_velocity_delta.detach().to(dtype=dtype, device=device)
    linear_velocity_delta = torch.nn.Parameter(linear_velocity_delta.clone())
    angular_velocity_delta = torch.nn.Parameter(angular_velocity_delta.clone())
    if initial_physics_params is None:
        physics_params = make_learnable_physics_params(args, dtype=dtype, device=device)
    else:
        physics_params = {
            name: torch.nn.Parameter(initial_physics_params[name].detach().to(dtype=dtype, device=device).clone())
            for name in PHYSICS_PARAM_NAMES
        }
    if initial_geometry_params is None:
        geometry_params = make_learnable_geometry_params(body, args, dtype=dtype, device=device)
    else:
        geometry_params = {
            name: torch.nn.Parameter(value.detach().to(dtype=dtype, device=device).clone())
            for name, value in initial_geometry_params.items()
        }
    initial_raw = {name: param.detach().clone() for name, param in physics_params.items()}
    initial_geometry_raw = {name: param.detach().clone() for name, param in geometry_params.items()}
    learnable = [linear_velocity_delta, angular_velocity_delta, *physics_params.values(), *geometry_params.values()]
    optimizer = torch.optim.Adam(learnable, lr=float(args.fit_physics_lr))
    history = []
    best = {
        "loss": float("inf"),
        "linear_velocity_delta": None,
        "angular_velocity_delta": None,
        "physics_params": None,
        "geometry_params": None,
    }
    for fit_iter in range(int(args.fit_physics_iters)):
        optimizer.zero_grad(set_to_none=True)
        needs_state_tensors = target_masks is not None or gaussian_render_loss is not None
        if not needs_state_tensors:
            pred_positions = rollout_tensor(
                payload,
                body,
                args,
                linear_velocity_delta=linear_velocity_delta,
                angular_velocity_delta=angular_velocity_delta,
                physics_params=physics_params,
                geometry_params=geometry_params,
                frames_limit=horizon if horizon > 0 else None,
            )
            mask_loss = torch.zeros((), dtype=dtype, device=device)
            gaussian_rgb_loss = torch.zeros((), dtype=dtype, device=device)
            gaussian_rgb_diagnostics = {}
        else:
            pred_states = rollout_state_tensors(
                payload,
                body,
                args,
                linear_velocity_delta=linear_velocity_delta,
                angular_velocity_delta=angular_velocity_delta,
                physics_params=physics_params,
                geometry_params=geometry_params,
                frames_limit=horizon if horizon > 0 else None,
            )
            pred_positions = pred_states["positions"]
            if target_masks is None:
                mask_loss = torch.zeros((), dtype=dtype, device=device)
            else:
                pred_masks = topdown_soft_silhouette(
                    pred_states["positions"],
                    pred_states["quaternions"],
                    half_extent=half_extent,
                    resolution=max(8, int(args.mask_loss_resolution)),
                    softness=float(args.mask_loss_softness),
                )
                mask_loss = F.binary_cross_entropy(
                    torch.clamp(pred_masks, 1e-5, 1.0 - 1e-5),
                    torch.clamp(target_masks, 0.0, 1.0),
                )
            if gaussian_render_loss is None:
                gaussian_rgb_loss = torch.zeros((), dtype=dtype, device=device)
                gaussian_rgb_diagnostics = {}
            else:
                gaussian_rgb_loss, gaussian_rgb_diagnostics = gaussian_render_loss(
                    pred_states["positions"][gaussian_render_indices],
                    pred_states["quaternions"][gaussian_render_indices],
                )
        position_loss = torch.mean((pred_positions - target_positions) ** 2)
        velocity_regularizer = float(args.fit_initial_velocity_l2) * (
            torch.mean(linear_velocity_delta**2) + 0.01 * torch.mean(angular_velocity_delta**2)
        )
        physics_regularizer = torch.zeros((), dtype=dtype, device=device)
        for name, param in physics_params.items():
            physics_regularizer = physics_regularizer + torch.mean((param - initial_raw[name]) ** 2)
        geometry_radius_regularizer = torch.zeros((), dtype=dtype, device=device)
        if "log_radius_offsets" in geometry_params:
            geometry_radius_regularizer = torch.mean(
                (geometry_params["log_radius_offsets"] - initial_geometry_raw["log_radius_offsets"]) ** 2
            )
        geometry_center_regularizer = torch.zeros((), dtype=dtype, device=device)
        if "center_offsets" in geometry_params:
            geometry_center_regularizer = torch.mean(
                (geometry_params["center_offsets"] - initial_geometry_raw["center_offsets"]) ** 2
            )
        loss = (
            float(args.fit_position_weight) * position_loss
            + float(args.mask_loss_weight) * mask_loss
            + float(args.gaussian_rgb_loss_weight) * gaussian_rgb_loss
            + velocity_regularizer
            + float(args.fit_physics_l2) * physics_regularizer
            + float(args.fit_geometry_radius_l2) * geometry_radius_regularizer
            + float(args.fit_geometry_center_l2) * geometry_center_regularizer
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            linear_velocity_delta.clamp_(-4.0, 4.0)
            angular_velocity_delta.clamp_(-20.0, 20.0)
            physics_params["mass_multiplier"].clamp_(inverse_softplus(0.25, dtype=dtype, device=device), inverse_softplus(4.0, dtype=dtype, device=device))
            physics_params["radius_multiplier"].clamp_(inverse_softplus(0.35, dtype=dtype, device=device), inverse_softplus(2.5, dtype=dtype, device=device))
            if "log_radius_offsets" in geometry_params:
                geometry_params["log_radius_offsets"].clamp_(
                    -abs(float(args.fit_geometry_max_log_radius_offset)),
                    abs(float(args.fit_geometry_max_log_radius_offset)),
                )
            if "center_offsets" in geometry_params:
                geometry_params["center_offsets"].clamp_(
                    -abs(float(args.fit_geometry_max_center_offset)),
                    abs(float(args.fit_geometry_max_center_offset)),
                )
        loss_value = float(loss.detach().cpu().item())
        serial_params = physics_params_to_serializable(physics_params)
        serial_geometry = geometry_params_to_serializable(
            geometry_params,
            max_log_radius_offset=float(args.fit_geometry_max_log_radius_offset),
            max_center_offset=float(args.fit_geometry_max_center_offset),
        ) or {}
        history.append(
            {
                "iter": fit_iter,
                "loss": loss_value,
                "position_mse": float(position_loss.detach().cpu().item()),
                "mask_bce": float(mask_loss.detach().cpu().item()),
                "gaussian_rgb_loss": float(gaussian_rgb_loss.detach().cpu().item()),
                "geometry_radius_l2": float(geometry_radius_regularizer.detach().cpu().item()),
                "geometry_center_l2": float(geometry_center_regularizer.detach().cpu().item()),
                "geometry_num_radius_offsets": int(geometry_params.get("log_radius_offsets", torch.empty(0)).numel()),
                "geometry_num_center_offsets": int(geometry_params.get("center_offsets", torch.empty(0)).numel()),
                "linear_delta_norm": float(torch.linalg.norm(linear_velocity_delta.detach()).cpu().item()),
                "angular_delta_norm": float(torch.linalg.norm(angular_velocity_delta.detach()).cpu().item()),
                **serial_params,
                "geometry": serial_geometry,
                **gaussian_rgb_diagnostics,
            }
        )
        if loss_value < best["loss"]:
            best["loss"] = loss_value
            best["linear_velocity_delta"] = linear_velocity_delta.detach().clone()
            best["angular_velocity_delta"] = angular_velocity_delta.detach().clone()
            best["physics_params"] = {name: param.detach().clone() for name, param in physics_params.items()}
            best["geometry_params"] = {name: param.detach().clone() for name, param in geometry_params.items()}
        if fit_iter == 0 or (fit_iter + 1) % 10 == 0 or fit_iter == int(args.fit_physics_iters) - 1:
            print(json.dumps(history[-1]))
    return (
        best["linear_velocity_delta"],
        best["angular_velocity_delta"],
        best["physics_params"],
        best["geometry_params"],
        {
            "history": history,
            "best_loss": best["loss"],
            "learned_physics": physics_params_to_serializable(best["physics_params"]),
            "learned_geometry": geometry_params_to_serializable(
                best["geometry_params"],
                max_log_radius_offset=float(args.fit_geometry_max_log_radius_offset),
                max_center_offset=float(args.fit_geometry_max_center_offset),
            ),
            "kind": "stage2_phys_geo_refinement",
            "image_space_supervision": {
                "enabled": target_masks is not None or gaussian_render_loss is not None,
                "gt_mask_dir": None if args.gt_mask_dir is None else str(args.gt_mask_dir.resolve()),
                "mask_loss_weight": float(args.mask_loss_weight),
                "mask_loss_resolution": int(args.mask_loss_resolution),
                "mask_loss_softness": float(args.mask_loss_softness),
                "gaussian_rgb_enabled": gaussian_render_loss is not None,
                "gaussian_rgb_dir": (
                    None
                    if (args.gaussian_rgb_dir is None and args.gt_rgb_dir is None)
                    else str((args.gaussian_rgb_dir if args.gaussian_rgb_dir is not None else args.gt_rgb_dir).resolve())
                ),
                "gaussian_rgb_loss_weight": float(args.gaussian_rgb_loss_weight),
                "gaussian_render_width": int(args.gaussian_render_width),
                "gaussian_render_height": int(args.gaussian_render_height),
                "gaussian_render_stride": int(args.gaussian_render_stride),
                "gaussian_render_loss": str(args.gaussian_render_loss),
                "gaussian_render_scale_multiplier": float(stage1_to_mujoco_scale),
            },
        },
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
    import imageio.v2 as imageio
    from gaussian_initiailization.tools.generate_mujoco_multi_dice_rollout import build_mjcf

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
    import imageio.v2 as imageio

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
    physics_fit_summary = None
    linear_velocity_delta = None
    angular_velocity_delta = None
    physics_params = None
    geometry_params = None
    loaded_refined_params = None
    loaded_refined_params_path = None
    if args.load_refined_params is not None:
        dice_count = int(payload.get("dice_count", len(payload["states"][0]["dice"])))
        (
            linear_velocity_delta,
            angular_velocity_delta,
            physics_params,
            geometry_params,
            loaded_refined_params,
        ) = read_refined_params(
            args.load_refined_params.resolve(),
            body=body,
            dice_count=dice_count,
            dtype=dtype,
            device=device,
        )
        loaded_refined_params_path = str(args.load_refined_params.resolve())
    if int(args.fit_iters) > 0:
        linear_velocity_delta, angular_velocity_delta, fit_summary = fit_stage2_initial_state(payload, body, args)
    if int(args.fit_physics_iters) > 0:
        (
            linear_velocity_delta,
            angular_velocity_delta,
            physics_params,
            geometry_params,
            physics_fit_summary,
        ) = fit_stage2_physics(
            payload,
            body,
            args,
            stage1_to_mujoco_scale=float(body_metadata["stage1_to_mujoco_scale"]),
            initial_linear_velocity_delta=linear_velocity_delta,
            initial_angular_velocity_delta=angular_velocity_delta,
            initial_physics_params=physics_params,
            initial_geometry_params=geometry_params,
        )
    elif physics_params is not None or geometry_params is not None:
        loaded_geometry_summary = None if loaded_refined_params is None else loaded_refined_params.get("geometry")
        physics_fit_summary = {
            "kind": "loaded_refined_params",
            "loaded_from": loaded_refined_params_path,
            "learned_physics": None if physics_params is None else physics_params_to_serializable(physics_params),
            "learned_geometry": (
                loaded_geometry_summary
                if loaded_geometry_summary is not None
                else geometry_params_to_serializable(
                    geometry_params,
                    max_log_radius_offset=float(args.fit_geometry_max_log_radius_offset),
                    max_center_offset=float(args.fit_geometry_max_center_offset),
                )
            ),
            "history": [],
            "best_loss": None,
        }
    predicted, metrics = rollout(
        payload,
        body,
        args,
        linear_velocity_delta=linear_velocity_delta,
        angular_velocity_delta=angular_velocity_delta,
        physics_params=physics_params,
        geometry_params=geometry_params,
    )
    pred_rgb_dir = None
    comparison_path = None
    if not bool(args.skip_render):
        if args.gt_rgb_dir is None:
            raise ValueError("--gt_rgb_dir is required unless --skip_render is set.")
        pred_rgb_dir = render_predicted_frames(payload, predicted, args, output_dir)
        comparison_path = output_dir / "gt_vs_stage2_predicted_rollout.gif"
        make_comparison_gif(args.gt_rgb_dir.resolve(), pred_rgb_dir, predicted, metrics, comparison_path, int(args.fps))
    saved_refined_params_path = None
    if args.save_refined_params is not None:
        saved_refined_params_path = str(args.save_refined_params.resolve())
        write_refined_params(
            args.save_refined_params.resolve(),
            trajectory=args.trajectory,
            stage1_ply=args.stage1_ply,
            stage1_body=body_metadata,
            linear_velocity_delta=linear_velocity_delta,
            angular_velocity_delta=angular_velocity_delta,
            physics_params=physics_params,
            geometry_params=geometry_params,
            fit_summary=physics_fit_summary,
            source="fit" if int(args.fit_physics_iters) > 0 else ("loaded" if loaded_refined_params is not None else "rollout"),
            args=args,
        )
    write_json(output_dir / "stage2_predicted_trajectory.json", {"states": predicted})
    summary = {
        "trajectory": str(args.trajectory.resolve()),
        "gt_rgb_dir": None if args.gt_rgb_dir is None else str(args.gt_rgb_dir.resolve()),
        "stage1_body": body_metadata,
        "metrics": metrics,
        "fit": fit_summary,
        "physics_fit": physics_fit_summary,
        "refined_params": {
            "loaded_from": loaded_refined_params_path,
            "saved_to": saved_refined_params_path,
        },
        "dynamics_backend": str(args.dynamics_backend),
        "friction_cone": {
            "model": str(args.stage2_friction_model),
            "num_directions": int(args.stage2_friction_num_directions),
            "dynamic_friction": float(args.pair_friction),
            "static_friction": None if float(args.stage2_static_friction) <= 0.0 else float(args.stage2_static_friction),
            "tangential_damping": float(args.stage2_tangential_damping),
            "transition_velocity": float(args.stage2_friction_transition_velocity),
        },
        "comparison_gif": None if comparison_path is None else str(comparison_path),
        "predicted_rollout_gif": None if pred_rgb_dir is None else str(output_dir / "stage2_predicted_rollout.gif"),
        "note": (
            "This is an end-to-end Stage1 collision-proxy to Stage2 N-body rollout result. "
            "When fit is present, Stage2 initial linear/angular velocities were optimized against the GT trajectory; "
            "when physics_fit is present, K/D/friction/radius-scale were also optimized through the Stage2 dynamics."
        ),
    }
    write_json(output_dir / "stage2_rollout_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
