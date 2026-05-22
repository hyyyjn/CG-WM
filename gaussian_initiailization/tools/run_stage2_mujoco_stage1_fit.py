from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_initiailization.stage2.differentiable_collision_detection import (
    GaussianCollisionBody,
    PlaneCollider,
    detect_gaussian_union_contacts,
    load_gaussian_collision_body_from_ply,
    make_floor_disk_query_points,
)
from gaussian_initiailization.stage2.differentiable_complementarity_free_contact_dynamics import (
    PairwiseGaussianBodyImpedanceDynamics,
    PairwiseImpedanceDynamicsConfig,
    RigidBodyState,
    smooth_weighted_max,
)
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the new Stage 2 contact dynamics against one MuJoCo episode, "
            "using a Stage 1 Gaussian object reconstructed from that same dataset."
        )
    )
    parser.add_argument("--episode_root", required=True, type=Path)
    parser.add_argument("--stage1_ply", default=None, type=Path)
    parser.add_argument("--stage1_model_path", default=None, type=Path)
    parser.add_argument("--output_dir", default=None, type=Path)
    parser.add_argument("--max_frames", default=160, type=int)
    parser.add_argument("--fit_iters", default=300, type=int)
    parser.add_argument("--lr", default=0.04, type=float)
    parser.add_argument("--radius_scale", default=1.0, type=float)
    parser.add_argument("--object_id", default=None, type=int)
    parser.add_argument("--foreground_threshold", default=None, type=float)
    parser.add_argument("--opacity_threshold", default=None, type=float)
    parser.add_argument("--max_primitives", default=None, type=int)
    parser.add_argument("--query_radius_scale", default=1.10, type=float)
    parser.add_argument("--query_rings", default=5, type=int)
    parser.add_argument("--query_angles", default=32, type=int)
    parser.add_argument("--contact_softness", default=2e-3, type=float)
    parser.add_argument("--smooth_max_temperature", default=1e-2, type=float)
    parser.add_argument("--inside_penalty", default=0.02, type=float)
    parser.add_argument("--inside_sharpness", default=50.0, type=float)
    parser.add_argument(
        "--dynamics",
        default="impedance",
        choices=("impedance", "restitution", "pairwise_impedance"),
        help=(
            "impedance: paper III-D-2 SoftPlus(-K(hJ̃b+ϕ)-D(J̃b)) form, learn (v0, g, K, D). "
            "restitution: legacy reflection+slop, learn (v0, g, e). "
            "pairwise_impedance: use PairwiseGaussianBodyImpedanceDynamics against a static Gaussian body B."
        ),
    )
    parser.add_argument("--init_stiffness", default=800.0, type=float)
    parser.add_argument("--init_damping", default=30.0, type=float)
    parser.add_argument("--mass", default=1.0, type=float)
    parser.add_argument("--pairwise_body_b_ply", default=None, type=Path)
    parser.add_argument("--pairwise_body_b_object_id", default=None, type=int)
    parser.add_argument("--pairwise_body_b_foreground_threshold", default=None, type=float)
    parser.add_argument("--pairwise_body_b_opacity_threshold", default=None, type=float)
    parser.add_argument("--pairwise_body_b_max_primitives", default=None, type=int)
    parser.add_argument(
        "--pairwise_static_position",
        default="0,0,0",
        type=str,
        help="Static body-B position for --dynamics pairwise_impedance, as x,y,z.",
    )
    parser.add_argument("--pairwise_num_contact_patches", default=4, type=int)
    parser.add_argument("--pairwise_broad_phase_margin", default=0.02, type=float)
    parser.add_argument("--pairwise_broad_phase_mode", default="sphere", choices=("sphere", "aabb"))
    parser.add_argument("--pairwise_friction_coefficient", default=0.0, type=float)
    parser.add_argument("--pairwise_tangential_damping", default=0.0, type=float)
    parser.add_argument(
        "--orientation_loss_weight",
        default=0.0,
        type=float,
        help="Weight for sign-invariant quaternion loss in --dynamics pairwise_impedance.",
    )
    parser.add_argument(
        "--fit_initial_angular_velocity",
        action="store_true",
        help="Learn the initial angular velocity from trajectory orientation loss in pairwise mode.",
    )
    parser.add_argument("--angular_velocity_l2_weight", default=1e-4, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gif_fps", default=24, type=int)
    return parser.parse_args()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def resolve_stage1_ply(stage1_ply: Path | None, stage1_model_path: Path | None) -> Path:
    if stage1_ply is not None:
        resolved = stage1_ply.resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved
    if stage1_model_path is None:
        raise ValueError("Pass --stage1_ply or --stage1_model_path.")

    model_path = stage1_model_path.resolve()
    candidates = list(model_path.glob("point_cloud/iteration_*/point_cloud.ply"))
    if not candidates:
        raise FileNotFoundError(f"No point_cloud/iteration_*/point_cloud.ply under {model_path}")

    def iteration_number(path: Path) -> int:
        match = re.search(r"iteration_(\d+)", str(path))
        return int(match.group(1)) if match else -1

    return max(candidates, key=iteration_number)


def load_target_positions(episode_root: Path, max_frames: int) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    trajectory_path = episode_root / "state" / "trajectory.json"
    payload = read_json(trajectory_path)
    states = payload["states"]
    if max_frames > 0:
        states = states[:max_frames]
    if len(states) < 3:
        raise ValueError("Need at least 3 trajectory states for Stage 2 fitting.")
    positions = torch.tensor([state["position"] for state in states], dtype=torch.float32)
    times = torch.tensor([float(state.get("time", idx)) for idx, state in enumerate(states)], dtype=torch.float32)
    return positions, times, states


def load_target_quaternions(states: list[dict]) -> torch.Tensor:
    quaternions = [
        state.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0])
        for state in states
    ]
    return normalize_quaternion(torch.tensor(quaternions, dtype=torch.float32))


def load_initial_angular_velocity(states: list[dict]) -> torch.Tensor:
    angular_velocity = states[0].get("angular_velocity", [0.0, 0.0, 0.0])
    return torch.tensor(angular_velocity, dtype=torch.float32)


def infer_dt(times: torch.Tensor) -> float:
    diffs = times[1:] - times[:-1]
    dt = float(torch.median(diffs).item())
    if dt <= 0.0:
        raise ValueError(f"Invalid non-positive dt inferred from trajectory: {dt}")
    return dt


def parse_vec3(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated values, got: {value}")
    return tuple(float(part) for part in parts)


def normalize_quaternion(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    return quaternion_wxyz / torch.clamp(torch.linalg.norm(quaternion_wxyz, dim=-1, keepdim=True), min=1e-12)


def quaternion_loss(predicted_wxyz: torch.Tensor, target_wxyz: torch.Tensor) -> torch.Tensor:
    predicted = normalize_quaternion(predicted_wxyz)
    target = normalize_quaternion(target_wxyz)
    alignment = torch.sum(predicted * target, dim=-1).abs().clamp(max=1.0)
    return torch.mean(1.0 - alignment * alignment)


def quaternion_rmse_degrees(predicted_wxyz: torch.Tensor, target_wxyz: torch.Tensor) -> float:
    predicted = normalize_quaternion(predicted_wxyz)
    target = normalize_quaternion(target_wxyz)
    alignment = torch.sum(predicted * target, dim=-1).abs().clamp(max=1.0)
    angles = 2.0 * torch.acos(alignment)
    return float(torch.sqrt(torch.mean(angles * angles)).detach().cpu().item() * 180.0 / np.pi)


def estimate_radius(local_centers: torch.Tensor, radii: torch.Tensor) -> float:
    xy_extent = torch.linalg.norm(local_centers[:, :2], dim=-1) + radii
    radius = float(torch.quantile(xy_extent.detach().cpu(), 0.98).item())
    return max(radius, 1e-3)


def load_stage1_body_arrays(
    path: Path,
    args: argparse.Namespace,
    *,
    object_id: int | None,
    foreground_threshold: float | None,
    opacity_threshold: float | None,
    max_primitives: int | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    body = load_gaussian_collision_body_from_ply(
        path,
        radius_scale=float(args.radius_scale),
        recenter=True,
        object_id=object_id,
        foreground_threshold=foreground_threshold,
        opacity_threshold=opacity_threshold,
        max_primitives=max_primitives,
        use_centers_as_queries=True,
        dtype=torch.float32,
        device=device,
    )
    return body.local_centers, body.radii


def _smooth_min_signed(values: torch.Tensor, temperature: float) -> torch.Tensor:
    return -temperature * torch.logsumexp(-values / temperature, dim=-1)


def simulate(
    initial_position: torch.Tensor,
    initial_velocity: torch.Tensor,
    gravity_z: torch.Tensor,
    restitution: torch.Tensor,
    local_centers: torch.Tensor,
    radii: torch.Tensor,
    floor_query_offsets_xy: torch.Tensor,
    *,
    steps: int,
    dt: float,
    contact_softness: float,
    smooth_max_temperature: float,
    inside_penalty: float = 0.02,
    inside_sharpness: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy reflection+slop dynamics (kept for --dynamics restitution)."""
    collider = PlaneCollider.floor(dtype=initial_position.dtype, device=initial_position.device)
    position = initial_position
    velocity = initial_velocity
    positions = [position]
    contact_gates = []
    gravity = torch.stack(
        (
            torch.zeros((), dtype=initial_position.dtype, device=initial_position.device),
            torch.zeros((), dtype=initial_position.dtype, device=initial_position.device),
            gravity_z,
        )
    )

    for _ in range(steps - 1):
        predicted_velocity = velocity + gravity * dt
        predicted_position = position + predicted_velocity * dt

        floor_points = torch.cat(
            (
                predicted_position[:2].unsqueeze(0) + floor_query_offsets_xy,
                torch.zeros((floor_query_offsets_xy.shape[0], 1), dtype=predicted_position.dtype, device=predicted_position.device),
            ),
            dim=-1,
        )
        gaussian_centers = local_centers + predicted_position.unsqueeze(0)
        contacts = detect_gaussian_union_contacts(
            floor_points,
            gaussian_centers,
            radii,
            collider.normal.to(dtype=predicted_position.dtype, device=predicted_position.device),
            softness=contact_softness,
            smooth_min_temperature=smooth_max_temperature,
            inside_penalty=inside_penalty,
            inside_sharpness=inside_sharpness,
        )
        penetration_depth = smooth_weighted_max(contacts.penetrations, smooth_max_temperature)
        contact_gate = smooth_weighted_max(contacts.contact_weights, smooth_max_temperature)
        normal = contacts.collider_normal.to(dtype=predicted_position.dtype, device=predicted_position.device)
        normal_velocity = torch.sum(predicted_velocity * normal)
        closing_speed = torch.nn.functional.softplus(-normal_velocity / contact_softness) * contact_softness
        velocity = predicted_velocity + contact_gate * (1.0 + restitution) * closing_speed * normal
        position = predicted_position + contact_gate * (penetration_depth + 1e-4) * normal

        positions.append(position)
        contact_gates.append(contact_gate)

    if contact_gates:
        gates = torch.stack(contact_gates)
    else:
        gates = torch.empty((0,), dtype=initial_position.dtype, device=initial_position.device)
    return torch.stack(positions, dim=0), gates


def simulate_impedance(
    initial_position: torch.Tensor,
    initial_velocity: torch.Tensor,
    gravity_z: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
    local_centers: torch.Tensor,
    radii: torch.Tensor,
    floor_query_offsets_xy: torch.Tensor,
    *,
    steps: int,
    dt: float,
    mass: float,
    contact_softness: float,
    smooth_min_temperature: float,
    inside_penalty: float,
    inside_sharpness: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paper III-D-2 impedance contact dynamics rolled out over ``steps`` frames.

    ``stiffness`` and ``damping`` are expected to already be positive (typically
    produced via ``torch.exp(log_K)`` in the fit loop) — no extra SoftPlus is
    applied here. Single contact pair, frictionless.
    """
    collider = PlaneCollider.floor(dtype=initial_position.dtype, device=initial_position.device)
    K = stiffness
    D = damping
    position = initial_position
    velocity = initial_velocity
    positions = [position]
    lambdas = []
    gravity = torch.stack(
        (
            torch.zeros((), dtype=initial_position.dtype, device=initial_position.device),
            torch.zeros((), dtype=initial_position.dtype, device=initial_position.device),
            gravity_z,
        )
    )

    for _ in range(steps - 1):
        b = velocity + dt * gravity

        floor_points = torch.cat(
            (
                position[:2].unsqueeze(0) + floor_query_offsets_xy,
                torch.zeros(
                    (floor_query_offsets_xy.shape[0], 1),
                    dtype=position.dtype,
                    device=position.device,
                ),
            ),
            dim=-1,
        )
        gaussian_centers = local_centers + position.unsqueeze(0)
        contacts = detect_gaussian_union_contacts(
            floor_points,
            gaussian_centers,
            radii,
            collider.normal.to(dtype=position.dtype, device=position.device),
            softness=contact_softness,
            smooth_min_temperature=smooth_min_temperature,
            inside_penalty=inside_penalty,
            inside_sharpness=inside_sharpness,
        )
        phi_agg = _smooth_min_signed(contacts.signed_distances, smooth_min_temperature)
        normal = contacts.collider_normal.to(dtype=position.dtype, device=position.device)
        Jb = torch.sum(b * normal)
        lambda_t = F.softplus(-K * (dt * Jb + phi_agg) - D * Jb)

        velocity = b + (dt / mass) * lambda_t * normal
        position = position + dt * velocity

        positions.append(position)
        lambdas.append(lambda_t)

    if lambdas:
        gates = torch.stack(lambdas)
    else:
        gates = torch.empty((0,), dtype=initial_position.dtype, device=initial_position.device)
    return torch.stack(positions, dim=0), gates


def simulate_pairwise_impedance(
    initial_position: torch.Tensor,
    initial_quaternion: torch.Tensor,
    initial_velocity: torch.Tensor,
    initial_angular_velocity: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
    local_centers_a: torch.Tensor,
    radii_a: torch.Tensor,
    local_centers_b: torch.Tensor,
    radii_b: torch.Tensor,
    static_position_b: torch.Tensor,
    *,
    steps: int,
    dt: float,
    mass: float,
    gravity_z: float,
    contact_softness: float,
    smooth_min_temperature: float,
    inside_penalty: float,
    inside_sharpness: float,
    num_contact_patches: int,
    broad_phase_margin: float,
    broad_phase_mode: str,
    friction_coefficient: float,
    tangential_damping: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    body_a = GaussianCollisionBody(local_centers_a, radii_a)
    body_b = GaussianCollisionBody(local_centers_b, radii_b)
    dynamics = PairwiseGaussianBodyImpedanceDynamics(
        body_a,
        body_b,
        stiffness=stiffness,
        damping=damping,
        config=PairwiseImpedanceDynamicsConfig(
            dt=float(dt),
            mass_a=float(mass),
            mass_b=float(mass),
            gravity=(0.0, 0.0, float(gravity_z)),
            dynamic_a=True,
            dynamic_b=False,
            contact_softness=float(contact_softness),
            smooth_min_temperature=float(smooth_min_temperature),
            inside_penalty=float(inside_penalty),
            inside_sharpness=float(inside_sharpness),
            num_contact_patches=int(num_contact_patches),
            broad_phase_margin=float(broad_phase_margin),
            broad_phase_mode=str(broad_phase_mode),
            friction_coefficient=float(friction_coefficient),
            tangential_damping=float(tangential_damping),
        ),
    )
    quaternion = normalize_quaternion(initial_quaternion.to(dtype=initial_position.dtype, device=initial_position.device))
    static_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=initial_position.dtype, device=initial_position.device)
    zeros = torch.zeros(3, dtype=initial_position.dtype, device=initial_position.device)
    state_a = RigidBodyState(initial_position, quaternion, initial_velocity, initial_angular_velocity)
    state_b = RigidBodyState(static_position_b, static_quaternion, zeros, zeros)
    positions = [state_a.position]
    quaternions = [state_a.quaternion_wxyz]
    gates = []
    for _ in range(steps - 1):
        state_a, state_b, diagnostics = dynamics.step(state_a, state_b)
        positions.append(state_a.position)
        quaternions.append(state_a.quaternion_wxyz)
        patch_weights = diagnostics["patch_weights"]
        gates.append(torch.max(patch_weights))

    if gates:
        gate_tensor = torch.stack(gates)
    else:
        gate_tensor = torch.empty((0,), dtype=initial_position.dtype, device=initial_position.device)
    return torch.stack(positions, dim=0), torch.stack(quaternions, dim=0), gate_tensor


def fit_stage2(
    target_positions: torch.Tensor,
    target_quaternions: torch.Tensor | None,
    initial_angular_velocity_hint: torch.Tensor,
    times: torch.Tensor,
    local_centers: torch.Tensor,
    radii: torch.Tensor,
    floor_query_offsets_xy: torch.Tensor,
    args: argparse.Namespace,
    *,
    pairwise_body_b: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, dict]:
    device = torch.device(args.device)
    target_positions = target_positions.to(device=device)
    if target_quaternions is not None:
        target_quaternions = normalize_quaternion(target_quaternions.to(device=device))
    initial_angular_velocity_hint = initial_angular_velocity_hint.to(device=device)
    times = times.to(device=device)
    local_centers = local_centers.to(device=device)
    radii = radii.to(device=device)
    floor_query_offsets_xy = floor_query_offsets_xy.to(device=device)
    dt = infer_dt(times.detach().cpu())

    dynamics_mode = str(args.dynamics)
    use_impedance = dynamics_mode == "impedance"
    use_pairwise = dynamics_mode == "pairwise_impedance"
    initial_position = target_positions[0].detach()
    initial_quaternion = (
        target_quaternions[0].detach()
        if target_quaternions is not None
        else torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    )
    finite_velocity = (target_positions[1] - target_positions[0]) / dt
    initial_velocity = torch.nn.Parameter(finite_velocity.detach().clone())
    initial_angular_velocity = torch.nn.Parameter(initial_angular_velocity_hint.detach().clone())
    gravity_z = torch.nn.Parameter(torch.tensor(-9.81, dtype=torch.float32, device=device))

    if use_pairwise:
        stiffness = torch.nn.Parameter(torch.tensor(float(args.init_stiffness), dtype=torch.float32, device=device))
        damping = torch.nn.Parameter(torch.tensor(float(args.init_damping), dtype=torch.float32, device=device))
        learnable = [initial_velocity, stiffness, damping]
        if bool(args.fit_initial_angular_velocity):
            learnable.append(initial_angular_velocity)
        log_K = None
        log_D = None
        raw_restitution = None
    elif use_impedance:
        # Reparameterise K = exp(log_K) so Adam can move K in log-space; this
        # avoids the K ~10^3 vs v0 ~10^0 scale mismatch where a shared lr keeps
        # K essentially frozen for the whole fit.
        log_K = torch.nn.Parameter(
            torch.log(torch.tensor(float(args.init_stiffness), dtype=torch.float32, device=device))
        )
        log_D = torch.nn.Parameter(
            torch.log(torch.tensor(float(args.init_damping), dtype=torch.float32, device=device))
        )
        learnable = [initial_velocity, gravity_z, log_K, log_D]
        raw_restitution = None
    else:
        raw_restitution = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
        log_K = None
        log_D = None
        stiffness = None
        damping = None
        learnable = [initial_velocity, gravity_z, raw_restitution]

    optimizer = torch.optim.Adam(learnable, lr=float(args.lr))

    def run(grad: bool):
        if use_pairwise:
            if pairwise_body_b is None:
                local_centers_b, radii_b = local_centers, radii
            else:
                local_centers_b, radii_b = pairwise_body_b
                local_centers_b = local_centers_b.to(device=device)
                radii_b = radii_b.to(device=device)
            static_position_b = torch.tensor(
                parse_vec3(args.pairwise_static_position),
                dtype=target_positions.dtype,
                device=device,
            )
            return simulate_pairwise_impedance(
                initial_position,
                initial_quaternion,
                initial_velocity if grad else initial_velocity.detach(),
                initial_angular_velocity if grad else initial_angular_velocity.detach(),
                stiffness if grad else stiffness.detach(),
                damping if grad else damping.detach(),
                local_centers,
                radii,
                local_centers_b,
                radii_b,
                static_position_b,
                steps=target_positions.shape[0],
                dt=dt,
                mass=float(args.mass),
                gravity_z=float(gravity_z.detach().cpu().item()),
                contact_softness=float(args.contact_softness),
                smooth_min_temperature=float(args.smooth_max_temperature),
                inside_penalty=float(args.inside_penalty),
                inside_sharpness=float(args.inside_sharpness),
                num_contact_patches=int(args.pairwise_num_contact_patches),
                broad_phase_margin=float(args.pairwise_broad_phase_margin),
                broad_phase_mode=str(args.pairwise_broad_phase_mode),
                friction_coefficient=float(args.pairwise_friction_coefficient),
                tangential_damping=float(args.pairwise_tangential_damping),
            )
        if use_impedance:
            K_tensor = torch.exp(log_K if grad else log_K.detach())
            D_tensor = torch.exp(log_D if grad else log_D.detach())
            predicted_positions, gates = simulate_impedance(
                initial_position,
                initial_velocity if grad else initial_velocity.detach(),
                gravity_z if grad else gravity_z.detach(),
                K_tensor,
                D_tensor,
                local_centers,
                radii,
                floor_query_offsets_xy,
                steps=target_positions.shape[0],
                dt=dt,
                mass=float(args.mass),
                contact_softness=float(args.contact_softness),
                smooth_min_temperature=float(args.smooth_max_temperature),
                inside_penalty=float(args.inside_penalty),
                inside_sharpness=float(args.inside_sharpness),
            )
            return predicted_positions, None, gates
        restitution = torch.sigmoid(raw_restitution if grad else raw_restitution.detach())
        predicted_positions, gates = simulate(
            initial_position,
            initial_velocity if grad else initial_velocity.detach(),
            gravity_z if grad else gravity_z.detach(),
            restitution,
            local_centers,
            radii,
            floor_query_offsets_xy,
            steps=target_positions.shape[0],
            dt=dt,
            contact_softness=float(args.contact_softness),
            smooth_max_temperature=float(args.smooth_max_temperature),
            inside_penalty=float(args.inside_penalty),
            inside_sharpness=float(args.inside_sharpness),
        )
        return predicted_positions, None, gates

    initial_loss = None
    final_loss = None
    for iteration in range(max(1, int(args.fit_iters))):
        optimizer.zero_grad(set_to_none=True)
        predicted, predicted_quaternions, gates = run(grad=True)
        position_loss = torch.mean((predicted - target_positions) ** 2)
        loss = position_loss + 1e-4 * torch.mean(initial_velocity * initial_velocity)
        orientation_loss = torch.tensor(0.0, dtype=target_positions.dtype, device=device)
        if (
            use_pairwise
            and predicted_quaternions is not None
            and target_quaternions is not None
            and float(args.orientation_loss_weight) > 0.0
        ):
            orientation_loss = quaternion_loss(predicted_quaternions, target_quaternions)
            loss = loss + float(args.orientation_loss_weight) * orientation_loss
        if use_pairwise and bool(args.fit_initial_angular_velocity):
            loss = loss + float(args.angular_velocity_l2_weight) * torch.mean(
                initial_angular_velocity * initial_angular_velocity
            )
        loss.backward()
        optimizer.step()
        if iteration == 0:
            initial_loss = float(position_loss.detach().cpu().item())
        final_loss = float(position_loss.detach().cpu().item())

    predicted, predicted_quaternions, gates = run(grad=False)
    contact_indices = torch.nonzero(gates.detach().cpu() > 0.5).flatten()
    first_contact_frame = int(contact_indices[0].item() + 1) if contact_indices.numel() else None
    rmse = torch.sqrt(torch.mean((predicted - target_positions) ** 2)).detach().cpu().item()
    diagnostics = {
        "dt": dt,
        "frames": int(target_positions.shape[0]),
        "dynamics": dynamics_mode,
        "num_primitives_a": int(local_centers.shape[0]),
        "radius_scale": float(args.radius_scale),
        "object_id": args.object_id,
        "foreground_threshold": args.foreground_threshold,
        "opacity_threshold": args.opacity_threshold,
        "max_primitives": args.max_primitives,
        "initial_position_loss": initial_loss,
        "final_position_loss": final_loss,
        "position_rmse": float(rmse),
        "learned_initial_velocity": initial_velocity.detach().cpu().tolist(),
        "learned_gravity_z": float(gravity_z.detach().cpu().item()),
        "first_contact_frame": first_contact_frame,
        "max_contact_gate": float(gates.detach().cpu().max().item()) if gates.numel() else 0.0,
    }
    if use_pairwise:
        diagnostics["learned_stiffness"] = float(F.softplus(stiffness.detach()).cpu().item())
        diagnostics["learned_damping"] = float(F.softplus(damping.detach()).cpu().item())
        diagnostics["pairwise_static_position"] = list(parse_vec3(args.pairwise_static_position))
        diagnostics["pairwise_num_contact_patches"] = int(args.pairwise_num_contact_patches)
        diagnostics["pairwise_broad_phase_mode"] = str(args.pairwise_broad_phase_mode)
        diagnostics["pairwise_friction_coefficient"] = float(args.pairwise_friction_coefficient)
        diagnostics["pairwise_tangential_damping"] = float(args.pairwise_tangential_damping)
        diagnostics["gravity_fit_enabled"] = False
        diagnostics["fit_initial_angular_velocity"] = bool(args.fit_initial_angular_velocity)
        diagnostics["learned_initial_angular_velocity"] = initial_angular_velocity.detach().cpu().tolist()
        diagnostics["orientation_loss_weight"] = float(args.orientation_loss_weight)
        diagnostics["angular_velocity_l2_weight"] = float(args.angular_velocity_l2_weight)
        if predicted_quaternions is not None and target_quaternions is not None:
            orientation_loss_value = quaternion_loss(predicted_quaternions, target_quaternions)
            diagnostics["orientation_loss"] = float(orientation_loss_value.detach().cpu().item())
            diagnostics["orientation_rmse_degrees"] = quaternion_rmse_degrees(predicted_quaternions, target_quaternions)
    elif use_impedance:
        diagnostics["learned_stiffness"] = float(torch.exp(log_K.detach()).cpu().item())
        diagnostics["learned_damping"] = float(torch.exp(log_D.detach()).cpu().item())
    else:
        diagnostics["learned_restitution"] = float(torch.sigmoid(raw_restitution.detach()).cpu().item())
    predicted_quaternions_cpu = predicted_quaternions.detach().cpu() if predicted_quaternions is not None else None
    return predicted.detach().cpu(), predicted_quaternions_cpu, gates.detach().cpu(), diagnostics


def draw_follow_view(
    target_positions: np.ndarray,
    predicted_positions: np.ndarray,
    contact_gates: np.ndarray,
    output_gif: Path,
    *,
    fps: int,
) -> None:
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 480
    frames = []
    z_min = min(float(target_positions[:, 2].min()), float(predicted_positions[:, 2].min()), 0.0)
    z_max = max(float(target_positions[:, 2].max()), float(predicted_positions[:, 2].max()), 1.0)
    z_pad = max(0.25, 0.15 * (z_max - z_min))
    z_min -= z_pad
    z_max += z_pad
    radius_px = 28
    floor_y = height - 74

    def z_to_y(z: float) -> int:
        t = (z - z_min) / max(z_max - z_min, 1e-6)
        return int(floor_y - t * (floor_y - 64))

    for idx, (target, pred) in enumerate(zip(target_positions, predicted_positions)):
        frame = Image.new("RGB", (width, height), (248, 247, 241))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, width, 48), fill=(28, 28, 26))
        draw.text((14, 16), f"Stage2 MuJoCo/Stage1 fit | frame {idx:03d}/{len(target_positions)-1:03d}", fill=(255, 255, 245))
        draw.line((56, floor_y, width - 56, floor_y), fill=(24, 24, 22), width=3)
        for grid_idx in range(8):
            y = floor_y + 16 + grid_idx * 14
            draw.line((0, y, width, y + 90), fill=(218, 214, 198), width=1)

        center_x = width // 2
        target_y = z_to_y(float(target[2]))
        pred_y = z_to_y(float(pred[2]))
        gate = float(contact_gates[idx - 1]) if idx > 0 and idx - 1 < len(contact_gates) else 0.0
        gate_color = (34, 120, 215) if gate < 0.5 else (220, 72, 48)
        draw.ellipse(
            (center_x - radius_px, target_y - radius_px, center_x + radius_px, target_y + radius_px),
            outline=(25, 82, 130),
            width=4,
            fill=(78, 150, 219),
        )
        draw.ellipse(
            (center_x - 7, pred_y - 7, center_x + 7, pred_y + 7),
            fill=gate_color,
            outline=(30, 30, 30),
        )
        draw.line((center_x - 34, target_y, center_x + 34, target_y), fill=(25, 82, 130), width=1)
        draw.text((18, height - 30), f"target_z={target[2]:.3f} pred_z={pred[2]:.3f} contact_gate={gate:.3f}", fill=(20, 20, 18))
        frames.append(frame)

    duration_ms = max(1, int(round(1000.0 / float(fps))))
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)


def main() -> None:
    args = parse_args()
    episode_root = args.episode_root.resolve()
    stage1_ply = resolve_stage1_ply(args.stage1_ply, args.stage1_model_path)
    output_dir = args.output_dir.resolve() if args.output_dir else episode_root / "stage2_mujoco_stage1_fit"

    target_positions_cpu, times_cpu, states = load_target_positions(episode_root, int(args.max_frames))
    target_quaternions_cpu = load_target_quaternions(states)
    initial_angular_velocity_hint_cpu = load_initial_angular_velocity(states)
    local_centers, radii = load_stage1_body_arrays(
        stage1_ply,
        args,
        object_id=args.object_id,
        foreground_threshold=args.foreground_threshold,
        opacity_threshold=args.opacity_threshold,
        max_primitives=args.max_primitives,
        device=torch.device(args.device),
    )
    pairwise_body_b = None
    if str(args.dynamics) == "pairwise_impedance" and args.pairwise_body_b_ply is not None:
        pairwise_body_b = load_stage1_body_arrays(
            args.pairwise_body_b_ply.resolve(),
            args,
            object_id=args.pairwise_body_b_object_id,
            foreground_threshold=args.pairwise_body_b_foreground_threshold,
            opacity_threshold=args.pairwise_body_b_opacity_threshold,
            max_primitives=args.pairwise_body_b_max_primitives,
            device=torch.device(args.device),
        )
    query_radius = estimate_radius(local_centers.detach().cpu(), radii.detach().cpu()) * float(args.query_radius_scale)
    floor_query_offsets_xy = make_floor_disk_query_points(
        query_radius,
        num_rings=int(args.query_rings),
        num_angles=int(args.query_angles),
        dtype=torch.float32,
        device=torch.device(args.device),
    )
    predicted_positions, predicted_quaternions, contact_gates, diagnostics = fit_stage2(
        target_positions_cpu,
        target_quaternions_cpu,
        initial_angular_velocity_hint_cpu,
        times_cpu,
        local_centers,
        radii,
        floor_query_offsets_xy,
        args,
        pairwise_body_b=pairwise_body_b,
    )

    trajectory_states = []
    for idx in range(predicted_positions.shape[0]):
        state_payload = {
            "frame_index": int(states[idx]["frame_index"]),
            "time": float(states[idx].get("time", idx)),
            "target_position": target_positions_cpu[idx].tolist(),
            "predicted_position": predicted_positions[idx].tolist(),
            "contact_gate": float(contact_gates[idx - 1].item()) if idx > 0 and idx - 1 < contact_gates.numel() else 0.0,
        }
        if predicted_quaternions is not None:
            state_payload["target_quaternion_wxyz"] = target_quaternions_cpu[idx].tolist()
            state_payload["predicted_quaternion_wxyz"] = predicted_quaternions[idx].tolist()
        trajectory_states.append(state_payload)

    trajectory = {
        "source_episode_root": str(episode_root),
        "stage1_ply": str(stage1_ply),
        "states": trajectory_states,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "fit_summary.json", diagnostics)
    write_json(output_dir / "predicted_trajectory.json", trajectory)
    gif_path = output_dir / "stage2_fit_follow_view.gif"
    draw_follow_view(
        target_positions_cpu.numpy(),
        predicted_positions.numpy(),
        contact_gates.numpy(),
        gif_path,
        fps=int(args.gif_fps),
    )
    diagnostics["output_dir"] = str(output_dir)
    diagnostics["diagnostic_gif"] = str(gif_path)
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
