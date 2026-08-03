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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2.differentiable_collision_detection import (
    GaussianCollisionBody,
    PlaneCollider,
    quat_wxyz_to_matrix,
    detect_gaussian_union_contacts,
    load_gaussian_collision_body_from_ply,
    make_floor_disk_query_points,
    make_gaussian_proxy_query_points,
    transform_local_points,
)
from stage2.differentiable_complementarity_free_contact_dynamics import (
    PairwiseGaussianBodyImpedanceDynamics,
    PairwiseImpedanceDynamicsConfig,
    RigidBodyState,
    smooth_weighted_max,
)
from stage2.experiment_provenance import write_experiment_bundle
from stage2.video_observations import (
    load_optional_evaluation_trajectory,
    load_video_observations,
    observation_summary,
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
    parser.add_argument(
        "--source_scene_manifest",
        default=None,
        type=Path,
        help="Original generic scene manifest, recorded for provenance when invoked through the adapter.",
    )
    parser.add_argument(
        "--evaluation_trajectory",
        default=None,
        type=Path,
        help=(
            "Optional pose-label trajectory used by the legacy supervised adapter and metrics. "
            "Defaults to episode_root/state/trajectory.json for now; video observations are loaded independently."
        ),
    )
    parser.add_argument("--stage1_ply", default=None, type=Path)
    parser.add_argument("--stage1_model_path", default=None, type=Path)
    parser.add_argument("--output_dir", default=None, type=Path)
    parser.add_argument("--max_frames", default=160, type=int)
    parser.add_argument("--fit_iters", default=300, type=int)
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Run a fixed-parameter rollout without optimizer steps (for holdout evaluation).",
    )
    parser.add_argument("--initial_state_json", default=None, type=Path)
    parser.add_argument("--prefit_initial_state", action="store_true")
    parser.add_argument("--prefit_pose_iters", default=100, type=int)
    parser.add_argument("--prefit_velocity_iters", default=100, type=int)
    parser.add_argument("--prefit_velocity_frames", default=3, type=int)
    parser.add_argument("--prefit_lr", default=0.01, type=float)
    parser.add_argument("--prefit_velocity_l2", default=1e-4, type=float)
    parser.add_argument("--prefit_position_init", default="0,0,0", type=str)
    parser.add_argument("--prefit_quaternion_init", default="1,0,0,0", type=str)
    parser.add_argument("--lr", default=0.04, type=float)
    parser.add_argument(
        "--log_every",
        default=0,
        type=int,
        help="Print position loss every N fit iterations (0 = off). Useful for convergence diagnosis.",
    )
    parser.add_argument("--radius_scale", default=1.0, type=float)
    parser.add_argument(
        "--gaussian_radius_convention",
        default="paper_r2s",
        choices=("paper_r2s", "legacy_r_equals_s"),
        help="Map Stage-I Gaussian scale s to collision radius (paper: r=2s).",
    )
    parser.add_argument(
        "--gaussian_scale_reduction",
        default="mean",
        choices=("strict", "mean", "max"),
        help="Convert legacy anisotropic PLY scale channels to one spherical scale.",
    )
    parser.add_argument("--gaussian_isotropic_tolerance", default=1e-4, type=float)
    parser.add_argument("--object_id", default=None, type=int)
    parser.add_argument("--foreground_threshold", default=None, type=float)
    parser.add_argument("--opacity_threshold", default=None, type=float)
    parser.add_argument("--max_primitives", default=None, type=int)
    parser.add_argument(
        "--disable_collision_bbox_calibration",
        action="store_true",
        help="Do not scale the Stage1 collision proxy to episode_manifest.normalization bbox.",
    )
    parser.add_argument(
        "--collision_bbox_margin",
        default=0.0,
        type=float,
        help="Inset each side of the manifest bbox by this many metres for the collision proxy.",
    )
    parser.add_argument(
        "--collision_bbox_margin_ratio",
        default=0.0,
        type=float,
        help="Inset each side of the manifest bbox by this fraction of each bbox axis length.",
    )
    parser.add_argument(
        "--collision_bbox_margin_z",
        default=None,
        type=float,
        help="Override the bbox inset on z only; useful when floor contact is inflated but planar contact should keep the full footprint.",
    )
    parser.add_argument(
        "--collision_bbox_margin_z_ratio",
        default=None,
        type=float,
        help="[Deprecated] Override the bbox inset on z as a fraction of bbox height. "
             "Prefer --floor_clip_slack instead.",
    )
    parser.add_argument(
        "--disable_floor_clip",
        action="store_true",
        help="Do not remove below-floor Gaussians. Use only if floor_clip causes problems.",
    )
    parser.add_argument(
        "--floor_clip_slack",
        default=0.0,
        type=float,
        help="Tolerance in metres for floor clip. A small positive value (e.g. 0.005) "
             "keeps Gaussians that barely touch the floor contact plane. Default 0.0.",
    )
    parser.add_argument(
        "--stage1_world_translation",
        default=None,
        type=str,
        help="Debug override: Stage1 object origin in world coordinates as x,y,z.",
    )
    parser.add_argument(
        "--stage1_world_rotation",
        default=None,
        type=str,
        help="Stage1 local-to-world 3x3 rotation matrix, row-major 9 values. Used with --stage1_world_translation.",
    )
    parser.add_argument("--query_radius_scale", default=1.10, type=float)
    parser.add_argument("--query_rings", default=5, type=int)
    parser.add_argument("--query_angles", default=32, type=int)
    parser.add_argument(
        "--query_mode",
        default="floor_disk",
        choices=("floor_disk", "body_surface", "body_lowest_k"),
        help=(
            "floor_disk: environment-side XY disk query points under the object, "
            "evaluated against the Gaussian union SDF (default, original behaviour). "
            "body_surface: object-side query points sampled on the Gaussian primitives "
            "(make_gaussian_proxy_query_points), evaluated against the floor plane. "
            "body_lowest_k: same object-side sampling as body_surface, but each step "
            "keeps only K contact patches. In floor dynamics these are the lowest "
            "(deepest) points; in pairwise_impedance they are the strongest bidirectional "
            "body-body patches."
        ),
    )
    parser.add_argument(
        "--body_query_dirs",
        default=6,
        type=int,
        help="Directions per Gaussian for --query_mode body_surface/body_lowest_k (6 = axis dirs, else Fibonacci).",
    )
    parser.add_argument(
        "--body_lowest_k",
        default=32,
        type=int,
        help=(
            "Number of lowest body-surface query points kept as the contact patch for "
            "--query_mode body_lowest_k. Smaller = tighter patch, weaker aggregate friction."
        ),
    )
    parser.add_argument(
        "--body_query_scheme",
        default="axis6",
        choices=("axis6", "fibonacci", "analytic"),
        help=(
            "How object-side query points are sampled per Gaussian primitive. "
            "axis6: 6 local axis directions (original). "
            "fibonacci: --body_query_dirs directions on a Fibonacci lattice (denser, more uniform). "
            "analytic: an exact support point per sphere; c_world-r*n against a plane, "
            "or the support point toward the closest target sphere for body-body contact. "
            "axis6/fibonacci sample in the LOCAL frame, so the sampled points rotate with the body "
            "while analytic is rotation-invariant for spherical primitives and needs one point per primitive."
        ),
    )
    parser.add_argument(
        "--floor_friction_mode",
        default="off",
        choices=("off", "fixed", "learned"),
        help=(
            "Coulomb friction at the floor contact point (restitution dynamics only). "
            "Slip uses ω×r with ω observed from the GT orientation sequence, so spin "
            "converts into planar motion on rim bounces. 'learned' fits the coefficient."
        ),
    )
    parser.add_argument(
        "--floor_friction_init",
        default=0.5,
        type=float,
        help="Initial (or fixed) friction coefficient for --floor_friction_mode.",
    )
    parser.add_argument(
        "--init_restitution",
        default=0.5,
        type=float,
        help="Initial restitution for --dynamics restitution. The contact loss landscape "
             "has local minima, so this (with --floor_friction_init) picks the basin.",
    )
    parser.add_argument(
        "--freeze_gravity",
        action="store_true",
        help="Keep gravity_z fixed at -9.81 instead of fitting it. Removes a compensation "
             "degeneracy when friction/restitution are learned jointly.",
    )
    parser.add_argument(
        "--substeps",
        default=1,
        type=int,
        help="Integration substeps per frame (restitution dynamics). >1 integrates the "
             "contact impulse finely instead of smearing it across a full 1/fps step.",
    )
    parser.add_argument("--contact_softness", default=2e-3, type=float)
    parser.add_argument("--smooth_max_temperature", default=1e-2, type=float)
    parser.add_argument("--inside_penalty", default=0.02, type=float)
    parser.add_argument("--inside_sharpness", default=50.0, type=float)
    parser.add_argument(
        "--dynamics",
        default="impedance",
        choices=("impedance", "restitution", "pairwise_impedance"),
        help=(
            "impedance: paper III-D-2 SoftPlus(-K(h*Jb+phi)-D*Jb) form, learn (v0, g, K, D). "
            "restitution: legacy reflection+slop, learn (v0, g, e). "
            "pairwise_impedance: use PairwiseGaussianBodyImpedanceDynamics against a static Gaussian body B."
        ),
    )
    parser.add_argument("--init_stiffness", default=800.0, type=float)
    parser.add_argument("--init_damping", default=30.0, type=float)
    parser.add_argument("--mass", default=1.0, type=float)
    parser.add_argument(
        "--pairwise_mass_mode",
        default="fixed",
        choices=("fixed", "learned"),
        help="Keep mass fixed or learn the positive body-A mass in pairwise dynamics.",
    )
    parser.add_argument("--pairwise_inertia_diag", default="1,1,1", type=str)
    parser.add_argument(
        "--pairwise_inertia_mode",
        default="fixed",
        choices=("fixed", "learned"),
        help="Keep or learn the positive diagonal body-frame inertia tensor.",
    )
    parser.add_argument("--pairwise_mass_l2_weight", default=1e-4, type=float)
    parser.add_argument("--pairwise_inertia_l2_weight", default=1e-4, type=float)
    parser.add_argument(
        "--actions_json",
        default=None,
        type=Path,
        help="Action trajectory JSON. Defaults to episode_root/actions/trajectory.json when present.",
    )
    parser.add_argument("--action_force_scale", default=1.0, type=float)
    parser.add_argument("--action_torque_scale", default=1.0, type=float)
    parser.add_argument(
        "--pairwise_body_b_trajectory_json",
        default=None,
        type=Path,
        help="Prescribed pose trajectory for a kinematic environment/robot body B.",
    )
    parser.add_argument(
        "--initial_velocity_source",
        default="finite_difference",
        choices=("finite_difference", "trajectory"),
        help="Initial linear velocity source for the rollout.",
    )
    parser.add_argument(
        "--freeze_initial_velocity",
        action="store_true",
        help="Use the selected initial velocity as a fixed rollout input instead of learning it.",
    )
    parser.add_argument(
        "--floor_tangential_damping",
        default=0.0,
        type=float,
        help="Contact-gated damping applied to floor-tangent velocity in floor dynamics.",
    )
    parser.add_argument("--pairwise_body_b_ply", default=None, type=Path)
    parser.add_argument("--pairwise_body_b_object_id", default=None, type=int)
    parser.add_argument("--pairwise_body_b_foreground_threshold", default=None, type=float)
    parser.add_argument("--pairwise_body_b_opacity_threshold", default=None, type=float)
    parser.add_argument("--pairwise_body_b_max_primitives", default=None, type=int)
    parser.add_argument("--pairwise_body_b_world_translation", default=None, type=str)
    parser.add_argument("--pairwise_body_b_world_rotation", default=None, type=str)
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
    parser.add_argument(
        "--pairwise_friction_mode",
        default="fixed",
        choices=("off", "fixed", "learned"),
        help="Use zero, fixed, or learned mu in pairwise contact dynamics.",
    )
    parser.add_argument(
        "--pairwise_contact_model",
        default="dual_cone",
        choices=("dual_cone", "projected"),
        help="Paper Appendix-C dual-cone dynamics or the previous projected-friction baseline.",
    )
    parser.add_argument("--pairwise_dual_cone_directions", default=4, type=int)
    parser.add_argument("--pairwise_tangential_damping", default=0.0, type=float)
    parser.add_argument(
        "--orientation_loss_weight",
        default=0.0,
        type=float,
        help="Weight for sign-invariant quaternion loss in --dynamics pairwise_impedance.",
    )
    parser.add_argument("--position_loss_weight", default=1.0, type=float)
    parser.add_argument(
        "--image_only_objective",
        action="store_true",
        help="Set position/orientation/posed-geometry supervision weights to zero; optimize from RGB and priors.",
    )
    parser.add_argument(
        "--fit_initial_angular_velocity",
        action="store_true",
        help="Learn the initial angular velocity from trajectory orientation loss in pairwise mode.",
    )
    parser.add_argument("--angular_velocity_l2_weight", default=1e-4, type=float)
    parser.add_argument(
        "--geometry_loss_weight",
        default=0.0,
        type=float,
        help="Weight for posed Gaussian-center MSE against the GT rigid pose.",
    )
    parser.add_argument("--geometry_loss_stride", default=1, type=int)
    parser.add_argument("--refine_geometry", action="store_true")
    parser.add_argument(
        "--geometry_gradient_route",
        default="collision_only",
        choices=("collision_only", "collision_and_render"),
        help=(
            "collision_only matches ContactGaussian-WM: renderer/direct supervision "
            "see detached geometry while image gradients still reach geometry through dynamics."
        ),
    )
    parser.add_argument("--geometry_center_l2_weight", default=1e-3, type=float)
    parser.add_argument("--geometry_radius_l2_weight", default=1e-3, type=float)
    parser.add_argument("--geometry_max_center_offset", default=0.02, type=float)
    parser.add_argument("--geometry_max_log_radius_offset", default=0.35, type=float)
    parser.add_argument("--geometry_min_radius", default=1e-4, type=float)
    parser.add_argument("--gaussian_rgb_loss_weight", default=0.0, type=float)
    parser.add_argument("--gaussian_rgb_dir", default=None, type=Path)
    parser.add_argument("--gaussian_mask_dir", default=None, type=Path)
    parser.add_argument("--gaussian_views_manifest", default=None, type=Path)
    parser.add_argument("--gaussian_render_stride", default=10, type=int)
    parser.add_argument("--gaussian_render_max_frames", default=0, type=int)
    parser.add_argument("--gaussian_render_width", default=160, type=int)
    parser.add_argument("--gaussian_render_height", default=120, type=int)
    parser.add_argument(
        "--gaussian_render_loss",
        default="l1_ssim",
        choices=("l1", "mse", "l1_ssim", "l1_loftr"),
    )
    parser.add_argument("--gaussian_render_ssim_weight", default=0.2, type=float)
    parser.add_argument("--gaussian_render_loftr_weight", default=0.1, type=float)
    parser.add_argument("--loftr_pretrained", default="outdoor", choices=("outdoor", "indoor"))
    parser.add_argument("--loftr_confidence_threshold", default=0.2, type=float)
    parser.add_argument("--loftr_max_matches", default=1024, type=int)
    parser.add_argument("--loftr_min_matches", default=8, type=int)
    parser.add_argument("--loftr_patch_radius", default=2, type=int)
    parser.add_argument("--gaussian_cam_distance", default=1.12, type=float)
    parser.add_argument("--gaussian_cam_height", default=0.66, type=float)
    parser.add_argument("--gaussian_cam_fovy_deg", default=40.0, type=float)
    parser.add_argument("--gaussian_render_white_background", action="store_true")
    parser.add_argument("--gaussian_render_scale_multiplier", default=1.0, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gif_fps", default=24, type=int)
    return parser.parse_args()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def route_geometry_for_supervision(
    centers: torch.Tensor,
    radii: torch.Tensor,
    route: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if route == "collision_only":
        return centers.detach(), radii.detach()
    if route == "collision_and_render":
        return centers, radii
    raise ValueError(f"Unknown geometry gradient route: {route!r}.")


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


def load_target_quaternions(states: list[dict]) -> torch.Tensor:
    quaternions = [
        state.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0])
        for state in states
    ]
    return normalize_quaternion(torch.tensor(quaternions, dtype=torch.float32))


def load_initial_angular_velocity(states: list[dict]) -> torch.Tensor:
    angular_velocity = states[0].get("angular_velocity", [0.0, 0.0, 0.0])
    return torch.tensor(angular_velocity, dtype=torch.float32)


def load_initial_linear_velocity(states: list[dict]) -> torch.Tensor:
    linear_velocity = states[0].get("linear_velocity")
    if linear_velocity is None:
        qvel = states[0].get("qvel")
        if qvel is not None and len(qvel) >= 3:
            linear_velocity = qvel[:3]
    if linear_velocity is None:
        raise ValueError("Trajectory state[0] has no linear_velocity/qvel needed for --initial_velocity_source trajectory.")
    return torch.tensor(linear_velocity, dtype=torch.float32)


def load_action_wrenches(
    path: Path | None,
    *,
    num_steps: int,
    force_scale: float,
    torque_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    forces = torch.zeros((num_steps, 3), dtype=torch.float32)
    torques = torch.zeros((num_steps, 3), dtype=torch.float32)
    if path is None or not path.exists():
        return forces, torques, {"source": "zero_wrench", "path": None, "num_actions": 0}
    payload = read_json(path)
    actions = payload.get("actions", payload if isinstance(payload, list) else None)
    if not isinstance(actions, list):
        raise ValueError(f"{path} must contain an 'actions' list.")
    for index, action in enumerate(actions[:num_steps]):
        wrench = action.get("generalized_force") or action.get("wrench")
        force = action.get("force_world") or action.get("force")
        torque = action.get("torque_world") or action.get("torque")
        control = action.get("control")
        if wrench is not None:
            if len(wrench) != 6:
                raise ValueError(f"actions[{index}].generalized_force/wrench must have 6 values.")
            force, torque = wrench[:3], wrench[3:]
        elif force is None and torque is None and control:
            if len(control) != 6:
                raise ValueError(
                    f"actions[{index}].control has {len(control)} values; "
                    "only a 6D world-frame wrench can be inferred automatically."
                )
            force, torque = control[:3], control[3:]
        if force is not None:
            forces[index] = torch.tensor(force, dtype=torch.float32) * float(force_scale)
        if torque is not None:
            torques[index] = torch.tensor(torque, dtype=torch.float32) * float(torque_scale)
    return forces, torques, {
        "source": "action_trajectory",
        "path": str(path.resolve()),
        "num_actions": min(len(actions), num_steps),
        "force_scale": float(force_scale),
        "torque_scale": float(torque_scale),
        "force_nonzero_steps": int(torch.any(forces != 0.0, dim=-1).sum().item()),
        "torque_nonzero_steps": int(torch.any(torques != 0.0, dim=-1).sum().item()),
    }


def infer_dt(times: torch.Tensor) -> float:
    diffs = times[1:] - times[:-1]
    dt = float(torch.median(diffs).item())
    if dt <= 0.0:
        raise ValueError(f"Invalid non-positive dt inferred from trajectory: {dt}")
    return dt


def load_kinematic_body_trajectory(
    path: Path,
    *,
    num_frames: int,
    dt: float,
) -> tuple[dict[str, torch.Tensor], dict]:
    payload = read_json(path)
    states = payload.get("states", payload.get("trajectory"))
    if not isinstance(states, list) or len(states) < num_frames:
        raise ValueError(f"{path} must contain at least {num_frames} states.")
    states = states[:num_frames]
    positions = torch.tensor([state["position"] for state in states], dtype=torch.float32)
    quaternions = normalize_quaternion(torch.tensor(
        [state.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]) for state in states],
        dtype=torch.float32,
    ))
    finite_linear = (positions[1:] - positions[:-1]) / float(dt)
    finite_linear = torch.cat((finite_linear, finite_linear[-1:]), dim=0)
    finite_angular = angular_velocity_from_quaternions(quaternions, dt)
    finite_angular = torch.cat((finite_angular, finite_angular[-1:]), dim=0)
    linear_velocities = torch.stack([
        torch.tensor(state.get("linear_velocity"), dtype=torch.float32)
        if state.get("linear_velocity") is not None else finite_linear[index]
        for index, state in enumerate(states)
    ])
    angular_velocities = torch.stack([
        torch.tensor(state.get("angular_velocity"), dtype=torch.float32)
        if state.get("angular_velocity") is not None else finite_angular[index]
        for index, state in enumerate(states)
    ])
    return {
        "positions": positions,
        "quaternions": quaternions,
        "linear_velocities": linear_velocities,
        "angular_velocities": angular_velocities,
    }, {
        "source": "kinematic_body_b_trajectory",
        "path": str(path.resolve()),
        "num_frames": num_frames,
        "velocity_source": "per-state values with finite-difference fallback",
    }


def parse_float_sequence(value: str, *, expected: int, label: str) -> tuple[float, ...]:
    raw = str(value).replace(",", " ").split()
    if len(raw) != expected:
        raise ValueError(f"{label} expects {expected} values, got {len(raw)}: {value}")
    return tuple(float(part) for part in raw)


def parse_optional_vec3(value: str | None, *, label: str) -> tuple[float, float, float] | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = parse_float_sequence(value, expected=3, label=label)
    return (parsed[0], parsed[1], parsed[2])


def parse_optional_matrix3(value: str | None, *, label: str) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = parse_float_sequence(value, expected=9, label=label)
    return (parsed[0:3], parsed[3:6], parsed[6:9])


def parse_vec3(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated values, got: {value}")
    return tuple(float(part) for part in parts)


def matrix3_to_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    array = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    return tuple(tuple(float(array[row, col]) for col in range(3)) for row in range(3))


def quaternion_wxyz_to_matrix_tuple(values) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    quat = torch.tensor(values, dtype=torch.float32)
    matrix = quat_wxyz_to_matrix(quat).detach().cpu().numpy()
    return matrix3_to_tuple(matrix)


def normalize_pose_contract(payload: dict, *, source: str) -> dict:
    translation = (
        payload.get("translation")
        or payload.get("world_translation")
        or payload.get("position")
    )
    if translation is None:
        raise ValueError(f"{source} must define translation/world_translation/position.")
    translation_tuple = parse_optional_vec3(",".join(str(v) for v in translation), label=f"{source}.translation")

    rotation = payload.get("rotation_matrix") or payload.get("world_rotation")
    if rotation is not None:
        rotation_array = np.asarray(rotation, dtype=np.float32)
        rotation_tuple = matrix3_to_tuple(rotation_array)
    elif payload.get("quaternion_wxyz") is not None:
        rotation_tuple = quaternion_wxyz_to_matrix_tuple(payload["quaternion_wxyz"])
    else:
        rotation_tuple = matrix3_to_tuple(np.eye(3, dtype=np.float32))

    return {
        "coordinate_frame": "world",
        "world_translation": translation_tuple,
        "world_rotation": rotation_tuple,
        "source": source,
    }


def load_stage1_coordinate_contract(episode_root: Path, *, translation_override, rotation_override) -> dict:
    if translation_override is not None:
        return {
            "coordinate_frame": "world",
            "world_translation": translation_override,
            "world_rotation": rotation_override,
            "source": "cli_override",
        }
    if rotation_override is not None:
        raise ValueError("--stage1_world_rotation requires --stage1_world_translation.")

    manifest_path = episode_root / "episode_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing episode manifest: {manifest_path}")
    manifest = read_json(manifest_path)

    body_contract = manifest.get("stage1_gaussian_body") or {}
    coordinate_frame = (
        body_contract.get("coordinate_frame")
        or manifest.get("stage1_coordinate_frame")
        or manifest.get("stage1_points_coordinate_frame")
    )
    world_pose = (
        body_contract.get("world_pose")
        or manifest.get("stage1_object_pose")
        or manifest.get("stage1_world_pose")
    )
    if world_pose is not None:
        return normalize_pose_contract(world_pose, source="episode_manifest.stage1_gaussian_body.world_pose")
    if coordinate_frame == "object_local":
        return {
            "coordinate_frame": "object_local",
            "world_translation": None,
            "world_rotation": None,
            "source": "episode_manifest.stage1_gaussian_body.coordinate_frame",
        }
    if coordinate_frame == "world":
        raise ValueError(
            "episode_manifest declares Stage1 Gaussian coordinates as world-frame but does not provide "
            "stage1_gaussian_body.world_pose."
        )

    raise ValueError(
        "Stage1 Gaussian coordinate frame is not declared. Add "
        "episode_manifest.stage1_gaussian_body={\"coordinate_frame\":\"object_local\"} for local PLYs, "
        "or {\"coordinate_frame\":\"world\", \"world_pose\":{...}} for world-frame Stage1 PLYs."
    )


def load_manifest_collision_bbox(episode_root: Path) -> dict | None:
    manifest_path = episode_root / "episode_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    normalization = manifest.get("normalization") or {}
    bbox_min = normalization.get("bbox_min")
    bbox_max = normalization.get("bbox_max")
    if bbox_min is None or bbox_max is None:
        return None
    scale = float(normalization.get("scale", 1.0))
    bbox_min = np.asarray(bbox_min, dtype=np.float32) * scale
    bbox_max = np.asarray(bbox_max, dtype=np.float32) * scale
    if bbox_min.shape != (3,) or bbox_max.shape != (3,):
        raise ValueError("episode_manifest.normalization bbox_min/bbox_max must be length-3 arrays.")
    if np.any(bbox_max <= bbox_min):
        raise ValueError("episode_manifest.normalization bbox_max must be greater than bbox_min on every axis.")
    return {
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "source": "episode_manifest.normalization",
    }


def floor_clip_collision_proxy(
    local_centers: torch.Tensor,
    radii: torch.Tensor,
    floor_z_local: float,
    slack: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Remove Gaussians whose bottom (center_z - radius) is below floor_z_local.

    This replaces collision_bbox_margin_z_ratio: instead of compressing the
    whole proxy vertically (which distorts top/side contact geometry too), we
    physically-motivated remove Gaussians that bleed below the floor contact
    plane.  Works for any object shape without per-object tuning.

    Args:
        local_centers: (N, 3) Gaussian centers in object-local frame.
        radii:         (N,) Gaussian radii.
        floor_z_local: z-coordinate of the floor contact plane in local frame.
                       For an object whose asset JSON has bbox_min[2] = -h,
                       pass floor_z_local = -h.
        slack:         Extra tolerance (metres).  A small positive value (e.g.
                       0.005) keeps Gaussians that barely touch the floor.
    """
    threshold = floor_z_local - slack
    above = (local_centers[:, 2] - radii) >= threshold
    n_removed = int((~above).sum().item())
    metadata = {
        "floor_clip_enabled": True,
        "floor_z_local": float(floor_z_local),
        "slack": float(slack),
        "n_removed": n_removed,
        "n_kept": int(above.sum().item()),
    }
    if n_removed > 0:
        print(
            f"[INFO] floor_clip: removed {n_removed} below-floor Gaussians "
            f"(floor_z_local={floor_z_local:.4f}, slack={slack:.4f})"
        )
    return local_centers[above], radii[above], metadata


def calibrate_collision_proxy_to_bbox(
    local_centers: torch.Tensor,
    radii: torch.Tensor,
    collision_bbox: dict,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    target_min = torch.as_tensor(collision_bbox["bbox_min"], dtype=local_centers.dtype, device=local_centers.device)
    target_max = torch.as_tensor(collision_bbox["bbox_max"], dtype=local_centers.dtype, device=local_centers.device)
    margin_xyz = torch.as_tensor(collision_bbox.get("margin_xyz", [0.0, 0.0, 0.0]), dtype=local_centers.dtype, device=local_centers.device)
    if torch.any(margin_xyz > 0.0):
        target_min = target_min + margin_xyz
        target_max = target_max - margin_xyz
        if torch.any(target_max <= target_min):
            raise ValueError(f"collision bbox margin {margin_xyz.detach().cpu().tolist()} collapses the manifest bbox.")
    current_min = torch.min(local_centers - radii.unsqueeze(-1), dim=0).values
    current_max = torch.max(local_centers + radii.unsqueeze(-1), dim=0).values
    current_center = (current_min + current_max) * 0.5
    target_center = (target_min + target_max) * 0.5
    current_extent = torch.clamp(current_max - current_min, min=1e-6)
    target_extent = target_max - target_min
    axis_scale = target_extent / current_extent
    radius_scale = torch.min(axis_scale)
    calibrated_centers = (local_centers - current_center.unsqueeze(0)) * axis_scale.unsqueeze(0) + target_center.unsqueeze(0)
    calibrated_radii = torch.clamp(radii * radius_scale, min=1e-6)
    metadata = {
        "enabled": True,
        "source": collision_bbox["source"],
        "current_bbox_min": current_min.detach().cpu().tolist(),
        "current_bbox_max": current_max.detach().cpu().tolist(),
        "target_bbox_min": target_min.detach().cpu().tolist(),
        "target_bbox_max": target_max.detach().cpu().tolist(),
        "margin_xyz": margin_xyz.detach().cpu().tolist(),
        "axis_scale": axis_scale.detach().cpu().tolist(),
        "radius_scale": float(radius_scale.detach().cpu().item()),
    }
    return calibrated_centers, calibrated_radii, metadata


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
    coordinate_contract: dict,
    collision_bbox: dict | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    world_translation = coordinate_contract["world_translation"]
    world_rotation = coordinate_contract["world_rotation"]
    body = load_gaussian_collision_body_from_ply(
        path,
        radius_scale=float(args.radius_scale),
        radius_convention=str(args.gaussian_radius_convention),
        scale_reduction=str(args.gaussian_scale_reduction),
        isotropic_tolerance=float(args.gaussian_isotropic_tolerance),
        recenter=False,
        object_id=object_id,
        foreground_threshold=foreground_threshold,
        opacity_threshold=opacity_threshold,
        max_primitives=max_primitives,
        use_centers_as_queries=True,
        world_translation=world_translation,
        world_rotation=world_rotation,
        dtype=torch.float32,
        device=device,
    )
    local_centers = body.local_centers
    radii = body.radii
    source_indices = body.source_indices

    collision_bbox_metadata = {"enabled": False}
    if collision_bbox is not None:
        local_centers, radii, collision_bbox_metadata = calibrate_collision_proxy_to_bbox(
            local_centers,
            radii,
            collision_bbox,
        )

    # Floor clip must run after bbox calibration so the manifest floor plane and
    # the Gaussian proxy are in the same local metric frame.
    floor_clip_metadata = {"floor_clip_enabled": False}
    if collision_bbox is not None and not getattr(args, "disable_floor_clip", False):
        floor_z_local = float(collision_bbox["bbox_min"][2])
        slack = float(getattr(args, "floor_clip_slack", 0.0))
        floor_keep = (local_centers[:, 2] - radii) >= (floor_z_local - slack)
        local_centers, radii, floor_clip_metadata = floor_clip_collision_proxy(
            local_centers, radii, floor_z_local, slack=slack
        )
        if source_indices is not None:
            source_indices = source_indices[floor_keep]
        if local_centers.shape[0] == 0:
            raise RuntimeError(
                "floor_clip removed ALL Gaussians. "
                "Try --floor_clip_slack 0.02 or --disable_floor_clip."
            )
    coordinate_mode = "world_pose" if world_translation is not None else "object_local"
    spherical_metadata_path = path.with_suffix(".spherical.json")
    spherical_metadata = (
        read_json(spherical_metadata_path) if spherical_metadata_path.exists() else None
    )
    metadata = {
        "coordinate_mode": coordinate_mode,
        "world_translation": None if world_translation is None else list(world_translation),
        "world_rotation": None if world_rotation is None else [list(row) for row in world_rotation],
        "coordinate_contract_source": coordinate_contract["source"],
        "gaussian_radius_convention": str(args.gaussian_radius_convention),
        "gaussian_scale_reduction": str(args.gaussian_scale_reduction),
        "gaussian_isotropic_tolerance": float(args.gaussian_isotropic_tolerance),
        "stage1_scale_to_collision_radius": (
            2.0 if args.gaussian_radius_convention == "paper_r2s" else 1.0
        ),
        "spherical_gaussian_metadata_path": (
            str(spherical_metadata_path) if spherical_metadata is not None else None
        ),
        "spherical_gaussian_metadata": spherical_metadata,
        "recenter": False,
        "floor_clip": floor_clip_metadata,
        "collision_bbox_calibration": collision_bbox_metadata,
        "source_indices": (
            None if source_indices is None else source_indices.detach().cpu().tolist()
        ),
    }
    return local_centers, radii, metadata


def oriented_centers(local_centers: torch.Tensor, position: torch.Tensor, quaternion_wxyz: torch.Tensor | None) -> torch.Tensor:
    if quaternion_wxyz is None:
        return local_centers + position.unsqueeze(0)
    return transform_local_points(local_centers, position, quaternion_wxyz=quaternion_wxyz)


def _smooth_min_signed(values: torch.Tensor, temperature: float) -> torch.Tensor:
    return -temperature * torch.logsumexp(-values / temperature, dim=-1)


def angular_velocity_from_quaternions(quaternions: torch.Tensor, dt: float) -> torch.Tensor:
    """World-frame angular velocity (T-1, 3) from unit quaternions wxyz (T, 4).

    Finite-difference: dq = q_{t+1} ⊗ conj(q_t), ω = axis * angle / dt.
    """
    q0 = quaternions[:-1]
    q1 = quaternions[1:]
    w0, x0, y0, z0 = q0.unbind(-1)
    w1, x1, y1, z1 = q1.unbind(-1)
    dw = w1 * w0 + x1 * x0 + y1 * y0 + z1 * z0
    dx = -w1 * x0 + x1 * w0 - y1 * z0 + z1 * y0
    dy = -w1 * y0 + x1 * z0 + y1 * w0 - z1 * x0
    dz = -w1 * z0 - x1 * y0 + y1 * x0 + z1 * w0
    sign = torch.where(dw < 0.0, -torch.ones_like(dw), torch.ones_like(dw))
    dw = dw * sign
    vec = torch.stack((dx, dy, dz), dim=-1) * sign.unsqueeze(-1)
    vec_norm = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp(min=1e-12)
    angle = 2.0 * torch.atan2(vec_norm.squeeze(-1), dw.clamp(min=-1.0, max=1.0))
    return (vec / vec_norm) * (angle / float(dt)).unsqueeze(-1)


def integrate_quaternion_sequence(
    initial_quaternion: torch.Tensor,
    angular_velocity: torch.Tensor,
    *,
    steps: int,
    dt: float,
) -> torch.Tensor:
    """Differentiably integrate a constant world-frame angular velocity."""
    quaternion = normalize_quaternion(initial_quaternion)
    quaternions = [quaternion]
    omega_quaternion = torch.cat((torch.zeros_like(angular_velocity[:1]), angular_velocity))
    for _ in range(max(0, int(steps) - 1)):
        w1, x1, y1, z1 = omega_quaternion.unbind(-1)
        w2, x2, y2, z2 = quaternion.unbind(-1)
        derivative = 0.5 * torch.stack(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )
        )
        quaternion = normalize_quaternion(quaternion + float(dt) * derivative)
        quaternions.append(quaternion)
    return torch.stack(quaternions, dim=0)


def _resolve_num_query_points(
    query_mode: str,
    body_query_scheme: str,
    body_lowest_k: int,
    local_query_points: torch.Tensor | None,
    num_primitives: int,
    num_floor_offsets: int,
) -> int:
    """Query points actually used per step (the contact patch size)."""
    if query_mode not in ("body_surface", "body_lowest_k"):
        return int(num_floor_offsets)
    # analytic builds one point per primitive on the fly; the others precompute a pool.
    pool = int(num_primitives) if body_query_scheme == "analytic" else int(local_query_points.shape[0])
    if query_mode == "body_lowest_k":
        return min(int(body_lowest_k), pool)
    return pool


def _floor_contact_response(
    predicted_position: torch.Tensor,
    predicted_velocity: torch.Tensor,
    orientation: torch.Tensor | None,
    omega: torch.Tensor | None,
    *,
    normal: torch.Tensor,
    restitution: torch.Tensor,
    local_centers: torch.Tensor,
    radii: torch.Tensor,
    floor_query_offsets_xy: torch.Tensor,
    contact_softness: float,
    smooth_max_temperature: float,
    inside_penalty: float,
    inside_sharpness: float,
    query_mode: str,
    local_query_points: torch.Tensor | None,
    friction_coefficient: torch.Tensor | None,
    body_lowest_k: int = 0,
    body_query_scheme: str = "axis6",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One frictional floor-contact impulse. Returns (velocity, gate, penetration).

    Friction is summed over every query point in contact (weighted by its own
    contact weight), and each point's slip carries its own ω×r_i lever, so a
    tilted rim contact produces the correct net tangential impulse and the spin
    it induces — not a single averaged-point approximation.

    ``body_lowest_k`` mode keeps only the K lowest world-z query points as the
    contact patch each step. body_surface sums friction over every primitive's
    surface samples (1080 pts), which over-broadens the contact patch and
    over-damps the induced planar motion; restricting to the real lowest points
    keeps the aggregate Coulomb friction physical.
    """
    if query_mode in ("body_surface", "body_lowest_k"):
        if body_query_scheme == "analytic":
            # A sphere's lowest point against a plane is c_world - r*n regardless of
            # body orientation. Local-frame direction sampling rotates with the body
            # and therefore misses it; this is exact and needs one point per primitive.
            centers_world = oriented_centers(local_centers, predicted_position, orientation)
            query_points = centers_world - radii.unsqueeze(-1) * normal.unsqueeze(0)
        else:
            query_points = oriented_centers(local_query_points, predicted_position, orientation)
        signed_distances = query_points[:, 2]
        if query_mode == "body_lowest_k":
            k = min(int(body_lowest_k), int(signed_distances.shape[0]))
            if k > 0:
                # Lowest = smallest signed z = deepest into / closest to the floor.
                _, keep_idx = torch.topk(signed_distances, k, largest=False)
                query_points = query_points[keep_idx]
                signed_distances = signed_distances[keep_idx]
        penetrations = F.softplus(-signed_distances / contact_softness) * contact_softness
        contact_weights = torch.sigmoid(-signed_distances / contact_softness)
    else:
        query_points = torch.cat(
            (
                predicted_position[:2].unsqueeze(0) + floor_query_offsets_xy,
                torch.zeros((floor_query_offsets_xy.shape[0], 1), dtype=predicted_position.dtype, device=predicted_position.device),
            ),
            dim=-1,
        )
        gaussian_centers = oriented_centers(local_centers, predicted_position, orientation)
        contacts = detect_gaussian_union_contacts(
            query_points,
            gaussian_centers,
            radii,
            normal,
            softness=contact_softness,
            smooth_min_temperature=smooth_max_temperature,
            inside_penalty=inside_penalty,
            inside_sharpness=inside_sharpness,
        )
        penetrations = contacts.penetrations
        contact_weights = contacts.contact_weights

    contact_gate = smooth_weighted_max(contact_weights, smooth_max_temperature)
    penetration_depth = smooth_weighted_max(penetrations, smooth_max_temperature)
    normal_velocity = torch.sum(predicted_velocity * normal)
    closing_speed = torch.nn.functional.softplus(-normal_velocity / contact_softness) * contact_softness
    normal_impulse = contact_gate * (1.0 + restitution) * closing_speed
    velocity = predicted_velocity + normal_impulse * normal

    if friction_coefficient is not None:
        levers = query_points - predicted_position.unsqueeze(0)
        if omega is not None:
            slip = velocity.unsqueeze(0) + torch.linalg.cross(omega.unsqueeze(0).expand_as(levers), levers)
        else:
            slip = velocity.unsqueeze(0).expand_as(levers)
        slip_tangent = slip - (slip @ normal).unsqueeze(-1) * normal
        slip_norm = torch.sqrt(torch.sum(slip_tangent * slip_tangent, dim=-1) + 1e-12)
        # Per-point Coulomb cap, distributing the (already aggregated) normal
        # impulse across contacting points by their weight share.
        weight_sum = torch.clamp(contact_weights.sum(), min=1e-9)
        per_point_normal = normal_impulse * contact_weights / weight_sum
        per_point_friction = torch.minimum(friction_coefficient * per_point_normal, slip_norm)
        friction_delta = torch.sum(
            (per_point_friction / slip_norm).unsqueeze(-1) * slip_tangent, dim=0
        )
        velocity = velocity - friction_delta
    return velocity, contact_gate, penetration_depth


def simulate(
    initial_position: torch.Tensor,
    initial_velocity: torch.Tensor,
    gravity_z: torch.Tensor,
    restitution: torch.Tensor,
    local_centers: torch.Tensor,
    radii: torch.Tensor,
    floor_query_offsets_xy: torch.Tensor,
    orientation_sequence: torch.Tensor | None,
    *,
    steps: int,
    dt: float,
    contact_softness: float,
    smooth_max_temperature: float,
    inside_penalty: float = 0.02,
    inside_sharpness: float = 50.0,
    floor_tangential_damping: float = 0.0,
    query_mode: str = "floor_disk",
    local_query_points: torch.Tensor | None = None,
    friction_coefficient: torch.Tensor | None = None,
    angular_velocity_sequence: torch.Tensor | None = None,
    substeps: int = 1,
    body_lowest_k: int = 0,
    body_query_scheme: str = "axis6",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy reflection+slop dynamics (kept for --dynamics restitution)."""
    if (
        query_mode in ("body_surface", "body_lowest_k")
        and body_query_scheme != "analytic"
        and local_query_points is None
    ):
        raise ValueError(f"query_mode='{query_mode}' requires local_query_points.")
    collider = PlaneCollider.floor(dtype=initial_position.dtype, device=initial_position.device)
    normal = collider.normal.to(dtype=initial_position.dtype, device=initial_position.device)
    position = initial_position
    velocity = initial_velocity
    positions = [position]
    contact_gates = []
    n_sub = max(1, int(substeps))
    dt_sub = dt / n_sub
    gravity = torch.stack(
        (
            torch.zeros((), dtype=initial_position.dtype, device=initial_position.device),
            torch.zeros((), dtype=initial_position.dtype, device=initial_position.device),
            gravity_z,
        )
    )

    for step_idx in range(steps - 1):
        orientation = None if orientation_sequence is None else orientation_sequence[min(step_idx + 1, orientation_sequence.shape[0] - 1)]
        omega = (
            angular_velocity_sequence[min(step_idx, angular_velocity_sequence.shape[0] - 1)]
            if angular_velocity_sequence is not None and angular_velocity_sequence.shape[0] > 0
            else None
        )
        step_gate = None
        # Split each frame into substeps so the contact impulse is integrated
        # finely instead of being smeared across a full 1/fps step.
        for _ in range(n_sub):
            predicted_velocity = velocity + gravity * dt_sub
            predicted_position = position + predicted_velocity * dt_sub
            velocity, contact_gate, penetration_depth = _floor_contact_response(
                predicted_position,
                predicted_velocity,
                orientation,
                omega,
                normal=normal,
                restitution=restitution,
                local_centers=local_centers,
                radii=radii,
                floor_query_offsets_xy=floor_query_offsets_xy,
                contact_softness=contact_softness,
                smooth_max_temperature=smooth_max_temperature,
                inside_penalty=inside_penalty,
                inside_sharpness=inside_sharpness,
                query_mode=query_mode,
                local_query_points=local_query_points,
                friction_coefficient=friction_coefficient,
                body_lowest_k=body_lowest_k,
                body_query_scheme=body_query_scheme,
            )
            if floor_tangential_damping > 0.0:
                tangent_velocity = velocity - torch.sum(velocity * normal) * normal
                damping_fraction = 1.0 - torch.exp(
                    torch.as_tensor(-float(floor_tangential_damping) * dt_sub, dtype=velocity.dtype, device=velocity.device)
                )
                velocity = velocity - contact_gate * damping_fraction * tangent_velocity
            position = predicted_position + contact_gate * (penetration_depth + 1e-4) * normal
            step_gate = contact_gate if step_gate is None else torch.maximum(step_gate, contact_gate)

        positions.append(position)
        contact_gates.append(step_gate)

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
    orientation_sequence: torch.Tensor | None,
    *,
    steps: int,
    dt: float,
    mass: float,
    contact_softness: float,
    smooth_min_temperature: float,
    inside_penalty: float,
    inside_sharpness: float,
    floor_tangential_damping: float = 0.0,
    query_mode: str = "floor_disk",
    local_query_points: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paper III-D-2 impedance contact dynamics rolled out over ``steps`` frames.

    ``stiffness`` and ``damping`` are expected to already be positive (typically
    produced via ``torch.exp(log_K)`` in the fit loop) — no extra SoftPlus is
    applied here. Single contact pair, frictionless.
    """
    if query_mode == "body_surface" and local_query_points is None:
        raise ValueError("query_mode='body_surface' requires local_query_points.")
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

    for step_idx in range(steps - 1):
        b = velocity + dt * gravity

        orientation = None if orientation_sequence is None else orientation_sequence[min(step_idx, orientation_sequence.shape[0] - 1)]
        if query_mode == "body_surface":
            # Object-side queries: surface points on the Gaussian primitives,
            # transformed to world, evaluated against the floor plane z=0.
            world_query_points = oriented_centers(local_query_points, position, orientation)
            phi_agg = _smooth_min_signed(world_query_points[:, 2], smooth_min_temperature)
        else:
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
            gaussian_centers = oriented_centers(local_centers, position, orientation)
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
        # Floor contact should push along the plane normal.  The Gaussian SDF
        # normal is still useful for object/object contact, but for a static
        # floor it leaks vertical impulse into XY motion on round proxies.
        normal = collider.normal.to(dtype=position.dtype, device=position.device)
        Jb = torch.sum(b * normal)
        lambda_t = F.softplus(-K * (dt * Jb + phi_agg) - D * Jb)

        velocity = b + (dt / mass) * lambda_t * normal
        if floor_tangential_damping > 0.0:
            contact_gate = torch.sigmoid(-phi_agg / contact_softness)
            tangent_velocity = velocity - torch.sum(velocity * normal) * normal
            damping_fraction = 1.0 - torch.exp(
                torch.as_tensor(-float(floor_tangential_damping) * dt, dtype=velocity.dtype, device=velocity.device)
            )
            velocity = velocity - contact_gate * damping_fraction * tangent_velocity
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
    mass_parameter: torch.Tensor | None,
    inertia_diag: tuple[float, float, float],
    inertia_parameter: torch.Tensor | None,
    external_forces: torch.Tensor,
    external_torques: torch.Tensor,
    body_b_trajectory: dict[str, torch.Tensor] | None,
    gravity_z: float,
    contact_softness: float,
    smooth_min_temperature: float,
    inside_penalty: float,
    inside_sharpness: float,
    num_contact_patches: int,
    broad_phase_margin: float,
    broad_phase_mode: str,
    friction_coefficient: float,
    friction_parameter: torch.Tensor | None,
    tangential_damping: float,
    contact_model: str,
    dual_cone_directions: int,
    body_query_scheme: str,
    body_query_directions: int,
    floor_query_radius_scale: float,
    floor_query_rings: int,
    floor_query_angles: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    body_a = GaussianCollisionBody(local_centers_a, radii_a)
    body_b = GaussianCollisionBody(local_centers_b, radii_b)
    dynamics = PairwiseGaussianBodyImpedanceDynamics(
        body_a,
        body_b,
        stiffness=stiffness,
        damping=damping,
        friction_coefficient=friction_parameter,
        mass_a=mass_parameter,
        inertia_diag_a=inertia_parameter,
        config=PairwiseImpedanceDynamicsConfig(
            dt=float(dt),
            mass_a=float(mass),
            inertia_diag_a=tuple(float(value) for value in inertia_diag),
            mass_b=float(mass),
            gravity=(0.0, 0.0, float(gravity_z)),
            dynamic_a=True,
            dynamic_b=False,
            kinematic_b=body_b_trajectory is not None,
            contact_softness=float(contact_softness),
            smooth_min_temperature=float(smooth_min_temperature),
            inside_penalty=float(inside_penalty),
            inside_sharpness=float(inside_sharpness),
            num_contact_patches=int(num_contact_patches),
            broad_phase_margin=float(broad_phase_margin),
            broad_phase_mode=str(broad_phase_mode),
            friction_coefficient=float(friction_coefficient),
            tangential_damping=float(tangential_damping),
            contact_model=str(contact_model),
            dual_cone_directions=int(dual_cone_directions),
            body_query_scheme=str(body_query_scheme),
            body_query_directions=int(body_query_directions),
            floor_query_radius_scale=float(floor_query_radius_scale),
            floor_query_rings=int(floor_query_rings),
            floor_query_angles=int(floor_query_angles),
        ),
    )
    quaternion = normalize_quaternion(initial_quaternion.to(dtype=initial_position.dtype, device=initial_position.device))
    static_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=initial_position.dtype, device=initial_position.device)
    zeros = torch.zeros(3, dtype=initial_position.dtype, device=initial_position.device)
    state_a = RigidBodyState(initial_position, quaternion, initial_velocity, initial_angular_velocity)
    state_b = (
        RigidBodyState(
            body_b_trajectory["positions"][0],
            body_b_trajectory["quaternions"][0],
            body_b_trajectory["linear_velocities"][0],
            body_b_trajectory["angular_velocities"][0],
        )
        if body_b_trajectory is not None
        else RigidBodyState(static_position_b, static_quaternion, zeros, zeros)
    )
    positions = [state_a.position]
    quaternions = [state_a.quaternion_wxyz]
    gates = []
    for step_index in range(steps - 1):
        if body_b_trajectory is not None:
            body_index = step_index + 1
            state_b = RigidBodyState(
                body_b_trajectory["positions"][body_index],
                body_b_trajectory["quaternions"][body_index],
                body_b_trajectory["linear_velocities"][body_index],
                body_b_trajectory["angular_velocities"][body_index],
            )
        state_a, state_b, diagnostics = dynamics.step(
            state_a,
            state_b,
            external_force_a=external_forces[step_index],
            external_torque_a=external_torques[step_index],
        )
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
    target_positions: torch.Tensor | None,
    target_quaternions: torch.Tensor | None,
    initial_linear_velocity_hint: torch.Tensor,
    initial_angular_velocity_hint: torch.Tensor,
    times: torch.Tensor,
    local_centers: torch.Tensor,
    radii: torch.Tensor,
    floor_query_offsets_xy: torch.Tensor,
    args: argparse.Namespace,
    *,
    pairwise_body_b: tuple[torch.Tensor, torch.Tensor] | None = None,
    stage1_metadata: dict | None = None,
    pairwise_body_b_metadata: dict | None = None,
    local_query_points: torch.Tensor | None = None,
    gaussian_render_loss=None,
    gaussian_render_indices: list[int] | None = None,
    initial_state_override: dict[str, torch.Tensor] | None = None,
    external_forces: torch.Tensor | None = None,
    external_torques: torch.Tensor | None = None,
    action_metadata: dict | None = None,
    body_b_trajectory: dict[str, torch.Tensor] | None = None,
    body_b_trajectory_metadata: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, dict]:
    device = torch.device(args.device)
    image_only = bool(args.image_only_objective)
    if target_positions is None and not image_only:
        raise ValueError("Pose-supervised fitting requires target_positions; use --image_only_objective without GT.")
    if target_positions is not None:
        target_positions = target_positions.to(device=device)
    if target_quaternions is not None:
        target_quaternions = normalize_quaternion(target_quaternions.to(device=device))
    initial_linear_velocity_hint = initial_linear_velocity_hint.to(device=device)
    initial_angular_velocity_hint = initial_angular_velocity_hint.to(device=device)
    times = times.to(device=device)
    local_centers = local_centers.to(device=device)
    radii = radii.to(device=device)
    dtype = local_centers.dtype
    num_frames = int(times.shape[0])
    if num_frames < 3:
        raise ValueError("Need at least 3 video observation frames for Stage 2 fitting.")
    external_forces = (
        torch.zeros((num_frames - 1, 3), dtype=dtype, device=device)
        if external_forces is None else external_forces.to(device=device, dtype=dtype)
    )
    external_torques = (
        torch.zeros_like(external_forces)
        if external_torques is None else external_torques.to(device=device, dtype=dtype)
    )
    if body_b_trajectory is not None:
        body_b_trajectory = {
            key: value.to(device=device, dtype=dtype)
            for key, value in body_b_trajectory.items()
        }
    floor_query_offsets_xy = floor_query_offsets_xy.to(device=device)
    if local_query_points is not None:
        local_query_points = local_query_points.to(device=device)
    dt = infer_dt(times.detach().cpu())

    dynamics_mode = str(args.dynamics)
    query_mode = str(args.query_mode)
    use_impedance = dynamics_mode == "impedance"
    use_pairwise = dynamics_mode == "pairwise_impedance"
    position_loss_weight = 0.0 if image_only else float(args.position_loss_weight)
    orientation_loss_weight = 0.0 if image_only else float(args.orientation_loss_weight)
    geometry_supervision_weight = 0.0 if image_only else float(args.geometry_loss_weight)
    if image_only and float(args.gaussian_rgb_loss_weight) <= 0.0:
        raise ValueError("--image_only_objective requires --gaussian_rgb_loss_weight > 0.")
    if bool(args.refine_geometry) and not use_pairwise:
        raise ValueError("--refine_geometry currently requires --dynamics pairwise_impedance.")
    base_local_centers = local_centers.detach().clone()
    base_radii = radii.detach().clone()
    center_offsets = torch.nn.Parameter(torch.zeros_like(base_local_centers))
    log_radius_offsets = torch.nn.Parameter(torch.zeros_like(base_radii))
    if target_positions is None and initial_state_override is None:
        raise ValueError(
            "GT-free image-only fitting requires --initial_state_json or --prefit_initial_state."
        )
    initial_position = (
        initial_state_override["position"].to(device=device).detach()
        if initial_state_override is not None
        else target_positions[0].detach()  # type: ignore[index]
    )
    initial_quaternion = (
        initial_state_override["quaternion_wxyz"].to(device=device).detach()
        if initial_state_override is not None
        else target_quaternions[0].detach()
        if target_quaternions is not None
        else torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    )
    finite_velocity = (
        (target_positions[1] - target_positions[0]) / dt
        if target_positions is not None
        else initial_linear_velocity_hint
    )
    initial_velocity_value = (
        initial_state_override["linear_velocity"].to(device=device)
        if initial_state_override is not None
        else initial_linear_velocity_hint
        if str(args.initial_velocity_source) == "trajectory"
        else finite_velocity
    )
    if bool(args.freeze_initial_velocity):
        initial_velocity = initial_velocity_value.detach().clone()
    else:
        initial_velocity = torch.nn.Parameter(initial_velocity_value.detach().clone())
    angular_velocity_value = (
        initial_state_override["angular_velocity"].to(device=device)
        if initial_state_override is not None
        else initial_angular_velocity_hint
    )
    initial_angular_velocity = torch.nn.Parameter(angular_velocity_value.detach().clone())
    gravity_z = torch.nn.Parameter(torch.tensor(-9.81, dtype=torch.float32, device=device))

    raw_mass = None
    raw_inertia = None
    inertia_init = (1.0, 1.0, 1.0)
    if use_pairwise:
        stiffness = torch.nn.Parameter(torch.tensor(float(args.init_stiffness), dtype=torch.float32, device=device))
        damping = torch.nn.Parameter(torch.tensor(float(args.init_damping), dtype=torch.float32, device=device))
        learnable = [stiffness, damping]
        if not bool(args.freeze_initial_velocity):
            learnable.insert(0, initial_velocity)
        if bool(args.fit_initial_angular_velocity):
            learnable.append(initial_angular_velocity)
        inertia_init = parse_vec3(args.pairwise_inertia_diag)
        if any(value <= 0.0 for value in inertia_init):
            raise ValueError("--pairwise_inertia_diag entries must be positive.")
        if float(args.mass) <= 0.0:
            raise ValueError("--mass must be positive.")
        raw_mass = None
        if str(args.pairwise_mass_mode) == "learned":
            raw_mass = torch.nn.Parameter(torch.tensor(
                float(np.log(np.expm1(float(args.mass)))),
                dtype=torch.float32,
                device=device,
            ))
            learnable.append(raw_mass)
        raw_inertia = None
        if str(args.pairwise_inertia_mode) == "learned":
            raw_inertia = torch.nn.Parameter(torch.tensor(
                [float(np.log(np.expm1(value))) for value in inertia_init],
                dtype=torch.float32,
                device=device,
            ))
            learnable.append(raw_inertia)
        pairwise_friction_mode = str(args.pairwise_friction_mode)
        raw_pairwise_friction = None
        if pairwise_friction_mode == "learned":
            friction_init = max(float(args.pairwise_friction_coefficient), 1e-4)
            raw_pairwise_friction = torch.nn.Parameter(torch.tensor(
                float(np.log(np.expm1(friction_init))),
                dtype=torch.float32,
                device=device,
            ))
            learnable.append(raw_pairwise_friction)
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
        learnable = [gravity_z, log_K, log_D]
        if not bool(args.freeze_initial_velocity):
            learnable.insert(0, initial_velocity)
        raw_restitution = None
    else:
        init_e = min(max(float(getattr(args, "init_restitution", 0.5)), 1e-4), 1.0 - 1e-4)
        raw_restitution = torch.nn.Parameter(
            torch.tensor(float(np.log(init_e / (1.0 - init_e))), dtype=torch.float32, device=device)
        )
        log_K = None
        log_D = None
        stiffness = None
        damping = None
        learnable = [raw_restitution] if bool(getattr(args, "freeze_gravity", False)) else [gravity_z, raw_restitution]
        if not bool(args.freeze_initial_velocity):
            learnable.insert(0, initial_velocity)
        pairwise_friction_mode = "off"
        raw_pairwise_friction = None
        raw_mass = None
        raw_inertia = None
        inertia_init = (1.0, 1.0, 1.0)

    if bool(args.refine_geometry):
        learnable.extend((center_offsets, log_radius_offsets))

    friction_mode = str(getattr(args, "floor_friction_mode", "off"))
    raw_friction = None
    if friction_mode != "off":
        if dynamics_mode != "restitution":
            raise ValueError("--floor_friction_mode currently supports --dynamics restitution only.")
        friction_init = max(float(args.floor_friction_init), 1e-4)
        raw_friction_init = float(np.log(np.expm1(friction_init)))
        raw_friction = torch.tensor(raw_friction_init, dtype=torch.float32, device=device)
        if friction_mode == "learned":
            raw_friction = torch.nn.Parameter(raw_friction)
            learnable.append(raw_friction)
    angular_velocity_sequence = (
        angular_velocity_from_quaternions(target_quaternions, dt)
        if friction_mode != "off" and target_quaternions is not None
        else initial_angular_velocity.unsqueeze(0).expand(num_frames - 1, -1)
        if friction_mode != "off"
        else None
    )

    optimizer = torch.optim.Adam(learnable, lr=float(args.lr))

    def effective_geometry(grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(args.refine_geometry):
            return base_local_centers, base_radii
        centers_delta = center_offsets if grad else center_offsets.detach()
        radius_delta = log_radius_offsets if grad else log_radius_offsets.detach()
        # Preserve the declared object-local frame instead of letting geometry
        # absorb a rigid translation that belongs to the pose state.
        centers_delta = centers_delta - torch.mean(centers_delta, dim=0, keepdim=True)
        centers = base_local_centers + centers_delta
        radii_current = base_radii * torch.exp(radius_delta)
        return centers, torch.clamp(radii_current, min=float(args.geometry_min_radius))

    def run(grad: bool):
        geometry_centers, geometry_radii = effective_geometry(grad)
        rollout_quaternions = (
            target_quaternions.detach()
            if target_quaternions is not None and not image_only
            else integrate_quaternion_sequence(
                initial_quaternion,
                initial_angular_velocity if grad else initial_angular_velocity.detach(),
                steps=num_frames,
                dt=dt,
            )
        )
        if use_pairwise:
            if pairwise_body_b is None:
                local_centers_b, radii_b = geometry_centers, geometry_radii
            else:
                local_centers_b, radii_b = pairwise_body_b
                local_centers_b = local_centers_b.to(device=device)
                radii_b = radii_b.to(device=device)
            static_position_b = torch.tensor(
                parse_vec3(args.pairwise_static_position),
                dtype=dtype,
                device=device,
            )
            pairwise_patch_count = (
                int(args.body_lowest_k)
                if query_mode == "body_lowest_k"
                else int(args.pairwise_num_contact_patches)
            )
            pairwise_query_scheme = (
                str(args.body_query_scheme)
                if query_mode in ("body_surface", "body_lowest_k")
                else "floor_disk"
            )
            return simulate_pairwise_impedance(
                initial_position,
                initial_quaternion,
                initial_velocity if grad else initial_velocity.detach(),
                initial_angular_velocity if grad else initial_angular_velocity.detach(),
                stiffness if grad else stiffness.detach(),
                damping if grad else damping.detach(),
                geometry_centers,
                geometry_radii,
                local_centers_b,
                radii_b,
                static_position_b,
                steps=num_frames,
                dt=dt,
                mass=float(args.mass),
                mass_parameter=(
                    raw_mass if grad or raw_mass is None else raw_mass.detach()
                ),
                inertia_diag=inertia_init,
                inertia_parameter=(
                    raw_inertia if grad or raw_inertia is None else raw_inertia.detach()
                ),
                external_forces=external_forces,
                external_torques=external_torques,
                body_b_trajectory=body_b_trajectory,
                gravity_z=float(gravity_z.detach().cpu().item()),
                contact_softness=float(args.contact_softness),
                smooth_min_temperature=float(args.smooth_max_temperature),
                inside_penalty=float(args.inside_penalty),
                inside_sharpness=float(args.inside_sharpness),
                num_contact_patches=pairwise_patch_count,
                broad_phase_margin=float(args.pairwise_broad_phase_margin),
                broad_phase_mode=str(args.pairwise_broad_phase_mode),
                friction_coefficient=(
                    0.0
                    if pairwise_friction_mode == "off"
                    else float(args.pairwise_friction_coefficient)
                ),
                friction_parameter=(
                    raw_pairwise_friction
                    if grad or raw_pairwise_friction is None
                    else raw_pairwise_friction.detach()
                ),
                tangential_damping=float(args.pairwise_tangential_damping),
                contact_model=str(args.pairwise_contact_model),
                dual_cone_directions=int(args.pairwise_dual_cone_directions),
                body_query_scheme=pairwise_query_scheme,
                body_query_directions=int(args.body_query_dirs),
                floor_query_radius_scale=float(args.query_radius_scale),
                floor_query_rings=int(args.query_rings),
                floor_query_angles=int(args.query_angles),
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
                rollout_quaternions,
                steps=num_frames,
                dt=dt,
                mass=float(args.mass),
                contact_softness=float(args.contact_softness),
                smooth_min_temperature=float(args.smooth_max_temperature),
                inside_penalty=float(args.inside_penalty),
                inside_sharpness=float(args.inside_sharpness),
                floor_tangential_damping=float(args.floor_tangential_damping),
                query_mode=query_mode,
                local_query_points=local_query_points,
            )
            return predicted_positions, rollout_quaternions, gates
        restitution = torch.sigmoid(raw_restitution if grad else raw_restitution.detach())
        friction_coefficient = None
        if raw_friction is not None:
            friction_coefficient = F.softplus(raw_friction if grad else raw_friction.detach())
        predicted_positions, gates = simulate(
            initial_position,
            initial_velocity if grad else initial_velocity.detach(),
            gravity_z if grad else gravity_z.detach(),
            restitution,
            local_centers,
            radii,
            floor_query_offsets_xy,
            rollout_quaternions,
            steps=num_frames,
            dt=dt,
            contact_softness=float(args.contact_softness),
            smooth_max_temperature=float(args.smooth_max_temperature),
            inside_penalty=float(args.inside_penalty),
            inside_sharpness=float(args.inside_sharpness),
            floor_tangential_damping=float(args.floor_tangential_damping),
            query_mode=query_mode,
            local_query_points=local_query_points,
            friction_coefficient=friction_coefficient,
            angular_velocity_sequence=angular_velocity_sequence,
            substeps=int(args.substeps),
            body_lowest_k=int(getattr(args, "body_lowest_k", 0)),
            body_query_scheme=str(getattr(args, "body_query_scheme", "axis6")),
        )
        # Restitution mode currently consumes the observed angular-velocity
        # sequence instead of integrating orientation. Returning the observed
        # quaternions still allows RGB/posed-geometry supervision to train the
        # predicted translational contact dynamics end-to-end.
        return predicted_positions, rollout_quaternions, gates

    geometry_stride = max(1, int(args.geometry_loss_stride))
    geometry_indices = torch.arange(
        0, num_frames, geometry_stride, device=device, dtype=torch.long
    )
    render_indices = gaussian_render_indices or []

    def auxiliary_losses(predicted_positions, predicted_quaternions):
        zero = torch.zeros((), dtype=dtype, device=device)
        geometry_loss = zero
        gaussian_rgb_loss = zero
        gaussian_rgb_diagnostics = {}
        supervision_centers, supervision_radii = route_geometry_for_supervision(
            *effective_geometry(torch.is_grad_enabled()),
            str(args.geometry_gradient_route),
        )
        if geometry_supervision_weight > 0.0:
            if predicted_quaternions is None or target_quaternions is None or target_positions is None:
                raise ValueError("--geometry_loss_weight requires pairwise quaternion rollout.")
            pred_geometry = transform_local_points(
                supervision_centers,
                predicted_positions[geometry_indices],
                quaternion_wxyz=predicted_quaternions[geometry_indices],
            )
            target_geometry = transform_local_points(
                supervision_centers,
                target_positions[geometry_indices],
                quaternion_wxyz=target_quaternions[geometry_indices],
            )
            geometry_loss = torch.mean((pred_geometry - target_geometry) ** 2)
        if gaussian_render_loss is not None:
            if predicted_quaternions is None:
                raise ValueError("Gaussian render loss requires pairwise quaternion rollout.")
            gaussian_rgb_loss, gaussian_rgb_diagnostics = gaussian_render_loss(
                predicted_positions[render_indices],
                predicted_quaternions[render_indices],
                geometry_centers=supervision_centers,
                geometry_radii=supervision_radii,
            )
        return geometry_loss, gaussian_rgb_loss, gaussian_rgb_diagnostics

    def geometry_regularization() -> torch.Tensor:
        if not bool(args.refine_geometry):
            return torch.zeros((), dtype=dtype, device=device)
        return (
            float(args.geometry_center_l2_weight) * torch.mean(center_offsets * center_offsets)
            + float(args.geometry_radius_l2_weight) * torch.mean(log_radius_offsets * log_radius_offsets)
        )

    def physical_regularization() -> torch.Tensor:
        regularizer = torch.zeros((), dtype=dtype, device=device)
        if raw_mass is not None:
            regularizer = regularizer + float(args.pairwise_mass_l2_weight) * (
                F.softplus(raw_mass) - float(args.mass)
            ) ** 2
        if raw_inertia is not None:
            inertia_reference = torch.tensor(inertia_init, dtype=dtype, device=device)
            regularizer = regularizer + float(args.pairwise_inertia_l2_weight) * torch.mean(
                (F.softplus(raw_inertia) - inertia_reference) ** 2
            )
        return regularizer

    initial_predicted, initial_quaternions, _ = run(grad=False)
    initial_position_tensor = (
        torch.mean((initial_predicted - target_positions) ** 2)
        if target_positions is not None
        else torch.zeros((), dtype=dtype, device=device)
    )
    initial_orientation_tensor = torch.tensor(0.0, dtype=dtype, device=device)
    if use_pairwise and initial_quaternions is not None and target_quaternions is not None:
        initial_orientation_tensor = quaternion_loss(initial_quaternions, target_quaternions)
    initial_geometry, initial_rgb, _ = auxiliary_losses(initial_predicted, initial_quaternions)
    initial_loss = (
        float(initial_position_tensor.detach().cpu().item()) if target_positions is not None else None
    )
    initial_objective = (
        position_loss_weight * initial_position_tensor
        + orientation_loss_weight * initial_orientation_tensor
        + 1e-4 * torch.mean(initial_velocity * initial_velocity)
        + geometry_supervision_weight * initial_geometry
        + float(args.gaussian_rgb_loss_weight) * initial_rgb
        + geometry_regularization()
        + physical_regularization()
    )
    if use_pairwise and bool(args.fit_initial_angular_velocity):
        initial_objective = initial_objective + float(args.angular_velocity_l2_weight) * torch.mean(
            initial_angular_velocity * initial_angular_velocity
        )
    best_loss = initial_loss
    best_objective = float(initial_objective.detach().cpu().item())
    best_state = [p.detach().clone() for p in learnable]
    best_iteration = 0
    last_optimizer_loss = None
    num_fit_iterations = 0 if bool(args.eval_only) else max(1, int(args.fit_iters))
    for iteration in range(num_fit_iterations):
        optimizer.zero_grad(set_to_none=True)
        predicted, predicted_quaternions, gates = run(grad=True)
        position_loss = (
            torch.mean((predicted - target_positions) ** 2)
            if target_positions is not None
            else torch.zeros((), dtype=dtype, device=device)
        )
        loss = position_loss_weight * position_loss + 1e-4 * torch.mean(initial_velocity * initial_velocity)
        orientation_loss = torch.tensor(0.0, dtype=dtype, device=device)
        if (
            use_pairwise
            and predicted_quaternions is not None
            and target_quaternions is not None
            and orientation_loss_weight > 0.0
        ):
            orientation_loss = quaternion_loss(predicted_quaternions, target_quaternions)
            loss = loss + orientation_loss_weight * orientation_loss
        if use_pairwise and bool(args.fit_initial_angular_velocity):
            loss = loss + float(args.angular_velocity_l2_weight) * torch.mean(
                initial_angular_velocity * initial_angular_velocity
            )
        geometry_loss, gaussian_rgb_loss, _ = auxiliary_losses(predicted, predicted_quaternions)
        loss = (
            loss
            + geometry_supervision_weight * geometry_loss
            + float(args.gaussian_rgb_loss_weight) * gaussian_rgb_loss
            + geometry_regularization()
            + physical_regularization()
        )
        current_position_loss = float(position_loss.detach().cpu().item())
        # The loss above was produced by the parameters as they exist before
        # optimizer.step(), so snapshot that same state. Saving after the step
        # pairs a loss with the wrong parameters by one iteration.
        current_objective = float(loss.detach().cpu().item())
        if current_objective < best_objective:
            best_loss = current_position_loss if target_positions is not None else None
            best_objective = current_objective
            best_state = [p.detach().clone() for p in learnable]
            best_iteration = iteration
        loss.backward()
        optimizer.step()
        if bool(args.refine_geometry):
            with torch.no_grad():
                center_offsets.clamp_(
                    -abs(float(args.geometry_max_center_offset)),
                    abs(float(args.geometry_max_center_offset)),
                )
                log_radius_offsets.clamp_(
                    -abs(float(args.geometry_max_log_radius_offset)),
                    abs(float(args.geometry_max_log_radius_offset)),
                )
        last_optimizer_loss = current_position_loss
        log_every = int(getattr(args, "log_every", 0) or 0)
        if log_every > 0 and (iteration % log_every == 0 or iteration == num_fit_iterations - 1):
            print(f"iter {iteration:4d}  position_loss {current_position_loss:.6f}", flush=True)

    # The final optimizer.step() creates one additional parameter state that was
    # not evaluated at the start of a loop iteration. Include it as a candidate.
    post_step_predicted, post_step_quaternions, _ = run(grad=False)
    post_step_loss = (
        float(torch.mean((post_step_predicted - target_positions) ** 2).detach().cpu().item())
        if target_positions is not None
        else 0.0
    )
    post_step_orientation = torch.tensor(0.0, dtype=dtype, device=device)
    if use_pairwise and post_step_quaternions is not None and target_quaternions is not None:
        post_step_orientation = quaternion_loss(post_step_quaternions, target_quaternions)
    post_geometry, post_rgb, _ = auxiliary_losses(post_step_predicted, post_step_quaternions)
    post_step_objective = position_loss_weight * post_step_loss + orientation_loss_weight * float(
        post_step_orientation.detach().cpu().item()
    )
    post_step_objective += 1e-4 * float(torch.mean(initial_velocity.detach() ** 2).cpu().item())
    post_step_objective += geometry_supervision_weight * float(post_geometry.detach().cpu().item())
    post_step_objective += float(args.gaussian_rgb_loss_weight) * float(post_rgb.detach().cpu().item())
    post_step_objective += float(geometry_regularization().detach().cpu().item())
    if use_pairwise and bool(args.fit_initial_angular_velocity):
        post_step_objective += float(args.angular_velocity_l2_weight) * float(
            torch.mean(initial_angular_velocity.detach() ** 2).cpu().item()
        )
    if post_step_objective < best_objective:
        best_loss = post_step_loss if target_positions is not None else None
        best_objective = post_step_objective
        best_state = [p.detach().clone() for p in learnable]
        best_iteration = num_fit_iterations
    if best_state is not None:
        with torch.no_grad():
            for param, best in zip(learnable, best_state):
                param.copy_(best)
    predicted, predicted_quaternions, gates = run(grad=False)
    final_geometry, final_rgb, final_rgb_diagnostics = auxiliary_losses(predicted, predicted_quaternions)
    refined_centers, refined_radii = effective_geometry(False)
    final_loss = (
        float(torch.mean((predicted - target_positions) ** 2).detach().cpu().item())
        if target_positions is not None
        else None
    )
    contact_indices = torch.nonzero(gates.detach().cpu() > 0.5).flatten()
    first_contact_frame = int(contact_indices[0].item() + 1) if contact_indices.numel() else None
    rmse = None if final_loss is None else float(np.sqrt(final_loss))
    diagnostics = {
        "dt": dt,
        "frames": num_frames,
        "dynamics": dynamics_mode,
        "query_mode": query_mode,
        "num_query_points": (
            int(args.body_lowest_k)
            if use_pairwise and query_mode == "body_lowest_k"
            else int(args.pairwise_num_contact_patches)
            if use_pairwise
            else _resolve_num_query_points(
                query_mode,
                str(getattr(args, "body_query_scheme", "axis6")),
                int(getattr(args, "body_lowest_k", 0)),
                local_query_points,
                int(local_centers.shape[0]),
                int(floor_query_offsets_xy.shape[0]),
            )
        ),
        "body_lowest_k": (
            int(getattr(args, "body_lowest_k", 0)) if query_mode == "body_lowest_k" else None
        ),
        "body_query_scheme": (
            str(getattr(args, "body_query_scheme", "axis6"))
            if query_mode in ("body_surface", "body_lowest_k")
            else None
        ),
        "num_primitives_a": int(local_centers.shape[0]),
        "radius_scale": float(args.radius_scale),
        "object_id": args.object_id,
        "foreground_threshold": args.foreground_threshold,
        "opacity_threshold": args.opacity_threshold,
        "max_primitives": args.max_primitives,
        "initial_position_loss": initial_loss,
        "final_position_loss": final_loss,
        "position_rmse": rmse,
        "best_iteration": best_iteration,
        "best_position_loss": best_loss,
        "best_objective": best_objective,
        "eval_only": bool(args.eval_only),
        "geometry_loss_weight": geometry_supervision_weight,
        "geometry_loss": float(final_geometry.detach().cpu().item()),
        "geometry_loss_stride": geometry_stride,
        "geometry_refinement_enabled": bool(args.refine_geometry),
        "geometry_gradient_route": str(args.geometry_gradient_route),
        "geometry_direct_render_gradient": str(args.geometry_gradient_route) == "collision_and_render",
        "geometry_center_l2_weight": float(args.geometry_center_l2_weight),
        "geometry_radius_l2_weight": float(args.geometry_radius_l2_weight),
        "geometry_center_offset_rms": float(
            torch.sqrt(torch.mean(center_offsets.detach() ** 2)).cpu().item()
        ),
        "geometry_log_radius_offset_rms": float(
            torch.sqrt(torch.mean(log_radius_offsets.detach() ** 2)).cpu().item()
        ),
        "refined_local_centers": refined_centers.detach().cpu().tolist(),
        "refined_radii": refined_radii.detach().cpu().tolist(),
        "gaussian_rgb_loss_weight": float(args.gaussian_rgb_loss_weight),
        "gaussian_rgb_loss": float(final_rgb.detach().cpu().item()),
        "gaussian_rgb_dir": (
            None
            if float(args.gaussian_rgb_loss_weight) <= 0.0
            else str((args.gaussian_rgb_dir or (args.episode_root / "rgb")).resolve())
        ),
        "gaussian_mask_dir": (
            None if args.gaussian_mask_dir is None else str(args.gaussian_mask_dir.resolve())
        ),
        "gaussian_views_manifest": (
            None if args.gaussian_views_manifest is None else str(args.gaussian_views_manifest.resolve())
        ),
        "gaussian_render_stride": int(args.gaussian_render_stride),
        "gaussian_render_max_frames": int(args.gaussian_render_max_frames),
        "gaussian_render_width": int(args.gaussian_render_width),
        "gaussian_render_height": int(args.gaussian_render_height),
        "gaussian_render_loss_type": str(args.gaussian_render_loss),
        "gaussian_render_ssim_weight": float(args.gaussian_render_ssim_weight),
        "gaussian_render_loftr_weight": float(args.gaussian_render_loftr_weight),
        "loftr_pretrained": str(args.loftr_pretrained),
        "gaussian_camera": {
            "distance": float(args.gaussian_cam_distance),
            "height": float(args.gaussian_cam_height),
            "fovy_deg": float(args.gaussian_cam_fovy_deg),
        },
        "image_only_objective": image_only,
        "ground_truth_trajectory_used_for_training": target_positions is not None,
        "effective_position_loss_weight": position_loss_weight,
        "effective_orientation_loss_weight": orientation_loss_weight,
        "effective_geometry_supervision_weight": geometry_supervision_weight,
        **final_rgb_diagnostics,
        "last_optimizer_position_loss": last_optimizer_loss,
        "post_step_position_loss": post_step_loss,
        "learned_initial_velocity": initial_velocity.detach().cpu().tolist(),
        "initial_velocity_source": str(args.initial_velocity_source),
        "initial_state_source": (
            "prefit_or_json" if initial_state_override is not None else "trajectory_state"
        ),
        "rollout_initial_position": initial_position.detach().cpu().tolist(),
        "rollout_initial_quaternion_wxyz": initial_quaternion.detach().cpu().tolist(),
        "freeze_initial_velocity": bool(args.freeze_initial_velocity),
        "learned_gravity_z": float(gravity_z.detach().cpu().item()),
        "floor_tangential_damping": float(args.floor_tangential_damping),
        "first_contact_frame": first_contact_frame,
        "max_contact_gate": float(gates.detach().cpu().max().item()) if gates.numel() else 0.0,
        "stage1_coordinate": stage1_metadata or {},
        "action_input": action_metadata or {"source": "zero_wrench"},
        "kinematic_body_b": body_b_trajectory_metadata or {"source": "static_body_b"},
    }
    if pairwise_body_b_metadata is not None:
        diagnostics["pairwise_body_b_coordinate"] = pairwise_body_b_metadata
    if use_pairwise:
        num_primitives_b = int(pairwise_body_b[0].shape[0]) if pairwise_body_b is not None else int(local_centers.shape[0])
        diagnostics["num_primitives_b"] = num_primitives_b
        diagnostics["learned_stiffness"] = float(F.softplus(stiffness.detach()).cpu().item())
        diagnostics["learned_damping"] = float(F.softplus(damping.detach()).cpu().item())
        diagnostics["pairwise_static_position"] = list(parse_vec3(args.pairwise_static_position))
        diagnostics["pairwise_num_contact_patches"] = (
            int(args.body_lowest_k)
            if query_mode == "body_lowest_k"
            else int(args.pairwise_num_contact_patches)
        )
        diagnostics["pairwise_body_query_scheme"] = (
            str(args.body_query_scheme)
            if query_mode in ("body_surface", "body_lowest_k")
            else "floor_disk"
        )
        diagnostics["pairwise_body_query_directions"] = (
            1 + int(args.query_rings) * int(args.query_angles)
            if diagnostics["pairwise_body_query_scheme"] == "floor_disk"
            else 1
            if diagnostics["pairwise_body_query_scheme"] == "analytic"
            else 6
            if diagnostics["pairwise_body_query_scheme"] == "axis6"
            else int(args.body_query_dirs)
        )
        if diagnostics["pairwise_body_query_scheme"] == "floor_disk":
            diagnostics["raw_query_candidates_a"] = 0
            diagnostics["raw_query_candidates_b"] = int(diagnostics["pairwise_body_query_directions"])
        else:
            diagnostics["raw_query_candidates_a"] = (
                int(local_centers.shape[0]) * int(diagnostics["pairwise_body_query_directions"])
            )
            diagnostics["raw_query_candidates_b"] = (
                num_primitives_b * int(diagnostics["pairwise_body_query_directions"])
            )
        diagnostics["raw_query_candidates_total"] = (
            diagnostics["raw_query_candidates_a"] + diagnostics["raw_query_candidates_b"]
        )
        diagnostics["pairwise_broad_phase_mode"] = str(args.pairwise_broad_phase_mode)
        diagnostics["pairwise_contact_model"] = str(args.pairwise_contact_model)
        diagnostics["pairwise_dual_cone_directions"] = int(args.pairwise_dual_cone_directions)
        diagnostics["pairwise_friction_mode"] = pairwise_friction_mode
        diagnostics["pairwise_friction_coefficient"] = (
            float(F.softplus(raw_pairwise_friction.detach()).cpu().item())
            if raw_pairwise_friction is not None
            else 0.0
            if pairwise_friction_mode == "off"
            else float(args.pairwise_friction_coefficient)
        )
        diagnostics["pairwise_tangential_damping"] = float(args.pairwise_tangential_damping)
        diagnostics["pairwise_mass_mode"] = str(args.pairwise_mass_mode)
        diagnostics["learned_mass"] = (
            float(F.softplus(raw_mass.detach()).cpu().item())
            if raw_mass is not None else float(args.mass)
        )
        diagnostics["pairwise_inertia_mode"] = str(args.pairwise_inertia_mode)
        diagnostics["learned_inertia_diag"] = (
            F.softplus(raw_inertia.detach()).cpu().tolist()
            if raw_inertia is not None else list(inertia_init)
        )
        diagnostics["pairwise_mass_l2_weight"] = float(args.pairwise_mass_l2_weight)
        diagnostics["pairwise_inertia_l2_weight"] = float(args.pairwise_inertia_l2_weight)
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
        diagnostics["floor_friction_mode"] = friction_mode
        diagnostics["substeps"] = int(args.substeps)
        if raw_friction is not None:
            diagnostics["learned_friction"] = float(F.softplus(raw_friction.detach()).cpu().item())
    predicted_quaternions_cpu = predicted_quaternions.detach().cpu() if predicted_quaternions is not None else None
    return predicted.detach().cpu(), predicted_quaternions_cpu, gates.detach().cpu(), diagnostics


def draw_follow_view(
    target_positions: np.ndarray,
    predicted_positions: np.ndarray,
    contact_gates: np.ndarray,
    output_gif: Path,
    *,
    fps: int,
    object_shape: str = "sphere",
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

    def draw_cube(draw: ImageDraw.ImageDraw, center_x: int, center_y: int, size: int, *, fill_front, fill_top, fill_side, outline):
        half = size // 2
        depth = max(8, size // 3)
        front = [
            (center_x - half, center_y - half),
            (center_x + half, center_y - half),
            (center_x + half, center_y + half),
            (center_x - half, center_y + half),
        ]
        back = [(x + depth, y - depth) for x, y in front]
        top = [front[0], front[1], back[1], back[0]]
        side = [front[1], front[2], back[2], back[1]]
        draw.polygon(top, fill=fill_top, outline=outline)
        draw.polygon(side, fill=fill_side, outline=outline)
        draw.polygon(front, fill=fill_front, outline=outline)
        draw.line(front + [front[0]], fill=outline, width=3)
        draw.line([front[0], back[0], back[1], front[1]], fill=outline, width=2)

    def draw_can(draw: ImageDraw.ImageDraw, center_x: int, center_y: int, width_px: int, height_px: int, *, body, stripe, cap, outline):
        left = center_x - width_px // 2
        right = center_x + width_px // 2
        top = center_y - height_px // 2
        bottom = center_y + height_px // 2
        cap_h = max(5, height_px // 8)
        draw.rectangle((left, top, right, bottom), fill=body, outline=outline)
        draw.ellipse((left, top - cap_h // 2, right, top + cap_h), fill=cap, outline=outline)
        draw.ellipse((left, bottom - cap_h, right, bottom + cap_h // 2), fill=(165, 165, 158), outline=outline)
        draw.rectangle((center_x - width_px // 5, top + cap_h, center_x + width_px // 5, bottom - cap_h), fill=stripe)
        draw.line((left + 4, center_y, right - 4, center_y), fill=(245, 245, 238), width=2)
        draw.rectangle((center_x - 5, top - 1, center_x + 8, top + 3), fill=(105, 105, 100))

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
        if object_shape == "box":
            draw_cube(
                draw,
                center_x - 56,
                target_y,
                44,
                fill_front=(78, 150, 219),
                fill_top=(126, 184, 232),
                fill_side=(45, 108, 172),
                outline=(25, 82, 130),
            )
            draw_cube(
                draw,
                center_x + 48,
                pred_y,
                30,
                fill_front=gate_color,
                fill_top=(245, 154, 103) if gate >= 0.5 else (88, 164, 226),
                fill_side=(172, 52, 44) if gate >= 0.5 else (30, 95, 160),
                outline=(30, 30, 30),
            )
            draw.line((center_x - 90, target_y + 26, center_x - 24, target_y + 26), fill=(25, 82, 130), width=1)
        elif object_shape == "cylinder":
            draw_can(
                draw,
                center_x - 56,
                target_y,
                42,
                70,
                body=(78, 150, 219),
                stripe=(232, 240, 250),
                cap=(190, 205, 214),
                outline=(25, 82, 130),
            )
            draw_can(
                draw,
                center_x + 48,
                pred_y,
                30,
                48,
                body=gate_color,
                stripe=(255, 250, 240),
                cap=(210, 210, 202),
                outline=(30, 30, 30),
            )
            draw.line((center_x - 84, target_y + 38, center_x - 28, target_y + 38), fill=(25, 82, 130), width=1)
        else:
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
    episode_manifest = read_json(episode_root / "episode_manifest.json")
    object_shape = str(episode_manifest.get("physics_shape", "sphere"))

    observations = load_video_observations(
        episode_root,
        max_frames=int(args.max_frames),
        rgb_dir=args.gaussian_rgb_dir,
        mask_dir=args.gaussian_mask_dir,
        views_manifest=args.gaussian_views_manifest,
        camera_defaults={
            "distance": float(args.gaussian_cam_distance),
            "height": float(args.gaussian_cam_height),
            "fovy_deg": float(args.gaussian_cam_fovy_deg),
        },
    )
    evaluation_path = (
        args.evaluation_trajectory.resolve()
        if args.evaluation_trajectory is not None
        else episode_root / "state" / "trajectory.json"
    )
    evaluation = load_optional_evaluation_trajectory(
        evaluation_path,
        max_frames=int(args.max_frames),
    )
    if evaluation is None and not bool(args.image_only_objective):
        raise ValueError(
            "Pose-supervised fitting requires --evaluation_trajectory. "
            "Use --image_only_objective with --initial_state_json or --prefit_initial_state for GT-free fitting."
        )
    if evaluation is not None and observations.num_frames != evaluation.num_frames:
        raise ValueError(
            f"RGB/evaluation frame count mismatch: {observations.num_frames} vs {evaluation.num_frames}."
        )
    states = (
        list(evaluation.states)
        if evaluation is not None
        else [
            {"frame_index": frame_index, "time": float(observations.times[index].item())}
            for index, frame_index in enumerate(observations.frame_indices)
        ]
    )
    target_positions_cpu = None if evaluation is None else evaluation.positions
    target_quaternions_cpu = None if evaluation is None else evaluation.quaternions_wxyz
    times_cpu = observations.times if bool(args.image_only_objective) else evaluation.times
    default_actions_path = episode_root / "actions" / "trajectory.json"
    actions_path = (
        args.actions_json.resolve()
        if args.actions_json is not None
        else default_actions_path.resolve()
        if default_actions_path.exists()
        else None
    )
    external_forces_cpu, external_torques_cpu, action_metadata = load_action_wrenches(
        actions_path,
        num_steps=observations.num_frames - 1,
        force_scale=float(args.action_force_scale),
        torque_scale=float(args.action_torque_scale),
    )
    body_b_trajectory = None
    body_b_trajectory_metadata = None
    if args.pairwise_body_b_trajectory_json is not None:
        if str(args.dynamics) != "pairwise_impedance":
            raise ValueError("--pairwise_body_b_trajectory_json requires --dynamics pairwise_impedance.")
        body_b_trajectory, body_b_trajectory_metadata = load_kinematic_body_trajectory(
            args.pairwise_body_b_trajectory_json.resolve(),
            num_frames=observations.num_frames,
            dt=infer_dt(times_cpu),
        )
    if str(args.initial_velocity_source) == "trajectory" and evaluation is not None:
        initial_linear_velocity_hint_cpu = load_initial_linear_velocity(states)
    else:
        initial_linear_velocity_hint_cpu = torch.zeros(3, dtype=torch.float32)
    initial_angular_velocity_hint_cpu = (
        load_initial_angular_velocity(states)
        if evaluation is not None and not bool(args.image_only_objective)
        else torch.zeros(3, dtype=torch.float32)
    )
    stage1_world_translation = parse_optional_vec3(args.stage1_world_translation, label="--stage1_world_translation")
    stage1_world_rotation = parse_optional_matrix3(args.stage1_world_rotation, label="--stage1_world_rotation")
    stage1_coordinate_contract = load_stage1_coordinate_contract(
        episode_root,
        translation_override=stage1_world_translation,
        rotation_override=stage1_world_rotation,
    )
    collision_bbox = None if bool(args.disable_collision_bbox_calibration) else load_manifest_collision_bbox(episode_root)
    if collision_bbox is not None:
        bbox_extent = np.asarray(collision_bbox["bbox_max"], dtype=np.float32) - np.asarray(collision_bbox["bbox_min"], dtype=np.float32)
        margin = float(args.collision_bbox_margin)
        margin_ratio = float(args.collision_bbox_margin_ratio)
        margin_xyz = (np.full(3, margin, dtype=np.float32) + bbox_extent * margin_ratio).astype(np.float32)
        margin_z = margin if args.collision_bbox_margin_z is None else float(args.collision_bbox_margin_z)
        if args.collision_bbox_margin_z_ratio is not None:
            margin_z = float(args.collision_bbox_margin_z_ratio) * float(bbox_extent[2])
        margin_xyz[2] = float(margin_z)
        collision_bbox["margin_xyz"] = margin_xyz.tolist()
    local_centers, radii, stage1_metadata = load_stage1_body_arrays(
        stage1_ply,
        args,
        object_id=args.object_id,
        foreground_threshold=args.foreground_threshold,
        opacity_threshold=args.opacity_threshold,
        max_primitives=args.max_primitives,
        coordinate_contract=stage1_coordinate_contract,
        collision_bbox=collision_bbox,
        device=torch.device(args.device),
    )
    pairwise_body_b = None
    pairwise_body_b_metadata = None
    if str(args.dynamics) == "pairwise_impedance" and args.pairwise_body_b_ply is not None:
        pairwise_body_b_translation = parse_optional_vec3(
            args.pairwise_body_b_world_translation,
            label="--pairwise_body_b_world_translation",
        )
        pairwise_body_b_rotation = parse_optional_matrix3(
            args.pairwise_body_b_world_rotation,
            label="--pairwise_body_b_world_rotation",
        )
        if pairwise_body_b_translation is None and pairwise_body_b_rotation is not None:
            raise ValueError("--pairwise_body_b_world_rotation requires --pairwise_body_b_world_translation.")
        pairwise_body_b_contract = {
            "coordinate_frame": "world" if pairwise_body_b_translation is not None else "object_local",
            "world_translation": pairwise_body_b_translation,
            "world_rotation": pairwise_body_b_rotation,
            "source": "cli_override" if pairwise_body_b_translation is not None else "implicit_pairwise_object_local",
        }
        local_centers_b, radii_b, pairwise_body_b_metadata = load_stage1_body_arrays(
            args.pairwise_body_b_ply.resolve(),
            args,
            object_id=args.pairwise_body_b_object_id,
            foreground_threshold=args.pairwise_body_b_foreground_threshold,
            opacity_threshold=args.pairwise_body_b_opacity_threshold,
            max_primitives=args.pairwise_body_b_max_primitives,
            coordinate_contract=pairwise_body_b_contract,
            collision_bbox=None,
            device=torch.device(args.device),
        )
        pairwise_body_b = (local_centers_b, radii_b)
    query_radius = estimate_radius(local_centers.detach().cpu(), radii.detach().cpu()) * float(args.query_radius_scale)
    floor_query_offsets_xy = make_floor_disk_query_points(
        query_radius,
        num_rings=int(args.query_rings),
        num_angles=int(args.query_angles),
        dtype=torch.float32,
        device=torch.device(args.device),
    )
    local_query_points = None
    if str(args.query_mode) in ("body_surface", "body_lowest_k"):
        scheme = str(args.body_query_scheme)
        if scheme == "analytic":
            # Query points depend on world orientation, so they are built per step
            # inside _floor_contact_response rather than precomputed in local frame.
            local_query_points = None
        else:
            # axis6 -> the 6 axis directions; fibonacci -> --body_query_dirs directions.
            dirs = 6 if scheme == "axis6" else max(int(args.body_query_dirs), 4)
            local_query_points = make_gaussian_proxy_query_points(
                local_centers,
                radii,
                directions_per_gaussian=dirs,
            )
    gaussian_render_loss = None
    gaussian_render_indices: list[int] = []
    if float(args.gaussian_rgb_loss_weight) > 0.0 or bool(args.prefit_initial_state):
        if torch.device(args.device).type != "cuda":
            raise ValueError("--gaussian_rgb_loss_weight requires --device cuda.")
        from stage2.differentiable_gaussian_render_loss import (
            GaussianRenderLossConfig,
            MultiViewStage2GaussianRenderLoss,
            Stage2GaussianRenderLoss,
        )

        render_stride = max(1, int(args.gaussian_render_stride))
        gaussian_render_indices = list(range(0, observations.num_frames, render_stride))
        if int(args.gaussian_render_max_frames) > 0 and len(gaussian_render_indices) > int(args.gaussian_render_max_frames):
            selected = np.linspace(
                0,
                len(gaussian_render_indices) - 1,
                int(args.gaussian_render_max_frames),
                dtype=np.int64,
            )
            gaussian_render_indices = [gaussian_render_indices[int(idx)] for idx in selected]
        if args.gaussian_views_manifest is None:
            view_specs = [{
                "rgb_dir": str((args.gaussian_rgb_dir or (episode_root / "rgb")).resolve()),
                "mask_dir": None if args.gaussian_mask_dir is None else str(args.gaussian_mask_dir.resolve()),
            }]
            view_base = episode_root
        else:
            view_manifest_path = args.gaussian_views_manifest.resolve()
            view_payload = read_json(view_manifest_path)
            view_specs = view_payload.get("views", [])
            view_base = view_manifest_path.parent
            if not view_specs:
                raise ValueError("--gaussian_views_manifest must contain a non-empty 'views' list.")
        render_views = []
        for spec in view_specs:
            rgb_path = Path(spec["rgb_dir"])
            rgb_path = (view_base / rgb_path).resolve() if not rgb_path.is_absolute() else rgb_path.resolve()
            mask_value = spec.get("mask_dir")
            mask_path = None if mask_value is None else Path(mask_value)
            if mask_path is not None:
                mask_path = (view_base / mask_path).resolve() if not mask_path.is_absolute() else mask_path.resolve()
            render_views.append(Stage2GaussianRenderLoss(
                stage1_ply=stage1_ply,
                gt_rgb_dir=rgb_path,
                frame_indices=[int(observations.frame_indices[idx]) for idx in gaussian_render_indices],
                gaussian_indices=stage1_metadata.get("source_indices"),
                gt_mask_dir=mask_path,
                config=GaussianRenderLossConfig(
                    image_width=max(16, int(spec.get("image_width", args.gaussian_render_width))),
                    image_height=max(16, int(spec.get("image_height", args.gaussian_render_height))),
                    cam_distance=float(spec.get("cam_distance", args.gaussian_cam_distance)),
                    cam_height=float(spec.get("cam_height", args.gaussian_cam_height)),
                    cam_fovy_deg=float(spec.get("cam_fovy_deg", args.gaussian_cam_fovy_deg)),
                    white_background=bool(spec.get("white_background", args.gaussian_render_white_background)),
                    scale_multiplier=float(args.gaussian_render_scale_multiplier),
                    collision_radius_to_gaussian_scale=(
                        0.5 if args.gaussian_radius_convention == "paper_r2s" else 1.0
                    ),
                    loss=str(args.gaussian_render_loss),
                    ssim_weight=float(args.gaussian_render_ssim_weight),
                    loftr_weight=float(args.gaussian_render_loftr_weight),
                    loftr_pretrained=str(args.loftr_pretrained),
                    loftr_confidence_threshold=float(args.loftr_confidence_threshold),
                    loftr_max_matches=int(args.loftr_max_matches),
                    loftr_min_matches=int(args.loftr_min_matches),
                    loftr_patch_radius=int(args.loftr_patch_radius),
                ),
                dtype=torch.float32,
                device=torch.device(args.device),
            ))
        gaussian_render_loss = (
            render_views[0] if len(render_views) == 1 else MultiViewStage2GaussianRenderLoss(render_views)
        )
    initial_state_override = None
    initial_state_prefit_report = None
    if args.initial_state_json is not None and bool(args.prefit_initial_state):
        raise ValueError("Use either --initial_state_json or --prefit_initial_state, not both.")
    if args.initial_state_json is not None:
        state_payload = read_json(args.initial_state_json.resolve())
        initial_state_override = {
            key: torch.tensor(state_payload[key], dtype=torch.float32)
            for key in ("position", "quaternion_wxyz", "linear_velocity", "angular_velocity")
        }
        args.freeze_initial_velocity = True
        args.fit_initial_angular_velocity = False
    elif bool(args.prefit_initial_state):
        from stage2.initial_state_estimation import estimate_initial_state_from_images

        prefit_frames = max(2, min(int(args.prefit_velocity_frames), observations.num_frames))
        prefit_indices = list(range(prefit_frames))
        prefit_views = []
        for spec in view_specs:
            rgb_path = Path(spec["rgb_dir"])
            rgb_path = (view_base / rgb_path).resolve() if not rgb_path.is_absolute() else rgb_path.resolve()
            mask_value = spec.get("mask_dir")
            mask_path = None if mask_value is None else Path(mask_value)
            if mask_path is not None:
                mask_path = (view_base / mask_path).resolve() if not mask_path.is_absolute() else mask_path.resolve()
            prefit_views.append(Stage2GaussianRenderLoss(
                stage1_ply=stage1_ply,
                gt_rgb_dir=rgb_path,
                frame_indices=[int(observations.frame_indices[idx]) for idx in prefit_indices],
                gaussian_indices=stage1_metadata.get("source_indices"),
                gt_mask_dir=mask_path,
                config=GaussianRenderLossConfig(
                    image_width=max(16, int(spec.get("image_width", args.gaussian_render_width))),
                    image_height=max(16, int(spec.get("image_height", args.gaussian_render_height))),
                    cam_distance=float(spec.get("cam_distance", args.gaussian_cam_distance)),
                    cam_height=float(spec.get("cam_height", args.gaussian_cam_height)),
                    cam_fovy_deg=float(spec.get("cam_fovy_deg", args.gaussian_cam_fovy_deg)),
                    white_background=bool(spec.get("white_background", args.gaussian_render_white_background)),
                    scale_multiplier=float(args.gaussian_render_scale_multiplier),
                    collision_radius_to_gaussian_scale=(
                        0.5 if args.gaussian_radius_convention == "paper_r2s" else 1.0
                    ),
                    loss=str(args.gaussian_render_loss),
                    ssim_weight=float(args.gaussian_render_ssim_weight),
                    loftr_weight=float(args.gaussian_render_loftr_weight),
                    loftr_pretrained=str(args.loftr_pretrained),
                    loftr_confidence_threshold=float(args.loftr_confidence_threshold),
                    loftr_max_matches=int(args.loftr_max_matches),
                    loftr_min_matches=int(args.loftr_min_matches),
                    loftr_patch_radius=int(args.loftr_patch_radius),
                ),
                dtype=torch.float32,
                device=torch.device(args.device),
            ))
        prefit_loss = prefit_views[0] if len(prefit_views) == 1 else MultiViewStage2GaussianRenderLoss(prefit_views)
        initial_state_override, initial_state_prefit_report = estimate_initial_state_from_images(
            prefit_loss,
            position_init=torch.tensor(parse_vec3(args.prefit_position_init), dtype=torch.float32, device=args.device),
            quaternion_init=torch.tensor(
                parse_float_sequence(args.prefit_quaternion_init, expected=4, label="--prefit_quaternion_init"),
                dtype=torch.float32,
                device=args.device,
            ),
            times=times_cpu[:prefit_frames].to(device=args.device),
            pose_iters=int(args.prefit_pose_iters),
            velocity_iters=int(args.prefit_velocity_iters),
            lr=float(args.prefit_lr),
            geometry_centers=local_centers,
            geometry_radii=radii,
            velocity_l2=float(args.prefit_velocity_l2),
        )
        args.freeze_initial_velocity = True
        args.fit_initial_angular_velocity = False
    if float(args.gaussian_rgb_loss_weight) <= 0.0:
        gaussian_render_loss = None
    predicted_positions, predicted_quaternions, contact_gates, diagnostics = fit_stage2(
        None if bool(args.image_only_objective) else target_positions_cpu,
        None if bool(args.image_only_objective) else target_quaternions_cpu,
        initial_linear_velocity_hint_cpu,
        initial_angular_velocity_hint_cpu,
        times_cpu,
        local_centers,
        radii,
        floor_query_offsets_xy,
        args,
        pairwise_body_b=pairwise_body_b,
        stage1_metadata=stage1_metadata,
        pairwise_body_b_metadata=pairwise_body_b_metadata,
        local_query_points=local_query_points,
        gaussian_render_loss=gaussian_render_loss,
        gaussian_render_indices=gaussian_render_indices,
        initial_state_override=initial_state_override,
        external_forces=external_forces_cpu,
        external_torques=external_torques_cpu,
        action_metadata=action_metadata,
        body_b_trajectory=body_b_trajectory,
        body_b_trajectory_metadata=body_b_trajectory_metadata,
    )
    diagnostics["training_observations"] = observation_summary(observations)
    diagnostics["evaluation_trajectory"] = {
        "path": None if evaluation is None else str(evaluation.path),
        "num_frames": 0 if evaluation is None else evaluation.num_frames,
        "used_by_legacy_training_objective": not bool(args.image_only_objective),
        "separated_from_video_observations": True,
    }
    if evaluation is not None:
        evaluation_mse = float(torch.mean((predicted_positions - evaluation.positions) ** 2).item())
        diagnostics["evaluation_position_rmse"] = float(np.sqrt(evaluation_mse))

    trajectory_states = []
    for idx in range(predicted_positions.shape[0]):
        state_payload = {
            "frame_index": int(observations.frame_indices[idx]),
            "time": float(times_cpu[idx].item()),
            "predicted_position": predicted_positions[idx].tolist(),
            "contact_gate": float(contact_gates[idx - 1].item()) if idx > 0 and idx - 1 < contact_gates.numel() else 0.0,
        }
        if evaluation is not None:
            state_payload["target_position"] = evaluation.positions[idx].tolist()
            state_payload["target_quaternion_wxyz"] = evaluation.quaternions_wxyz[idx].tolist()
        if predicted_quaternions is not None:
            # Restitution dynamics does not predict orientation; it reads omega
            # from the GT quaternion sequence. Persist the GT orientation so the
            # trajectory renderer tumbles the object consistently across query
            # modes (orientation is the controlled variable — only the predicted
            # position differs between modes).
            state_payload["predicted_quaternion_wxyz"] = predicted_quaternions[idx].tolist()
        if idx < external_forces_cpu.shape[0]:
            state_payload["action_force_world"] = external_forces_cpu[idx].tolist()
            state_payload["action_torque_world"] = external_torques_cpu[idx].tolist()
        if body_b_trajectory is not None:
            state_payload["body_b_position"] = body_b_trajectory["positions"][idx].tolist()
            state_payload["body_b_quaternion_wxyz"] = body_b_trajectory["quaternions"][idx].tolist()
        trajectory_states.append(state_payload)

    trajectory = {
        "source_episode_root": str(episode_root),
        "stage1_ply": str(stage1_ply),
        "states": trajectory_states,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "stage2_fit_follow_view.gif"
    diagnostics["output_dir"] = str(output_dir)
    diagnostics["diagnostic_gif"] = str(gif_path) if evaluation is not None else None
    diagnostics["experiment_bundle"] = str(output_dir / "experiment_bundle.json")
    initial_state_path = output_dir / "initial_state_estimate.json"
    if initial_state_override is not None:
        diagnostics["initial_state_source"] = (
            "image_prefit" if initial_state_prefit_report is not None else "initial_state_json"
        )
        initial_state_payload = {
            key: value.detach().cpu().tolist() for key, value in initial_state_override.items()
        }
        initial_state_payload["source"] = diagnostics["initial_state_source"]
        initial_state_payload["prefit"] = initial_state_prefit_report
        write_json(initial_state_path, initial_state_payload)
        diagnostics["initial_state_estimate"] = str(initial_state_path)
    refined_geometry_path = output_dir / "refined_geometry.json"
    write_json(
        refined_geometry_path,
        {
            "enabled": bool(args.refine_geometry),
            "source_indices": stage1_metadata.get("source_indices"),
            "local_centers": diagnostics["refined_local_centers"],
            "radii": diagnostics["refined_radii"],
        },
    )
    diagnostics["refined_geometry"] = str(refined_geometry_path)
    write_json(output_dir / "fit_summary.json", diagnostics)
    write_json(output_dir / "predicted_trajectory.json", trajectory)
    if evaluation is not None:
        draw_follow_view(
            evaluation.positions.numpy(),
            predicted_positions.numpy(),
            contact_gates.numpy(),
            gif_path,
            fps=int(args.gif_fps),
            object_shape=object_shape,
        )
    bundle_path = write_experiment_bundle(
        output_dir=output_dir,
        repo_root=REPO_ROOT,
        args=args,
        episode_manifest_path=episode_root / "episode_manifest.json",
        input_paths={
            "stage1_ply": stage1_ply,
            "source_scene_manifest": args.source_scene_manifest,
            "evaluation_trajectory": None if evaluation is None else evaluation.path,
            "pairwise_body_b_ply": args.pairwise_body_b_ply,
            "initial_state_json": args.initial_state_json,
            "actions_json": actions_path,
            "pairwise_body_b_trajectory_json": args.pairwise_body_b_trajectory_json,
        },
        result_paths={
            "fit_summary": output_dir / "fit_summary.json",
            "predicted_trajectory": output_dir / "predicted_trajectory.json",
            **({"diagnostic_gif": gif_path} if evaluation is not None else {}),
            "refined_geometry": refined_geometry_path,
            **(
                {"initial_state_estimate": initial_state_path}
                if initial_state_override is not None
                else {}
            ),
        },
    )
    diagnostics["experiment_bundle"] = str(bundle_path)
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
