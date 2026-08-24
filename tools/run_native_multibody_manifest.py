"""Run an N-body Gaussian-union rollout directly from a scene manifest."""

from __future__ import annotations

import argparse
import math
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2.differentiable_collision_detection import (
    GaussianCollisionBody,
    PlaneCollider,
    load_gaussian_collision_body_from_ply,
)
from stage2.differentiable_complementarity_free_contact_dynamics import (
    GaussianPlaneContactPair,
    MultiBodyGaussianImpedanceDynamics,
    MultiBodyImpedanceDynamicsConfig,
    RigidBodyState,
)
from stage2.scene_manifest import load_scene_manifest, manifest_summary
from stage2.differentiable_gaussian_render_loss import (
    GaussianRenderLossConfig,
    MultiBodyStage2GaussianRenderLoss,
)
from stage2.initial_state_estimation import constant_velocity_poses
from stage2.pipeline_modes import (
    Stage2PipelineMode, resolve_stage2_mode, validate_stage2_mode_options,
)
from stage2.video_observations import load_optional_evaluation_trajectory


def _positive_raw(value: float) -> torch.Tensor:
    value = max(float(value), 1e-8)
    return torch.tensor(math.log(math.expm1(value)) if value < 20 else value)


def _parameter_initial(value, default: float) -> float:
    if isinstance(value, dict):
        return float(value.get("initial", default))
    return float(default if value is None else value)


def _pair_impedance_initial(pair, key: str, *, bodies, masses, id_to_index, default: float) -> float:
    explicit = pair.parameters.get(key)
    if explicit is not None:
        return _parameter_initial(explicit, default)
    prior = dict(pair.parameters.get("impedance_prior") or {})
    if not prior:
        return float(default)
    time_constant = float(prior.get("time_constant", 0.02))
    damping_ratio = float(prior.get("damping_ratio", 1.0))
    if time_constant <= 0.0 or damping_ratio < 0.0:
        raise ValueError("impedance_prior requires time_constant > 0 and damping_ratio >= 0")
    i, j = id_to_index[pair.body_a], id_to_index[pair.body_b]
    dynamic_masses = [
        float(masses[index]) for index in (i, j) if bodies[index].role == "dynamic"
    ]
    if not dynamic_masses:
        return float(default)
    effective_mass = (
        dynamic_masses[0] if len(dynamic_masses) == 1
        else 1.0 / sum(1.0 / mass for mass in dynamic_masses)
    )
    if key == "stiffness":
        return effective_mass / (time_constant ** 2)
    if key == "damping":
        return 2.0 * damping_ratio * effective_mass / time_constant
    raise ValueError(f"unsupported impedance parameter {key!r}")


def _inertia_component_raw(diagonal) -> torch.Tensor:
    """Map principal inertia to positive components enforcing triangle inequalities."""
    i_x, i_y, i_z = (float(value) for value in diagonal)
    components = (
        0.5 * (i_x + i_y - i_z),
        0.5 * (i_x + i_z - i_y),
        0.5 * (i_y + i_z - i_x),
    )
    return torch.stack([_positive_raw(value) for value in components])


def _effective_inertia(raw: torch.Tensor) -> torch.Tensor:
    components = F.softplus(raw)
    a, b, c = components.unbind(dim=-1)
    return torch.stack((a + b, a + c, b + c), dim=-1)


def _gaussian_union_inertia_diagonal(
    collision: GaussianCollisionBody, mass: float
) -> tuple[float, float, float]:
    """Approximate inertia about the Gaussian union's volume-weighted COM.

    Each collision Gaussian is treated as a solid sphere with mass proportional
    to r^3. This is object-agnostic and gives the correct physical units kg m^2.
    """
    centers = collision.local_centers.detach().to(dtype=torch.float64)
    radii = collision.radii.detach().to(dtype=torch.float64).clamp_min(1e-8)
    weights = radii.pow(3)
    weights = weights / weights.sum().clamp_min(1e-12)
    center_of_mass = torch.sum(weights[:, None] * centers, dim=0)
    offsets = centers - center_of_mass
    squared_distance = torch.sum(offsets.square(), dim=-1)
    orbital = squared_distance[:, None] - offsets.square()
    sphere = (2.0 / 5.0) * radii.square()[:, None]
    diagonal = float(mass) * torch.sum(weights[:, None] * (orbital + sphere), dim=0)
    diagonal = diagonal.clamp_min(1e-8)
    return tuple(float(value) for value in diagonal.cpu())


def _state_path(manifest, body) -> Path:
    value = body.initialization.get("state_json")
    if not value:
        raise ValueError(f"body {body.id!r} requires initialization.state_json")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else manifest.path.parent / path).resolve()


def _load_state(manifest, body, *, device: torch.device) -> RigidBodyState:
    payload = json.loads(_state_path(manifest, body).read_text(encoding="utf-8"))
    state_frame = body.initialization.get("state_frame")
    if "states" in payload:
        if state_frame is None:
            raise ValueError(
                f"body {body.id!r} state_json contains a trajectory; "
                "initialization.state_frame is required"
            )
        requested_frame = int(state_frame)
        matches = [
            state for state in payload["states"]
            if int(state.get("frame_index", -1)) == requested_frame
        ]
        if len(matches) != 1:
            raise ValueError(
                f"body {body.id!r} trajectory has {len(matches)} states for frame "
                f"{requested_frame}; expected exactly one"
            )
        payload = matches[0]
    elif state_frame is not None and int(state_frame) != 0:
        raise ValueError(
            f"body {body.id!r} state_frame can only be nonzero for trajectory state_json"
        )
    def tensor(name, default):
        return torch.tensor(payload.get(name, default), dtype=torch.float32, device=device)
    return RigidBodyState(
        tensor("position", [0, 0, 0]), tensor("quaternion_wxyz", [1, 0, 0, 0]),
        tensor("linear_velocity", [0, 0, 0]), tensor("angular_velocity", [0, 0, 0]),
    )


def _canonical_offset(body, *, device: torch.device) -> torch.Tensor:
    value = body.initialization.get("canonical_offset", [0.0, 0.0, 0.0])
    offset = torch.as_tensor(value, dtype=torch.float32, device=device)
    if offset.shape != (3,):
        raise ValueError(f"body {body.id!r} initialization.canonical_offset must have length 3")
    return offset


def _simulation_timing(manifest) -> tuple[float, int, float]:
    """Return actual observation dt, substep count, and physics integration dt.

    A recorded simulator dataset commonly rounds the number of fixed physics
    steps per video frame.  Preserve that fixed timestep instead of shrinking it
    to force the nominal FPS interval; otherwise the replay slowly drifts from
    the recorded timestamps.
    """
    if manifest.observations.fps is None:
        raise ValueError("native runner currently requires observations.fps")
    nominal_frame_dt = 1.0 / float(manifest.observations.fps)
    requested_dt = float(manifest.simulation.get("physics_timestep", nominal_frame_dt))
    if requested_dt <= 0.0:
        raise ValueError("simulation.physics_timestep must be positive")
    declared_substeps = manifest.simulation.get("steps_per_frame")
    substeps = (
        max(1, int(round(nominal_frame_dt / requested_dt)))
        if declared_substeps is None else int(declared_substeps)
    )
    if substeps < 1:
        raise ValueError("simulation.steps_per_frame must be at least 1")
    return requested_dt * substeps, substeps, requested_dt


def _step_observation_frame(states, dynamics, *, external_wrenches=None):
    """Integrate exactly one RGB-frame interval using differentiable substeps."""
    diagnostics = {}
    for _ in range(int(getattr(dynamics, "frame_substeps", 1))):
        states, diagnostics = dynamics.step(states, external_wrenches=external_wrenches)
    return states, diagnostics


def build_native_runtime(
    manifest, *, device: torch.device, pipeline_mode: str | None = None
):
    paper_mode = pipeline_mode == Stage2PipelineMode.PAPER_COMPATIBLE.value
    bodies = list(manifest.all_bodies)
    if len(bodies) < 2:
        raise ValueError("native multi-body execution requires at least two bodies")
    unsupported = [body.id for body in bodies if body.collision.type not in {"gaussian_union", "plane"}]
    if unsupported:
        raise ValueError(
            "native N-body dynamics supports gaussian_union and static plane collision; "
            f"unsupported bodies: {unsupported}."
        )
    kinematic = [body.id for body in bodies if body.role == "kinematic"]
    if kinematic:
        raise ValueError(
            "kinematic trajectory playback is not yet implemented by the native runner; "
            f"unsupported bodies: {kinematic}"
        )
    id_to_index = {body.id: index for index, body in enumerate(bodies)}
    if not manifest.contact_pairs:
        raise ValueError("native N-body dynamics requires at least one declared contact_pair")
    models = {pair.model for pair in manifest.contact_pairs}
    if len(models) != 1:
        raise ValueError("all contact_pairs must currently use the same contact model")
    if paper_mode and next(iter(models)) != "dual_cone":
        raise ValueError("paper_compatible mode requires contact_pairs[].model='dual_cone'")

    collision_bodies, states, masses, inertia, generalized_damping = [], [], [], [], []
    for body in bodies:
        params = body.collision.parameters
        render = body.render
        if paper_mode and body.collision.type == "gaussian_union":
            selection = str(params.get("primitive_selection", "top_score"))
            experimental_keys = {
                key for key in (
                    "max_primitives", "primitive_selection", "geometry_feature_weight",
                    "support_trim_quantile", "max_radius"
                ) if params.get(key) is not None
            }
            if selection in {"support_surface", "geometry_feature_support"}:
                experimental_keys.add("primitive_selection")
            if experimental_keys:
                raise ValueError(
                    f"paper_compatible body {body.id!r} cannot use experimental collision "
                    f"proxy options: {sorted(experimental_keys)}"
                )
        if body.collision.type == "gaussian_union":
            ply = body.collision.gaussian_ply or body.render.gaussian_ply
            max_primitives = params.get("max_primitives")
            loaded_collision = load_gaussian_collision_body_from_ply(
                ply,
                radius_convention=str(params.get("radius_convention", "paper_r2s")),
                radius_scale=float(params.get("radius_scale", 1.0)),
                max_primitives=(
                    None if max_primitives is None and paper_mode
                    else int(512 if max_primitives is None else max_primitives)
                ),
                primitive_selection=str(params.get("primitive_selection", "top_score")),
                support_trim_quantile=float(params.get("support_trim_quantile", 0.0)),
                max_radius=(
                    None if params.get("max_radius") is None
                    else float(params["max_radius"])
                ),
                geometry_feature_weight=float(params.get("geometry_feature_weight", 1.0)),
                foreground_threshold=None if render is None else render.foreground_threshold,
                opacity_threshold=None if render is None else render.opacity_threshold,
                object_id=None if render is None else render.object_id,
                device=device,
            )
            offset = _canonical_offset(body, device=device)
            collision_bodies.append(GaussianCollisionBody(
                loaded_collision.local_centers - offset,
                loaded_collision.radii,
                None if loaded_collision.local_query_points is None else loaded_collision.local_query_points - offset,
                loaded_collision.source_indices,
            ))
        else:
            placeholder = torch.zeros((1, 3), dtype=torch.float32, device=device)
            collision_bodies.append(GaussianCollisionBody(
                placeholder, torch.full((1,), 1e-6, device=device), placeholder
            ))
        states.append(_load_state(manifest, body, device=device))
        masses.append(_parameter_initial(body.physics.get("mass"), 1.0))
        generalized_damping.append(
            _parameter_initial(body.physics.get("generalized_damping"), 0.0)
            if body.role == "dynamic" else 0.0
        )
        inertia_value = body.physics.get("inertia", {})
        if "initial_diagonal" in inertia_value:
            inertia.append(tuple(float(v) for v in inertia_value["initial_diagonal"]))
        elif body.role == "dynamic" and body.collision.type == "gaussian_union":
            inertia.append(_gaussian_union_inertia_diagonal(collision_bodies[-1], masses[-1]))
        else:
            inertia.append((1.0, 1.0, 1.0))

    stiffness = torch.stack([
        _positive_raw(_pair_impedance_initial(
            pair, "stiffness", bodies=bodies, masses=masses,
            id_to_index=id_to_index, default=800.0,
        ))
        for pair in manifest.contact_pairs
    ]).to(device)
    damping = torch.stack([
        _positive_raw(_pair_impedance_initial(
            pair, "damping", bodies=bodies, masses=masses,
            id_to_index=id_to_index, default=30.0,
        ))
        for pair in manifest.contact_pairs
    ]).to(device)
    friction = torch.stack([
        _positive_raw(_parameter_initial(pair.parameters.get("friction"), 0.4))
        for pair in manifest.contact_pairs
    ]).to(device)
    actual_frame_dt, frame_substeps, integration_dt = _simulation_timing(manifest)
    gaussian_pairs, gaussian_parameter_indices, plane_pairs = [], [], []
    for parameter_index, pair in enumerate(manifest.contact_pairs):
        index_a, index_b = id_to_index[pair.body_a], id_to_index[pair.body_b]
        body_a, body_b = bodies[index_a], bodies[index_b]
        if body_a.collision.type == "gaussian_union" and body_b.collision.type == "gaussian_union":
            gaussian_pairs.append((index_a, index_b))
            gaussian_parameter_indices.append(parameter_index)
            continue
        plane_index, gaussian_index = (
            (index_a, index_b) if body_a.collision.type == "plane" else (index_b, index_a)
        )
        plane_body, gaussian_body = bodies[plane_index], bodies[gaussian_index]
        if plane_body.collision.type != "plane" or gaussian_body.collision.type != "gaussian_union":
            raise ValueError(f"unsupported contact pair {pair.body_a!r}, {pair.body_b!r}")
        if paper_mode:
            raise ValueError(
                "paper_compatible mode requires Gaussian-union geometry for both bodies; "
                "analytic plane contact is an experimental baseline"
            )
        if plane_body.role != "static":
            raise ValueError(f"analytic plane body {plane_body.id!r} must have role='static'")
        normal = torch.tensor(
            plane_body.collision.parameters.get("normal", [0, 0, 1]),
            dtype=torch.float32, device=device,
        )
        plane_pairs.append(GaussianPlaneContactPair(
            body_index=gaussian_index, plane_index=plane_index,
            collider=PlaneCollider(normal, float(plane_body.collision.parameters.get("height", 0.0))),
            parameter_index=parameter_index,
        ))
    paper_collision = dict(manifest.training.get("paper_collision") or {})
    config = MultiBodyImpedanceDynamicsConfig(
        dt=integration_dt, masses=tuple(masses), inertia_diags=tuple(inertia),
        generalized_damping=tuple(generalized_damping),
        dynamic_flags=tuple(body.role == "dynamic" for body in bodies),
        kinematic_flags=tuple(body.role == "kinematic" for body in bodies),
        candidate_pair_mode="all", contact_model=next(iter(models)),
        smooth_min_temperature=float(paper_collision.get("smooth_min_temperature", 1e-2)),
        inside_penalty=float(paper_collision.get("inside_penalty", 0.02)),
        inside_sharpness=float(paper_collision.get("inside_sharpness", 50.0)),
        primitive_locality_margin=(
            None if paper_collision.get("primitive_locality_margin") is None
            else float(paper_collision["primitive_locality_margin"])
        ),
        # The fixed-inside transform stabilizes learned Gaussian-union SDFs.
        # An analytic plane already has an exact signed distance, so applying
        # the blend there creates false penetration for exterior points.
        plane_fixed_penetration=False,
        paper_closed_form_contact=bool(paper_mode),
    )
    dynamics = MultiBodyGaussianImpedanceDynamics(
        collision_bodies, names=[body.id for body in bodies], candidate_pairs=gaussian_pairs,
        candidate_pair_parameter_indices=gaussian_parameter_indices,
        plane_contact_pairs=plane_pairs,
        stiffness=stiffness, damping=damping,
        friction_coefficient=friction, config=config,
        mass_parameters=torch.stack([_positive_raw(value) for value in masses]).to(device),
        inertia_parameters=torch.stack([_inertia_component_raw(value) for value in inertia]).to(device),
    )
    dynamics.frame_substeps = frame_substeps
    dynamics.observation_frame_dt = actual_frame_dt
    dynamics.nominal_observation_frame_dt = 1.0 / float(manifest.observations.fps)
    dynamics.requested_physics_timestep = float(
        manifest.simulation.get("physics_timestep", integration_dt)
    )
    return bodies, tuple(states), dynamics


def rollout_manifest(
    manifest, *, steps: int, device: torch.device, pipeline_mode: str = "image_only"
) -> dict:
    mode_contract = resolve_stage2_mode(pipeline_mode)
    bodies, states, dynamics = build_native_runtime(
        manifest, device=device, pipeline_mode=mode_contract.mode.value
    )
    action_wrenches, action_report = _load_action_wrenches(
        manifest, bodies, max_frame=steps, device=device
    )
    frames = []
    for frame in range(steps + 1):
        frames.append({"frame": frame, "bodies": {
            body.id: state.to_serializable() for body, state in zip(bodies, states)
        }})
        if frame < steps:
            states, _ = _step_observation_frame(
                states, dynamics, external_wrenches=action_wrenches[frame]
            )
    collision_profile = {
        "type": (
            "paper_lse_sigmoid_fixed_penetration_gaussian_raw_plane"
            if mode_contract.mode is Stage2PipelineMode.PAPER_COMPATIBLE else "legacy"
        ),
        "smooth_min_temperature": float(dynamics.config.smooth_min_temperature),
        "inside_penalty": float(dynamics.config.inside_penalty),
        "inside_sharpness": float(dynamics.config.inside_sharpness),
        "applies_to": ["gaussian_union"],
        "analytic_plane_distance": "raw_signed_distance",
    }
    dynamics_profile = {
        "type": "paper_complementarity_free_dual_cone" if dynamics.config.paper_closed_form_contact else "legacy",
        "dual_cone_directions": int(dynamics.config.dual_cone_directions),
        "unified_normal_and_friction": bool(dynamics.config.paper_closed_form_contact),
        "generalized_damping": list(dynamics.config.generalized_damping or ()),
        "observation_frame_dt": float(dynamics.observation_frame_dt),
        "nominal_observation_frame_dt": float(dynamics.nominal_observation_frame_dt),
        "requested_physics_timestep": float(dynamics.requested_physics_timestep),
        "integration_dt": float(dynamics.config.dt),
        "substeps_per_frame": int(dynamics.frame_substeps),
    }
    return {"schema_version": 1, "pipeline_mode": mode_contract.mode.value,
            "pipeline_contract": mode_contract.to_serializable(),
            "actions": action_report,
            "collision_profile": collision_profile,
            "contact_dynamics_profile": dynamics_profile,
            "manifest": manifest_summary(manifest),
            "ground_truth_trajectory_used": False, "frames": frames}


def _rgb_frame_indices(rgb_dir: Path, *, stride: int, max_frames: int) -> list[int]:
    indices = sorted(int(path.stem) for path in rgb_dir.glob("*.png") if path.stem.isdigit())
    indices = indices[::max(1, int(stride))]
    if max_frames > 0:
        indices = indices[:max_frames]
    if not indices:
        raise ValueError(f"no numeric PNG frames found in {rgb_dir}")
    return indices


def _load_action_wrenches(manifest, bodies, *, max_frame: int, device: torch.device):
    """Load world-frame [force, torque] actions for transitions t -> t+1."""
    spec = dict(manifest.actions or {})
    action_type = str(spec.get("type", "zero_wrench"))
    wrenches = torch.zeros(
        (int(max_frame) + 1, len(bodies), 6), dtype=torch.float32, device=device
    )
    if action_type == "zero_wrench":
        return wrenches, {
            "type": "zero_wrench", "source": "manifest" if manifest.actions else "implicit",
            "frame_convention": "wrench[t] drives transition t_to_t_plus_1",
            "coordinate_frame": "world",
        }
    if action_type != "wrench_sequence":
        raise ValueError(f"unsupported actions.type {action_type!r}")
    value = spec.get("path")
    if not value:
        raise ValueError("actions.type='wrench_sequence' requires actions.path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (manifest.path.parent / path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    coordinate_frame = str(payload.get("coordinate_frame", "world")) if isinstance(payload, dict) else "world"
    if coordinate_frame != "world":
        raise ValueError("action wrench coordinate_frame must currently be 'world'")
    frames = payload.get("frames") if isinstance(payload, dict) else payload
    if not isinstance(frames, list):
        raise ValueError("action wrench file must contain a frames list")
    body_to_index = {body.id: index for index, body in enumerate(bodies)}
    nonzero_frames = set()
    for entry in frames:
        frame = int(entry["frame"])
        if frame < 0:
            raise ValueError("action frame indices must be non-negative")
        if frame > max_frame:
            continue
        mapping = entry.get("bodies", {})
        for body_id, wrench in mapping.items():
            if body_id not in body_to_index:
                raise ValueError(f"action references unknown body {body_id!r}")
            force = wrench.get("force", [0.0, 0.0, 0.0])
            torque = wrench.get("torque", [0.0, 0.0, 0.0])
            if len(force) != 3 or len(torque) != 3:
                raise ValueError("action force and torque must each contain three values")
            wrenches[frame, body_to_index[body_id]] = torch.tensor(
                [*force, *torque], dtype=torch.float32, device=device
            )
            nonzero_frames.add(frame)
    return wrenches, {
        "type": "wrench_sequence", "source": str(path),
        "frame_convention": "wrench[t] drives transition t_to_t_plus_1",
        "coordinate_frame": coordinate_frame,
        "nonzero_frame_count": len(nonzero_frames),
    }


def _render_config_from_manifest(
    manifest, *, width: int, height: int, image_loss: str,
    loftr_weight: float = 0.1, loftr_pretrained: str = "outdoor",
    loftr_confidence_threshold: float = 0.2, loftr_max_matches: int = 1024,
    loftr_min_matches: int = 8, loftr_patch_radius: int = 2,
    silhouette_weight: float = 0.0, silhouette_fp_weight: float = 1.0,
    silhouette_fn_weight: float = 1.0,
):
    camera = {}
    if manifest.observations.camera_manifest is not None:
        payload = json.loads(manifest.observations.camera_manifest.read_text(encoding="utf-8"))
        camera = dict(payload.get("camera") or payload)
    training_camera = manifest.training.get("camera")
    if isinstance(training_camera, dict):
        camera.update(training_camera)
    target = camera.get("target")
    background_rgb = camera.get("background_rgb")
    return GaussianRenderLossConfig(
        image_width=width,
        image_height=height,
        loss=image_loss,
        cam_distance=float(camera.get("distance", 1.12)),
        cam_height=float(camera.get("height", 0.66)),
        cam_fovy_deg=float(camera.get("fovy_deg", 40.0)),
        camera_target=None if target is None else tuple(float(value) for value in target),
        background_rgb=(
            None if background_rgb is None
            else tuple(float(value) for value in background_rgb)
        ),
        loftr_weight=float(loftr_weight),
        loftr_pretrained=str(loftr_pretrained),
        loftr_confidence_threshold=float(loftr_confidence_threshold),
        loftr_max_matches=int(loftr_max_matches),
        loftr_min_matches=int(loftr_min_matches),
        loftr_patch_radius=int(loftr_patch_radius),
        silhouette_weight=float(silhouette_weight),
        silhouette_false_positive_weight=float(silhouette_fp_weight),
        silhouette_false_negative_weight=float(silhouette_fn_weight),
    )


def _rollout_selected(
    initial_states, dynamics, frame_indices: list[int], action_wrenches: torch.Tensor | None = None
):
    selected = set(frame_indices)
    states = tuple(initial_states)
    positions, quaternions = [], []
    for frame in range(max(frame_indices) + 1):
        if frame in selected:
            positions.append(torch.stack([state.position for state in states]))
            quaternions.append(torch.stack([state.quaternion_wxyz for state in states]))
        if frame < max(frame_indices):
            wrench = None if action_wrenches is None else action_wrenches[frame]
            states, _ = _step_observation_frame(
                states, dynamics, external_wrenches=wrench
            )
    return torch.stack(positions), torch.stack(quaternions), states


def _temporal_window(
    frame_indices: list[int], *, iteration: int, window_frames: int, window_step: int
) -> tuple[list[int], list[int]]:
    """Return a deterministic cyclic window and its target-array indices."""
    count = len(frame_indices)
    if window_frames <= 0 or window_frames >= count:
        return list(frame_indices), list(range(count))
    width = max(1, int(window_frames))
    step = max(1, int(window_step))
    max_start = count - width
    starts = list(range(0, max_start + 1, step))
    if starts[-1] != max_start:
        starts.append(max_start)
    start = starts[int(iteration) % len(starts)]
    local_indices = list(range(start, start + width))
    return [frame_indices[index] for index in local_indices], local_indices


def _estimate_contact_frames(initial_states, dynamics, frame_indices, action_wrenches) -> list[int]:
    """Estimate contact frames from the current model without reading pose labels."""
    states = tuple(initial_states)
    selected, maximum = set(frame_indices), max(frame_indices)
    contacts = []
    with torch.no_grad():
        for frame in range(maximum):
            wrench = None if action_wrenches is None else action_wrenches[frame]
            states, diagnostics = _step_observation_frame(
                states, dynamics, external_wrenches=wrench
            )
            plane_contacts = diagnostics.get("plane_contacts") or []
            if plane_contacts:
                is_contact = any(
                    bool((entry["contact_gate"] > 0.5).any().detach())
                    for entry in plane_contacts
                )
            else:
                is_contact = bool(diagnostics.get("active_edges"))
            if frame + 1 in selected and is_contact:
                contacts.append(frame + 1)
    return contacts


def _contact_curriculum_selection(
    frame_indices: list[int], contact_frames: list[int], *, budget: int, iteration: int,
) -> tuple[list[int], list[int]]:
    if budget <= 0 or budget >= len(frame_indices) or not contact_frames:
        return list(frame_indices), list(range(len(frame_indices)))
    anchor = contact_frames[iteration % len(contact_frames)]
    ranked = sorted(range(len(frame_indices)), key=lambda idx: (abs(frame_indices[idx] - anchor), idx))
    chosen = sorted(ranked[:max(3, int(budget))])
    return [frame_indices[idx] for idx in chosen], chosen


def prefit_native_initial_states(
    bodies, base_states, render_loss, *, frame_indices: list[int], fps: float,
    pose_iters: int, velocity_iters: int, velocity_frames: int, lr: float,
    velocity_l2: float,
):
    """Image-only pose then free-flight velocity prefit for all dynamic bodies."""
    dynamic_indices = [index for index, body in enumerate(bodies) if body.role == "dynamic"]
    if not dynamic_indices:
        raise ValueError("initial-state prefit requires at least one dynamic body")
    frame_count = max(2, min(int(velocity_frames), len(frame_indices)))
    selected_frames = frame_indices[:frame_count]
    relative_times = torch.tensor(
        [(frame - selected_frames[0]) / float(fps) for frame in selected_frames],
        dtype=base_states[0].position.dtype, device=base_states[0].position.device,
    )

    pose_parameters = []
    for index in dynamic_indices:
        pose_parameters.extend((
            torch.nn.Parameter(base_states[index].position.detach().clone()),
            torch.nn.Parameter(base_states[index].quaternion_wxyz.detach().clone()),
        ))
    optimizer = torch.optim.Adam(pose_parameters, lr=float(lr))
    pose_history, best_pose = [], None
    for iteration in range(max(1, int(pose_iters))):
        optimizer.zero_grad(set_to_none=True)
        positions = torch.stack([
            pose_parameters[2 * dynamic_indices.index(index)] if index in dynamic_indices else state.position
            for index, state in enumerate(base_states)
        ]).unsqueeze(0)
        quaternions = torch.stack([
            F.normalize(pose_parameters[2 * dynamic_indices.index(index) + 1], dim=-1)
            if index in dynamic_indices else state.quaternion_wxyz
            for index, state in enumerate(base_states)
        ]).unsqueeze(0)
        loss, diagnostics = render_loss(positions, quaternions)
        loss.backward()
        value = float(loss.detach().cpu())
        pose_history.append({"iteration": iteration, "loss": value, **diagnostics})
        if best_pose is None or value < best_pose["loss"]:
            best_pose = {"loss": value, "values": [value_.detach().clone() for value_ in pose_parameters]}
        optimizer.step()

    fitted_pose_states = list(base_states)
    for local_index, body_index in enumerate(dynamic_indices):
        fitted_pose_states[body_index] = RigidBodyState(
            best_pose["values"][2 * local_index],
            F.normalize(best_pose["values"][2 * local_index + 1], dim=-1),
            base_states[body_index].linear_velocity,
            base_states[body_index].angular_velocity,
        )

    velocity_parameters = []
    for index in dynamic_indices:
        velocity_parameters.extend((
            torch.nn.Parameter(base_states[index].linear_velocity.detach().clone()),
            torch.nn.Parameter(base_states[index].angular_velocity.detach().clone()),
        ))
    optimizer = torch.optim.Adam(velocity_parameters, lr=float(lr))
    velocity_history, best_velocity = [], None
    for iteration in range(max(1, int(velocity_iters))):
        optimizer.zero_grad(set_to_none=True)
        body_positions, body_quaternions = [], []
        for body_index, state in enumerate(fitted_pose_states):
            if body_index in dynamic_indices:
                local_index = dynamic_indices.index(body_index)
                positions, quaternions = constant_velocity_poses(
                    state.position, state.quaternion_wxyz,
                    velocity_parameters[2 * local_index], velocity_parameters[2 * local_index + 1],
                    relative_times,
                )
            else:
                positions = state.position.unsqueeze(0).expand(frame_count, -1)
                quaternions = state.quaternion_wxyz.unsqueeze(0).expand(frame_count, -1)
            body_positions.append(positions)
            body_quaternions.append(quaternions)
        positions = torch.stack(body_positions, dim=1)
        quaternions = torch.stack(body_quaternions, dim=1)
        image_loss, diagnostics = render_loss(positions, quaternions)
        regularizer = float(velocity_l2) * torch.stack([
            torch.mean(parameter ** 2) * (0.01 if index % 2 else 1.0)
            for index, parameter in enumerate(velocity_parameters)
        ]).mean()
        loss = image_loss + regularizer
        loss.backward()
        value = float(loss.detach().cpu())
        velocity_history.append({"iteration": iteration, "loss": value, **diagnostics})
        if best_velocity is None or value < best_velocity["loss"]:
            best_velocity = {"loss": value, "values": [value_.detach().clone() for value_ in velocity_parameters]}
        optimizer.step()

    fitted_states = list(fitted_pose_states)
    for local_index, body_index in enumerate(dynamic_indices):
        state = fitted_pose_states[body_index]
        fitted_states[body_index] = RigidBodyState(
            state.position, state.quaternion_wxyz,
            best_velocity["values"][2 * local_index], best_velocity["values"][2 * local_index + 1],
        )
    return tuple(fitted_states), {
        "supervision": "image_only",
        "ground_truth_trajectory_used": False,
        "frame_indices": selected_frames,
        "pose_best_loss": best_pose["loss"],
        "velocity_best_loss": best_velocity["loss"],
        "pose_history": pose_history,
        "velocity_history": velocity_history,
        "states": {body.id: state.to_serializable() for body, state in zip(bodies, fitted_states)},
    }


def refined_geometry_overrides(
    bodies, collision_bodies, render_assets, center_deltas, log_radius_deltas, *,
    max_center_delta: float, max_log_radius_delta: float,
    gradient_route: str = "collision_only",
):
    """Build one shared differentiable geometry for collision and rendering."""
    if gradient_route not in {"collision_only", "collision_and_render"}:
        raise ValueError(
            "gradient_route must be 'collision_only' or 'collision_and_render'"
        )
    refined_collision, render_centers, render_radii, stats = [], [], [], {}
    for index, (body, collision, asset) in enumerate(zip(bodies, collision_bodies, render_assets)):
        if center_deltas[index] is None:
            refined_collision.append(collision)
            render_centers.append(None)
            render_radii.append(None)
            continue
        source_indices = collision.source_indices
        if source_indices is None:
            raise ValueError(f"body {body.id!r} geometry refinement requires collision source_indices")
        asset_source_indices = (
            torch.arange(asset.num_gaussians, dtype=torch.long, device=asset.xyz.device)
            if asset.source_indices is None else asset.source_indices
        )
        render_indices = torch.searchsorted(asset_source_indices, source_indices)
        valid_mapping = (
            render_indices.numel() == source_indices.numel()
            and bool(torch.all(render_indices < asset.num_gaussians))
            and bool(torch.equal(asset_source_indices[render_indices], source_indices))
        )
        if not valid_mapping:
            raise ValueError(
                f"body {body.id!r} collision/render filters do not share PLY index space; "
                "use matching object/opacity/foreground filters"
            )
        center_offset = float(max_center_delta) * torch.tanh(center_deltas[index])
        log_radius_offset = float(max_log_radius_delta) * torch.tanh(log_radius_deltas[index])
        centers = collision.local_centers + center_offset
        radii = collision.radii * torch.exp(log_radius_offset)
        refined_collision.append(type(collision)(centers, radii, centers, source_indices))
        render_proxy_centers = centers if gradient_route == "collision_and_render" else centers.detach()
        render_proxy_radii = radii if gradient_route == "collision_and_render" else radii.detach()
        full_centers = asset.xyz.index_copy(0, render_indices, render_proxy_centers)
        base_render_radii = asset.activated_scaling.mean(dim=-1) * 2.0
        full_radii = base_render_radii.index_copy(0, render_indices, render_proxy_radii)
        render_centers.append(full_centers)
        render_radii.append(full_radii)
        stats[body.id] = {
            "num_refined": int(centers.shape[0]),
            "center_delta_rms": float(torch.sqrt(torch.mean(center_offset.detach() ** 2)).cpu()),
            "center_delta_max": float(torch.max(torch.linalg.norm(center_offset.detach(), dim=-1)).cpu()),
            "log_radius_delta_rms": float(torch.sqrt(torch.mean(log_radius_offset.detach() ** 2)).cpu()),
            "radius_scale_min": float(torch.exp(log_radius_offset.detach()).min().cpu()),
            "radius_scale_max": float(torch.exp(log_radius_offset.detach()).max().cpu()),
        }
    return tuple(refined_collision), render_centers, render_radii, stats


def loss_gradient_attribution(
    loss_terms: dict[str, torch.Tensor],
    parameter_groups: dict[str, list[torch.Tensor]],
) -> dict[str, dict[str, float]]:
    """Return per-loss gradient norms without modifying ``parameter.grad``."""
    unique_parameters = []
    parameter_indices = {}
    group_indices = {}
    for group_name, parameters in parameter_groups.items():
        indices = []
        for parameter in parameters:
            if not parameter.requires_grad:
                continue
            identity = id(parameter)
            if identity not in parameter_indices:
                parameter_indices[identity] = len(unique_parameters)
                unique_parameters.append(parameter)
            indices.append(parameter_indices[identity])
        group_indices[group_name] = indices
    attribution = {}
    for loss_name, term in loss_terms.items():
        if unique_parameters and term.requires_grad:
            gradients = torch.autograd.grad(
                term, unique_parameters, retain_graph=True, allow_unused=True,
            )
        else:
            gradients = (None,) * len(unique_parameters)
        group_norms = {}
        for group_name, indices in group_indices.items():
            squared_norm = sum(
                torch.sum(gradients[index].detach() ** 2)
                for index in indices if gradients[index] is not None
            )
            group_norms[group_name] = (
                0.0 if isinstance(squared_norm, int)
                else float(torch.sqrt(squared_norm).cpu())
            )
        attribution[loss_name] = group_norms
    return attribution


def fit_native_image_only(
    manifest, *, fit_iters: int, lr: float, stride: int, max_frames: int,
    width: int, height: int, image_loss: str, device: torch.device,
    render_output_dir: Path | None = None,
    render_loss_factory=MultiBodyStage2GaussianRenderLoss,
    prefit_initial_state: bool = False, prefit_pose_iters: int = 100,
    prefit_velocity_iters: int = 100, prefit_velocity_frames: int = 3,
    prefit_lr: float = 0.01, prefit_velocity_l2: float = 1e-4,
    refine_geometry: bool = False, geometry_lr: float = 1e-3,
    geometry_center_l2: float = 1e-3, geometry_radius_l2: float = 1e-3,
    geometry_max_center_delta: float = 0.01, geometry_max_log_radius_delta: float = 0.2,
    geometry_gradient_route: str = "collision_only",
    loftr_weight: float = 0.1, loftr_pretrained: str = "outdoor",
    loftr_confidence_threshold: float = 0.2, loftr_max_matches: int = 1024,
    loftr_min_matches: int = 8, loftr_patch_radius: int = 2,
    learn_mass_inertia: bool = True, mass_inertia_lr: float | None = None,
    mass_l2: float = 1e-4, inertia_l2: float = 1e-4,
    temporal_window_frames: int = 0, temporal_window_step: int = 1,
    physics_warmup_fraction: float = 0.2,
    freeze_initial_state_after_warmup: bool = True,
    contact_parameter_l2: float = 1e-4,
    trajectory_stability_weight: float = 1e-4,
    max_dynamic_displacement: float = 3.0,
    stiffness_lr: float | None = None, damping_lr: float | None = None,
    friction_lr: float | None = None, contact_curriculum_frames: int = 8,
    silhouette_weight: float = 0.005, silhouette_fp_weight: float = 2.0,
    silhouette_fn_weight: float = 1.0,
    gradient_attribution: bool = False, gradient_attribution_interval: int = 1,
    physics_override: dict | None = None,
    learn_contact_parameters: bool = True,
    evaluation_only: bool = False,
    pipeline_mode: str | None = None,
) -> dict:
    if not 0.0 <= float(physics_warmup_fraction) <= 1.0:
        raise ValueError("physics_warmup_fraction must be in [0, 1]")
    if min(float(contact_parameter_l2), float(trajectory_stability_weight)) < 0.0:
        raise ValueError("stability regularization weights must be non-negative")
    if float(max_dynamic_displacement) <= 0.0:
        raise ValueError("max_dynamic_displacement must be positive")
    if int(gradient_attribution_interval) < 1:
        raise ValueError("gradient_attribution_interval must be at least 1")
    if pipeline_mode is None:
        pipeline_mode = (
            "experimental"
            if temporal_window_frames > 0 or geometry_gradient_route != "collision_only"
            else "image_only"
        )
    mode_contract = resolve_stage2_mode(pipeline_mode)
    requested_geometry_gradient_route = str(geometry_gradient_route)
    effective_geometry_gradient_route = (
        "collision_only"
        if mode_contract.mode is Stage2PipelineMode.PAPER_COMPATIBLE
        else requested_geometry_gradient_route
    )
    validate_stage2_mode_options(
        mode_contract, prefit_initial_state=prefit_initial_state,
        temporal_window_frames=temporal_window_frames,
        geometry_gradient_route=effective_geometry_gradient_route,
    )
    paper_supervision = mode_contract.mode is Stage2PipelineMode.PAPER_COMPATIBLE
    if paper_supervision and not learn_mass_inertia and not evaluation_only:
        raise ValueError(
            "paper_compatible jointly optimizes M, mu, K, D and Gaussian geometry; "
            "mass/inertia cannot be frozen"
        )
    if device.type != "cuda" and render_loss_factory is MultiBodyStage2GaussianRenderLoss:
        raise ValueError("native image-only fitting requires --device cuda for the Gaussian rasterizer")
    bodies, base_states, dynamics = build_native_runtime(
        manifest, device=device, pipeline_mode=mode_contract.mode.value
    )
    if physics_override:
        def inverse_softplus(value: float) -> torch.Tensor:
            if value <= 0.0:
                raise ValueError("fixed physics values must be positive")
            tensor = torch.tensor(value, dtype=torch.float32, device=device)
            return tensor if value > 20.0 else torch.log(torch.expm1(tensor))
        pairs = physics_override.get("contact_pairs", [])
        by_edge = {(item["body_a"], item["body_b"]): item for item in pairs}
        for index, pair in enumerate(manifest.contact_pairs):
            item = by_edge.get((pair.body_a, pair.body_b)) or by_edge.get((pair.body_b, pair.body_a))
            if item:
                for key, target in (
                    ("stiffness", dynamics.stiffness), ("damping", dynamics.damping),
                    ("friction", dynamics.friction_coefficient),
                ):
                    if key in item:
                        target[index] = inverse_softplus(float(item[key]))
    missing_render = [body.id for body in bodies if body.render is None]
    if missing_render:
        raise ValueError(f"image-only fitting requires render.gaussian_ply for every body: {missing_render}")
    frame_indices = _rgb_frame_indices(manifest.observations.rgb_dir, stride=stride, max_frames=max_frames)
    action_wrenches, action_report = _load_action_wrenches(
        manifest, bodies, max_frame=max(frame_indices), device=device
    )
    effective_refine_geometry = bool(refine_geometry or paper_supervision)
    effective_image_loss = "l1_loftr" if paper_supervision else image_loss
    effective_loftr_weight = 1.0 if paper_supervision else float(loftr_weight)
    effective_mask_dir = None if paper_supervision else manifest.observations.instance_mask_dir
    render_loss = render_loss_factory(
        stage1_plys=[body.render.gaussian_ply for body in bodies],
        gt_rgb_dir=manifest.observations.rgb_dir,
        gt_mask_dir=effective_mask_dir,
        frame_indices=frame_indices,
        config=_render_config_from_manifest(
            manifest, width=width, height=height, image_loss=effective_image_loss,
            loftr_weight=effective_loftr_weight, loftr_pretrained=loftr_pretrained,
            loftr_confidence_threshold=loftr_confidence_threshold,
            loftr_max_matches=loftr_max_matches, loftr_min_matches=loftr_min_matches,
            loftr_patch_radius=loftr_patch_radius,
            silhouette_weight=0.0 if paper_supervision else silhouette_weight,
            silhouette_fp_weight=silhouette_fp_weight,
            silhouette_fn_weight=silhouette_fn_weight,
        ),
        render_filters=[{
            "opacity_threshold": body.render.opacity_threshold,
            "foreground_threshold": body.render.foreground_threshold,
            "object_id": body.render.object_id,
            "canonical_offset": _canonical_offset(body, device=device).detach().cpu().tolist(),
        } for body in bodies],
        device=device,
    )

    prefit_report = None
    if prefit_initial_state:
        base_states, prefit_report = prefit_native_initial_states(
            bodies, base_states, render_loss, frame_indices=frame_indices,
            fps=float(manifest.observations.fps or 30.0), pose_iters=prefit_pose_iters,
            velocity_iters=prefit_velocity_iters, velocity_frames=prefit_velocity_frames,
            lr=prefit_lr, velocity_l2=prefit_velocity_l2,
        )

    trainable, state_parameters = [], []
    learn_initial_state = mode_contract.mode is not Stage2PipelineMode.PAPER_COMPATIBLE
    for body, state in zip(bodies, base_states):
        if body.role == "dynamic" and learn_initial_state:
            values = [torch.nn.Parameter(value.detach().clone()) for value in (
                state.position, state.quaternion_wxyz, state.linear_velocity, state.angular_velocity
            )]
            trainable.extend(values)
            state_parameters.append(values)
        else:
            state_parameters.append(None)
    initial_contact_values = tuple(
        F.softplus(value.detach()).clone() for value in (
            dynamics.stiffness, dynamics.damping, dynamics.friction_coefficient
        )
    )
    log_stiffness, log_damping, log_friction = [
        torch.nn.Parameter(torch.log(torch.clamp(value, min=1e-8)))
        for value in initial_contact_values
    ]
    contact_log_parameters = [log_stiffness, log_damping, log_friction]

    def sync_contact_parameters() -> None:
        values = [torch.exp(parameter) for parameter in contact_log_parameters]
        dynamics.stiffness, dynamics.damping, dynamics.friction_coefficient = [
            torch.where(
                value > 20.0, value,
                torch.log(torch.expm1(torch.clamp(value, max=20.0))),
            )
            for value in values
        ]

    sync_contact_parameters()
    initial_mass_raw = dynamics.mass_parameters.detach().clone()
    initial_inertia_raw = dynamics.inertia_parameters.detach().clone()
    dynamics.mass_parameters = torch.nn.Parameter(initial_mass_raw.clone())
    dynamics.inertia_parameters = torch.nn.Parameter(initial_inertia_raw.clone())
    material_parameters = []
    if learn_mass_inertia:
        material_parameters.extend((dynamics.mass_parameters, dynamics.inertia_parameters))
    center_deltas, log_radius_deltas = [], []
    geometry_parameters = []
    for body, collision in zip(bodies, dynamics.bodies):
        if effective_refine_geometry and body.role == "dynamic" and body.collision.type == "gaussian_union":
            center_delta = torch.nn.Parameter(torch.zeros_like(collision.local_centers))
            radius_delta = torch.nn.Parameter(torch.zeros_like(collision.radii))
            center_deltas.append(center_delta)
            log_radius_deltas.append(radius_delta)
            geometry_parameters.extend((center_delta, radius_delta))
        else:
            center_deltas.append(None)
            log_radius_deltas.append(None)
    parameter_groups = [group for group in (
        {"params": trainable, "lr": float(lr)},
        {"params": geometry_parameters, "lr": float(geometry_lr)},
        {"params": material_parameters, "lr": float(lr if mass_inertia_lr is None else mass_inertia_lr)},
    ) if group["params"]]
    if learn_contact_parameters:
        parameter_groups.extend((
            {"params": [log_stiffness], "lr": float(lr if stiffness_lr is None else stiffness_lr)},
            {"params": [log_damping], "lr": float(lr if damping_lr is None else damping_lr)},
            {"params": [log_friction], "lr": float(lr if friction_lr is None else friction_lr)},
        ))
    optimizer = torch.optim.Adam(parameter_groups) if parameter_groups else None
    all_parameters = [
        parameter for group in ([] if optimizer is None else optimizer.param_groups)
        for parameter in group["params"]
    ]
    initial_contact_logs = tuple(value.detach().clone() for value in contact_log_parameters)
    warmup_iterations = min(
        int(fit_iters), max(0, int(round(float(fit_iters) * float(physics_warmup_fraction))))
    )
    history = []
    best_loss = float("inf")
    best_parameters = None
    base_collision_bodies = tuple(dynamics.bodies)
    geometry_stats = {}
    estimated_contact_frames = _estimate_contact_frames(
        base_states, dynamics, frame_indices, action_wrenches
    )
    for iteration in range(int(fit_iters)):
        if optimizer is None:
            raise ValueError("fit_iters > 0 requires at least one learnable parameter")
        optimizer.zero_grad(set_to_none=True)
        sync_contact_parameters()
        initial_states = []
        for base, params in zip(base_states, state_parameters):
            if params is None:
                initial_states.append(base)
            else:
                position, quaternion, linear_velocity, angular_velocity = params
                initial_states.append(RigidBodyState(
                    position, F.normalize(quaternion, dim=-1), linear_velocity, angular_velocity
                ))
        render_geometry_centers = render_geometry_radii = None
        if effective_refine_geometry:
            refined_collision, render_geometry_centers, render_geometry_radii, geometry_stats = refined_geometry_overrides(
                bodies, base_collision_bodies, render_loss.base_assets,
                center_deltas, log_radius_deltas,
                max_center_delta=geometry_max_center_delta,
                max_log_radius_delta=geometry_max_log_radius_delta,
                gradient_route=effective_geometry_gradient_route,
            )
            dynamics.bodies = refined_collision
        training_frames, target_indices = _temporal_window(
            frame_indices, iteration=iteration, window_frames=temporal_window_frames,
            window_step=temporal_window_step,
        )
        if iteration >= warmup_iterations and temporal_window_frames <= 0:
            training_frames, target_indices = _contact_curriculum_selection(
                frame_indices, estimated_contact_frames,
                budget=int(contact_curriculum_frames), iteration=iteration - warmup_iterations,
            )
        positions, quaternions, _ = _rollout_selected(
            initial_states, dynamics, training_frames, action_wrenches
        )
        if effective_refine_geometry:
            loss, diagnostics = render_loss(
                positions, quaternions,
                geometry_centers=render_geometry_centers,
                geometry_radii=render_geometry_radii,
                target_indices=target_indices,
            )
        else:
            loss, diagnostics = render_loss(
                positions, quaternions, target_indices=target_indices
            )
        loss_terms = dict(getattr(render_loss, "last_loss_terms", {"image": loss}))
        if geometry_parameters:
            center_regularizer = torch.stack([torch.mean(value ** 2) for value in center_deltas if value is not None]).mean()
            radius_regularizer = torch.stack([torch.mean(value ** 2) for value in log_radius_deltas if value is not None]).mean()
            loss_terms["geometry_center_l2"] = float(geometry_center_l2) * center_regularizer
            loss_terms["geometry_radius_l2"] = float(geometry_radius_l2) * radius_regularizer
            loss = loss + loss_terms["geometry_center_l2"] + loss_terms["geometry_radius_l2"]
        if material_parameters:
            dynamic_mask = torch.tensor(
                [body.role == "dynamic" for body in bodies], device=device, dtype=torch.bool
            )
            loss_terms["mass_l2"] = float(mass_l2) * torch.mean(
                (dynamics.mass_parameters[dynamic_mask] - initial_mass_raw[dynamic_mask]) ** 2
            )
            loss_terms["inertia_l2"] = float(inertia_l2) * torch.mean(
                (dynamics.inertia_parameters[dynamic_mask] - initial_inertia_raw[dynamic_mask]) ** 2
            )
            loss = loss + loss_terms["mass_l2"] + loss_terms["inertia_l2"]
        contact_regularizer = sum(
            torch.mean((current - initial) ** 2)
            for current, initial in zip(
                contact_log_parameters, initial_contact_logs,
            )
        )
        loss_terms["contact_parameter_l2"] = float(contact_parameter_l2) * contact_regularizer
        loss = loss + loss_terms["contact_parameter_l2"]
        dynamic_indices = [index for index, body in enumerate(bodies) if body.role == "dynamic"]
        stability_regularizer = torch.zeros((), dtype=loss.dtype, device=loss.device)
        if dynamic_indices and positions.shape[0] > 1:
            dynamic_positions = positions[:, dynamic_indices]
            origin = torch.stack([initial_states[index].position for index in dynamic_indices])
            displacement = torch.linalg.norm(dynamic_positions - origin.unsqueeze(0), dim=-1)
            excess = F.softplus(displacement - float(max_dynamic_displacement))
            frame_steps = torch.tensor(
                [max(1, training_frames[index + 1] - training_frames[index])
                 for index in range(len(training_frames) - 1)],
                dtype=loss.dtype, device=loss.device,
            ).reshape(-1, 1, 1)
            velocity = (dynamic_positions[1:] - dynamic_positions[:-1]) / frame_steps
            stability_regularizer = torch.mean(excess ** 2) + 0.01 * torch.mean(velocity ** 2)
            loss_terms["trajectory_stability"] = float(trajectory_stability_weight) * stability_regularizer
            loss = loss + loss_terms["trajectory_stability"]
        if not bool(torch.isfinite(loss).detach()):
            raise FloatingPointError(f"non-finite Stage2 loss at iteration {iteration}")
        attribution = None
        if gradient_attribution and iteration % int(gradient_attribution_interval) == 0:
            attribution = loss_gradient_attribution(loss_terms, {
                "initial_state": trainable,
                "geometry": geometry_parameters,
                "mass_inertia": material_parameters,
                "log_stiffness": [log_stiffness],
                "log_damping": [log_damping],
                "log_friction": [log_friction],
            })
        loss.backward()
        curriculum_phase = "state_warmup" if iteration < warmup_iterations else "joint_physics"
        frozen_parameter_groups = []
        if iteration < warmup_iterations:
            physics_parameters = [
                *contact_log_parameters,
                *material_parameters, *geometry_parameters,
            ]
            for parameter in physics_parameters:
                parameter.grad = None
            frozen_parameter_groups.append("physics")
        elif freeze_initial_state_after_warmup:
            for parameter in trainable:
                parameter.grad = None
            frozen_parameter_groups.append("initial_state")
        gradient_norm = torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=10.0)
        physics_gradient_norms = {
            name: 0.0 if parameter.grad is None else float(parameter.grad.detach().norm().cpu())
            for name, parameter in zip(
                ("log_stiffness", "log_damping", "log_friction"), contact_log_parameters
            )
        }
        current_loss = float(loss.detach().cpu())
        # Warm-up and joint-physics losses are not comparable: they optimize
        # different parameter groups and may use different frame curricula.
        # Never restore a warm-up checkpoint over learned contact physics.
        if iteration >= warmup_iterations and current_loss < best_loss:
            best_loss = current_loss
            best_parameters = [parameter.detach().clone() for parameter in all_parameters]
        optimizer.step()
        history.append({
            "iteration": iteration, "loss": current_loss,
            "curriculum_phase": curriculum_phase,
            "frozen_parameter_groups": frozen_parameter_groups,
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "physics_gradient_norms": physics_gradient_norms,
            "loss_gradient_attribution": attribution,
            "contact_regularizer": float(contact_regularizer.detach().cpu()),
            "trajectory_stability_regularizer": float(stability_regularizer.detach().cpu()),
            "training_frame_indices": training_frames,
            "training_target_indices": target_indices,
            "estimated_contact_frames": estimated_contact_frames,
            "contact_frames_in_batch": [
                frame for frame in training_frames if frame in set(estimated_contact_frames)
            ],
            **diagnostics,
        })

    if best_parameters is not None:
        with torch.no_grad():
            for parameter, best_value in zip(all_parameters, best_parameters):
                parameter.copy_(best_value)
        sync_contact_parameters()

    fitted_states = []
    for base, params in zip(base_states, state_parameters):
        fitted_states.append(base if params is None else RigidBodyState(
            params[0].detach(), F.normalize(params[1].detach(), dim=-1),
            params[2].detach(), params[3].detach(),
        ))
    render_geometry_centers = render_geometry_radii = None
    refined_geometry_payload = {}
    if effective_refine_geometry:
        refined_collision, render_geometry_centers, render_geometry_radii, geometry_stats = refined_geometry_overrides(
            bodies, base_collision_bodies, render_loss.base_assets,
            center_deltas, log_radius_deltas,
            max_center_delta=geometry_max_center_delta,
            max_log_radius_delta=geometry_max_log_radius_delta,
            gradient_route=effective_geometry_gradient_route,
        )
        dynamics.bodies = refined_collision
        refined_geometry_payload = {
            body.id: {
                "source_indices": collision.source_indices.detach().cpu().tolist(),
                "local_centers": collision.local_centers.detach().cpu().tolist(),
                "radii": collision.radii.detach().cpu().tolist(),
            }
            for body, collision, delta in zip(bodies, refined_collision, center_deltas)
            if delta is not None
        }
    positions, quaternions, _ = _rollout_selected(
        fitted_states, dynamics, frame_indices, action_wrenches
    )
    with torch.no_grad():
        rendered = (
            render_loss.render_sequence(
                positions, quaternions,
                geometry_centers=render_geometry_centers,
                geometry_radii=render_geometry_radii,
            )
            if effective_refine_geometry else render_loss.render_sequence(positions, quaternions)
        )
        targets = render_loss.targets[: rendered.shape[0]]
        evaluation_l1 = torch.mean(torch.abs(rendered - targets))
        evaluation_mse = torch.mean((rendered - targets) ** 2)
        evaluation_psnr = -10.0 * torch.log10(torch.clamp(evaluation_mse, min=1e-12))
        masks = None if render_loss.masks is None else render_loss.masks[: rendered.shape[0]]
        if masks is None:
            foreground_l1, foreground_mse, foreground_psnr = evaluation_l1, evaluation_mse, evaluation_psnr
        else:
            foreground_denominator = torch.clamp(masks.sum() * rendered.shape[1], min=1.0)
            foreground_l1 = (torch.abs(rendered - targets) * masks).sum() / foreground_denominator
            foreground_mse = (((rendered - targets) ** 2) * masks).sum() / foreground_denominator
            foreground_psnr = -10.0 * torch.log10(torch.clamp(foreground_mse, min=1e-12))
    trajectory_evaluation = None
    evaluation_trajectory = load_optional_evaluation_trajectory(manifest.evaluation_trajectory)
    if evaluation_trajectory is not None:
        frame_to_target = {
            frame: index for index, frame in enumerate(evaluation_trajectory.frame_indices)
        }
        matched = [
            (pred_index, frame_to_target[frame]) for pred_index, frame in enumerate(frame_indices)
            if frame in frame_to_target
        ]
        dynamic_indices = [index for index, body in enumerate(bodies) if body.role == "dynamic"]
        if matched and dynamic_indices:
            pred_indices = torch.tensor([item[0] for item in matched], device=device)
            target_indices = torch.tensor([item[1] for item in matched])
            predicted_position = positions[pred_indices, dynamic_indices[0]]
            target_position = evaluation_trajectory.positions[target_indices].to(device)
            predicted_quaternion = F.normalize(
                quaternions[pred_indices, dynamic_indices[0]], dim=-1
            )
            target_quaternion = evaluation_trajectory.quaternions_wxyz[target_indices].to(device)
            translation = torch.linalg.norm(predicted_position - target_position, dim=-1)
            dots = torch.clamp(torch.abs(torch.sum(
                predicted_quaternion * target_quaternion, dim=-1
            )), max=1.0)
            rotation = 2.0 * torch.acos(dots)
            trajectory_evaluation = {
                "body_id": bodies[dynamic_indices[0]].id,
                "matched_frames": len(matched),
                "translation_error_mean_m": float(translation.mean().detach().cpu()),
                "translation_error_rmse_m": float(torch.sqrt(torch.mean(translation ** 2)).detach().cpu()),
                "rotation_error_mean_rad": float(rotation.mean().detach().cpu()),
                "rotation_error_rmse_rad": float(torch.sqrt(torch.mean(rotation ** 2)).detach().cpu()),
            }
    artifacts = None
    if render_output_dir is not None:
        render_output_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir = render_output_dir / "prediction"
        comparison_dir = render_output_dir / "comparison"
        prediction_dir.mkdir(exist_ok=True)
        comparison_dir.mkdir(exist_ok=True)
        prediction_images, comparison_images = [], []
        for local_index, frame_index in enumerate(frame_indices):
            prediction_array = (rendered[local_index].detach().cpu().permute(1, 2, 0).numpy() * 255.0)
            target_array = (targets[local_index].detach().cpu().permute(1, 2, 0).numpy() * 255.0)
            prediction_image = Image.fromarray(np.clip(prediction_array, 0, 255).astype(np.uint8))
            target_image = Image.fromarray(np.clip(target_array, 0, 255).astype(np.uint8))
            comparison_image = Image.new("RGB", (prediction_image.width * 2, prediction_image.height))
            comparison_image.paste(target_image, (0, 0))
            comparison_image.paste(prediction_image, (prediction_image.width, 0))
            prediction_image.save(prediction_dir / f"{frame_index:06d}.png")
            comparison_image.save(comparison_dir / f"{frame_index:06d}.png")
            prediction_images.append(prediction_image)
            comparison_images.append(comparison_image)
        duration_ms = max(1, int(round(1000.0 / float(manifest.observations.fps or 30.0))))
        prediction_images[0].save(
            render_output_dir / "prediction.gif", save_all=True,
            append_images=prediction_images[1:], duration=duration_ms, loop=0,
        )
        comparison_images[0].save(
            render_output_dir / "gt_left_prediction_right.gif", save_all=True,
            append_images=comparison_images[1:], duration=duration_ms, loop=0,
        )
        artifacts = {
            "prediction_dir": str(prediction_dir.resolve()),
            "comparison_dir": str(comparison_dir.resolve()),
            "prediction_gif": str((render_output_dir / "prediction.gif").resolve()),
            "comparison_gif": str((render_output_dir / "gt_left_prediction_right.gif").resolve()),
        }
    return {
        "schema_version": 1,
        "pipeline_mode": mode_contract.mode.value,
        "pipeline_contract": mode_contract.to_serializable(),
        "paper_compatibility": {
            "status": "stage2_core_complete" if mode_contract.mode is Stage2PipelineMode.PAPER_COMPATIBLE else "not_applicable",
            "completed_components": (
                [
                    "mode_contract", "fixed_known_initial_state",
                    "frame_aligned_world_wrench_actions",
                    "lse_sigmoid_fixed_penetration_collision",
                    "complementarity_free_dual_cone_contact",
                    "full_image_l1_plus_loftr_supervision",
                    "collision_only_geometry_gradient",
                ]
                if mode_contract.mode is Stage2PipelineMode.PAPER_COMPATIBLE else []
            ),
            "note": (
                "The staged paper-compatible Stage2 core path is active."
                if mode_contract.mode is Stage2PipelineMode.PAPER_COMPATIBLE else None
            ),
        },
        "initial_state_learning": {
            "enabled": bool(learn_initial_state),
            "source": "optimizer" if learn_initial_state else "manifest_state_json",
            "bodies": {
                body.id: {
                    "state_json": str(_state_path(manifest, body)),
                    "state_frame": body.initialization.get("state_frame"),
                }
                for body in bodies
            },
        },
        "canonical_alignment": {
            body.id: {
                "asset_origin_offset": _canonical_offset(body, device=device).detach().cpu().tolist(),
                "applied_to": ["render_gaussians", "collision_gaussians", "collision_query_points"],
            }
            for body in bodies
        },
        "actions": action_report,
        "collision_profile": {
            "type": (
                "paper_lse_sigmoid_fixed_penetration_gaussian_raw_plane"
                if paper_supervision else "legacy"
            ),
            "smooth_min_temperature": float(dynamics.config.smooth_min_temperature),
            "inside_penalty": float(dynamics.config.inside_penalty),
            "inside_sharpness": float(dynamics.config.inside_sharpness),
            "applies_to": ["gaussian_union"],
            "analytic_plane_distance": "raw_signed_distance",
        },
        "contact_dynamics_profile": {
            "type": (
                "paper_complementarity_free_dual_cone"
                if dynamics.config.paper_closed_form_contact else "legacy"
            ),
            "dual_cone_directions": int(dynamics.config.dual_cone_directions),
            "unified_normal_and_friction": bool(dynamics.config.paper_closed_form_contact),
            "update": "A*lambda=rhs; v_next=b+h*M_inv*J_dual_T*lambda",
            "implicit_matrix": "A=I+h*(h*K+D)*J_dual*M_inv*J_dual_T",
            "generalized_damping": list(dynamics.config.generalized_damping or ()),
            "observation_frame_dt": float(dynamics.observation_frame_dt),
            "nominal_observation_frame_dt": float(dynamics.nominal_observation_frame_dt),
            "requested_physics_timestep": float(dynamics.requested_physics_timestep),
            "integration_dt": float(dynamics.config.dt),
            "substeps_per_frame": int(dynamics.frame_substeps),
        },
        "manifest": manifest_summary(manifest),
        "supervision": "paper_full_image" if paper_supervision else "image_only",
        "ground_truth_trajectory_used_for_training": False,
        "frame_indices": frame_indices,
        "temporal_window": {
            "enabled": bool(0 < temporal_window_frames < len(frame_indices)),
            "frames": int(temporal_window_frames),
            "step": int(temporal_window_step),
            "schedule": "cyclic_contiguous",
            "full_evaluation_frames": len(frame_indices),
        },
        "fit_iterations": int(fit_iters),
        "optimization_stability": {
            "physics_warmup_fraction": float(physics_warmup_fraction),
            "physics_warmup_iterations": int(warmup_iterations),
            "freeze_initial_state_after_warmup": bool(freeze_initial_state_after_warmup),
            "contact_parameter_l2": float(contact_parameter_l2),
            "trajectory_stability_weight": float(trajectory_stability_weight),
            "max_dynamic_displacement": float(max_dynamic_displacement),
            "best_loss_restored": best_parameters is not None,
            "best_loss": None if best_parameters is None else float(best_loss),
            "best_state_scope": "joint_physics_only",
            "gradient_clip_norm": 10.0,
            "learn_contact_parameters": bool(learn_contact_parameters),
            "physics_override_applied": bool(physics_override),
            "contact_parameterization": "log_positive",
            "stiffness_lr": float(lr if stiffness_lr is None else stiffness_lr),
            "damping_lr": float(lr if damping_lr is None else damping_lr),
            "friction_lr": float(lr if friction_lr is None else friction_lr),
            "estimated_contact_frames": estimated_contact_frames,
            "contact_curriculum_frames": int(contact_curriculum_frames),
            "silhouette_weight": 0.0 if paper_supervision else float(silhouette_weight),
            "silhouette_false_positive_weight": float(silhouette_fp_weight),
            "silhouette_false_negative_weight": float(silhouette_fn_weight),
            "gradient_attribution": bool(gradient_attribution),
            "gradient_attribution_interval": int(gradient_attribution_interval),
        },
        "image_loss_config": {
            "requested_type": str(image_loss),
            "type": str(effective_image_loss),
            "full_image": bool(paper_supervision),
            "gt_mask_used_for_loss": not paper_supervision and effective_mask_dir is not None,
            "requested_loftr_weight": float(loftr_weight),
            "loftr_weight": float(effective_loftr_weight),
            "loftr_pretrained": str(loftr_pretrained),
            "loftr_confidence_threshold": float(loftr_confidence_threshold),
            "loftr_max_matches": int(loftr_max_matches),
            "loftr_min_matches": int(loftr_min_matches),
            "loftr_patch_radius": int(loftr_patch_radius),
        },
        "initial_state_prefit": prefit_report,
        "mass_inertia_learning": {
            "enabled": bool(learn_mass_inertia),
            "lr": float(lr if mass_inertia_lr is None else mass_inertia_lr),
            "mass_l2": float(mass_l2),
            "inertia_l2": float(inertia_l2),
            "parameterization": "positive_mass_and_triangle_constrained_principal_inertia",
        },
        "geometry_refinement": {
            "requested": bool(refine_geometry),
            "enabled": bool(effective_refine_geometry),
            "enabled_by_pipeline_mode": bool(paper_supervision and not refine_geometry),
            "requested_gradient_route": requested_geometry_gradient_route,
            "gradient_route": effective_geometry_gradient_route,
            "renderer_geometry_detached": bool(
                effective_refine_geometry and effective_geometry_gradient_route == "collision_only"
            ),
            "center_l2": float(geometry_center_l2),
            "radius_l2": float(geometry_radius_l2),
            "max_center_delta": float(geometry_max_center_delta),
            "max_log_radius_delta": float(geometry_max_log_radius_delta),
            "bodies": geometry_stats,
            "refined_collision_geometry": refined_geometry_payload,
        },
        "loss_history": history,
        "evaluation": {
            "rgb_l1": float(evaluation_l1.detach().cpu()),
            "rgb_mse": float(evaluation_mse.detach().cpu()),
            "rgb_psnr": float(evaluation_psnr.detach().cpu()),
            "foreground_l1": float(foreground_l1.detach().cpu()),
            "foreground_mse": float(foreground_mse.detach().cpu()),
            "foreground_psnr": float(foreground_psnr.detach().cpu()),
            "uses_foreground_mask": masks is not None,
            "num_frames": len(frame_indices),
            "trajectory": trajectory_evaluation,
        },
        "artifacts": artifacts,
        "render_gaussian_counts": {
            body.id: {
                "before": int(render_loss.unfiltered_gaussian_counts[index]),
                "after": int(render_loss.filtered_gaussian_counts[index]),
            }
            for index, body in enumerate(bodies)
        } if hasattr(render_loss, "unfiltered_gaussian_counts") else None,
        "learned_contact_pairs": [
            {
                "body_a": pair.body_a,
                "body_b": pair.body_b,
                "model": pair.model,
                "stiffness": float(F.softplus(dynamics.stiffness[index]).detach().cpu()),
                "damping": float(F.softplus(dynamics.damping[index]).detach().cpu()),
                "friction": float(F.softplus(dynamics.friction_coefficient[index]).detach().cpu()),
            }
            for index, pair in enumerate(manifest.contact_pairs)
        ],
        "learned_body_physics": {
            body.id: {
                "role": body.role,
                "learned": bool(learn_mass_inertia and body.role == "dynamic"),
                "initialization": (
                    "manifest" if "initial_diagonal" in body.physics.get("inertia", {})
                    else "gaussian_union_geometry" if body.role == "dynamic" and body.collision.type == "gaussian_union"
                    else "static_default"
                ),
                "initial_mass": float(F.softplus(initial_mass_raw[index]).detach().cpu()),
                "initial_inertia_diagonal": _effective_inertia(
                    initial_inertia_raw[index]
                ).detach().cpu().tolist(),
                "mass": float(F.softplus(dynamics.mass_parameters[index]).detach().cpu()),
                "inertia_diagonal": _effective_inertia(
                    dynamics.inertia_parameters[index]
                ).detach().cpu().tolist(),
            }
            for index, body in enumerate(bodies)
        },
        "initial_states": {body.id: state.to_serializable() for body, state in zip(bodies, fitted_states)},
        "sampled_trajectory": [
            {"frame": frame, "bodies": {
                body.id: {"position": positions[index, body_index].detach().cpu().tolist(),
                          "quaternion_wxyz": quaternions[index, body_index].detach().cpu().tolist()}
                for body_index, body in enumerate(bodies)}}
            for index, frame in enumerate(frame_indices)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", default=60, type=int)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--fit_iters", default=0, type=int)
    parser.add_argument(
        "--pipeline_mode", default=None,
        choices=tuple(mode.value for mode in Stage2PipelineMode),
    )
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--render_stride", default=1, type=int)
    parser.add_argument("--render_max_frames", default=0, type=int)
    parser.add_argument("--render_width", default=160, type=int)
    parser.add_argument("--render_height", default=120, type=int)
    parser.add_argument("--image_loss", default="l1", choices=("l1", "mse", "l1_ssim", "l1_loftr"))
    parser.add_argument("--loftr_weight", default=0.1, type=float)
    parser.add_argument("--loftr_pretrained", default="outdoor", choices=("outdoor", "indoor"))
    parser.add_argument("--loftr_confidence_threshold", default=0.2, type=float)
    parser.add_argument("--loftr_max_matches", default=1024, type=int)
    parser.add_argument("--loftr_min_matches", default=8, type=int)
    parser.add_argument("--loftr_patch_radius", default=2, type=int)
    parser.add_argument("--render_output_dir", default=None, type=Path)
    parser.add_argument("--prefit_initial_state", action="store_true")
    parser.add_argument("--prefit_pose_iters", default=100, type=int)
    parser.add_argument("--prefit_velocity_iters", default=100, type=int)
    parser.add_argument("--prefit_velocity_frames", default=3, type=int)
    parser.add_argument("--prefit_lr", default=0.01, type=float)
    parser.add_argument("--prefit_velocity_l2", default=1e-4, type=float)
    parser.add_argument("--refine_geometry", action="store_true")
    parser.add_argument("--geometry_lr", default=1e-3, type=float)
    parser.add_argument("--geometry_center_l2", default=1e-3, type=float)
    parser.add_argument("--geometry_radius_l2", default=1e-3, type=float)
    parser.add_argument("--geometry_max_center_delta", default=0.01, type=float)
    parser.add_argument("--geometry_max_log_radius_delta", default=0.2, type=float)
    parser.add_argument("--freeze_mass_inertia", action="store_true")
    parser.add_argument("--mass_inertia_lr", default=None, type=float)
    parser.add_argument("--mass_l2", default=1e-4, type=float)
    parser.add_argument("--inertia_l2", default=1e-4, type=float)
    parser.add_argument("--temporal_window_frames", default=0, type=int)
    parser.add_argument("--temporal_window_step", default=1, type=int)
    parser.add_argument("--physics_warmup_fraction", default=0.2, type=float)
    parser.add_argument(
        "--keep_initial_state_trainable_after_warmup", action="store_true",
        help="Disable the default initial-state freeze after physics warm-up.",
    )
    parser.add_argument("--contact_parameter_l2", default=1e-4, type=float)
    parser.add_argument("--trajectory_stability_weight", default=1e-4, type=float)
    parser.add_argument("--max_dynamic_displacement", default=3.0, type=float)
    parser.add_argument("--stiffness_lr", default=None, type=float)
    parser.add_argument("--damping_lr", default=None, type=float)
    parser.add_argument("--friction_lr", default=None, type=float)
    parser.add_argument("--contact_curriculum_frames", default=8, type=int)
    parser.add_argument("--silhouette_weight", default=0.005, type=float)
    parser.add_argument("--silhouette_fp_weight", default=2.0, type=float)
    parser.add_argument("--silhouette_fn_weight", default=1.0, type=float)
    parser.add_argument("--gradient_attribution", action="store_true")
    parser.add_argument("--gradient_attribution_interval", default=1, type=int)
    parser.add_argument(
        "--geometry_gradient_route", default="collision_only",
        choices=("collision_only", "collision_and_render"),
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.prefit_initial_state and args.render_max_frames == 1:
        parser.error("initial-state prefit needs at least two rendered frames")
    manifest = load_scene_manifest(args.manifest)
    device = torch.device(args.device)
    result = (
        fit_native_image_only(
            manifest, fit_iters=args.fit_iters, lr=args.lr, stride=args.render_stride,
            max_frames=args.render_max_frames, width=args.render_width, height=args.render_height,
            image_loss=args.image_loss, device=device, render_output_dir=args.render_output_dir,
            prefit_initial_state=args.prefit_initial_state,
            prefit_pose_iters=args.prefit_pose_iters,
            prefit_velocity_iters=args.prefit_velocity_iters,
            prefit_velocity_frames=args.prefit_velocity_frames,
            prefit_lr=args.prefit_lr, prefit_velocity_l2=args.prefit_velocity_l2,
            refine_geometry=args.refine_geometry, geometry_lr=args.geometry_lr,
            geometry_center_l2=args.geometry_center_l2,
            geometry_radius_l2=args.geometry_radius_l2,
            geometry_max_center_delta=args.geometry_max_center_delta,
            geometry_max_log_radius_delta=args.geometry_max_log_radius_delta,
            geometry_gradient_route=args.geometry_gradient_route,
            learn_mass_inertia=not args.freeze_mass_inertia,
            mass_inertia_lr=args.mass_inertia_lr,
            mass_l2=args.mass_l2, inertia_l2=args.inertia_l2,
            temporal_window_frames=args.temporal_window_frames,
            temporal_window_step=args.temporal_window_step,
            physics_warmup_fraction=args.physics_warmup_fraction,
            freeze_initial_state_after_warmup=(
                not args.keep_initial_state_trainable_after_warmup
            ),
            contact_parameter_l2=args.contact_parameter_l2,
            trajectory_stability_weight=args.trajectory_stability_weight,
            max_dynamic_displacement=args.max_dynamic_displacement,
            stiffness_lr=args.stiffness_lr, damping_lr=args.damping_lr,
            friction_lr=args.friction_lr,
            contact_curriculum_frames=args.contact_curriculum_frames,
            silhouette_weight=args.silhouette_weight,
            silhouette_fp_weight=args.silhouette_fp_weight,
            silhouette_fn_weight=args.silhouette_fn_weight,
            gradient_attribution=args.gradient_attribution,
            gradient_attribution_interval=args.gradient_attribution_interval,
            pipeline_mode=args.pipeline_mode,
            loftr_weight=args.loftr_weight, loftr_pretrained=args.loftr_pretrained,
            loftr_confidence_threshold=args.loftr_confidence_threshold,
            loftr_max_matches=args.loftr_max_matches,
            loftr_min_matches=args.loftr_min_matches,
            loftr_patch_radius=args.loftr_patch_radius,
        )
        if args.fit_iters > 0 or args.prefit_initial_state
        else rollout_manifest(
            manifest, steps=args.steps, device=device, pipeline_mode=args.pipeline_mode
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "frames": len(result.get("frames", result.get("sampled_trajectory", []))),
                      "fit_iterations": args.fit_iters,
                      "body_ids": result["manifest"]["body_ids"] + result["manifest"]["environment_ids"]}, indent=2))


if __name__ == "__main__":
    main()
