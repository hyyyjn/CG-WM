from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from .differentiable_collision_detection import (
    CollisionEngineConfig,
    DifferentiableCollisionEngine,
    GaussianCollisionBody,
    BodyPairContacts,
    PlaneCollider,
    detect_gaussian_union_contacts,
    detect_plane_contacts,
    detect_sphere_floor_contacts,
)


@dataclass(frozen=True)
class RigidState:
    """Translational rigid state for the first Stage 2 smoke test."""

    position: torch.Tensor
    linear_velocity: torch.Tensor

    def to_serializable(self) -> dict[str, list[float]]:
        return {
            "position": self.position.detach().cpu().tolist(),
            "linear_velocity": self.linear_velocity.detach().cpu().tolist(),
        }


@dataclass(frozen=True)
class RigidBodyState:
    """Rigid body state with translation and quaternion orientation."""

    position: torch.Tensor
    quaternion_wxyz: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor

    def to_serializable(self) -> dict[str, list[float]]:
        return {
            "position": self.position.detach().cpu().tolist(),
            "quaternion_wxyz": self.quaternion_wxyz.detach().cpu().tolist(),
            "linear_velocity": self.linear_velocity.detach().cpu().tolist(),
            "angular_velocity": self.angular_velocity.detach().cpu().tolist(),
        }


@dataclass(frozen=True)
class ContactDynamicsConfig:
    dt: float = 1.0 / 60.0
    mass: float = 1.0
    restitution: float = 0.0
    acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)
    contact_softness: float = 1e-3
    contact_gate_softness: float = 2e-3
    smooth_max_temperature: float = 1e-2
    position_slop: float = 1e-4


def _as_vec3(values: Iterable[float], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(list(values), dtype=dtype, device=device)
    if tensor.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got shape {tuple(tensor.shape)}.")
    return tensor


def smooth_weighted_max(values: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    weights = torch.softmax(values / temperature, dim=-1)
    return torch.sum(weights * values, dim=-1)


class ComplementarityFreeContactDynamics:
    """A small differentiable contact integrator for floor collisions.

    The integrator predicts constant-velocity motion, detects soft plane
    penetration at query points, and applies smooth normal position/velocity
    corrections. It avoids LCP/complementarity branching, which keeps the demo
    differentiable end-to-end.
    """

    def __init__(
        self,
        collider: PlaneCollider,
        local_query_points: torch.Tensor,
        config: Optional[ContactDynamicsConfig] = None,
    ):
        self.collider = collider
        self.local_query_points = local_query_points
        self.config = config or ContactDynamicsConfig()

    def step(self, state: RigidState) -> tuple[RigidState, dict[str, torch.Tensor]]:
        cfg = self.config
        if cfg.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if cfg.mass <= 0.0:
            raise ValueError("mass must be positive.")

        acceleration = _as_vec3(
            cfg.acceleration,
            dtype=state.position.dtype,
            device=state.position.device,
        )
        predicted_velocity = state.linear_velocity + acceleration * cfg.dt
        predicted_position = state.position + predicted_velocity * cfg.dt
        contacts = detect_plane_contacts(
            predicted_position,
            self.local_query_points,
            self.collider,
            softness=cfg.contact_softness,
        )

        penetration_depth = smooth_weighted_max(contacts.penetrations, cfg.smooth_max_temperature)
        contact_gate = smooth_weighted_max(contacts.contact_weights, cfg.smooth_max_temperature)
        normal = contacts.collider_normal.to(dtype=state.position.dtype, device=state.position.device)

        normal_velocity = torch.sum(predicted_velocity * normal, dim=-1)
        closing_speed = F.softplus(-normal_velocity / cfg.contact_gate_softness) * cfg.contact_gate_softness
        velocity_delta = contact_gate * (1.0 + cfg.restitution) * closing_speed * normal
        corrected_velocity = predicted_velocity + velocity_delta / cfg.mass

        position_delta = contact_gate * (penetration_depth + cfg.position_slop) * normal
        corrected_position = predicted_position + position_delta

        diagnostics = {
            "contact_gate": contact_gate,
            "max_penetration": contacts.max_penetration,
            "mean_penetration": torch.mean(contacts.penetrations),
            "smooth_penetration": penetration_depth,
            "min_signed_distance": contacts.min_signed_distance,
            "normal_velocity": normal_velocity,
        }
        return RigidState(corrected_position, corrected_velocity), diagnostics


def rollout(
    initial_state: RigidState,
    dynamics: ComplementarityFreeContactDynamics,
    num_steps: int,
) -> tuple[list[RigidState], list[dict[str, torch.Tensor]]]:
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1.")
    states = [initial_state]
    diagnostics = []
    state = initial_state
    for _ in range(num_steps):
        state, step_diagnostics = dynamics.step(state)
        states.append(state)
        diagnostics.append(step_diagnostics)
    return states, diagnostics


class SphereFloorQueryContactDynamics:
    """Sphere/floor contact dynamics using environment-side floor query points."""

    def __init__(
        self,
        collider: PlaneCollider,
        floor_query_offsets_xy: torch.Tensor,
        *,
        radius: float,
        config: Optional[ContactDynamicsConfig] = None,
    ):
        self.collider = collider
        self.floor_query_offsets_xy = floor_query_offsets_xy
        self.radius = float(radius)
        self.config = config or ContactDynamicsConfig()

    def step(self, state: RigidState) -> tuple[RigidState, dict[str, torch.Tensor]]:
        cfg = self.config
        acceleration = _as_vec3(
            cfg.acceleration,
            dtype=state.position.dtype,
            device=state.position.device,
        )
        predicted_velocity = state.linear_velocity + acceleration * cfg.dt
        predicted_position = state.position + predicted_velocity * cfg.dt
        contacts = detect_sphere_floor_contacts(
            predicted_position,
            self.floor_query_offsets_xy,
            self.collider,
            radius=self.radius,
            softness=cfg.contact_softness,
        )

        penetration_depth = smooth_weighted_max(contacts.penetrations, cfg.smooth_max_temperature)
        contact_gate = smooth_weighted_max(contacts.contact_weights, cfg.smooth_max_temperature)
        normal = contacts.collider_normal.to(dtype=state.position.dtype, device=state.position.device)
        normal_velocity = torch.sum(predicted_velocity * normal, dim=-1)
        closing_speed = F.softplus(-normal_velocity / cfg.contact_gate_softness) * cfg.contact_gate_softness
        corrected_velocity = predicted_velocity + contact_gate * (1.0 + cfg.restitution) * closing_speed * normal
        corrected_position = predicted_position + contact_gate * (penetration_depth + cfg.position_slop) * normal
        diagnostics = {
            "contact_gate": contact_gate,
            "max_penetration": contacts.max_penetration,
            "mean_penetration": torch.mean(contacts.penetrations),
            "smooth_penetration": penetration_depth,
            "min_signed_distance": contacts.min_signed_distance,
            "normal_velocity": normal_velocity,
            "contact_point": contacts.contact_point,
        }
        return RigidState(corrected_position, corrected_velocity), diagnostics


@dataclass(frozen=True)
class ImpedanceContactDynamicsConfig:
    """Per-step config for the paper's impedance-form contact dynamics.

    The contact force is ``λ = SoftPlus(-K·(h·J̃·b + ϕ̃) - D·(J̃·b))`` where
    ``b = v + h·M⁻¹·τ`` carries the non-contact forces (gravity here),
    ``J̃·b`` is the velocity in the contact normal direction (paper III-D-2,
    Appendix C, with a single contact pair and no friction), and ``ϕ̃`` is the
    aggregated signed distance over the active query points.

    Stiffness K and damping D live in :class:`ImpedanceFloorContactDynamics`
    itself so they can be ``nn.Parameter``s in a fit loop.
    """

    dt: float = 1.0 / 60.0
    mass: float = 1.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    contact_softness: float = 1e-3
    smooth_min_temperature: float = 1e-2
    inside_penalty: float = 0.02
    inside_sharpness: float = 50.0
    query_radius_floor: float = 0.0


@dataclass(frozen=True)
class PairwiseImpedanceDynamicsConfig:
    """Multi-contact impedance dynamics config for a pair of Gaussian bodies."""

    dt: float = 1.0 / 60.0
    mass_a: float = 1.0
    mass_b: float = 1.0
    inertia_diag_a: tuple[float, float, float] = (1.0, 1.0, 1.0)
    inertia_diag_b: tuple[float, float, float] = (1.0, 1.0, 1.0)
    inertia_matrix_a: Optional[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = None
    inertia_matrix_b: Optional[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = None
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    dynamic_a: bool = True
    dynamic_b: bool = True
    kinematic_b: bool = False
    contact_softness: float = 1e-3
    smooth_min_temperature: float = 1e-2
    inside_penalty: float = 0.02
    inside_sharpness: float = 50.0
    num_contact_patches: int = 4
    broad_phase_margin: float = 0.02
    broad_phase_mode: str = "sphere"
    patch_selection: str = "spatial"
    body_query_scheme: str = "axis6"
    body_query_directions: int = 6
    floor_query_radius_scale: float = 1.1
    floor_query_rings: int = 5
    floor_query_angles: int = 32
    linear_damping: float = 0.0
    angular_damping: float = 0.0
    friction_coefficient: float = 0.0
    static_friction_coefficient: Optional[float] = None
    tangential_damping: float = 0.0
    friction_softness: float = 1e-6
    friction_transition_velocity: float = 1e-3
    contact_model: str = "dual_cone"
    dual_cone_directions: int = 4


@dataclass(frozen=True)
class MultiBodyImpedanceDynamicsConfig:
    """N-body Gaussian impedance dynamics using the Stage 2 contact graph."""

    dt: float = 1.0 / 60.0
    masses: tuple[float, ...] | None = None
    inertia_diags: tuple[tuple[float, float, float], ...] | None = None
    generalized_damping: tuple[float, ...] | None = None
    dynamic_flags: tuple[bool, ...] | None = None
    kinematic_flags: tuple[bool, ...] | None = None
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    contact_softness: float = 1e-3
    smooth_min_temperature: float = 1e-2
    inside_penalty: float = 0.02
    inside_sharpness: float = 50.0
    plane_fixed_penetration: bool = False
    num_contact_patches: int = 4
    broad_phase_margin: float = 0.02
    broad_phase_mode: str = "aabb"
    patch_selection: str = "spatial"
    candidate_pair_mode: str = "spatial_hash"
    spatial_hash_cell_size: float = 0.15
    contact_threshold: float = 0.5
    linear_damping: float = 0.0
    angular_damping: float = 0.0
    friction_coefficient: float = 0.0
    static_friction_coefficient: Optional[float] = None
    tangential_damping: float = 0.0
    friction_softness: float = 1e-6
    friction_transition_velocity: float = 1e-3
    contact_model: str = "dual_cone"
    dual_cone_directions: int = 4
    paper_closed_form_contact: bool = False


@dataclass(frozen=True)
class GaussianPlaneContactPair:
    """One Gaussian rigid body contacting an analytic static plane."""

    body_index: int
    plane_index: int
    collider: PlaneCollider
    parameter_index: int


class GaussianUnionFloorContactDynamics:
    """Contact dynamics using floor queries against spherical Gaussian collision geometry."""

    def __init__(
        self,
        collider: PlaneCollider,
        floor_query_offsets_xy: torch.Tensor,
        local_gaussian_centers: torch.Tensor,
        gaussian_radii: torch.Tensor,
        *,
        world_rotation: Optional[torch.Tensor] = None,
        config: Optional[ContactDynamicsConfig] = None,
    ):
        self.collider = collider
        self.floor_query_offsets_xy = floor_query_offsets_xy
        self.local_gaussian_centers = local_gaussian_centers
        self.gaussian_radii = gaussian_radii
        self.world_rotation = world_rotation
        self.config = config or ContactDynamicsConfig()

    def step(self, state: RigidState) -> tuple[RigidState, dict[str, torch.Tensor]]:
        cfg = self.config
        acceleration = _as_vec3(cfg.acceleration, dtype=state.position.dtype, device=state.position.device)
        predicted_velocity = state.linear_velocity + acceleration * cfg.dt
        predicted_position = state.position + predicted_velocity * cfg.dt
        collider = self.collider.on_like(predicted_position)

        offsets = self.floor_query_offsets_xy.to(dtype=state.position.dtype, device=state.position.device)
        floor_points = torch.cat(
            (
                predicted_position[:2].unsqueeze(0) + offsets,
                torch.full((offsets.shape[0], 1), collider.height, dtype=state.position.dtype, device=state.position.device),
            ),
            dim=-1,
        )
        local = self.local_gaussian_centers.to(dtype=state.position.dtype, device=state.position.device)
        if self.world_rotation is not None:
            local = local @ self.world_rotation.to(dtype=local.dtype, device=local.device).T
        gaussian_centers = local + predicted_position.unsqueeze(0)
        contacts = detect_gaussian_union_contacts(
            floor_points,
            gaussian_centers,
            self.gaussian_radii,
            collider.normal,
            softness=cfg.contact_softness,
        )

        penetration_depth = smooth_weighted_max(contacts.penetrations, cfg.smooth_max_temperature)
        contact_gate = smooth_weighted_max(contacts.contact_weights, cfg.smooth_max_temperature)
        normal = contacts.collider_normal.to(dtype=state.position.dtype, device=state.position.device)
        normal_velocity = torch.sum(predicted_velocity * normal, dim=-1)
        closing_speed = F.softplus(-normal_velocity / cfg.contact_gate_softness) * cfg.contact_gate_softness
        corrected_velocity = predicted_velocity + contact_gate * (1.0 + cfg.restitution) * closing_speed * normal
        corrected_position = predicted_position + contact_gate * (penetration_depth + cfg.position_slop) * normal
        diagnostics = {
            "contact_gate": contact_gate,
            "max_penetration": contacts.max_penetration,
            "mean_penetration": torch.mean(contacts.penetrations),
            "smooth_penetration": penetration_depth,
            "min_signed_distance": contacts.min_signed_distance,
            "normal_velocity": normal_velocity,
            "contact_point": contacts.contact_point,
            "contact_normal": normal,
        }
        return RigidState(corrected_position, corrected_velocity), diagnostics


def _smooth_min(values: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    return -temperature * torch.logsumexp(-values / temperature, dim=-1)


def _normalize_quaternion(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    return quaternion_wxyz / torch.clamp(torch.linalg.norm(quaternion_wxyz, dim=-1, keepdim=True), min=1e-12)


def _quat_mul_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = lhs.unbind(dim=-1)
    w2, x2, y2, z2 = rhs.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _integrate_quaternion_wxyz(quaternion_wxyz: torch.Tensor, angular_velocity: torch.Tensor, dt: float) -> torch.Tensor:
    zeros = torch.zeros_like(angular_velocity[..., :1])
    omega_quat = torch.cat((zeros, angular_velocity), dim=-1)
    q_dot = 0.5 * _quat_mul_wxyz(omega_quat, quaternion_wxyz)
    return _normalize_quaternion(quaternion_wxyz + float(dt) * q_dot)


def _cross(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return torch.cross(lhs, rhs, dim=-1)


def _quat_rotate_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion = _normalize_quaternion(quaternion)
    scalar = quaternion[..., :1]
    imaginary = quaternion[..., 1:]
    twice_cross = 2.0 * _cross(imaginary, vector)
    return vector + scalar * twice_cross + _cross(imaginary, twice_cross)


def _body_inertia_angular_acceleration(
    torque_world: torch.Tensor,
    angular_velocity_world: torch.Tensor,
    quaternion_wxyz: torch.Tensor,
    inertia_diag_body: torch.Tensor,
) -> torch.Tensor:
    """Euler rigid-body equation for a body-frame diagonal inertia tensor."""
    inverse_quaternion = torch.cat(
        (quaternion_wxyz[..., :1], -quaternion_wxyz[..., 1:]), dim=-1
    )
    torque_body = _quat_rotate_wxyz(inverse_quaternion, torque_world)
    omega_body = _quat_rotate_wxyz(inverse_quaternion, angular_velocity_world)
    angular_momentum_body = inertia_diag_body * omega_body
    coriolis_body = _cross(omega_body, angular_momentum_body)
    acceleration_body = _safe_inverse_vec(inertia_diag_body) * (
        torque_body - coriolis_body
    )
    return _quat_rotate_wxyz(quaternion_wxyz, acceleration_body)


def _safe_inverse_vec(values: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.clamp(values, min=1e-12)


def _inertia_angular_delta(
    torque: torch.Tensor,
    *,
    inertia_diag: Iterable[float],
    inertia_matrix: Optional[Iterable[Iterable[float]]],
    dynamic: bool,
) -> torch.Tensor:
    if not dynamic:
        return torch.zeros_like(torque)
    if inertia_matrix is None:
        if isinstance(inertia_diag, torch.Tensor):
            inertia = inertia_diag.to(dtype=torque.dtype, device=torque.device)
            if inertia.shape != (3,):
                raise ValueError(f"inertia_diag must have shape (3,), got {tuple(inertia.shape)}.")
        else:
            inertia = _as_vec3(inertia_diag, dtype=torque.dtype, device=torque.device)
        return _safe_inverse_vec(inertia) * torque

    matrix = torch.as_tensor(inertia_matrix, dtype=torque.dtype, device=torque.device)
    if matrix.shape != (3, 3):
        raise ValueError(f"inertia_matrix must have shape (3, 3), got {tuple(matrix.shape)}.")
    regularizer = torch.eye(3, dtype=torque.dtype, device=torque.device) * 1e-9
    return torch.linalg.solve(matrix + regularizer, torque.unsqueeze(-1)).squeeze(-1)


def _tangent_basis(normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    abs_normals = torch.abs(normals)
    use_x = abs_normals[..., 0] < 0.9
    ref_x = torch.zeros_like(normals)
    ref_x[..., 0] = 1.0
    ref_y = torch.zeros_like(normals)
    ref_y[..., 1] = 1.0
    reference = torch.where(use_x.unsqueeze(-1), ref_x, ref_y)
    tangent_1 = _cross(normals, reference)
    tangent_1 = tangent_1 / torch.clamp(torch.linalg.norm(tangent_1, dim=-1, keepdim=True), min=1e-12)
    tangent_2 = _cross(normals, tangent_1)
    tangent_2 = tangent_2 / torch.clamp(torch.linalg.norm(tangent_2, dim=-1, keepdim=True), min=1e-12)
    return tangent_1, tangent_2


def contact_jacobian_rows(
    directions: torch.Tensor,
    lever_a: torch.Tensor,
    lever_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return rigid contact Jacobian rows ``[d, r_a×d, -d, -r_b×d]``.

    Leading dimensions are preserved, so directions may contain patch and
    friction-facet axes.  Omitting ``lever_b`` builds a body/static-plane row.
    """
    angular_a = _cross(lever_a, directions)
    if lever_b is None:
        return torch.cat((directions, angular_a), dim=-1)
    angular_b = _cross(lever_b, directions)
    return torch.cat((directions, angular_a, -directions, -angular_b), dim=-1)


def rigid_inverse_mass_matrix(
    mass: torch.Tensor,
    inertia_diag_body: torch.Tensor,
    quaternion_wxyz: torch.Tensor,
    *,
    dynamic: bool,
) -> torch.Tensor:
    """6×6 world-frame inverse generalized mass for one rigid body."""
    dtype, device = mass.dtype, mass.device
    if not dynamic:
        return torch.zeros((6, 6), dtype=dtype, device=device)
    basis = torch.eye(3, dtype=dtype, device=device)
    world_axes = torch.stack([
        _quat_rotate_wxyz(quaternion_wxyz, basis[index]) for index in range(3)
    ], dim=-1)
    inverse_inertia = world_axes @ torch.diag(_safe_inverse_vec(inertia_diag_body)) @ world_axes.T
    result = torch.zeros((6, 6), dtype=dtype, device=device)
    result[:3, :3] = torch.eye(3, dtype=dtype, device=device) / torch.clamp(mass, min=1e-12)
    result[3:, 3:] = inverse_inertia
    return result


def contact_delassus_matrix(jacobian: torch.Tensor, inverse_mass: torch.Tensor) -> torch.Tensor:
    """Compute the coupled contact-space effective-mass matrix J M⁻¹ Jᵀ."""
    rows = jacobian.reshape(-1, jacobian.shape[-1])
    return rows @ inverse_mass @ rows.T


def _friction_cone_forces(
    tangential_velocity: torch.Tensor,
    weights: torch.Tensor,
    normal_lambdas: torch.Tensor,
    *,
    tangential_damping: torch.Tensor,
    dynamic_mu: torch.Tensor,
    static_mu: torch.Tensor | None,
    friction_softness: float,
    transition_velocity: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Soft Coulomb cone projection for tangential damping forces.

    Inside the cone, the raw damping force is kept nearly unchanged (sticking).
    Outside the cone, it is smoothly projected onto ``||f_t|| <= mu * lambda_n``
    along the opposite slip direction (sliding).
    """

    raw_friction = -tangential_damping * weights.unsqueeze(-1) * tangential_velocity
    raw_norm = torch.linalg.norm(raw_friction, dim=-1, keepdim=True)
    slip_speed = torch.linalg.norm(tangential_velocity, dim=-1, keepdim=True)
    if static_mu is None:
        effective_mu = dynamic_mu
        static_gate = torch.zeros_like(raw_norm)
    else:
        transition = max(float(transition_velocity), 1e-9)
        static_gate = torch.sigmoid((transition - slip_speed) / transition)
        effective_mu = dynamic_mu + (torch.clamp(static_mu, min=dynamic_mu) - dynamic_mu) * static_gate
    cone_radius = effective_mu * normal_lambdas.unsqueeze(-1)
    softness = max(float(friction_softness), 1e-12)
    sliding_gate = torch.sigmoid((raw_norm - cone_radius) / softness)
    projected_scale = cone_radius / torch.clamp(raw_norm, min=1e-12)
    scale = (1.0 - sliding_gate) + sliding_gate * projected_scale
    friction = raw_friction * scale
    diagnostics = {
        "raw_friction_force": raw_friction,
        "raw_friction_norm": raw_norm.squeeze(-1),
        "friction_cone_radius": cone_radius.squeeze(-1),
        "friction_cone_scale": scale.squeeze(-1),
        "friction_sliding_gate": sliding_gate.squeeze(-1),
        "friction_static_gate": static_gate.squeeze(-1),
        "effective_friction_coefficient": torch.as_tensor(effective_mu, dtype=raw_friction.dtype, device=raw_friction.device),
        "slip_speed": slip_speed.squeeze(-1),
    }
    return friction, diagnostics


class PairwiseGaussianBodyImpedanceDynamics:
    """Multi-contact impedance dynamics for two Gaussian collision bodies.

    This class consumes the `BodyPairContacts.patch_*` output from
    `DifferentiableCollisionEngine`. Each patch contributes a normal impedance
    force plus optional Coulomb-limited tangential damping to both linear and
    angular velocity. It is intended as the
    first rigid-body dynamics bridge for object-object contact; existing
    floor-only smoke tests keep using the older classes below.
    """

    def __init__(
        self,
        body_a: GaussianCollisionBody,
        body_b: GaussianCollisionBody,
        *,
        stiffness: torch.Tensor,
        damping: torch.Tensor,
        friction_coefficient: torch.Tensor | None = None,
        mass_a: torch.Tensor | None = None,
        inertia_diag_a: torch.Tensor | None = None,
        config: Optional[PairwiseImpedanceDynamicsConfig] = None,
    ):
        self.body_a = body_a
        self.body_b = body_b
        self.stiffness = stiffness
        self.damping = damping
        self.friction_coefficient = friction_coefficient
        self.mass_a = mass_a
        self.inertia_diag_a = inertia_diag_a
        self.config = config or PairwiseImpedanceDynamicsConfig()
        self.collision_engine = DifferentiableCollisionEngine(
            CollisionEngineConfig(
                softness=self.config.contact_softness,
                smooth_min_temperature=self.config.smooth_min_temperature,
                inside_penalty=self.config.inside_penalty,
                inside_sharpness=self.config.inside_sharpness,
                num_contact_patches=self.config.num_contact_patches,
                broad_phase_margin=self.config.broad_phase_margin,
                broad_phase_mode=self.config.broad_phase_mode,
                patch_selection=self.config.patch_selection,
                body_query_scheme=self.config.body_query_scheme,
                body_query_directions=self.config.body_query_directions,
                floor_query_radius_scale=self.config.floor_query_radius_scale,
                floor_query_rings=self.config.floor_query_rings,
                floor_query_angles=self.config.floor_query_angles,
            )
        )

    def _predict_free(
        self,
        state: RigidBodyState,
        *,
        mass: float,
        dynamic: bool,
        external_force: torch.Tensor | None = None,
        external_torque: torch.Tensor | None = None,
        kinematic: bool = False,
    ) -> RigidBodyState:
        cfg = self.config
        if cfg.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if mass <= 0.0 and dynamic:
            raise ValueError("dynamic bodies require positive mass.")

        gravity = _as_vec3(cfg.gravity, dtype=state.position.dtype, device=state.position.device)
        if dynamic:
            effective_mass = (
                torch.as_tensor(float(mass), dtype=state.position.dtype, device=state.position.device)
                if self.mass_a is None
                else F.softplus(self.mass_a).to(dtype=state.position.dtype, device=state.position.device)
            )
            force = (
                torch.zeros_like(state.position)
                if external_force is None
                else external_force.to(dtype=state.position.dtype, device=state.position.device)
            )
            torque = (
                torch.zeros_like(state.position)
                if external_torque is None
                else external_torque.to(dtype=state.position.dtype, device=state.position.device)
            )
            linear_velocity = state.linear_velocity + cfg.dt * (
                gravity + force / torch.clamp(effective_mass, min=1e-9)
            )
            linear_velocity = linear_velocity * max(0.0, 1.0 - float(cfg.linear_damping) * cfg.dt)
            effective_inertia = (
                torch.as_tensor(cfg.inertia_diag_a, dtype=state.position.dtype, device=state.position.device)
                if self.inertia_diag_a is None
                else F.softplus(self.inertia_diag_a).to(dtype=state.position.dtype, device=state.position.device)
            )
            angular_acceleration = _body_inertia_angular_acceleration(
                torque,
                state.angular_velocity,
                state.quaternion_wxyz,
                effective_inertia,
            )
            angular_velocity = state.angular_velocity + cfg.dt * angular_acceleration
            angular_velocity = angular_velocity * max(0.0, 1.0 - float(cfg.angular_damping) * cfg.dt)
            position = state.position + cfg.dt * linear_velocity
            quaternion = _integrate_quaternion_wxyz(state.quaternion_wxyz, angular_velocity, cfg.dt)
        elif kinematic:
            linear_velocity = state.linear_velocity
            angular_velocity = state.angular_velocity
            position = state.position
            quaternion = _normalize_quaternion(state.quaternion_wxyz)
        else:
            linear_velocity = torch.zeros_like(state.linear_velocity)
            angular_velocity = torch.zeros_like(state.angular_velocity)
            position = state.position
            quaternion = _normalize_quaternion(state.quaternion_wxyz)
        return RigidBodyState(position, quaternion, linear_velocity, angular_velocity)

    def _apply_patch_forces(
        self,
        state_a: RigidBodyState,
        state_b: RigidBodyState,
        contacts: BodyPairContacts,
    ) -> tuple[RigidBodyState, RigidBodyState, dict[str, torch.Tensor]]:
        cfg = self.config
        dtype = state_a.position.dtype
        device = state_a.position.device

        normals = contacts.patch_normals.to(dtype=dtype, device=device)
        points = contacts.patch_points.to(dtype=dtype, device=device)
        weights = contacts.patch_weights.to(dtype=dtype, device=device)
        phi = contacts.patch_signed_distances.to(dtype=dtype, device=device)

        r_a = points - state_a.position.unsqueeze(-2)
        r_b = points - state_b.position.unsqueeze(-2)
        velocity_a_at_patch = state_a.linear_velocity.unsqueeze(-2) + _cross(state_a.angular_velocity.unsqueeze(-2), r_a)
        velocity_b_at_patch = state_b.linear_velocity.unsqueeze(-2) + _cross(state_b.angular_velocity.unsqueeze(-2), r_b)
        relative_velocity = velocity_a_at_patch - velocity_b_at_patch
        normal_velocity = torch.sum(relative_velocity * normals, dim=-1)
        tangent_1, tangent_2 = _tangent_basis(normals)
        tangent_velocity_1 = torch.sum(relative_velocity * tangent_1, dim=-1)
        tangent_velocity_2 = torch.sum(relative_velocity * tangent_2, dim=-1)
        tangential_velocity = (
            tangent_velocity_1.unsqueeze(-1) * tangent_1
            + tangent_velocity_2.unsqueeze(-1) * tangent_2
        )

        K = F.softplus(self.stiffness).to(dtype=dtype, device=device)
        D = F.softplus(self.damping).to(dtype=dtype, device=device)
        mu = (
            torch.as_tensor(float(cfg.friction_coefficient), dtype=dtype, device=device)
            if self.friction_coefficient is None
            else F.softplus(self.friction_coefficient).to(dtype=dtype, device=device)
        )
        if cfg.contact_model == "dual_cone":
            num_directions = int(cfg.dual_cone_directions)
            if num_directions < 2:
                raise ValueError("dual_cone_directions must be at least 2.")
            angles = torch.arange(num_directions, dtype=dtype, device=device)
            angles = angles * (2.0 * torch.pi / float(num_directions))
            tangent_directions = (
                torch.cos(angles).reshape(1, -1, 1) * tangent_1.unsqueeze(-2)
                + torch.sin(angles).reshape(1, -1, 1) * tangent_2.unsqueeze(-2)
            )
            # Appendix C: each row of J~ is J^n - mu J^d.  Applying
            # J~^T lambda therefore produces normal and frictional force in
            # one closed-form term rather than projecting a separate force.
            dual_faces = normals.unsqueeze(-2) - mu * tangent_directions
            contact_jacobian = contact_jacobian_rows(
                dual_faces,
                r_a.unsqueeze(-2).expand_as(dual_faces),
                r_b.unsqueeze(-2).expand_as(dual_faces),
            )
            generalized_velocity = torch.cat((
                state_a.linear_velocity, state_a.angular_velocity,
                state_b.linear_velocity, state_b.angular_velocity,
            ))
            dual_velocity = torch.matmul(contact_jacobian, generalized_velocity)
            dual_phi = phi.unsqueeze(-1).expand_as(dual_velocity)
            lambda_raw = F.softplus(
                -K * (cfg.dt * dual_velocity + dual_phi) - D * dual_velocity
            )
            facet_lambdas = weights.unsqueeze(-1) * lambda_raw
            forces = torch.sum(facet_lambdas.unsqueeze(-1) * dual_faces, dim=-2)
            normal_forces = (
                torch.sum(facet_lambdas, dim=-1).unsqueeze(-1) * normals
            )
            friction_forces = forces - normal_forces
            lambdas = torch.sum(facet_lambdas, dim=-1)
            friction_diagnostics = {
                "dual_cone_faces": dual_faces,
                "dual_cone_velocity": dual_velocity,
                "dual_cone_lambda": facet_lambdas,
                "effective_friction_coefficient": mu,
            }
        elif cfg.contact_model == "projected":
            contact_jacobian = contact_jacobian_rows(normals, r_a, r_b)
            lambda_raw = F.softplus(
                -K * (cfg.dt * normal_velocity + phi) - D * normal_velocity
            )
            lambdas = weights * lambda_raw
            normal_forces = lambdas.unsqueeze(-1) * normals
            friction_forces, friction_diagnostics = _friction_cone_forces(
                tangential_velocity,
                weights,
                lambdas,
                tangential_damping=torch.as_tensor(float(cfg.tangential_damping), dtype=dtype, device=device),
                dynamic_mu=mu,
                static_mu=(
                    None
                    if cfg.static_friction_coefficient is None
                    else torch.as_tensor(float(cfg.static_friction_coefficient), dtype=dtype, device=device)
                ),
                friction_softness=float(cfg.friction_softness),
                transition_velocity=float(cfg.friction_transition_velocity),
            )
            forces = normal_forces + friction_forces
        else:
            raise ValueError(
                f"Unknown contact_model {cfg.contact_model!r}; expected 'dual_cone' or 'projected'."
            )
        total_force = torch.sum(forces, dim=-2)
        torque_a = torch.sum(_cross(r_a, forces), dim=-2)
        torque_b = torch.sum(_cross(r_b, -forces), dim=-2)

        effective_mass_a = (
            torch.as_tensor(float(cfg.mass_a), dtype=dtype, device=device)
            if self.mass_a is None
            else F.softplus(self.mass_a).to(dtype=dtype, device=device)
        )
        effective_inertia_a = (
            cfg.inertia_diag_a
            if self.inertia_diag_a is None
            else F.softplus(self.inertia_diag_a).to(dtype=dtype, device=device)
        )
        inv_mass_a = 0.0 if not cfg.dynamic_a else 1.0 / torch.clamp(effective_mass_a, min=1e-9)
        inv_mass_b = 0.0 if not cfg.dynamic_b else 1.0 / float(cfg.mass_b)
        angular_delta_a = _inertia_angular_delta(
            torque_a,
            inertia_diag=effective_inertia_a,
            inertia_matrix=cfg.inertia_matrix_a,
            dynamic=cfg.dynamic_a,
        )
        angular_delta_b = _inertia_angular_delta(
            torque_b,
            inertia_diag=cfg.inertia_diag_b,
            inertia_matrix=cfg.inertia_matrix_b,
            dynamic=cfg.dynamic_b,
        )
        inverse_mass_a = rigid_inverse_mass_matrix(
            effective_mass_a,
            torch.as_tensor(effective_inertia_a, dtype=dtype, device=device),
            state_a.quaternion_wxyz,
            dynamic=cfg.dynamic_a,
        )
        inverse_mass_b = rigid_inverse_mass_matrix(
            torch.as_tensor(float(cfg.mass_b), dtype=dtype, device=device),
            torch.as_tensor(cfg.inertia_diag_b, dtype=dtype, device=device),
            state_b.quaternion_wxyz,
            dynamic=cfg.dynamic_b,
        )
        inverse_mass = torch.block_diag(inverse_mass_a, inverse_mass_b)
        delassus = contact_delassus_matrix(contact_jacobian, inverse_mass)

        velocity_a = state_a.linear_velocity + cfg.dt * inv_mass_a * total_force
        velocity_b = state_b.linear_velocity - cfg.dt * inv_mass_b * total_force
        angular_a = state_a.angular_velocity + cfg.dt * angular_delta_a
        angular_b = state_b.angular_velocity + cfg.dt * angular_delta_b

        position_a = state_a.position + cfg.dt * (velocity_a - state_a.linear_velocity)
        position_b = state_b.position + cfg.dt * (velocity_b - state_b.linear_velocity)
        quaternion_a = _integrate_quaternion_wxyz(state_a.quaternion_wxyz, angular_a, cfg.dt)
        quaternion_b = _integrate_quaternion_wxyz(state_b.quaternion_wxyz, angular_b, cfg.dt)

        next_a = RigidBodyState(position_a, quaternion_a, velocity_a, angular_a)
        next_b = RigidBodyState(position_b, quaternion_b, velocity_b, angular_b)
        diagnostics = {
            "contacts": contacts,
            "lambda": lambdas,
            "lambda_raw": lambda_raw,
            "contact_model": cfg.contact_model,
            "normal_velocity": normal_velocity,
            "tangent_basis_1": tangent_1,
            "tangent_basis_2": tangent_2,
            "tangential_velocity": tangential_velocity,
            "tangent_velocity_components": torch.stack((tangent_velocity_1, tangent_velocity_2), dim=-1),
            "friction_force": friction_forces,
            **friction_diagnostics,
            "normal_force": normal_forces,
            "patch_force": forces,
            "patch_weights": weights,
            "patch_penetrations": contacts.patch_penetrations,
            "patch_signed_distances": phi,
            "total_force_on_a": total_force,
            "torque_on_a": torque_a,
            "torque_on_b": torque_b,
            "angular_delta_a": angular_delta_a,
            "angular_delta_b": angular_delta_b,
            "inertia_matrix_a": None if cfg.inertia_matrix_a is None else torch.as_tensor(cfg.inertia_matrix_a, dtype=dtype, device=device),
            "inertia_matrix_b": None if cfg.inertia_matrix_b is None else torch.as_tensor(cfg.inertia_matrix_b, dtype=dtype, device=device),
            "broad_phase_overlaps": contacts.broad_phase_overlaps,
            "effective_mass_a": effective_mass_a,
            "effective_inertia_diag_a": torch.as_tensor(
                effective_inertia_a, dtype=dtype, device=device
            ),
            "contact_jacobian": contact_jacobian,
            "inverse_generalized_mass": inverse_mass,
            "delassus_matrix": delassus,
            "contact_effective_mass": 1.0 / torch.clamp(torch.diagonal(delassus), min=1e-12),
        }
        return next_a, next_b, diagnostics

    def step(
        self,
        state_a: RigidBodyState,
        state_b: RigidBodyState,
        *,
        external_force_a: torch.Tensor | None = None,
        external_torque_a: torch.Tensor | None = None,
    ) -> tuple[RigidBodyState, RigidBodyState, dict[str, torch.Tensor]]:
        predicted_a = self._predict_free(
            state_a,
            mass=self.config.mass_a,
            dynamic=self.config.dynamic_a,
            external_force=external_force_a,
            external_torque=external_torque_a,
        )
        predicted_b = self._predict_free(
            state_b,
            mass=self.config.mass_b,
            dynamic=self.config.dynamic_b,
            kinematic=self.config.kinematic_b,
        )
        contacts = self.collision_engine.body_pair_contacts(
            self.body_a,
            predicted_a.position,
            self.body_b,
            predicted_b.position,
            quaternion_a_wxyz=predicted_a.quaternion_wxyz,
            quaternion_b_wxyz=predicted_b.quaternion_wxyz,
        )
        return self._apply_patch_forces(predicted_a, predicted_b, contacts)


class MultiBodyGaussianImpedanceDynamics:
    """Sparse N-body impedance dynamics over Gaussian collision bodies.

    This is the multi-object counterpart to
    :class:`PairwiseGaussianBodyImpedanceDynamics`: it predicts free motion once
    for every body, builds the Stage 2 pairwise contact graph, then accumulates
    the same impedance/friction patch forces over all active graph edges.
    """

    def __init__(
        self,
        bodies: Iterable[GaussianCollisionBody],
        *,
        stiffness: torch.Tensor,
        damping: torch.Tensor,
        friction_coefficient: torch.Tensor | None = None,
        tangential_damping: torch.Tensor | None = None,
        mass_multiplier: torch.Tensor | None = None,
        mass_parameters: torch.Tensor | None = None,
        inertia_parameters: torch.Tensor | None = None,
        names: Iterable[str] | None = None,
        candidate_pairs: Iterable[tuple[int, int]] | None = None,
        candidate_pair_parameter_indices: Iterable[int] | None = None,
        plane_contact_pairs: Iterable[GaussianPlaneContactPair] | None = None,
        config: Optional[MultiBodyImpedanceDynamicsConfig] = None,
    ):
        self.bodies = tuple(bodies)
        if len(self.bodies) < 2:
            raise ValueError("MultiBodyGaussianImpedanceDynamics needs at least two bodies.")
        self.names = None if names is None else tuple(str(name) for name in names)
        if self.names is not None and len(self.names) != len(self.bodies):
            raise ValueError(f"names must have length {len(self.bodies)}, got {len(self.names)}.")
        self.candidate_pairs = None if candidate_pairs is None else tuple(candidate_pairs)
        self.candidate_pair_parameter_indices = (
            None if candidate_pair_parameter_indices is None
            else tuple(int(index) for index in candidate_pair_parameter_indices)
        )
        if self.candidate_pair_parameter_indices is not None and (
            self.candidate_pairs is None
            or len(self.candidate_pair_parameter_indices) != len(self.candidate_pairs)
        ):
            raise ValueError("candidate_pair_parameter_indices must align with candidate_pairs")
        self.plane_contact_pairs = tuple(plane_contact_pairs or ())
        self.stiffness = stiffness
        self.damping = damping
        self.friction_coefficient = friction_coefficient
        self.tangential_damping = tangential_damping
        self.mass_multiplier = mass_multiplier
        self.mass_parameters = mass_parameters
        self.inertia_parameters = inertia_parameters
        self.config = config or MultiBodyImpedanceDynamicsConfig()

    def _edge_parameter(self, value: torch.Tensor, edge_pair: tuple[int, int], name: str) -> torch.Tensor:
        """Select a scalar shared parameter or the value aligned with candidate_pairs."""
        if value.ndim == 0 or value.numel() == 1:
            return value.reshape(())
        if self.candidate_pairs is None:
            raise ValueError(f"vector {name} requires explicit candidate_pairs")
        expected = len(self.candidate_pairs)
        if self.candidate_pair_parameter_indices is not None:
            expected = max(self.candidate_pair_parameter_indices + tuple(
                pair.parameter_index for pair in self.plane_contact_pairs
            ), default=-1) + 1
        if value.ndim != 1 or value.numel() != expected:
            raise ValueError(
                f"{name} must be scalar or have one value per candidate pair "
                f"({expected}), got shape {tuple(value.shape)}"
            )
        canonical = tuple(sorted(edge_pair))
        pair_to_index = {
            tuple(sorted((int(pair[0]), int(pair[1])))): index
            for index, pair in enumerate(self.candidate_pairs)
        }
        local_index = pair_to_index[canonical]
        parameter_index = (
            local_index if self.candidate_pair_parameter_indices is None
            else self.candidate_pair_parameter_indices[local_index]
        )
        return value[parameter_index]

    def _masses(self) -> tuple[float, ...]:
        if self.config.masses is None:
            return tuple(1.0 for _ in self.bodies)
        if len(self.config.masses) != len(self.bodies):
            raise ValueError(f"masses must have length {len(self.bodies)}, got {len(self.config.masses)}.")
        return tuple(float(mass) for mass in self.config.masses)

    def _inertia_diags(self) -> tuple[tuple[float, float, float], ...]:
        if self.config.inertia_diags is None:
            return tuple((1.0, 1.0, 1.0) for _ in self.bodies)
        if len(self.config.inertia_diags) != len(self.bodies):
            raise ValueError(
                f"inertia_diags must have length {len(self.bodies)}, got {len(self.config.inertia_diags)}."
            )
        return tuple(tuple(float(v) for v in inertia) for inertia in self.config.inertia_diags)

    def _dynamic_flags(self) -> tuple[bool, ...]:
        if self.config.dynamic_flags is None:
            return tuple(True for _ in self.bodies)
        if len(self.config.dynamic_flags) != len(self.bodies):
            raise ValueError(
                f"dynamic_flags must have length {len(self.bodies)}, got {len(self.config.dynamic_flags)}."
            )
        return tuple(bool(flag) for flag in self.config.dynamic_flags)

    def _kinematic_flags(self) -> tuple[bool, ...]:
        if self.config.kinematic_flags is None:
            return tuple(False for _ in self.bodies)
        if len(self.config.kinematic_flags) != len(self.bodies):
            raise ValueError(
                f"kinematic_flags must have length {len(self.bodies)}, "
                f"got {len(self.config.kinematic_flags)}."
            )
        return tuple(bool(flag) for flag in self.config.kinematic_flags)

    def _generalized_damping(self) -> tuple[float, ...]:
        if self.config.generalized_damping is None:
            return tuple(0.0 for _ in self.bodies)
        if len(self.config.generalized_damping) != len(self.bodies):
            raise ValueError(
                f"generalized_damping must have length {len(self.bodies)}, "
                f"got {len(self.config.generalized_damping)}"
            )
        values = tuple(float(value) for value in self.config.generalized_damping)
        if any(value < 0.0 for value in values):
            raise ValueError("generalized_damping values must be non-negative")
        return values

    def _predict_free(
        self,
        states: Iterable[RigidBodyState],
        masses: list[torch.Tensor],
        inertia_diags: list[torch.Tensor],
        generalized_damping: tuple[float, ...],
        dynamic_flags: tuple[bool, ...],
        kinematic_flags: tuple[bool, ...],
        external_wrenches: torch.Tensor | None = None,
    ) -> tuple[RigidBodyState, ...]:
        cfg = self.config
        states = tuple(states)
        if len(states) != len(self.bodies):
            raise ValueError(f"states must have length {len(self.bodies)}, got {len(states)}.")
        if cfg.dt <= 0.0:
            raise ValueError("dt must be positive.")
        predicted = []
        if external_wrenches is not None and external_wrenches.shape != (len(self.bodies), 6):
            raise ValueError("external_wrenches must have shape (num_bodies, 6)")
        for index, (state, mass, inertia_diag, damping, dynamic, kinematic) in enumerate(zip(
            states, masses, inertia_diags, generalized_damping, dynamic_flags, kinematic_flags
        )):
            if bool((mass <= 0.0).detach()) and dynamic:
                raise ValueError("dynamic bodies require positive mass.")
            gravity = _as_vec3(cfg.gravity, dtype=state.position.dtype, device=state.position.device)
            if dynamic:
                wrench = (
                    torch.zeros(6, dtype=state.position.dtype, device=state.position.device)
                    if external_wrenches is None else external_wrenches[index].to(
                        dtype=state.position.dtype, device=state.position.device
                    )
                )
                damping_tensor = torch.as_tensor(
                    damping, dtype=state.position.dtype, device=state.position.device
                )
                linear_acceleration = gravity + wrench[:3] / torch.clamp(mass, min=1e-9)
                angular_acceleration = _body_inertia_angular_acceleration(
                    wrench[3:], state.angular_velocity, state.quaternion_wxyz, inertia_diag
                )
                linear_velocity = state.linear_velocity + cfg.dt * linear_acceleration
                # MuJoCo integrates joint damping implicitly. This is essential
                # for small rigid-body inertias where explicit -d*v is unstable.
                linear_velocity = linear_velocity / (
                    1.0 + cfg.dt * damping_tensor / torch.clamp(mass, min=1e-9)
                )
                linear_velocity = linear_velocity * max(0.0, 1.0 - float(cfg.linear_damping) * cfg.dt)
                angular_velocity = state.angular_velocity + cfg.dt * angular_acceleration
                inverse_quaternion = torch.cat((
                    state.quaternion_wxyz[..., :1], -state.quaternion_wxyz[..., 1:]
                ), dim=-1)
                angular_velocity_body = _quat_rotate_wxyz(
                    inverse_quaternion, angular_velocity
                )
                angular_velocity_body = angular_velocity_body / (
                    1.0 + cfg.dt * damping_tensor / torch.clamp(inertia_diag, min=1e-9)
                )
                angular_velocity = _quat_rotate_wxyz(
                    state.quaternion_wxyz, angular_velocity_body
                )
                angular_velocity = angular_velocity * max(0.0, 1.0 - float(cfg.angular_damping) * cfg.dt)
                position = state.position + cfg.dt * linear_velocity
                quaternion = _integrate_quaternion_wxyz(state.quaternion_wxyz, angular_velocity, cfg.dt)
            elif kinematic:
                linear_velocity = state.linear_velocity
                angular_velocity = state.angular_velocity
                position = state.position
                quaternion = _normalize_quaternion(state.quaternion_wxyz)
            else:
                linear_velocity = torch.zeros_like(state.linear_velocity)
                angular_velocity = torch.zeros_like(state.angular_velocity)
                position = state.position
                quaternion = _normalize_quaternion(state.quaternion_wxyz)
            predicted.append(RigidBodyState(position, quaternion, linear_velocity, angular_velocity))
        return tuple(predicted)

    def step(
        self, states: Iterable[RigidBodyState], *, external_wrenches: torch.Tensor | None = None
    ) -> tuple[tuple[RigidBodyState, ...], dict[str, object]]:
        from .differentiable_contact_graph import build_pairwise_contact_graph

        cfg = self.config
        masses = self._masses()
        inertia_diags = self._inertia_diags()
        generalized_damping = self._generalized_damping()
        dynamic_flags = self._dynamic_flags()
        kinematic_flags = self._kinematic_flags()
        if any(dynamic and kinematic for dynamic, kinematic in zip(dynamic_flags, kinematic_flags)):
            raise ValueError("A body cannot be both dynamic and kinematic.")
        states = tuple(states)
        if not states:
            raise ValueError("states must not be empty")
        dtype = states[0].position.dtype
        device = states[0].position.device
        if self.mass_parameters is not None:
            if self.mass_parameters.shape != (len(self.bodies),):
                raise ValueError("mass_parameters must have one raw value per body")
            mass_tensors = list(F.softplus(self.mass_parameters).to(dtype=dtype, device=device))
        else:
            mass_scale = (
                torch.ones((), dtype=dtype, device=device)
                if self.mass_multiplier is None
                else F.softplus(self.mass_multiplier).to(dtype=dtype, device=device)
            )
            mass_tensors = [
                torch.as_tensor(float(mass), dtype=dtype, device=device) * (mass_scale if dynamic else 1.0)
                for mass, dynamic in zip(masses, dynamic_flags)
            ]
        if self.inertia_parameters is not None:
            if self.inertia_parameters.shape != (len(self.bodies), 3):
                raise ValueError("inertia_parameters must have shape (num_bodies, 3)")
            components = F.softplus(self.inertia_parameters).to(dtype=dtype, device=device)
            a, b, c = components.unbind(dim=-1)
            inertia_tensors = list(torch.stack((a + b, a + c, b + c), dim=-1))
        else:
            inertia_tensors = [
                torch.as_tensor(values, dtype=dtype, device=device) for values in inertia_diags
            ]
        predicted = self._predict_free(
            states, mass_tensors, inertia_tensors, generalized_damping,
            dynamic_flags, kinematic_flags,
            external_wrenches=external_wrenches,
        )
        names = self.names or tuple(f"body_{idx}" for idx in range(len(self.bodies)))
        graph = build_pairwise_contact_graph(
            self.bodies,
            predicted,
            names=names,
            dynamic_flags=dynamic_flags,
            candidate_pairs=self.candidate_pairs,
            candidate_pair_mode=cfg.candidate_pair_mode,
            spatial_hash_cell_size=float(cfg.spatial_hash_cell_size),
            collision_config=CollisionEngineConfig(
                softness=float(cfg.contact_softness),
                smooth_min_temperature=float(cfg.smooth_min_temperature),
                inside_penalty=float(cfg.inside_penalty),
                inside_sharpness=float(cfg.inside_sharpness),
                num_contact_patches=int(cfg.num_contact_patches),
                broad_phase_margin=float(cfg.broad_phase_margin),
                broad_phase_mode=str(cfg.broad_phase_mode),
                patch_selection=str(cfg.patch_selection),
            ),
            include_inactive=False,
            contact_threshold=float(cfg.contact_threshold),
        )

        force_accum = [torch.zeros(3, dtype=dtype, device=device) for _ in predicted]
        torque_accum = [torch.zeros(3, dtype=dtype, device=device) for _ in predicted]
        K_all = F.softplus(self.stiffness).to(dtype=dtype, device=device)
        D_all = F.softplus(self.damping).to(dtype=dtype, device=device)
        mu_all = (
            torch.as_tensor(float(cfg.friction_coefficient), dtype=dtype, device=device)
            if self.friction_coefficient is None
            else F.softplus(self.friction_coefficient).to(dtype=dtype, device=device)
        )
        tangential_damping = (
            torch.as_tensor(float(cfg.tangential_damping), dtype=dtype, device=device)
            if self.tangential_damping is None
            else F.softplus(self.tangential_damping).to(dtype=dtype, device=device)
        )
        static_mu = (
            None
            if cfg.static_friction_coefficient is None
            else torch.as_tensor(float(cfg.static_friction_coefficient), dtype=dtype, device=device)
        )
        active_edges = []
        lambda_terms = []
        friction_terms = []
        jacobian_terms = []
        for edge in graph.active_edges(contact_threshold=float(cfg.contact_threshold)):
            contacts = edge.contacts
            i, j = int(edge.body_i), int(edge.body_j)
            K = self._edge_parameter(K_all, (i, j), "stiffness")
            D = self._edge_parameter(D_all, (i, j), "damping")
            mu = self._edge_parameter(mu_all, (i, j), "friction_coefficient")
            state_i = predicted[i]
            state_j = predicted[j]
            normals = contacts.patch_normals.to(dtype=dtype, device=device)
            points = contacts.patch_points.to(dtype=dtype, device=device)
            weights = contacts.patch_weights.to(dtype=dtype, device=device)
            phi = contacts.patch_signed_distances.to(dtype=dtype, device=device)
            r_i = points - state_i.position.unsqueeze(-2)
            r_j = points - state_j.position.unsqueeze(-2)
            inverse_i = rigid_inverse_mass_matrix(
                mass_tensors[i], inertia_tensors[i], state_i.quaternion_wxyz,
                dynamic=dynamic_flags[i],
            )
            inverse_j = rigid_inverse_mass_matrix(
                mass_tensors[j], inertia_tensors[j], state_j.quaternion_wxyz,
                dynamic=dynamic_flags[j],
            )
            inverse_pair_mass = torch.block_diag(inverse_i, inverse_j)
            velocity_i = state_i.linear_velocity.unsqueeze(-2) + _cross(state_i.angular_velocity.unsqueeze(-2), r_i)
            velocity_j = state_j.linear_velocity.unsqueeze(-2) + _cross(state_j.angular_velocity.unsqueeze(-2), r_j)
            relative_velocity = velocity_i - velocity_j
            normal_velocity = torch.sum(relative_velocity * normals, dim=-1)
            tangent_1, tangent_2 = _tangent_basis(normals)
            tangent_velocity_1 = torch.sum(relative_velocity * tangent_1, dim=-1)
            tangent_velocity_2 = torch.sum(relative_velocity * tangent_2, dim=-1)
            tangential_velocity = tangent_velocity_1.unsqueeze(-1) * tangent_1 + tangent_velocity_2.unsqueeze(-1) * tangent_2
            if cfg.contact_model == "dual_cone":
                num_directions = int(cfg.dual_cone_directions)
                if num_directions < 2:
                    raise ValueError("dual_cone_directions must be at least 2.")
                angles = torch.arange(num_directions, dtype=dtype, device=device)
                angles = angles * (2.0 * torch.pi / float(num_directions))
                tangent_directions = (
                    torch.cos(angles).reshape(1, -1, 1) * tangent_1.unsqueeze(-2)
                    + torch.sin(angles).reshape(1, -1, 1) * tangent_2.unsqueeze(-2)
                )
                dual_faces = normals.unsqueeze(-2) - mu * tangent_directions
                contact_jacobian = contact_jacobian_rows(
                    dual_faces,
                    r_i.unsqueeze(-2).expand_as(dual_faces),
                    r_j.unsqueeze(-2).expand_as(dual_faces),
                )
                generalized_velocity = torch.cat((
                    state_i.linear_velocity, state_i.angular_velocity,
                    state_j.linear_velocity, state_j.angular_velocity,
                ))
                dual_velocity = torch.matmul(contact_jacobian, generalized_velocity)
                rhs = (
                    -K * (cfg.dt * dual_velocity + phi.unsqueeze(-1))
                    - D * dual_velocity
                ).reshape(-1)
                delassus = contact_delassus_matrix(contact_jacobian, inverse_pair_mass)
                implicit_matrix = torch.eye(
                    rhs.numel(), dtype=dtype, device=device
                ) + cfg.dt * (cfg.dt * K + D) * delassus
                lambda_raw = F.softplus(
                    torch.linalg.solve(implicit_matrix, rhs.unsqueeze(-1)).squeeze(-1)
                ).reshape_as(dual_velocity)
                facet_lambdas = weights.unsqueeze(-1) * lambda_raw
                patch_forces = torch.sum(
                    facet_lambdas.unsqueeze(-1) * dual_faces, dim=-2
                )
                lambdas = torch.sum(facet_lambdas, dim=-1)
                normal_forces = lambdas.unsqueeze(-1) * normals
                friction = patch_forces - normal_forces
                friction_diagnostics = {
                    "dual_cone_faces": dual_faces,
                    "dual_cone_velocity": dual_velocity,
                    "dual_cone_lambda": facet_lambdas,
                    "effective_friction_coefficient": mu,
                    "implicit_contact_matrix": implicit_matrix,
                }
            elif cfg.contact_model == "projected":
                contact_jacobian = contact_jacobian_rows(normals, r_i, r_j)
                lambda_raw = F.softplus(
                    -K * (cfg.dt * normal_velocity + phi) - D * normal_velocity
                )
                lambdas = weights * lambda_raw
                normal_forces = lambdas.unsqueeze(-1) * normals
                friction, friction_diagnostics = _friction_cone_forces(
                    tangential_velocity,
                    weights,
                    lambdas,
                    tangential_damping=tangential_damping,
                    dynamic_mu=mu,
                    static_mu=static_mu,
                    friction_softness=float(cfg.friction_softness),
                    transition_velocity=float(cfg.friction_transition_velocity),
                )
                patch_forces = normal_forces + friction
            else:
                raise ValueError(
                    f"Unknown contact_model {cfg.contact_model!r}; expected 'dual_cone' or 'projected'."
                )
            total_force_on_i = torch.sum(patch_forces, dim=-2)
            torque_on_i = torch.sum(_cross(r_i, patch_forces), dim=-2)
            torque_on_j = torch.sum(_cross(r_j, -patch_forces), dim=-2)
            force_accum[i] = force_accum[i] + total_force_on_i
            force_accum[j] = force_accum[j] - total_force_on_i
            torque_accum[i] = torque_accum[i] + torque_on_i
            torque_accum[j] = torque_accum[j] + torque_on_j
            active_edges.append((i, j))
            lambda_terms.append(lambdas)
            delassus = contact_delassus_matrix(contact_jacobian, inverse_pair_mass)
            jacobian_terms.append({
                "edge": (i, j),
                "jacobian": contact_jacobian,
                "delassus": delassus,
                "effective_mass": 1.0 / torch.clamp(torch.diagonal(delassus), min=1e-12),
            })
            friction_terms.append(
                {
                    "edge": (i, j),
                    "effective_stiffness": K,
                    "effective_damping": D,
                    "effective_friction_coefficient": mu,
                    "friction_force": friction,
                    "normal_force": normal_forces,
                    "tangential_velocity": tangential_velocity,
                    **friction_diagnostics,
                }
            )

        plane_terms = []
        for plane_pair in self.plane_contact_pairs:
            body_index = int(plane_pair.body_index)
            if not dynamic_flags[body_index]:
                continue
            state = predicted[body_index]
            body = self.bodies[body_index]
            collider = plane_pair.collider.on_like(state.position)
            normal = collider.normal / torch.clamp(torch.linalg.norm(collider.normal), min=1e-12)
            centers = body.world_centers(state.position, quaternion_wxyz=state.quaternion_wxyz)
            radii = body.radii.to(dtype=dtype, device=device)
            raw_signed_distances = torch.sum(centers * normal, dim=-1) - collider.height - radii
            if cfg.plane_fixed_penetration:
                from .differentiable_collision_detection import fixed_penetration_signed_distance
                signed_distances = fixed_penetration_signed_distance(
                    raw_signed_distances,
                    inside_penalty=float(cfg.inside_penalty),
                    inside_sharpness=float(cfg.inside_sharpness),
                )
            else:
                signed_distances = raw_signed_distances
            temperature = max(float(cfg.smooth_min_temperature), 1e-8)
            surface_points = centers - radii.unsqueeze(-1) * normal
            patch_count = min(max(1, int(cfg.num_contact_patches)), int(signed_distances.numel()))
            phi, patch_indices = torch.topk(signed_distances, k=patch_count, largest=False)
            patch_weights = torch.softmax(-phi / temperature, dim=0)
            contact_gate = torch.sigmoid(-phi / max(float(cfg.contact_softness), 1e-8))
            points = surface_points[patch_indices]
            levers = points - state.position.unsqueeze(0)
            point_velocity = state.linear_velocity.unsqueeze(0) + _cross(
                state.angular_velocity.unsqueeze(0), levers
            )
            normals = normal.unsqueeze(0).expand_as(points)
            normal_velocity = torch.sum(point_velocity * normals, dim=-1)
            tangent_velocity = point_velocity - normal_velocity.unsqueeze(-1) * normals
            parameter_index = int(plane_pair.parameter_index)
            K = F.softplus(self.stiffness.reshape(-1)[parameter_index])
            D = F.softplus(self.damping.reshape(-1)[parameter_index])
            mu = (
                torch.as_tensor(float(cfg.friction_coefficient), dtype=dtype, device=device)
                if self.friction_coefficient is None
                else F.softplus(self.friction_coefficient.reshape(-1)[parameter_index])
            )
            inverse_body_mass = rigid_inverse_mass_matrix(
                mass_tensors[body_index], inertia_tensors[body_index],
                state.quaternion_wxyz, dynamic=True,
            )
            if cfg.contact_model == "dual_cone":
                num_directions = int(cfg.dual_cone_directions)
                if num_directions < 2:
                    raise ValueError("dual_cone_directions must be at least 2.")
                tangent_1, tangent_2 = _tangent_basis(normals)
                angles = torch.arange(num_directions, dtype=dtype, device=device)
                angles = angles * (2.0 * torch.pi / float(num_directions))
                tangent_directions = (
                    torch.cos(angles).reshape(1, -1, 1) * tangent_1.unsqueeze(-2)
                    + torch.sin(angles).reshape(1, -1, 1) * tangent_2.unsqueeze(-2)
                )
                dual_faces = normals.unsqueeze(-2) - mu * tangent_directions
                plane_jacobian = contact_jacobian_rows(
                    dual_faces, levers.unsqueeze(-2).expand_as(dual_faces)
                )
                generalized_velocity = torch.cat((state.linear_velocity, state.angular_velocity))
                dual_velocity = torch.matmul(plane_jacobian, generalized_velocity)
                rhs = (
                    -K * (cfg.dt * dual_velocity + phi.unsqueeze(-1))
                    - D * dual_velocity
                ).reshape(-1)
                plane_delassus = contact_delassus_matrix(
                    plane_jacobian, inverse_body_mass
                )
                implicit_matrix = torch.eye(
                    rhs.numel(), dtype=dtype, device=device
                ) + cfg.dt * (cfg.dt * K + D) * plane_delassus
                lambda_raw = F.softplus(
                    torch.linalg.solve(implicit_matrix, rhs.unsqueeze(-1)).squeeze(-1)
                ).reshape_as(dual_velocity)
                facet_lambdas = patch_weights.unsqueeze(-1) * contact_gate.unsqueeze(-1) * lambda_raw
                patch_forces = torch.sum(
                    facet_lambdas.unsqueeze(-1) * dual_faces, dim=-2
                )
                normal_lambda = torch.sum(facet_lambdas, dim=-1)
                normal_forces = normal_lambda.unsqueeze(-1) * normals
                friction_forces = patch_forces - normal_forces
                friction_diagnostics = {
                    "dual_cone_faces": dual_faces,
                    "dual_cone_velocity": dual_velocity,
                    "dual_cone_lambda": facet_lambdas,
                }
            elif cfg.contact_model == "projected":
                normal_lambda = patch_weights * contact_gate * F.softplus(
                    -K * (cfg.dt * normal_velocity + phi) - D * normal_velocity
                )
                tangent_speed = torch.linalg.norm(tangent_velocity, dim=-1)
                friction_direction = -tangent_velocity / torch.clamp(
                    tangent_speed.unsqueeze(-1), min=float(cfg.friction_softness)
                )
                friction_magnitude = mu * normal_lambda * torch.tanh(
                    tangent_speed / max(float(cfg.friction_transition_velocity), 1e-8)
                )
                normal_forces = normal_lambda.unsqueeze(-1) * normals
                friction_forces = friction_magnitude.unsqueeze(-1) * friction_direction
                patch_forces = normal_forces + friction_forces
                plane_jacobian = contact_jacobian_rows(normals, levers)
                friction_diagnostics = {}
            else:
                raise ValueError(
                    f"Unknown contact_model {cfg.contact_model!r}; expected 'dual_cone' or 'projected'."
                )
            force = torch.sum(patch_forces, dim=0)
            torque = torch.sum(_cross(levers, patch_forces), dim=0)
            force_accum[body_index] = force_accum[body_index] + force
            torque_accum[body_index] = torque_accum[body_index] + torque
            plane_delassus = contact_delassus_matrix(plane_jacobian, inverse_body_mass)
            active_edges.append((body_index, int(plane_pair.plane_index)))
            lambda_terms.append(normal_lambda)
            plane_terms.append({
                "edge": (body_index, int(plane_pair.plane_index)),
                "parameter_index": parameter_index,
                "signed_distance": phi,
                "raw_signed_distance": raw_signed_distances[patch_indices],
                "fixed_penetration_enabled": bool(cfg.plane_fixed_penetration),
                "contact_gate": contact_gate,
                "contact_points": points,
                "contact_point": torch.sum(patch_weights.unsqueeze(-1) * points, dim=0),
                "normal_velocity": normal_velocity,
                "normal_force": normal_forces,
                "friction_force": friction_forces,
                "patch_force": patch_forces,
                "patch_weights": patch_weights,
                "contact_jacobian": plane_jacobian,
                "delassus_matrix": plane_delassus,
                "contact_effective_mass": 1.0 / torch.clamp(
                    torch.diagonal(plane_delassus), min=1e-12
                ),
                "effective_stiffness": K,
                "effective_damping": D,
                "effective_friction_coefficient": mu,
                "contact_model": cfg.contact_model,
                "paper_closed_form_contact": bool(cfg.paper_closed_form_contact),
                "implicit_contact_matrix": (
                    implicit_matrix if cfg.contact_model == "dual_cone" else None
                ),
                **friction_diagnostics,
            })

        next_states = []
        for idx, state in enumerate(predicted):
            if not dynamic_flags[idx]:
                next_states.append(state)
                continue
            inv_mass = 1.0 / torch.clamp(mass_tensors[idx], min=1e-9)
            angular_delta = _inertia_angular_delta(
                torque_accum[idx],
                inertia_diag=inertia_tensors[idx],
                inertia_matrix=None,
                dynamic=True,
            )
            linear_velocity = state.linear_velocity + cfg.dt * inv_mass * force_accum[idx]
            angular_velocity = state.angular_velocity + cfg.dt * angular_delta
            position = state.position + cfg.dt * (linear_velocity - state.linear_velocity)
            quaternion = _integrate_quaternion_wxyz(state.quaternion_wxyz, angular_velocity, cfg.dt)
            next_states.append(RigidBodyState(position, quaternion, linear_velocity, angular_velocity))

        diagnostics = {
            "graph": graph,
            "active_edges": active_edges,
            "lambda": lambda_terms,
            "friction": friction_terms,
            "plane_contacts": plane_terms,
            "force_accum": force_accum,
            "torque_accum": torque_accum,
            "contact_jacobians": jacobian_terms,
            "effective_masses": mass_tensors,
            "effective_inertia_diags": inertia_tensors,
            "generalized_damping": generalized_damping,
        }
        return tuple(next_states), diagnostics


class ImpedanceFloorContactDynamics:
    """Paper III-D-2 closed-form impedance dynamics, frictionless single-pair.

    For each step:

      b_t       = v_t + h · M⁻¹ · τ(q_t, v_t, a_t)        (here τ = m·g)
      ϕ̃_t      = smooth_min over query-point signed distances
      ñ_t      = contact-weighted, plane-oriented normal (paper III-D-1)
      J̃·b_t   = b_t · ñ_t                                 (no friction tangents)
      λ_t       = SoftPlus(-K · (h · J̃·b_t + ϕ̃_t) - D · (J̃·b_t))
      v_{t+1}  = b_t + (h / m) · λ_t · ñ_t
      q_{t+1}  = q_t + h · v_{t+1}

    Stiffness ``stiffness`` and damping ``damping`` are exposed as raw tensors
    so a fit loop can wrap them in :class:`torch.nn.Parameter` and learn them
    end-to-end. They are clamped to be non-negative via SoftPlus internally so
    even unconstrained Adam updates stay physical.
    """

    def __init__(
        self,
        collider: PlaneCollider,
        floor_query_offsets_xy: torch.Tensor,
        local_gaussian_centers: torch.Tensor,
        gaussian_radii: torch.Tensor,
        *,
        stiffness: torch.Tensor,
        damping: torch.Tensor,
        world_rotation: Optional[torch.Tensor] = None,
        config: Optional[ImpedanceContactDynamicsConfig] = None,
    ):
        self.collider = collider
        self.floor_query_offsets_xy = floor_query_offsets_xy
        self.local_gaussian_centers = local_gaussian_centers
        self.gaussian_radii = gaussian_radii
        self.stiffness = stiffness
        self.damping = damping
        self.world_rotation = world_rotation
        self.config = config or ImpedanceContactDynamicsConfig()

    def step(self, state: RigidState) -> tuple[RigidState, dict[str, torch.Tensor]]:
        cfg = self.config
        gravity = _as_vec3(cfg.gravity, dtype=state.position.dtype, device=state.position.device)
        b = state.linear_velocity + cfg.dt * gravity

        collider = self.collider.on_like(state.position)
        offsets = self.floor_query_offsets_xy.to(dtype=state.position.dtype, device=state.position.device)
        floor_points = torch.cat(
            (
                state.position[:2].unsqueeze(0) + offsets,
                torch.full(
                    (offsets.shape[0], 1),
                    collider.height,
                    dtype=state.position.dtype,
                    device=state.position.device,
                ),
            ),
            dim=-1,
        )
        local = self.local_gaussian_centers.to(dtype=state.position.dtype, device=state.position.device)
        if self.world_rotation is not None:
            local = local @ self.world_rotation.to(dtype=local.dtype, device=local.device).T
        gaussian_centers = local + state.position.unsqueeze(0)
        contacts = detect_gaussian_union_contacts(
            floor_points,
            gaussian_centers,
            self.gaussian_radii,
            collider.normal,
            softness=cfg.contact_softness,
            smooth_min_temperature=cfg.smooth_min_temperature,
            inside_penalty=cfg.inside_penalty,
            inside_sharpness=cfg.inside_sharpness,
        )

        phi_agg = _smooth_min(contacts.signed_distances, cfg.smooth_min_temperature)
        normal = contacts.collider_normal.to(dtype=state.position.dtype, device=state.position.device)
        normal_velocity_b = torch.sum(b * normal)

        K = F.softplus(self.stiffness)
        D = F.softplus(self.damping)
        lambda_t = F.softplus(
            -K * (cfg.dt * normal_velocity_b + phi_agg) - D * normal_velocity_b
        )

        velocity_next = b + (cfg.dt / cfg.mass) * lambda_t * normal
        position_next = state.position + cfg.dt * velocity_next

        diagnostics = {
            "phi_agg": phi_agg,
            "lambda": lambda_t,
            "normal_velocity_b": normal_velocity_b,
            "contact_normal": normal,
            "stiffness": K.detach(),
            "damping": D.detach(),
            "max_penetration": contacts.max_penetration,
            "min_signed_distance": contacts.min_signed_distance,
        }
        return RigidState(position_next, velocity_next), diagnostics
