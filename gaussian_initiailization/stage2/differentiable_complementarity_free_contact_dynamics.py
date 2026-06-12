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
    contact_softness: float = 1e-3
    smooth_min_temperature: float = 1e-2
    inside_penalty: float = 0.02
    inside_sharpness: float = 50.0
    num_contact_patches: int = 4
    broad_phase_margin: float = 0.02
    broad_phase_mode: str = "sphere"
    patch_selection: str = "spatial"
    linear_damping: float = 0.0
    angular_damping: float = 0.0
    friction_coefficient: float = 0.0
    static_friction_coefficient: Optional[float] = None
    tangential_damping: float = 0.0
    friction_softness: float = 1e-6
    friction_transition_velocity: float = 1e-3


@dataclass(frozen=True)
class MultiBodyImpedanceDynamicsConfig:
    """N-body Gaussian impedance dynamics using the Stage 2 contact graph."""

    dt: float = 1.0 / 60.0
    masses: tuple[float, ...] | None = None
    inertia_diags: tuple[tuple[float, float, float], ...] | None = None
    dynamic_flags: tuple[bool, ...] | None = None
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    contact_softness: float = 1e-3
    smooth_min_temperature: float = 1e-2
    inside_penalty: float = 0.02
    inside_sharpness: float = 50.0
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
        config: Optional[PairwiseImpedanceDynamicsConfig] = None,
    ):
        self.body_a = body_a
        self.body_b = body_b
        self.stiffness = stiffness
        self.damping = damping
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
            )
        )

    def _predict_free(self, state: RigidBodyState, *, mass: float, dynamic: bool) -> RigidBodyState:
        cfg = self.config
        if cfg.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if mass <= 0.0 and dynamic:
            raise ValueError("dynamic bodies require positive mass.")

        gravity = _as_vec3(cfg.gravity, dtype=state.position.dtype, device=state.position.device)
        if dynamic:
            linear_velocity = state.linear_velocity + cfg.dt * gravity
            linear_velocity = linear_velocity * max(0.0, 1.0 - float(cfg.linear_damping) * cfg.dt)
            angular_velocity = state.angular_velocity * max(0.0, 1.0 - float(cfg.angular_damping) * cfg.dt)
            position = state.position + cfg.dt * linear_velocity
            quaternion = _integrate_quaternion_wxyz(state.quaternion_wxyz, angular_velocity, cfg.dt)
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
        lambda_raw = F.softplus(-K * (cfg.dt * normal_velocity + phi) - D * normal_velocity)
        lambdas = weights * lambda_raw
        normal_forces = lambdas.unsqueeze(-1) * normals
        friction_forces, friction_diagnostics = _friction_cone_forces(
            tangential_velocity,
            weights,
            lambdas,
            tangential_damping=torch.as_tensor(float(cfg.tangential_damping), dtype=dtype, device=device),
            dynamic_mu=torch.as_tensor(float(cfg.friction_coefficient), dtype=dtype, device=device),
            static_mu=(
                None
                if cfg.static_friction_coefficient is None
                else torch.as_tensor(float(cfg.static_friction_coefficient), dtype=dtype, device=device)
            ),
            friction_softness=float(cfg.friction_softness),
            transition_velocity=float(cfg.friction_transition_velocity),
        )
        forces = normal_forces + friction_forces
        total_force = torch.sum(forces, dim=-2)
        torque_a = torch.sum(_cross(r_a, forces), dim=-2)
        torque_b = torch.sum(_cross(r_b, -forces), dim=-2)

        inv_mass_a = 0.0 if not cfg.dynamic_a else 1.0 / float(cfg.mass_a)
        inv_mass_b = 0.0 if not cfg.dynamic_b else 1.0 / float(cfg.mass_b)
        angular_delta_a = _inertia_angular_delta(
            torque_a,
            inertia_diag=cfg.inertia_diag_a,
            inertia_matrix=cfg.inertia_matrix_a,
            dynamic=cfg.dynamic_a,
        )
        angular_delta_b = _inertia_angular_delta(
            torque_b,
            inertia_diag=cfg.inertia_diag_b,
            inertia_matrix=cfg.inertia_matrix_b,
            dynamic=cfg.dynamic_b,
        )

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
        }
        return next_a, next_b, diagnostics

    def step(self, state_a: RigidBodyState, state_b: RigidBodyState) -> tuple[RigidBodyState, RigidBodyState, dict[str, torch.Tensor]]:
        predicted_a = self._predict_free(state_a, mass=self.config.mass_a, dynamic=self.config.dynamic_a)
        predicted_b = self._predict_free(state_b, mass=self.config.mass_b, dynamic=self.config.dynamic_b)
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
        names: Iterable[str] | None = None,
        config: Optional[MultiBodyImpedanceDynamicsConfig] = None,
    ):
        self.bodies = tuple(bodies)
        if len(self.bodies) < 2:
            raise ValueError("MultiBodyGaussianImpedanceDynamics needs at least two bodies.")
        self.names = None if names is None else tuple(str(name) for name in names)
        if self.names is not None and len(self.names) != len(self.bodies):
            raise ValueError(f"names must have length {len(self.bodies)}, got {len(self.names)}.")
        self.stiffness = stiffness
        self.damping = damping
        self.friction_coefficient = friction_coefficient
        self.tangential_damping = tangential_damping
        self.mass_multiplier = mass_multiplier
        self.config = config or MultiBodyImpedanceDynamicsConfig()

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

    def _predict_free(
        self,
        states: Iterable[RigidBodyState],
        masses: tuple[float, ...],
        dynamic_flags: tuple[bool, ...],
    ) -> tuple[RigidBodyState, ...]:
        cfg = self.config
        states = tuple(states)
        if len(states) != len(self.bodies):
            raise ValueError(f"states must have length {len(self.bodies)}, got {len(states)}.")
        if cfg.dt <= 0.0:
            raise ValueError("dt must be positive.")
        predicted = []
        for state, mass, dynamic in zip(states, masses, dynamic_flags):
            if mass <= 0.0 and dynamic:
                raise ValueError("dynamic bodies require positive mass.")
            gravity = _as_vec3(cfg.gravity, dtype=state.position.dtype, device=state.position.device)
            if dynamic:
                linear_velocity = state.linear_velocity + cfg.dt * gravity
                linear_velocity = linear_velocity * max(0.0, 1.0 - float(cfg.linear_damping) * cfg.dt)
                angular_velocity = state.angular_velocity * max(0.0, 1.0 - float(cfg.angular_damping) * cfg.dt)
                position = state.position + cfg.dt * linear_velocity
                quaternion = _integrate_quaternion_wxyz(state.quaternion_wxyz, angular_velocity, cfg.dt)
            else:
                linear_velocity = torch.zeros_like(state.linear_velocity)
                angular_velocity = torch.zeros_like(state.angular_velocity)
                position = state.position
                quaternion = _normalize_quaternion(state.quaternion_wxyz)
            predicted.append(RigidBodyState(position, quaternion, linear_velocity, angular_velocity))
        return tuple(predicted)

    def step(self, states: Iterable[RigidBodyState]) -> tuple[tuple[RigidBodyState, ...], dict[str, object]]:
        from .differentiable_contact_graph import build_pairwise_contact_graph

        cfg = self.config
        masses = self._masses()
        inertia_diags = self._inertia_diags()
        dynamic_flags = self._dynamic_flags()
        predicted = self._predict_free(states, masses, dynamic_flags)
        names = self.names or tuple(f"body_{idx}" for idx in range(len(self.bodies)))
        graph = build_pairwise_contact_graph(
            self.bodies,
            predicted,
            names=names,
            dynamic_flags=dynamic_flags,
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

        dtype = predicted[0].position.dtype
        device = predicted[0].position.device
        mass_scale = (
            torch.ones((), dtype=dtype, device=device)
            if self.mass_multiplier is None
            else F.softplus(self.mass_multiplier).to(dtype=dtype, device=device)
        )
        mass_tensors = [
            torch.as_tensor(float(mass), dtype=dtype, device=device) * (mass_scale if dynamic else 1.0)
            for mass, dynamic in zip(masses, dynamic_flags)
        ]
        force_accum = [torch.zeros(3, dtype=dtype, device=device) for _ in predicted]
        torque_accum = [torch.zeros(3, dtype=dtype, device=device) for _ in predicted]
        K = F.softplus(self.stiffness).to(dtype=dtype, device=device)
        D = F.softplus(self.damping).to(dtype=dtype, device=device)
        mu = (
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
        for edge in graph.active_edges(contact_threshold=float(cfg.contact_threshold)):
            contacts = edge.contacts
            i, j = int(edge.body_i), int(edge.body_j)
            state_i = predicted[i]
            state_j = predicted[j]
            normals = contacts.patch_normals.to(dtype=dtype, device=device)
            points = contacts.patch_points.to(dtype=dtype, device=device)
            weights = contacts.patch_weights.to(dtype=dtype, device=device)
            phi = contacts.patch_signed_distances.to(dtype=dtype, device=device)
            r_i = points - state_i.position.unsqueeze(-2)
            r_j = points - state_j.position.unsqueeze(-2)
            velocity_i = state_i.linear_velocity.unsqueeze(-2) + _cross(state_i.angular_velocity.unsqueeze(-2), r_i)
            velocity_j = state_j.linear_velocity.unsqueeze(-2) + _cross(state_j.angular_velocity.unsqueeze(-2), r_j)
            relative_velocity = velocity_i - velocity_j
            normal_velocity = torch.sum(relative_velocity * normals, dim=-1)
            tangent_1, tangent_2 = _tangent_basis(normals)
            tangent_velocity_1 = torch.sum(relative_velocity * tangent_1, dim=-1)
            tangent_velocity_2 = torch.sum(relative_velocity * tangent_2, dim=-1)
            tangential_velocity = tangent_velocity_1.unsqueeze(-1) * tangent_1 + tangent_velocity_2.unsqueeze(-1) * tangent_2
            lambda_raw = F.softplus(-K * (cfg.dt * normal_velocity + phi) - D * normal_velocity)
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
            total_force_on_i = torch.sum(patch_forces, dim=-2)
            torque_on_i = torch.sum(_cross(r_i, patch_forces), dim=-2)
            torque_on_j = torch.sum(_cross(r_j, -patch_forces), dim=-2)
            force_accum[i] = force_accum[i] + total_force_on_i
            force_accum[j] = force_accum[j] - total_force_on_i
            torque_accum[i] = torque_accum[i] + torque_on_i
            torque_accum[j] = torque_accum[j] + torque_on_j
            active_edges.append((i, j))
            lambda_terms.append(lambdas)
            friction_terms.append(
                {
                    "edge": (i, j),
                    "friction_force": friction,
                    "normal_force": normal_forces,
                    "tangential_velocity": tangential_velocity,
                    **friction_diagnostics,
                }
            )

        next_states = []
        for idx, state in enumerate(predicted):
            if not dynamic_flags[idx]:
                next_states.append(state)
                continue
            inv_mass = 1.0 / torch.clamp(mass_tensors[idx], min=1e-9)
            scaled_inertia = tuple(float(value) * float(masses[idx]) for value in inertia_diags[idx])
            angular_delta = _inertia_angular_delta(
                torque_accum[idx],
                inertia_diag=scaled_inertia,
                inertia_matrix=None,
                dynamic=True,
            ) / torch.clamp(mass_scale, min=1e-9)
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
            "force_accum": force_accum,
            "torque_accum": torque_accum,
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
