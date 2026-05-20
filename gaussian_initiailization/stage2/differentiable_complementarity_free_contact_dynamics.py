from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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

    def to_serializable(self) -> Dict[str, List[float]]:
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

    def to_serializable(self) -> Dict[str, List[float]]:
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
    acceleration: Tuple[float, float, float] = (0.0, 0.0, 0.0)
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

    def step(self, state: RigidState) -> Tuple[RigidState, Dict[str, torch.Tensor]]:
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
) -> Tuple[List[RigidState], List[Dict[str, torch.Tensor]]]:
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

    def step(self, state: RigidState) -> Tuple[RigidState, Dict[str, torch.Tensor]]:
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
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
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
    inertia_diag_a: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    inertia_diag_b: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    dynamic_a: bool = True
    dynamic_b: bool = True
    contact_softness: float = 1e-3
    smooth_min_temperature: float = 1e-2
    inside_penalty: float = 0.02
    inside_sharpness: float = 50.0
    num_contact_patches: int = 4
    broad_phase_margin: float = 0.02
    patch_selection: str = "spatial"
    linear_damping: float = 0.0
    angular_damping: float = 0.0


class GaussianUnionFloorContactDynamics:
    """Contact dynamics using floor queries against spherical Gaussian collision geometry."""

    def __init__(
        self,
        collider: PlaneCollider,
        floor_query_offsets_xy: torch.Tensor,
        local_gaussian_centers: torch.Tensor,
        gaussian_radii: torch.Tensor,
        *,
        config: Optional[ContactDynamicsConfig] = None,
    ):
        self.collider = collider
        self.floor_query_offsets_xy = floor_query_offsets_xy
        self.local_gaussian_centers = local_gaussian_centers
        self.gaussian_radii = gaussian_radii
        self.config = config or ContactDynamicsConfig()

    def step(self, state: RigidState) -> Tuple[RigidState, Dict[str, torch.Tensor]]:
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
        gaussian_centers = self.local_gaussian_centers.to(
            dtype=state.position.dtype,
            device=state.position.device,
        ) + predicted_position.unsqueeze(0)
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


class PairwiseGaussianBodyImpedanceDynamics:
    """Multi-contact impedance dynamics for two Gaussian collision bodies.

    This class consumes the `BodyPairContacts.patch_*` output from
    `DifferentiableCollisionEngine`. Each patch contributes a frictionless
    normal force to both linear and angular velocity. It is intended as the
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
    ) -> tuple[RigidBodyState, RigidBodyState, Dict[str, torch.Tensor]]:
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

        K = F.softplus(self.stiffness).to(dtype=dtype, device=device)
        D = F.softplus(self.damping).to(dtype=dtype, device=device)
        lambda_raw = F.softplus(-K * (cfg.dt * normal_velocity + phi) - D * normal_velocity)
        lambdas = weights * lambda_raw
        forces = lambdas.unsqueeze(-1) * normals
        total_force = torch.sum(forces, dim=-2)
        torque_a = torch.sum(_cross(r_a, forces), dim=-2)
        torque_b = torch.sum(_cross(r_b, -forces), dim=-2)

        inv_mass_a = 0.0 if not cfg.dynamic_a else 1.0 / float(cfg.mass_a)
        inv_mass_b = 0.0 if not cfg.dynamic_b else 1.0 / float(cfg.mass_b)
        inertia_a = _as_vec3(cfg.inertia_diag_a, dtype=dtype, device=device)
        inertia_b = _as_vec3(cfg.inertia_diag_b, dtype=dtype, device=device)
        inv_inertia_a = torch.zeros_like(inertia_a) if not cfg.dynamic_a else _safe_inverse_vec(inertia_a)
        inv_inertia_b = torch.zeros_like(inertia_b) if not cfg.dynamic_b else _safe_inverse_vec(inertia_b)

        velocity_a = state_a.linear_velocity + cfg.dt * inv_mass_a * total_force
        velocity_b = state_b.linear_velocity - cfg.dt * inv_mass_b * total_force
        angular_a = state_a.angular_velocity + cfg.dt * inv_inertia_a * torque_a
        angular_b = state_b.angular_velocity + cfg.dt * inv_inertia_b * torque_b

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
            "patch_weights": weights,
            "patch_penetrations": contacts.patch_penetrations,
            "patch_signed_distances": phi,
            "total_force_on_a": total_force,
            "torque_on_a": torque_a,
            "torque_on_b": torque_b,
            "broad_phase_overlaps": contacts.broad_phase_overlaps,
        }
        return next_a, next_b, diagnostics

    def step(self, state_a: RigidBodyState, state_b: RigidBodyState) -> tuple[RigidBodyState, RigidBodyState, Dict[str, torch.Tensor]]:
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
        config: Optional[ImpedanceContactDynamicsConfig] = None,
    ):
        self.collider = collider
        self.floor_query_offsets_xy = floor_query_offsets_xy
        self.local_gaussian_centers = local_gaussian_centers
        self.gaussian_radii = gaussian_radii
        self.stiffness = stiffness
        self.damping = damping
        self.config = config or ImpedanceContactDynamicsConfig()

    def step(self, state: RigidState) -> Tuple[RigidState, Dict[str, torch.Tensor]]:
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
        gaussian_centers = self.local_gaussian_centers.to(
            dtype=state.position.dtype, device=state.position.device
        ) + state.position.unsqueeze(0)
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
