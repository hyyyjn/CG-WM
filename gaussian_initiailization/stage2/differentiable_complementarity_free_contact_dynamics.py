from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .differentiable_collision_detection import (
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
