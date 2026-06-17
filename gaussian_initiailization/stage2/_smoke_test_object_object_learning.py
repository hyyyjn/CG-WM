"""Object-object Stage 2 learning smoke test.

This test isolates a two-body Gaussian contact and checks that object-object
contact losses produce finite gradients for geometry and physics parameters.
It then runs a short fit against a synthetic target contact generated from a
known spherical Gaussian proxy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_initiailization.stage2.differentiable_collision_detection import (  # noqa: E402
    CollisionEngineConfig,
    DifferentiableCollisionEngine,
    GaussianCollisionBody,
    compare_gaussian_union_normal_modes,
    make_sphere_query_points,
)
from gaussian_initiailization.stage2.differentiable_complementarity_free_contact_dynamics import (  # noqa: E402
    MultiBodyGaussianImpedanceDynamics,
    MultiBodyImpedanceDynamicsConfig,
    RigidBodyState,
)

OUT_DIR = REPO_ROOT / "gaussian_initiailization" / "stage2" / "_outputs" / "object_object_learning"


def inverse_softplus(value: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(float(value), dtype=dtype, device=device)
    return torch.log(torch.expm1(torch.clamp(x, min=1e-8)))


def make_state(position: torch.Tensor, velocity: torch.Tensor) -> RigidBodyState:
    return RigidBodyState(
        position=position,
        quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=position.dtype, device=position.device),
        linear_velocity=velocity,
        angular_velocity=torch.zeros(3, dtype=position.dtype, device=position.device),
    )


def make_body(center: torch.Tensor, radius: torch.Tensor) -> GaussianCollisionBody:
    return GaussianCollisionBody(center.reshape(1, 3), radius.reshape(1), local_query_points=None)


def body_pair_contact_loss(
    body: GaussianCollisionBody,
    state_a: RigidBodyState,
    state_b: RigidBodyState,
    target: dict[str, torch.Tensor],
    *,
    body_b: GaussianCollisionBody | None = None,
    normal_mode: str = "signed_distance",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    engine = DifferentiableCollisionEngine(
        CollisionEngineConfig(
            softness=2e-3,
            smooth_min_temperature=1e-2,
            inside_penalty=0.02,
            inside_sharpness=50.0,
            num_contact_patches=3,
            broad_phase_margin=0.02,
            broad_phase_mode="sphere",
            patch_selection="soft",
            normal_mode=normal_mode,
        )
    )
    contacts = engine.body_pair_contacts(body, state_a.position, body if body_b is None else body_b, state_b.position)
    loss = (
        F.mse_loss(contacts.patch_weights, target["patch_weights"])
        + F.mse_loss(contacts.patch_signed_distances, target["patch_signed_distances"])
        + 0.1 * F.mse_loss(contacts.patch_normals, target["patch_normals"])
    )
    diagnostics = {
        "max_patch_weight": torch.max(contacts.patch_weights),
        "min_signed_distance": torch.min(contacts.patch_signed_distances),
        "normal_alignment": torch.mean(torch.sum(contacts.patch_normals * target["patch_normals"], dim=-1)),
    }
    return loss, diagnostics


def dynamics_loss(
    body: GaussianCollisionBody,
    state_a: RigidBodyState,
    state_b: RigidBodyState,
    target_next_position: torch.Tensor,
    target_next_velocity: torch.Tensor,
    *,
    body_b: GaussianCollisionBody | None = None,
    raw_stiffness: torch.Tensor,
    raw_damping: torch.Tensor,
    raw_friction: torch.Tensor,
    raw_tangential_damping: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dynamics = MultiBodyGaussianImpedanceDynamics(
        [body, body if body_b is None else body_b],
        stiffness=raw_stiffness,
        damping=raw_damping,
        friction_coefficient=raw_friction,
        tangential_damping=raw_tangential_damping,
        config=MultiBodyImpedanceDynamicsConfig(
            dt=1.0 / 60.0,
            masses=(1.0, 1.0),
            dynamic_flags=(True, False),
            gravity=(0.0, 0.0, 0.0),
            contact_softness=2e-3,
            smooth_min_temperature=1e-2,
            inside_penalty=0.02,
            inside_sharpness=50.0,
            normal_mode="signed_distance",
            num_contact_patches=3,
            broad_phase_margin=0.02,
            broad_phase_mode="sphere",
            patch_selection="soft",
            candidate_pair_mode="all",
            contact_threshold=0.05,
            friction_softness=1e-6,
        ),
    )
    next_states, diagnostics = dynamics.step([state_a, state_b])
    next_a = next_states[0]
    loss = F.mse_loss(next_a.position, target_next_position) + F.mse_loss(next_a.linear_velocity, target_next_velocity)
    if diagnostics["lambda"]:
        lambda_stack = torch.stack(diagnostics["lambda"])
        edge_gates = torch.stack(diagnostics["edge_gates"])
    else:
        lambda_stack = torch.zeros(1, dtype=state_a.position.dtype, device=state_a.position.device)
        edge_gates = torch.zeros(1, dtype=state_a.position.dtype, device=state_a.position.device)
    return loss, {
        "next_position": next_a.position,
        "next_velocity": next_a.linear_velocity,
        "max_lambda": torch.max(lambda_stack),
        "max_edge_gate": torch.max(edge_gates),
    }


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def geometry_error(center: torch.Tensor, radius: torch.Tensor, true_body: GaussianCollisionBody) -> dict[str, float]:
    true_center = true_body.local_centers.reshape(-1, 3)[0].to(dtype=center.dtype, device=center.device)
    true_radius = true_body.radii.reshape(-1)[0].to(dtype=radius.dtype, device=radius.device)
    return {
        "center_l2": scalar(torch.linalg.norm(center - true_center)),
        "radius_abs": scalar(torch.abs(radius - true_radius)),
    }


def run_geometry_only_fit(
    *,
    true_body: GaussianCollisionBody,
    state_a: RigidBodyState,
    state_b: RigidBodyState,
    target: dict[str, torch.Tensor],
    target_next_position: torch.Tensor,
    target_next_velocity: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> dict:
    """Fit only Gaussian center/radius against object-object contact targets."""

    center = torch.nn.Parameter(torch.tensor([0.018, -0.012, 0.006], dtype=dtype, device=device))
    raw_radius = torch.nn.Parameter(inverse_softplus(0.076, dtype=dtype, device=device))
    params = [center, raw_radius]
    fixed_stiffness = inverse_softplus(40.0, dtype=dtype, device=device)
    fixed_damping = inverse_softplus(4.0, dtype=dtype, device=device)
    fixed_friction = inverse_softplus(0.25, dtype=dtype, device=device)
    fixed_tangential_damping = inverse_softplus(1.5, dtype=dtype, device=device)

    def current_body() -> GaussianCollisionBody:
        return make_body(center, torch.clamp(F.softplus(raw_radius), min=1e-4))

    def objective() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        body = current_body()
        contact_loss, contact_diag = body_pair_contact_loss(body, state_a, state_b, target, body_b=true_body)
        dyn_loss, dyn_diag = dynamics_loss(
            body,
            state_a,
            state_b,
            target_next_position,
            target_next_velocity,
            body_b=true_body,
            raw_stiffness=fixed_stiffness,
            raw_damping=fixed_damping,
            raw_friction=fixed_friction,
            raw_tangential_damping=fixed_tangential_damping,
        )
        radius = torch.clamp(F.softplus(raw_radius), min=1e-4)
        prior = 1e-4 * torch.sum(center * center) + 1e-4 * radius * radius
        return contact_loss + dyn_loss + prior, contact_loss, dyn_loss, contact_diag, dyn_diag

    initial_loss, initial_contact_loss, initial_dyn_loss, initial_contact_diag, initial_dyn_diag = objective()
    initial_loss.backward()
    gradient_report = {
        "center_grad_norm": scalar(torch.linalg.norm(center.grad)),
        "radius_grad_abs": scalar(torch.abs(raw_radius.grad)),
        "all_gradients_finite": all(torch.isfinite(param.grad).all().item() for param in params),
    }
    for param in params:
        param.grad = None

    initial_radius = torch.clamp(F.softplus(raw_radius.detach()), min=1e-4)
    initial_geometry_error = geometry_error(center.detach(), initial_radius, true_body)
    optimizer = torch.optim.Adam(params, lr=0.025)
    best = {
        "loss": float("inf"),
        "step": None,
        "center": None,
        "radius": None,
        "contact_loss": None,
        "dynamics_loss": None,
        "contact_diag": None,
        "dyn_diag": None,
    }
    history = []
    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss, contact_loss, dyn_loss, contact_diag, dyn_diag = objective()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            raw_radius.clamp_(inverse_softplus(0.03, dtype=dtype, device=device), inverse_softplus(0.2, dtype=dtype, device=device))
            fit_objective = float((contact_loss + dyn_loss).detach().cpu().item())
            if fit_objective < best["loss"]:
                best.update(
                    {
                        "loss": fit_objective,
                        "step": int(step),
                        "center": center.detach().clone(),
                        "radius": F.softplus(raw_radius).detach().clone(),
                        "contact_loss": contact_loss.detach().clone(),
                        "dynamics_loss": dyn_loss.detach().clone(),
                        "contact_diag": {key: value.detach().clone() for key, value in contact_diag.items()},
                        "dyn_diag": {key: value.detach().clone() for key, value in dyn_diag.items()},
                    }
                )
        if step in (0, 1, 2, 9, 24, 49, 99):
            radius = torch.clamp(F.softplus(raw_radius), min=1e-4)
            errors = geometry_error(center.detach(), radius.detach(), true_body)
            history.append(
                {
                    "step": int(step),
                    "loss": scalar(loss),
                    "contact_loss": scalar(contact_loss),
                    "dynamics_loss": scalar(dyn_loss),
                    "center": center.detach().cpu().tolist(),
                    "radius": scalar(radius),
                    "center_l2_error": errors["center_l2"],
                    "radius_abs_error": errors["radius_abs"],
                    "max_patch_weight": scalar(contact_diag["max_patch_weight"]),
                    "max_edge_gate": scalar(dyn_diag["max_edge_gate"]),
                }
            )

    best_geometry_error = geometry_error(best["center"], best["radius"], true_body)
    best_contact_diag = best["contact_diag"]
    best_dyn_diag = best["dyn_diag"]
    return {
        "initial_total_loss": scalar(initial_loss.detach()),
        "initial_contact_loss": scalar(initial_contact_loss.detach()),
        "initial_dynamics_loss": scalar(initial_dyn_loss.detach()),
        "initial_geometry_error": initial_geometry_error,
        "initial_max_patch_weight": scalar(initial_contact_diag["max_patch_weight"]),
        "initial_max_edge_gate": scalar(initial_dyn_diag["max_edge_gate"]),
        "best_step": int(best["step"]),
        "best_total_loss": float(best["loss"]),
        "best_contact_loss": scalar(best["contact_loss"]),
        "best_dynamics_loss": scalar(best["dynamics_loss"]),
        "best_geometry_error": best_geometry_error,
        "geometry_error_improved": bool(
            best_geometry_error["center_l2"] < initial_geometry_error["center_l2"]
            and best_geometry_error["radius_abs"] < initial_geometry_error["radius_abs"]
        ),
        "loss_improved": bool(float(best["loss"]) < scalar(initial_loss.detach())),
        "gradient_report": gradient_report,
        "best_center": best["center"].detach().cpu().tolist(),
        "best_radius": scalar(best["radius"]),
        "best_max_patch_weight": scalar(best_contact_diag["max_patch_weight"]),
        "best_min_signed_distance": scalar(best_contact_diag["min_signed_distance"]),
        "best_normal_alignment": scalar(best_contact_diag["normal_alignment"]),
        "best_next_position": best_dyn_diag["next_position"].detach().cpu().tolist(),
        "best_next_velocity": best_dyn_diag["next_velocity"].detach().cpu().tolist(),
        "history": history,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    dtype = torch.float32
    device = torch.device("cpu")

    state_a = make_state(
        torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device),
        torch.tensor([0.35, 0.04, 0.0], dtype=dtype, device=device),
    )
    state_b = make_state(
        torch.tensor([0.185, 0.0, 0.0], dtype=dtype, device=device),
        torch.zeros(3, dtype=dtype, device=device),
    )

    true_body = make_body(torch.zeros(3, dtype=dtype, device=device), torch.tensor(0.1, dtype=dtype, device=device))
    engine = DifferentiableCollisionEngine(
        CollisionEngineConfig(
            softness=2e-3,
            smooth_min_temperature=1e-2,
            num_contact_patches=3,
            broad_phase_margin=0.02,
            broad_phase_mode="sphere",
            patch_selection="soft",
            normal_mode="signed_distance",
        )
    )
    with torch.no_grad():
        target_contacts = engine.body_pair_contacts(true_body, state_a.position, true_body, state_b.position)
        target = {
            "patch_weights": target_contacts.patch_weights.detach(),
            "patch_signed_distances": target_contacts.patch_signed_distances.detach(),
            "patch_normals": target_contacts.patch_normals.detach(),
        }
        target_dynamics = MultiBodyGaussianImpedanceDynamics(
            [true_body, true_body],
            stiffness=torch.tensor(40.0, dtype=dtype, device=device),
            damping=torch.tensor(4.0, dtype=dtype, device=device),
            friction_coefficient=torch.tensor(0.25, dtype=dtype, device=device),
            tangential_damping=torch.tensor(1.5, dtype=dtype, device=device),
            config=MultiBodyImpedanceDynamicsConfig(
                dt=1.0 / 60.0,
                masses=(1.0, 1.0),
                dynamic_flags=(True, False),
                gravity=(0.0, 0.0, 0.0),
                normal_mode="signed_distance",
                num_contact_patches=3,
                broad_phase_margin=0.02,
                broad_phase_mode="sphere",
                patch_selection="soft",
                candidate_pair_mode="all",
                contact_threshold=0.05,
            ),
        )
        target_next, _ = target_dynamics.step([state_a, state_b])
        target_next_position = target_next[0].position.detach()
        target_next_velocity = target_next[0].linear_velocity.detach()

    local_center = torch.nn.Parameter(torch.tensor([0.008, -0.006, 0.003], dtype=dtype, device=device))
    raw_radius = torch.nn.Parameter(inverse_softplus(0.092, dtype=dtype, device=device))
    raw_stiffness = torch.nn.Parameter(inverse_softplus(18.0, dtype=dtype, device=device))
    raw_damping = torch.nn.Parameter(inverse_softplus(1.0, dtype=dtype, device=device))
    raw_friction = torch.nn.Parameter(inverse_softplus(0.005, dtype=dtype, device=device))
    raw_tangential_damping = torch.nn.Parameter(inverse_softplus(0.5, dtype=dtype, device=device))
    params = [local_center, raw_radius, raw_stiffness, raw_damping, raw_friction, raw_tangential_damping]

    def current_body() -> GaussianCollisionBody:
        radius = torch.clamp(F.softplus(raw_radius), min=1e-4)
        return make_body(local_center, radius)

    contact_loss0, contact_diag0 = body_pair_contact_loss(current_body(), state_a, state_b, target)
    dyn_loss0, dyn_diag0 = dynamics_loss(
        current_body(),
        state_a,
        state_b,
        target_next_position,
        target_next_velocity,
        raw_stiffness=raw_stiffness,
        raw_damping=raw_damping,
        raw_friction=raw_friction,
        raw_tangential_damping=raw_tangential_damping,
    )
    total0 = contact_loss0 + dyn_loss0
    total0.backward()
    gradient_report = {
        "center_grad_norm": scalar(torch.linalg.norm(local_center.grad)),
        "radius_grad_abs": scalar(torch.abs(raw_radius.grad)),
        "stiffness_grad_abs": scalar(torch.abs(raw_stiffness.grad)),
        "damping_grad_abs": scalar(torch.abs(raw_damping.grad)),
        "friction_grad_abs": scalar(torch.abs(raw_friction.grad)),
        "tangential_damping_grad_abs": scalar(torch.abs(raw_tangential_damping.grad)),
        "all_gradients_finite": all(torch.isfinite(param.grad).all().item() for param in params),
    }
    for param in params:
        param.grad = None

    optimizer = torch.optim.Adam(params, lr=0.04)
    history = []
    best = {
        "loss": float("inf"),
        "step": None,
        "center": None,
        "radius": None,
        "stiffness": None,
        "damping": None,
        "friction": None,
        "tangential_damping": None,
        "contact_loss": None,
        "dynamics_loss": None,
        "contact_diag": None,
        "dyn_diag": None,
    }
    for step in range(80):
        optimizer.zero_grad(set_to_none=True)
        body = current_body()
        contact_loss, contact_diag = body_pair_contact_loss(body, state_a, state_b, target)
        dyn_loss, dyn_diag = dynamics_loss(
            body,
            state_a,
            state_b,
            target_next_position,
            target_next_velocity,
            raw_stiffness=raw_stiffness,
            raw_damping=raw_damping,
            raw_friction=raw_friction,
            raw_tangential_damping=raw_tangential_damping,
        )
        regularizer = 1e-3 * torch.sum(local_center * local_center)
        loss = contact_loss + dyn_loss + regularizer
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            raw_radius.clamp_(inverse_softplus(0.03, dtype=dtype, device=device), inverse_softplus(0.2, dtype=dtype, device=device))
            objective = float((contact_loss + dyn_loss).detach().cpu().item())
            if objective < best["loss"]:
                best.update(
                    {
                        "loss": objective,
                        "step": int(step),
                        "center": local_center.detach().clone(),
                        "radius": F.softplus(raw_radius).detach().clone(),
                        "stiffness": F.softplus(raw_stiffness).detach().clone(),
                        "damping": F.softplus(raw_damping).detach().clone(),
                        "friction": F.softplus(raw_friction).detach().clone(),
                        "tangential_damping": F.softplus(raw_tangential_damping).detach().clone(),
                        "contact_loss": contact_loss.detach().clone(),
                        "dynamics_loss": dyn_loss.detach().clone(),
                        "contact_diag": {key: value.detach().clone() for key, value in contact_diag.items()},
                        "dyn_diag": {key: value.detach().clone() for key, value in dyn_diag.items()},
                    }
                )
        if step in (0, 1, 2, 9, 19, 39, 79):
            history.append(
                {
                    "step": int(step),
                    "loss": scalar(loss),
                    "contact_loss": scalar(contact_loss),
                    "dynamics_loss": scalar(dyn_loss),
                    "radius": scalar(F.softplus(raw_radius)),
                    "center": local_center.detach().cpu().tolist(),
                    "max_patch_weight": scalar(contact_diag["max_patch_weight"]),
                    "max_edge_gate": scalar(dyn_diag["max_edge_gate"]),
                    "max_lambda": scalar(dyn_diag["max_lambda"]),
                }
            )

    normal_compare = compare_gaussian_union_normal_modes(
        make_sphere_query_points(0.1, num_points=8, dtype=dtype, device=device),
        true_body.local_centers,
        true_body.radii,
    )
    best_contact_diag = best["contact_diag"]
    best_dyn_diag = best["dyn_diag"]
    geometry_only = run_geometry_only_fit(
        true_body=true_body,
        state_a=state_a,
        state_b=state_b,
        target=target,
        target_next_position=target_next_position,
        target_next_velocity=target_next_velocity,
        dtype=dtype,
        device=device,
    )
    summary = {
        "initial_total_loss": scalar(total0.detach()),
        "initial_contact_loss": scalar(contact_loss0.detach()),
        "initial_dynamics_loss": scalar(dyn_loss0.detach()),
        "best_step": int(best["step"]),
        "best_total_loss": float(best["loss"]),
        "best_contact_loss": scalar(best["contact_loss"]),
        "best_dynamics_loss": scalar(best["dynamics_loss"]),
        "loss_improved": bool(float(best["loss"]) < scalar(total0.detach())),
        "gradient_report": gradient_report,
        "target_patch_weights": target["patch_weights"].detach().cpu().tolist(),
        "target_next_position": target_next_position.detach().cpu().tolist(),
        "target_next_velocity": target_next_velocity.detach().cpu().tolist(),
        "best_center": best["center"].detach().cpu().tolist(),
        "best_radius": scalar(best["radius"]),
        "best_stiffness": scalar(best["stiffness"]),
        "best_damping": scalar(best["damping"]),
        "best_friction": scalar(best["friction"]),
        "best_tangential_damping": scalar(best["tangential_damping"]),
        "best_max_patch_weight": scalar(best_contact_diag["max_patch_weight"]),
        "best_min_signed_distance": scalar(best_contact_diag["min_signed_distance"]),
        "best_normal_alignment": scalar(best_contact_diag["normal_alignment"]),
        "best_next_position": best_dyn_diag["next_position"].detach().cpu().tolist(),
        "best_next_velocity": best_dyn_diag["next_velocity"].detach().cpu().tolist(),
        "signed_distance_vs_autograd_normal_max_deg": scalar(normal_compare["signed_distance_vs_autograd_deg"].max()),
        "phi_soft_vs_autograd_normal_max_deg": scalar(normal_compare["phi_soft_vs_autograd_deg"].max()),
        "geometry_only": geometry_only,
        "history": history,
    }
    output_path = OUT_DIR / "object_object_learning_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "output_path": str(output_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
