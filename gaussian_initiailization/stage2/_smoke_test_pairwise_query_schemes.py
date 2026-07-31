"""Regression checks for pairwise Gaussian-body query schemes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_initiailization.stage2.differentiable_collision_detection import (  # noqa: E402
    CollisionEngineConfig,
    DifferentiableCollisionEngine,
    GaussianCollisionBody,
)
from gaussian_initiailization.stage2.differentiable_complementarity_free_contact_dynamics import (  # noqa: E402
    PairwiseGaussianBodyImpedanceDynamics,
    PairwiseImpedanceDynamicsConfig,
    RigidBodyState,
)


def main() -> None:
    dtype = torch.float64
    body_a = GaussianCollisionBody(
        torch.zeros((1, 3), dtype=dtype),
        torch.tensor([0.5], dtype=dtype),
    )
    body_b = GaussianCollisionBody(
        torch.zeros((1, 3), dtype=dtype),
        torch.tensor([0.4], dtype=dtype),
    )
    position_a = torch.zeros(3, dtype=dtype)
    position_b = torch.tensor([0.8, 0.0, 0.0], dtype=dtype)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    tilted = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=dtype)

    expected_counts = {"axis6": 6, "fibonacci": 10, "analytic": 1, "floor_disk": 9}
    scheme_results = {}
    for scheme, expected_count in expected_counts.items():
        engine = DifferentiableCollisionEngine(
            CollisionEngineConfig(
                softness=1e-3,
                smooth_min_temperature=1e-3,
                num_contact_patches=1,
                patch_selection="topk",
                body_query_scheme=scheme,
                body_query_directions=10,
                floor_query_rings=2,
                floor_query_angles=4,
            )
        )
        contacts = engine.body_pair_contacts(
            body_a,
            position_a,
            body_b,
            position_b,
            quaternion_a_wxyz=tilted,
            quaternion_b_wxyz=identity,
        )
        query_count = int(contacts.a_to_b.query_points.shape[-2])
        if query_count != expected_count:
            raise AssertionError(f"{scheme}: expected {expected_count} A queries, got {query_count}.")
        if not torch.isfinite(contacts.patch_signed_distances).all():
            raise AssertionError(f"{scheme}: non-finite contact distance.")
        scheme_results[scheme] = {
            "query_count_a": query_count,
            "min_primitive_distance_a_to_b": float(
                contacts.a_to_b.primitive_signed_distances.min().detach()
            ),
            "patch_signed_distance": float(contacts.patch_signed_distances.min().detach()),
        }

    # The analytic support query points exactly toward the other sphere, so its
    # raw point-to-sphere distance equals the sphere-pair separation: 0.8-0.5-0.4.
    torch.testing.assert_close(
        torch.tensor(scheme_results["analytic"]["min_primitive_distance_a_to_b"], dtype=dtype),
        torch.tensor(-0.1, dtype=dtype),
        rtol=0.0,
        atol=1e-12,
    )

    dynamics_results = {}
    for scheme in expected_counts:
        differentiable_position = position_a.clone().requires_grad_(True)
        dynamics = PairwiseGaussianBodyImpedanceDynamics(
            body_a,
            body_b,
            stiffness=torch.tensor(200.0, dtype=dtype),
            damping=torch.tensor(10.0, dtype=dtype),
            config=PairwiseImpedanceDynamicsConfig(
                dt=1.0 / 120.0,
                gravity=(0.0, 0.0, 0.0),
                dynamic_a=True,
                dynamic_b=False,
                num_contact_patches=1,
                broad_phase_margin=0.1,
                body_query_scheme=scheme,
                body_query_directions=10,
                floor_query_rings=2,
                floor_query_angles=4,
            ),
        )
        zeros = torch.zeros(3, dtype=dtype)
        state_a = RigidBodyState(differentiable_position, tilted, zeros, zeros)
        state_b = RigidBodyState(position_b, identity, zeros, zeros)
        next_a, _, diagnostics = dynamics.step(state_a, state_b)
        loss = next_a.position.sum() + diagnostics["lambda"].sum()
        loss.backward()
        if differentiable_position.grad is None or not torch.isfinite(differentiable_position.grad).all():
            raise AssertionError(f"{scheme}: non-finite position gradient.")
        dynamics_results[scheme] = {
            "next_position": next_a.position.detach().tolist(),
            "position_gradient": differentiable_position.grad.detach().tolist(),
        }

    # Paper Appendix-C regression: friction is part of J~ = Jn - mu*Jd,
    # so a tangential slip must produce a finite gradient with respect to mu.
    raw_mu = torch.tensor(-1.5, dtype=dtype, requires_grad=True)
    dual_dynamics = PairwiseGaussianBodyImpedanceDynamics(
        body_a,
        body_b,
        stiffness=torch.tensor(200.0, dtype=dtype),
        damping=torch.tensor(10.0, dtype=dtype),
        friction_coefficient=raw_mu,
        config=PairwiseImpedanceDynamicsConfig(
            dt=1.0 / 120.0,
            gravity=(0.0, 0.0, 0.0),
            dynamic_a=True,
            dynamic_b=False,
            num_contact_patches=1,
            broad_phase_margin=0.1,
            body_query_scheme="analytic",
            contact_model="dual_cone",
            dual_cone_directions=4,
        ),
    )
    slip = torch.tensor([0.0, 0.4, 0.0], dtype=dtype)
    next_a, _, dual_diagnostics = dual_dynamics.step(
        RigidBodyState(position_a, identity, slip, torch.zeros(3, dtype=dtype)),
        RigidBodyState(position_b, identity, torch.zeros(3, dtype=dtype), torch.zeros(3, dtype=dtype)),
    )
    next_a.linear_velocity[1].backward()
    if raw_mu.grad is None or not torch.isfinite(raw_mu.grad) or raw_mu.grad.abs() <= 0:
        raise AssertionError(f"dual-cone friction has invalid mu gradient: {raw_mu.grad}.")

    print(
        json.dumps(
            {
                "schemes": scheme_results,
                "dynamics": dynamics_results,
                "finite_gradients": True,
                "dual_cone": {
                    "directions": int(dual_diagnostics["dual_cone_faces"].shape[-2]),
                    "mu_gradient": float(raw_mu.grad),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
