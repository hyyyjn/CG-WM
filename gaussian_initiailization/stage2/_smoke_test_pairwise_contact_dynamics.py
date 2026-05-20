"""Smoke test for pairwise Gaussian-body multi-contact dynamics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_initiailization.stage2.differentiable_collision_detection import (  # noqa: E402
    GaussianCollisionBody,
    make_box_surface_query_points,
)
from gaussian_initiailization.stage2.differentiable_complementarity_free_contact_dynamics import (  # noqa: E402
    PairwiseGaussianBodyImpedanceDynamics,
    PairwiseImpedanceDynamicsConfig,
    RigidBodyState,
)

OUT_DIR = REPO_ROOT / "gaussian_initiailization" / "stage2" / "_outputs" / "pairwise_contact_dynamics"


def make_box_body(half_extent: float = 0.1, resolution: int = 3) -> GaussianCollisionBody:
    query_points = make_box_surface_query_points(
        [half_extent, half_extent, half_extent],
        grid_resolution=resolution,
        dtype=torch.float32,
    )
    spacing = 2.0 * float(half_extent) / float(max(resolution - 1, 1))
    radii = torch.full((query_points.shape[0],), spacing * 0.25, dtype=torch.float32)
    return GaussianCollisionBody(query_points, radii, query_points)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    body = make_box_body()
    dynamics = PairwiseGaussianBodyImpedanceDynamics(
        body,
        body,
        stiffness=torch.tensor(250.0),
        damping=torch.tensor(10.0),
        config=PairwiseImpedanceDynamicsConfig(
            gravity=(0.0, 0.0, 0.0),
            dynamic_a=True,
            dynamic_b=False,
            num_contact_patches=4,
            broad_phase_margin=0.02,
        ),
    )
    state_a = RigidBodyState(
        position=torch.tensor([0.0, 0.0, 0.0], requires_grad=True),
        quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        linear_velocity=torch.tensor([1.0, 0.0, 0.0]),
        angular_velocity=torch.tensor([0.0, 0.0, 0.0]),
    )
    state_b = RigidBodyState(
        position=torch.tensor([0.15, 0.0, 0.0]),
        quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        linear_velocity=torch.tensor([0.0, 0.0, 0.0]),
        angular_velocity=torch.tensor([0.0, 0.0, 0.0]),
    )

    next_a, next_b, diagnostics = dynamics.step(state_a, state_b)
    loss = next_a.position.sum() + next_a.linear_velocity.sum() + diagnostics["lambda"].sum()
    loss.backward()

    summary = {
        "broad_phase_overlaps": bool(diagnostics["broad_phase_overlaps"].detach().cpu().item()),
        "lambda": diagnostics["lambda"].detach().cpu().tolist(),
        "patch_weights": diagnostics["patch_weights"].detach().cpu().tolist(),
        "next_a_position": next_a.position.detach().cpu().tolist(),
        "next_a_linear_velocity": next_a.linear_velocity.detach().cpu().tolist(),
        "next_b_position": next_b.position.detach().cpu().tolist(),
        "gradient_is_finite": bool(torch.isfinite(state_a.position.grad).all().detach().cpu().item()),
    }
    output_path = OUT_DIR / "pairwise_contact_dynamics_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "output_path": str(output_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
