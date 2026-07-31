"""Regression test for paper-style collision-only geometry gradients."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_initiailization.tools.run_stage2_mujoco_stage1_fit import (  # noqa: E402
    route_geometry_for_supervision,
)


def gradients(route: str) -> tuple[torch.Tensor, torch.Tensor]:
    centers = torch.ones((2, 3), requires_grad=True)
    radii = torch.ones(2, requires_grad=True)
    render_centers, render_radii = route_geometry_for_supervision(
        centers, radii, route
    )
    # 3*c and 2*r stand in for the path image -> state -> collision geometry.
    loss = (3.0 * centers + render_centers).sum()
    loss = loss + (2.0 * radii + render_radii).sum()
    loss.backward()
    return centers.grad, radii.grad


def main() -> None:
    collision_centers, collision_radii = gradients("collision_only")
    direct_centers, direct_radii = gradients("collision_and_render")
    torch.testing.assert_close(collision_centers, torch.full_like(collision_centers, 3.0))
    torch.testing.assert_close(collision_radii, torch.full_like(collision_radii, 2.0))
    torch.testing.assert_close(direct_centers, torch.full_like(direct_centers, 4.0))
    torch.testing.assert_close(direct_radii, torch.full_like(direct_radii, 3.0))
    print({
        "collision_only_center_gradient": collision_centers[0].tolist(),
        "collision_and_render_center_gradient": direct_centers[0].tolist(),
        "direct_geometry_gradient_blocked": True,
    })


if __name__ == "__main__":
    main()
