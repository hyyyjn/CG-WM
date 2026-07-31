"""CPU regression test for the ContactGaussian-WM spherical scale convention."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_initiailization.stage2.differentiable_collision_detection import (  # noqa: E402
    _collision_radii_from_log_channels,
)
from gaussian_initiailization.stage2.renderable_gaussian_asset import (  # noqa: E402
    RenderableGaussianAsset,
)


def main() -> None:
    scales = np.asarray([[0.1, 0.1, 0.1], [0.1, 0.2, 0.3]], dtype=np.float32)
    paper_radii = _collision_radii_from_log_channels(
        np.log(scales),
        radius_scale=1.0,
        radius_convention="paper_r2s",
        scale_reduction="mean",
        isotropic_tolerance=1e-4,
    )
    np.testing.assert_allclose(paper_radii, [0.2, 0.4], rtol=1e-6)

    try:
        _collision_radii_from_log_channels(
            np.log(scales),
            radius_scale=1.0,
            radius_convention="paper_r2s",
            scale_reduction="strict",
            isotropic_tolerance=1e-4,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("strict loading must reject anisotropic scale channels")

    count = 2
    asset = RenderableGaussianAsset(
        xyz=torch.zeros((count, 3)),
        features_dc=torch.zeros((count, 1, 3)),
        features_rest=torch.zeros((count, 0, 3)),
        opacity=torch.zeros((count, 1)),
        scaling=torch.zeros((count, 3)),
        rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        features_geo=torch.zeros((count, 3)),
        foreground_logit=torch.zeros((count, 1)),
        object_ids=torch.zeros((count,), dtype=torch.int32),
        sh_degree=0,
    )
    collision_radii = torch.tensor([0.2, 0.4], requires_grad=True)
    spherical = asset.with_spherical_geometry(
        asset.xyz,
        collision_radii,
        radius_to_scale=0.5,
    )
    rendered_scales = torch.exp(spherical.scaling)
    torch.testing.assert_close(
        rendered_scales,
        torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]),
    )
    rendered_scales.sum().backward()
    assert collision_radii.grad is not None
    assert torch.isfinite(collision_radii.grad).all()
    print({
        "paper_collision_radii": paper_radii.tolist(),
        "renderer_scales": rendered_scales.detach().tolist(),
        "strict_rejects_anisotropic": True,
        "radius_gradient_finite": True,
    })


if __name__ == "__main__":
    main()
