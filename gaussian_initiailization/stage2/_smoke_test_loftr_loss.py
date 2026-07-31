"""CPU regression test for LoFTR-guided patch loss without pretrained weights."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_initiailization.stage2.differentiable_loftr_loss import (  # noqa: E402
    LoFTRCorrespondenceLoss,
)


class FakeMatcher(torch.nn.Module):
    def forward(self, batch):
        device = batch["image0"].device
        dtype = batch["image0"].dtype
        points = torch.tensor(
            [[3, 3], [7, 3], [3, 7], [7, 7]], dtype=dtype, device=device
        )
        return {
            "keypoints0": points,
            "keypoints1": points,
            "confidence": torch.tensor([0.9, 0.8, 0.7, 0.6], device=device),
            "batch_indexes": torch.zeros(4, dtype=torch.long, device=device),
        }


def main() -> None:
    torch.manual_seed(3)
    target = torch.rand((1, 3, 12, 12))
    rendered = torch.roll(target, shifts=1, dims=-1).detach().requires_grad_(True)
    criterion = LoFTRCorrespondenceLoss(
        matcher=FakeMatcher(),
        min_matches=4,
        max_matches=4,
        patch_radius=1,
    )
    loss, diagnostics = criterion(rendered, target)
    loss.backward()
    assert diagnostics["loftr_matches"] == 4
    assert rendered.grad is not None and torch.isfinite(rendered.grad).all()
    assert float(rendered.grad.abs().sum()) > 0.0
    print({
        **diagnostics,
        "render_gradient_finite": True,
        "render_gradient_l1": float(rendered.grad.abs().sum()),
    })


if __name__ == "__main__":
    main()
