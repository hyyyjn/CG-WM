import unittest

import torch

from stage2.differentiable_loftr_loss import LoFTRCorrespondenceLoss


class FakeMatcher(torch.nn.Module):
    def forward(self, inputs):
        device = inputs["image0"].device
        return {
            "keypoints0": torch.tensor([[3.0, 3.0], [5.0, 5.0]], device=device),
            "keypoints1": torch.tensor([[3.0, 3.0], [5.0, 5.0]], device=device),
            "confidence": torch.tensor([0.9, 0.1], device=device),
            "batch_indexes": torch.tensor([0, 0], dtype=torch.long, device=device),
        }


class LoFTRLossTests(unittest.TestCase):
    def test_correspondence_selection_keeps_render_gradient(self):
        rendered = torch.rand(1, 3, 8, 8, requires_grad=True)
        target = torch.rand(1, 3, 8, 8)
        loss_module = LoFTRCorrespondenceLoss(
            matcher=FakeMatcher(), confidence_threshold=0.5,
            min_matches=1, max_matches=4, patch_radius=1,
        )
        loss, diagnostics = loss_module(rendered, target)
        loss.backward()
        self.assertEqual(diagnostics["loftr_raw_matches"], 2)
        self.assertEqual(diagnostics["loftr_confidence_matches"], 1)
        self.assertEqual(diagnostics["loftr_matches"], 1)
        self.assertGreater(float(rendered.grad.abs().sum()), 0.0)

    def test_insufficient_matches_returns_connected_zero(self):
        rendered = torch.rand(1, 3, 8, 8, requires_grad=True)
        loss_module = LoFTRCorrespondenceLoss(
            matcher=FakeMatcher(), confidence_threshold=0.5,
            min_matches=2, max_matches=4, patch_radius=1,
        )
        loss, diagnostics = loss_module(rendered, torch.rand_like(rendered))
        loss.backward()
        self.assertFalse(diagnostics["loftr_sufficient_matches"])
        self.assertEqual(float(rendered.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
