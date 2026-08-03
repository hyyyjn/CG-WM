from __future__ import annotations

import unittest

import torch

from stage2.differentiable_collision_detection import (
    evaluate_gaussian_union_sdf, fixed_penetration_signed_distance,
    _spatial_coverage_indices,
)
from stage2.differentiable_gaussian_render_loss import (
    GaussianRenderLossConfig, gaussian_image_loss,
)


class PaperCollisionProfileTests(unittest.TestCase):
    def test_spatial_coverage_sampling_keeps_both_shape_extremes(self):
        centers = torch.tensor([
            [-1.0, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0], [1.0, 0.0, 0.0],
        ]).numpy()
        indices = _spatial_coverage_indices(centers, 3)
        selected = centers[indices, 0]
        self.assertAlmostEqual(float(selected.min()), -1.0)
        self.assertAlmostEqual(float(selected.max()), 1.0)
        self.assertIn(0.0, selected.tolist())

    def test_deep_penetration_converges_to_fixed_penalty(self):
        phi = torch.tensor([-1.0, 0.2], requires_grad=True)
        transformed = fixed_penetration_signed_distance(
            phi, inside_penalty=0.02, inside_sharpness=50.0
        )
        self.assertAlmostEqual(float(transformed[0].detach()), -0.02, places=5)
        self.assertAlmostEqual(float(transformed[1].detach()), 0.2, places=4)
        transformed.sum().backward()
        self.assertTrue(bool(torch.isfinite(phi.grad).all()))

    def test_gaussian_union_uses_lse_then_fixed_inside_transform(self):
        result = evaluate_gaussian_union_sdf(
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.tensor([0.5, 0.1]), smooth_min_temperature=0.01,
            inside_penalty=0.02, inside_sharpness=50.0,
        )
        self.assertLess(float(result.phi_soft[0]), -0.49)
        self.assertAlmostEqual(float(result.signed_distances[0]), -0.02, places=4)

    def test_full_image_loss_penalizes_prediction_outside_gt_mask(self):
        rendered = torch.zeros((1, 3, 4, 4))
        rendered[:, :, 0, 0] = 1.0
        target = torch.zeros_like(rendered)
        mask = torch.zeros((1, 1, 4, 4))
        config = GaussianRenderLossConfig(image_width=4, image_height=4, loss="l1")
        full_loss, _ = gaussian_image_loss(rendered, target, config=config)
        masked_loss, _ = gaussian_image_loss(
            rendered, target, config=config, masks=mask, background=torch.zeros(3)
        )
        self.assertGreater(float(full_loss), 0.0)
        self.assertEqual(float(masked_loss), 0.0)


if __name__ == "__main__":
    unittest.main()
