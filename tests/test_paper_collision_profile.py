from __future__ import annotations

import unittest

import torch

from stage2.differentiable_collision_detection import (
    CollisionEngineConfig, DifferentiableCollisionEngine, GaussianCollisionBody,
    evaluate_gaussian_union_sdf, fixed_penetration_signed_distance,
    _spatial_coverage_indices, _support_surface_indices, _trim_support_outliers,
    _geometry_feature_support_indices,
)
from stage2.differentiable_gaussian_render_loss import (
    GaussianRenderLossConfig, gaussian_image_loss,
)


class PaperCollisionProfileTests(unittest.TestCase):
    def test_local_broad_phase_matches_dense_contact_for_distant_primitives(self):
        centers_a = torch.tensor([[0.0, 0.0, 0.11], [5.0, 5.0, 5.0]], requires_grad=True)
        centers_b = torch.tensor([[0.0, 0.0, 0.0], [-5.0, -5.0, -5.0]], requires_grad=True)
        radii = torch.tensor([0.1, 0.1])
        body_a = GaussianCollisionBody(centers_a, radii, centers_a)
        body_b = GaussianCollisionBody(centers_b, radii, centers_b)
        kwargs = {
            "position_a": torch.zeros(3), "position_b": torch.zeros(3),
        }
        dense = DifferentiableCollisionEngine(CollisionEngineConfig()).body_pair_contacts(
            body_a, body_b=body_b, **kwargs
        )
        local = DifferentiableCollisionEngine(CollisionEngineConfig(
            primitive_locality_margin=0.05
        )).body_pair_contacts(body_a, body_b=body_b, **kwargs)
        torch.testing.assert_close(
            local.patch_signed_distances, dense.patch_signed_distances, atol=1e-5, rtol=1e-5
        )
        local.patch_signed_distances.sum().backward()
        self.assertTrue(bool(torch.isfinite(centers_a.grad).all()))
        self.assertTrue(bool(torch.isfinite(centers_b.grad).all()))

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

    def test_support_surface_sampling_prioritizes_rims_over_interior(self):
        angles = torch.arange(16, dtype=torch.float32) * (2.0 * torch.pi / 16.0)
        top = torch.stack((torch.cos(angles), torch.sin(angles), torch.ones_like(angles)), dim=1)
        bottom = top.clone()
        bottom[:, 2] = -1.0
        interior = torch.zeros(32, 3)
        centers = torch.cat((top, bottom, interior), dim=0).numpy()
        selected = _support_surface_indices(centers, 16)
        self.assertTrue(all(index < 32 for index in selected.tolist()))
        values = centers[selected, 2]
        self.assertAlmostEqual(float(values.min()), -1.0)
        self.assertAlmostEqual(float(values.max()), 1.0)

    def test_support_outlier_trimming_removes_visual_decorations(self):
        regular = torch.linspace(-1.0, 1.0, 101).unsqueeze(1).repeat(1, 3)
        centers = torch.cat((regular, torch.tensor([[10.0, 0.0, 0.0]])), dim=0).numpy()
        mask = _trim_support_outliers(centers, 0.01)
        self.assertFalse(bool(mask[-1]))
        self.assertGreater(int(mask.sum()), 90)

    def test_geometry_feature_support_keeps_surface_and_feature_diversity(self):
        centers = torch.tensor([
            [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0], [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        ]).numpy()
        features = torch.zeros(8, 2).numpy()
        features[7, 1] = 10.0
        selected = _geometry_feature_support_indices(centers, features, 7, feature_weight=1.0)
        self.assertIn(7, selected.tolist())
        self.assertGreaterEqual(sum(index < 6 for index in selected.tolist()), 5)

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
