from __future__ import annotations

import unittest

import torch

from stage2.renderable_gaussian_asset import RenderableGaussianAsset


class RenderableGaussianFilterTests(unittest.TestCase):
    def asset(self):
        count = 4
        return RenderableGaussianAsset(
            xyz=torch.zeros(count, 3), features_dc=torch.zeros(count, 1, 3),
            features_rest=torch.zeros(count, 0, 3),
            opacity=torch.tensor([[-5.0], [0.0], [5.0], [5.0]]),
            scaling=torch.zeros(count, 3),
            rotation=torch.tensor([[1.0, 0, 0, 0]]).repeat(count, 1),
            features_geo=torch.zeros(count, 0),
            foreground_logit=torch.tensor([[5.0], [5.0], [-5.0], [5.0]]),
            object_ids=torch.tensor([0, 1, 1, 2], dtype=torch.int32), sh_degree=0,
        )

    def test_combines_opacity_foreground_and_object_filters(self):
        filtered = self.asset().filter(
            opacity_threshold=0.4, foreground_threshold=0.5, object_id=1
        )
        self.assertEqual(filtered.num_gaussians, 1)
        self.assertEqual(int(filtered.object_ids[0]), 1)

    def test_rejects_empty_filter_result(self):
        with self.assertRaisesRegex(ValueError, "No render Gaussians"):
            self.asset().filter(object_id=99)

    def test_canonical_offset_only_translates_local_centers(self):
        asset = self.asset()
        asset.xyz[:] = torch.tensor([1.0, 2.0, 3.0])
        aligned = asset.subtract_local_offset([0.25, -0.5, 1.0])
        torch.testing.assert_close(
            aligned.xyz, torch.tensor([[0.75, 2.5, 2.0]]).repeat(4, 1)
        )
        torch.testing.assert_close(aligned.scaling, asset.scaling)
        torch.testing.assert_close(aligned.rotation, asset.rotation)
        torch.testing.assert_close(aligned.source_indices, asset.source_indices)

    def test_canonical_offset_requires_three_values(self):
        with self.assertRaisesRegex(ValueError, "three values"):
            self.asset().subtract_local_offset([0.1, 0.2])

    def test_render_config_accepts_explicit_rgb_background(self):
        from stage2.differentiable_gaussian_render_loss import GaussianRenderLossConfig

        config = GaussianRenderLossConfig(background_rgb=(0.8, 0.86, 0.93))
        self.assertEqual(config.background_rgb, (0.8, 0.86, 0.93))


if __name__ == "__main__":
    unittest.main()
