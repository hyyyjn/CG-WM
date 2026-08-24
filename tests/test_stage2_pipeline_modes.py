from __future__ import annotations

import unittest

from stage2.pipeline_modes import (
    Stage2PipelineMode, resolve_stage2_mode, validate_stage2_mode_options,
)


class Stage2PipelineModeTests(unittest.TestCase):
    def test_paper_contract_declares_known_state_and_paper_loss(self):
        contract = resolve_stage2_mode("paper_compatible")
        self.assertEqual(contract.mode, Stage2PipelineMode.PAPER_COMPATIBLE)
        self.assertEqual(contract.initial_state_source, "known_manifest_state")
        self.assertEqual(contract.supervision, "full_image_l1_plus_loftr")

    def test_paper_mode_rejects_experimental_options(self):
        with self.assertRaisesRegex(ValueError, "known initial state"):
            validate_stage2_mode_options(
                resolve_stage2_mode("paper_compatible"),
                prefit_initial_state=True, temporal_window_frames=0,
                geometry_gradient_route="collision_only",
            )
        with self.assertRaisesRegex(ValueError, "experimental mode"):
            validate_stage2_mode_options(
                resolve_stage2_mode("paper_compatible"),
                prefit_initial_state=False, temporal_window_frames=8,
                geometry_gradient_route="collision_only",
            )

    def test_paper_mode_requires_collision_only_geometry_gradient(self):
        validate_stage2_mode_options(
            resolve_stage2_mode("paper_compatible"),
            prefit_initial_state=False, temporal_window_frames=0,
            geometry_gradient_route="collision_only",
        )
        with self.assertRaisesRegex(ValueError, "collision_only"):
            validate_stage2_mode_options(
                resolve_stage2_mode("paper_compatible"),
                prefit_initial_state=False, temporal_window_frames=0,
                geometry_gradient_route="collision_and_render",
            )

    def test_experimental_mode_allows_ablation_options(self):
        validate_stage2_mode_options(
            resolve_stage2_mode("experimental"),
            prefit_initial_state=True, temporal_window_frames=8,
            geometry_gradient_route="collision_and_render",
        )


if __name__ == "__main__":
    unittest.main()
