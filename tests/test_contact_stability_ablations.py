import unittest

from tools.run_contact_stability_ablations import build_experiments


class ContactStabilityAblationTests(unittest.TestCase):
    def test_ablation_matrix_has_requested_variants(self):
        experiments = build_experiments()
        by_group = {}
        for experiment in experiments:
            by_group.setdefault(experiment.group, []).append(experiment)

        self.assertEqual([item.name for item in by_group["stabilization"]], ["legacy", "stabilized"])
        self.assertEqual(len(by_group["lr"]), 4)
        self.assertEqual(
            [(item.stiffness_lr, item.damping_lr, item.friction_lr) for item in by_group["lr"]],
            [(0.002, 0.002, 0.002), (0.005, 0.005, 0.002),
             (0.010, 0.010, 0.002), (0.010, 0.005, 0.001)],
        )
        self.assertEqual(
            [item.silhouette_weight for item in by_group["silhouette"]],
            [0.0, 0.002, 0.005, 0.01, 0.02],
        )
        self.assertEqual(
            [item.name for item in by_group["attribution"]],
            ["gradient_attribution_recommended"],
        )

    def test_legacy_and_stabilized_settings_are_explicit(self):
        legacy, stabilized = build_experiments()[:2]
        self.assertEqual(legacy.physics_warmup_fraction, 0.0)
        self.assertEqual(legacy.contact_curriculum_frames, 0)
        self.assertEqual(legacy.silhouette_weight, 0.0)
        self.assertFalse(legacy.freeze_initial_state_after_warmup)
        self.assertEqual(stabilized.physics_warmup_fraction, 0.2)
        self.assertEqual(stabilized.contact_curriculum_frames, 8)
        self.assertEqual(stabilized.silhouette_weight, 0.005)
        self.assertTrue(stabilized.freeze_initial_state_after_warmup)
        self.assertEqual(
            (legacy.stiffness_lr, legacy.damping_lr, legacy.friction_lr),
            (stabilized.stiffness_lr, stabilized.damping_lr, stabilized.friction_lr),
        )

    def test_run_names_are_unique_for_resumable_outputs(self):
        names = [experiment.name for experiment in build_experiments()]
        self.assertEqual(len(names), len(set(names)))
