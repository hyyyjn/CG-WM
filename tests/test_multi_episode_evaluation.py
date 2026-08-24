import json
import tempfile
import unittest
from pathlib import Path

from tools.run_multi_episode_evaluation import (
    aggregate_episode_metrics, extract_contact_parameters,
    load_multi_episode_protocol, materialize_episode_manifest,
)


class MultiEpisodeEvaluationTests(unittest.TestCase):
    def test_repository_protocol_includes_test_001_and_another_episode(self):
        protocol = load_multi_episode_protocol(Path("configs/multi_episode_test_protocol.json"))
        self.assertEqual([item["id"] for item in protocol["episodes"]], ["test_000", "test_001"])

    def test_extracts_named_ablation_physics(self):
        payload = {"experiments": [{
            "experiment": {"name": "chosen"},
            "train": {"learned_contact_pairs": [{"stiffness": 1.0}]},
        }]}
        self.assertEqual(
            extract_contact_parameters(payload, "chosen")["contact_pairs"][0]["stiffness"], 1.0
        )

    def test_aggregate_reports_dispersion_and_metric_aware_worst_episode(self):
        episodes = [
            {"id": "test_000", "metrics": {"score": 1.0, "rgb_l1": 0.2, "rgb_psnr": 20.0}},
            {"id": "test_001", "metrics": {"score": 2.0, "rgb_l1": 0.3, "rgb_psnr": 10.0}},
        ]
        aggregate = aggregate_episode_metrics(episodes)
        self.assertEqual(aggregate["score"]["worst_episode"], "test_001")
        self.assertEqual(aggregate["rgb_psnr"]["worst_episode"], "test_001")
        self.assertAlmostEqual(aggregate["score"]["mean"], 1.5)
        self.assertAlmostEqual(aggregate["score"]["std"], 0.5)

    def test_materialized_manifest_uses_episode_initial_state_and_absolute_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "episode"
            for directory in (episode / "rgb", episode / "masks", episode / "state"):
                directory.mkdir(parents=True, exist_ok=True)
            (episode / "state" / "trajectory.json").write_text("{}")
            (episode / "episode_manifest.json").write_text(json.dumps({
                "fps": 30, "initial_state": {"position": [1, 2, 3]},
            }))
            template = root / "template.json"
            template.write_text(json.dumps({
                "scene_id": "template",
                "bodies": [{"id": "body", "role": "dynamic",
                            "render": {"gaussian_ply": "asset.ply"},
                            "initialization": {"state_json": "old.json"}}],
                "environment": [],
            }))
            output = materialize_episode_manifest(
                template, {"id": "test_001", "episode_root": episode}, root / "generated"
            )
            payload = json.loads(output.read_text())
            self.assertEqual(payload["scene_id"], "test_001")
            self.assertTrue(Path(payload["bodies"][0]["render"]["gaussian_ply"]).is_absolute())
            state = json.loads(Path(payload["bodies"][0]["initialization"]["state_json"]).read_text())
            self.assertEqual(state["position"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
