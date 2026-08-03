from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from stage2.video_observations import (
    load_optional_evaluation_trajectory,
    load_video_observations,
)
from tools.run_stage2_mujoco_stage1_fit import integrate_quaternion_sequence


class VideoObservationTests(unittest.TestCase):
    def test_observations_do_not_require_or_read_trajectory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rgb").mkdir()
            (root / "masks").mkdir()
            (root / "episode_manifest.json").write_text(json.dumps({"fps": 20}), encoding="utf-8")
            for index in (0, 2, 4):
                (root / "rgb" / f"{index:06d}.png").write_bytes(b"not decoded by loader")
                (root / "masks" / f"{index:06d}.png").write_bytes(b"mask")

            observations = load_video_observations(root)

            self.assertEqual(observations.frame_indices, (0, 2, 4))
            self.assertTrue(torch.allclose(observations.times, torch.tensor([0.0, 0.1, 0.2])))
            self.assertFalse(observations.source["trajectory_read"])
            self.assertEqual(len(observations.mask_paths), 3)

    def test_evaluation_trajectory_is_optional(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertIsNone(load_optional_evaluation_trajectory(root / "missing.json"))
            path = root / "trajectory.json"
            path.write_text(
                json.dumps({
                    "states": [
                        {"frame_index": index, "time": index * 0.05, "position": [0, 0, index]}
                        for index in range(3)
                    ]
                }),
                encoding="utf-8",
            )
            evaluation = load_optional_evaluation_trajectory(path)
            self.assertIsNotNone(evaluation)
            self.assertEqual(evaluation.num_frames, 3)
            self.assertEqual(tuple(evaluation.positions[:, 2].tolist()), (0.0, 1.0, 2.0))

    def test_gt_free_orientation_rollout_is_differentiable(self):
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])
        angular_velocity = torch.tensor([0.0, 0.0, 1.0], requires_grad=True)
        sequence = integrate_quaternion_sequence(
            quaternion,
            angular_velocity,
            steps=4,
            dt=0.1,
        )
        self.assertEqual(tuple(sequence.shape), (4, 4))
        self.assertTrue(torch.allclose(torch.linalg.norm(sequence, dim=-1), torch.ones(4), atol=1e-6))
        sequence[-1, 3].backward()
        self.assertIsNotNone(angular_velocity.grad)
        self.assertGreater(float(torch.linalg.norm(angular_velocity.grad)), 0.0)


if __name__ == "__main__":
    unittest.main()
