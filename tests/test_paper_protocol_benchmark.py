from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.run_paper_protocol_benchmark import geometric_candidates, load_protocol


class PaperProtocolBenchmarkTests(unittest.TestCase):
    def test_protocol_resolves_paths_and_requires_holdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "protocol.json"
            path.write_text(json.dumps({
                "train_manifest": "train.json", "holdout_manifests": ["test.json"]
            }))
            result = load_protocol(path)
            self.assertEqual(result["train_manifest"], (root / "train.json").resolve())
            self.assertEqual(result["holdout_manifests"], [(root / "test.json").resolve()])

    def test_cem_candidates_are_seeded_and_keep_initial_candidate(self):
        initial = {"contact_pairs": [{
            "body_a": "a", "body_b": "b", "stiffness": 10.0,
            "damping": 2.0, "friction": 0.5,
        }]}
        first = geometric_candidates(initial, 4, 7, 1.0)
        second = geometric_candidates(initial, 4, 7, 1.0)
        self.assertEqual(first, second)
        self.assertEqual(first[0], initial)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(item["contact_pairs"][0]["stiffness"] > 0 for item in first))


if __name__ == "__main__":
    unittest.main()
