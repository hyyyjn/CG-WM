from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stage2.scene_manifest import load_scene_manifest
from tools.run_contactgaussian_pipeline import compile_manifest_command


class ManifestPipelineAdapterTests(unittest.TestCase):
    def make_manifest(self, root: Path, *, dynamic_count: int = 1) -> Path:
        (root / "rgb").mkdir()
        (root / "rgb" / "000000.png").write_bytes(b"frame")
        (root / "asset.ply").write_bytes(b"ply")
        (root / "state.json").write_text(
            json.dumps({
                "position": [0, 0, 1],
                "quaternion_wxyz": [1, 0, 0, 0],
                "linear_velocity": [0, 0, 0],
                "angular_velocity": [0, 0, 0],
            }),
            encoding="utf-8",
        )
        bodies = []
        for index in range(dynamic_count):
            bodies.append({
                "id": f"unseen_name_{index}",
                "role": "dynamic",
                "render": {"gaussian_ply": "asset.ply"},
                "collision": {"type": "gaussian_union", "max_primitives": 32},
                "initialization": {"state_json": "state.json"},
            })
        payload = {
            "scene_id": "generic",
            "bodies": bodies,
            "environment": [{
                "id": "not_called_floor",
                "role": "static",
                "collision": {"type": "plane", "normal": [0, 0, 1], "height": 0},
            }],
            "observations": {"rgb_dir": "rgb", "fps": 25},
            "contact_pairs": [{
                "body_a": "unseen_name_0",
                "body_b": "not_called_floor",
                "model": "dual_cone",
            }],
            "training": {"supervision": "image_only", "fit_iterations": 7},
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_compiles_generic_ids_to_image_only_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = load_scene_manifest(self.make_manifest(root))
            command, compatibility = compile_manifest_command(
                manifest,
                output_dir=root / "result",
                python_executable="python-test",
                compatibility_root=root / "compat",
            )
            self.assertIn("--image_only_objective", command)
            self.assertIn("--initial_state_json", command)
            self.assertIn("--disable_collision_bbox_calibration", command)
            self.assertEqual(command[command.index("--fit_iters") + 1], "7")
            self.assertEqual(compatibility["adapter"]["dynamic_body_id"], "unseen_name_0")
            self.assertFalse(compatibility["adapter"]["uses_object_shape_presets"])

    def test_rejects_multi_dynamic_until_native_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.make_manifest(root, dynamic_count=2)
            payload = json.loads(path.read_text())
            payload["contact_pairs"].append({
                "body_a": "unseen_name_1",
                "body_b": "not_called_floor",
            })
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path)
            with self.assertRaisesRegex(ValueError, "exactly one dynamic body"):
                compile_manifest_command(
                    manifest,
                    output_dir=root / "result",
                    python_executable="python-test",
                    compatibility_root=root / "compat",
                )


if __name__ == "__main__":
    unittest.main()
