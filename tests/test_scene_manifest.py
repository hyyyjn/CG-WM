from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stage2.scene_manifest import load_scene_manifest, manifest_summary, validate_scene_manifest


class SceneManifestTests(unittest.TestCase):
    def make_scene(self, root: Path) -> Path:
        (root / "rgb").mkdir()
        (root / "rgb" / "000000.png").write_bytes(b"frame")
        (root / "object.ply").write_bytes(b"ply")
        payload = {
            "version": 1,
            "scene_id": "anything",
            "bodies": [
                {
                    "id": "arbitrary_dynamic_id",
                    "role": "dynamic",
                    "render": {"gaussian_ply": "object.ply"},
                    "collision": {"type": "gaussian_union"},
                }
            ],
            "environment": [
                {
                    "id": "arbitrary_environment_id",
                    "role": "static",
                    "collision": {"type": "plane", "normal": [0, 0, 1], "height": 0},
                }
            ],
            "observations": {"rgb_dir": "rgb", "fps": 30},
            "contact_pairs": [
                {
                    "body_a": "arbitrary_dynamic_id",
                    "body_b": "arbitrary_environment_id",
                    "model": "dual_cone",
                }
            ],
        }
        path = root / "scene.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_relative_paths_without_shape_presets(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_scene(Path(temporary))
            manifest = load_scene_manifest(path)
            self.assertEqual(manifest.body("arbitrary_dynamic_id").role, "dynamic")
            self.assertEqual(manifest.body("arbitrary_environment_id").collision.type, "plane")
            self.assertTrue(manifest.bodies[0].render.gaussian_ply.is_absolute())
            self.assertFalse(manifest_summary(manifest)["uses_object_shape_presets"])

    def test_reports_unknown_pair_and_duplicate_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_scene(Path(temporary))
            payload = json.loads(path.read_text())
            payload["environment"][0]["id"] = "arbitrary_dynamic_id"
            payload["contact_pairs"][0]["body_b"] = "missing"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path, validate=False)
            errors = validate_scene_manifest(manifest)
            self.assertTrue(any("unique" in error for error in errors))
            self.assertTrue(any("unknown body_b" in error for error in errors))

    def test_rejects_invalid_plane_and_time_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_scene(Path(temporary))
            payload = json.loads(path.read_text())
            payload["environment"][0]["collision"]["normal"] = [0, 0, 0]
            payload["observations"] = {"rgb_dir": "rgb", "timestamps": [0.0, 0.1, 0.1]}
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path, validate=False)
            errors = validate_scene_manifest(manifest)
            self.assertTrue(any("non-zero" in error for error in errors))
            self.assertTrue(any("strictly increasing" in error for error in errors))

    def test_rejects_invalid_fixed_step_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_scene(Path(temporary))
            payload = json.loads(path.read_text())
            payload["simulation"] = {"physics_timestep": 0.002, "steps_per_frame": 0}
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_scene_manifest(path, validate=False)
            errors = validate_scene_manifest(manifest)
            self.assertTrue(any("steps_per_frame" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
