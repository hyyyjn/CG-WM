from __future__ import annotations

import unittest

import mujoco
import numpy as np

from tools.generate_mujoco_fall_dataset import build_mjcf


class MujocoDatasetMassTests(unittest.TestCase):
    def test_cylinder_uses_manifest_mass_not_fixed_density(self):
        xml = build_mjcf(
            half_extents=np.array([0.07, 0.07, 0.1]), physics_shape="cylinder",
            object_mass=0.35, object_rgba="1 0 0 1", floor_rgba="1 1 1 1",
            gravity=-9.81, timestep=0.002, ground_size=3.0,
            camera_distance=2.3, camera_height=0.75,
            camera_target_z=0.64, camera_target_x=0.0, camera_target_y=0.0,
            camera_fovy=42.0, sphere_solref="0.01 1", box_solref="",
            cylinder_solref="", floor_solref="", sphere_friction="0.5 0.01 0.001",
            box_friction="0.5 0.01 0.001", cylinder_friction="0.55 0.02 0.001",
            freejoint_damping=0.05, object_rgb1="0.8 0.1 0.1",
            object_rgb2="0.2 0.0 0.0",
            box_face_colors="1 0 0;0 1 0;0 0 1;1 1 0;1 0 1;0 1 1",
            visual_model="cola_can", floor_rgb1="0.8 0.8 0.8",
            floor_rgb2="0.3 0.3 0.3", floor_texrepeat=8,
            sky_rgb1="0.8 0.8 0.9", sky_rgb2="0.4 0.5 0.7",
        )
        model = mujoco.MjModel.from_xml_string(xml)
        body_id = int(model.jnt_bodyid[0])
        self.assertAlmostEqual(float(model.body_mass[body_id]), 0.35, places=6)
        expected = np.array([
            0.35 * (3 * 0.07**2 + 0.2**2) / 12,
            0.35 * (3 * 0.07**2 + 0.2**2) / 12,
            0.5 * 0.35 * 0.07**2,
        ])
        np.testing.assert_allclose(model.body_inertia[body_id], expected, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
