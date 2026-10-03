"""Geometry and measurement pieces of the video -> physics path."""

import math
import unittest

import numpy as np
import torch

from stage1.silhouette_geometry import mask_signed_distance
from stage2.object_geometry import plane_support_indices
from stage2.pose_space_fit import make_pose_track, pose_residual_loss, window_is_static
from tests.test_mujoco_parity import HALF_HEIGHT, RADIUS, cylinder_spheres


class SilhouetteDistanceTest(unittest.TestCase):
    def test_coverage_mask_gives_subpixel_edge(self):
        # Half plane covering pixel-index coordinates x < 10.3: pixel 10 spans [9.5, 10.5] and is 80% covered.
        mask = np.zeros((16, 24), dtype=np.float32)
        mask[:, :10] = 1.0
        mask[:, 10] = 0.8
        distance = mask_signed_distance(mask)
        for column in (4, 8, 13):
            self.assertAlmostEqual(float(distance[8, column]), 10.3 - column, delta=0.12)

    def test_binary_mask_edge_is_between_pixel_centres(self):
        mask = np.zeros((16, 24), dtype=np.float32)
        mask[:, :10] = 1.0  # last inside centre 9, first outside centre 10
        distance = mask_signed_distance(mask)
        self.assertAlmostEqual(float(distance[8, 6]), 9.5 - 6, delta=0.12)


class FlatFaceSupportTest(unittest.TestCase):
    """A reconstructed flat face bulges by less than the reconstruction can resolve."""

    DIRECTIONS = 65536
    TOLERANCE = 0.00065

    def domed(self, height):
        body = cylinder_spheres()
        centers = body.local_centers.clone()
        top = centers[:, 2] > HALF_HEIGHT - 0.0031
        rho = centers[top, :2].norm(dim=1)
        centers[top, 2] += height * (1.0 - (rho / RADIUS) ** 2)
        return centers, body.radii, top

    def inner_top(self, centers, top, chosen):
        picked = torch.zeros(centers.shape[0], dtype=torch.bool)
        picked[chosen] = True
        return int((picked & top & (centers[:, :2].norm(dim=1) < 0.5 * RADIUS)).sum())

    def test_unresolved_dome_is_supported_on_its_border(self):
        centers, radii, top = self.domed(0.0006)
        plain = plane_support_indices(centers, radii, num_directions=self.DIRECTIONS)
        self.assertGreater(self.inner_top(centers, top, plain), 0)
        resolved = plane_support_indices(centers, radii, num_directions=self.DIRECTIONS, flat_tolerance=self.TOLERANCE)
        self.assertEqual(self.inner_top(centers, top, resolved), 0)
        # the rim keeps its support
        axis = torch.tensor([math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)])
        full = (centers @ axis + radii).max()
        kept = (centers[resolved] @ axis + radii[resolved]).max()
        self.assertAlmostEqual(float(full), float(kept), delta=1e-5)

    def test_resolved_dome_stays(self):
        centers, radii, top = self.domed(0.004)  # 4 mm: a real feature of the shape
        resolved = plane_support_indices(centers, radii, num_directions=self.DIRECTIONS, flat_tolerance=self.TOLERANCE)
        self.assertGreater(self.inner_top(centers, top, resolved), 0)


class PoseMeasurementTest(unittest.TestCase):
    def track(self, positions):
        count = positions.shape[0]
        quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1)
        return make_pose_track("t", positions, quaternions, torch.full((count,), 1600.0),
                               camera_position=[0.0, -2.0, 0.0], fovy_deg=45.0, image_height=480,
                               frame_dt=0.034, physics_dt=0.002, substeps=17)

    def test_depth_is_weighted_far_below_image_plane_motion(self):
        track = self.track(torch.zeros(3, 3))
        focal = 0.5 * 480 / math.tan(math.radians(22.5))
        weight = track.position_weight[0]
        across = float((weight @ torch.tensor([1.0, 0.0, 0.0])).norm())
        along = float((weight @ torch.tensor([0.0, 1.0, 0.0])).norm())
        self.assertAlmostEqual(across, focal / 2.0, delta=1e-3 * focal)
        self.assertAlmostEqual(along, math.sqrt(1600.0 / math.pi) / 2.0, delta=1e-3)
        self.assertGreater(across / along, 10.0)

    def test_residual_and_rest_detection(self):
        still = self.track(torch.zeros(11, 3))
        loss, frames = pose_residual_loss(still.positions, still.quaternions, still, 0)
        self.assertEqual(frames, 11)
        self.assertLess(float(loss), 1e-9)
        self.assertTrue(window_is_static(still, 0, 11))
        moving = self.track(torch.linspace(0.0, 0.05, 11).reshape(-1, 1) * torch.tensor([[1.0, 0.0, 0.0]]))
        self.assertFalse(window_is_static(moving, 0, 11))


if __name__ == "__main__":
    unittest.main()
