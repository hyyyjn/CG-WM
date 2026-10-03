"""Stage 2 dynamics against MuJoCo itself (the dataset generator)."""

import math
import unittest

import numpy as np
import torch

from stage2.differentiable_collision_detection import GaussianCollisionBody, PlaneCollider
from stage2.differentiable_complementarity_free_contact_dynamics import (
    GaussianPlaneContactPair,
    MultiBodyGaussianImpedanceDynamics,
    MultiBodyImpedanceDynamicsConfig,
    RigidBodyState,
    angular_velocity_body_to_world,
    select_convex_plane_contacts,
    solimp_impedance,
)
from stage2.object_geometry import plane_support_indices, solid_mass_properties

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None

RADIUS, HALF_HEIGHT, MASS, DAMPING = 0.07, 0.1, 0.35, 0.05
XML = f"""<mujoco><option timestep="0.002" gravity="0 0 -9.81" integrator="Euler"/>
<worldbody><geom name="floor" type="plane" size="4 4 .1"/>
<body name="can" pos="0 0 2"><joint type="free" damping="{DAMPING}"/>
<geom type="cylinder" size="{RADIUS} {HALF_HEIGHT}" mass="{MASS}" friction="0.55 0.02 0.001"/></body>
</worldbody></mujoco>"""


def cylinder_spheres(sphere_radius=0.003, spacing=0.004):
    points = []
    zs = np.arange(-HALF_HEIGHT + sphere_radius, HALF_HEIGHT - sphere_radius + 1e-9, spacing)
    count = int(round(2 * math.pi * (RADIUS - sphere_radius) / spacing))
    angles = np.arange(count) * 2 * math.pi / count
    for z in zs:
        points += [((RADIUS - sphere_radius) * math.cos(a), (RADIUS - sphere_radius) * math.sin(a), z) for a in angles]
    for z in (-HALF_HEIGHT + sphere_radius, HALF_HEIGHT - sphere_radius):
        for ring in np.arange(0.0, RADIUS - sphere_radius + 1e-9, spacing):
            n = max(1, int(round(2 * math.pi * ring / spacing)))
            points += [(ring * math.cos(2 * math.pi * k / n), ring * math.sin(2 * math.pi * k / n), z) for k in range(n)]
    centers = torch.tensor(points, dtype=torch.float32)
    return GaussianCollisionBody(centers, torch.full((len(points),), sphere_radius), centers)


def stage2_dynamics(body, inertia, *, contact_model="soft_constraint", timeconst=0.02, dampratio=1.0, friction=1.0):
    floor = GaussianCollisionBody(torch.zeros(1, 3), torch.tensor([1e-6]), torch.zeros(1, 3))
    config = MultiBodyImpedanceDynamicsConfig(
        dt=0.002, masses=(MASS, 1.0), inertia_diags=(tuple(inertia), (1, 1, 1)),
        generalized_damping=(DAMPING, 0.0), dynamic_flags=(True, False), kinematic_flags=(False, False),
        candidate_pair_mode="all", contact_model=contact_model,
    )
    raw = lambda value: torch.log(torch.expm1(torch.tensor(float(value)))).reshape(1)
    return MultiBodyGaussianImpedanceDynamics(
        [body, floor], stiffness=raw(timeconst), damping=raw(dampratio), friction_coefficient=raw(friction),
        candidate_pairs=[], config=config,
        plane_contact_pairs=[GaussianPlaneContactPair(0, 1, PlaneCollider(torch.tensor([0.0, 0.0, 1.0]), 0.0), 0)],
    )


@unittest.skipIf(mujoco is None, "mujoco is not installed")
class MujocoParityTests(unittest.TestCase):
    def mujoco_rollout(self, qpos, qvel, steps):
        model = mujoco.MjModel.from_xml_string(XML)
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = qpos, qvel
        out = []
        for _ in range(steps):
            mujoco.mj_step(model, data)
            out.append((data.qpos[:3].copy(), data.qpos[3:7].copy()))
        return model, out

    def stage2_rollout(self, dynamics, qpos, qvel, steps):
        q = torch.tensor(qpos[3:7], dtype=torch.float32)
        state = RigidBodyState(
            torch.tensor(qpos[:3], dtype=torch.float32), q, torch.tensor(qvel[:3], dtype=torch.float32),
            angular_velocity_body_to_world(q, torch.tensor(qvel[3:6], dtype=torch.float32)),
        )
        states = (state, RigidBodyState(torch.zeros(3), torch.tensor([1.0, 0, 0, 0]), torch.zeros(3), torch.zeros(3)))
        out = []
        with torch.no_grad():
            for _ in range(steps):
                states, _ = dynamics.step(states)
                out.append((states[0].position.numpy(), states[0].quaternion_wxyz.numpy()))
        return out

    @staticmethod
    def rotation_deg(a, b):
        return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(a / np.linalg.norm(a), b))))))

    def test_free_flight_with_joint_damping_matches(self):
        qpos = np.array([0, 0, 5.0, 0.9888, -0.0887, -0.1067, 0.0552])
        qpos[3:] /= np.linalg.norm(qpos[3:])
        qvel = np.array([0.33, -0.2, 0.0, 2.3, 0.44, 4.35])
        model, reference = self.mujoco_rollout(qpos, qvel, 204)
        predicted = self.stage2_rollout(stage2_dynamics(cylinder_spheres(), model.body_inertia[1]), qpos, qvel, 204)
        for (p_ref, q_ref), (p, q) in zip(reference, predicted):
            self.assertLess(np.linalg.norm(p_ref - p), 1e-5)
            self.assertLess(self.rotation_deg(q_ref, q), 0.05)

    def test_soft_constraint_impact_and_rest_match(self):
        tilt = math.radians(5.0)
        qpos = np.array([0.0, 0.0, 0.4, math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0])
        qvel = np.array([0.3, 0.0, -1.0, 0.0, 0.0, 0.0])
        model, reference = self.mujoco_rollout(qpos, qvel, 600)
        predicted = self.stage2_rollout(stage2_dynamics(cylinder_spheres(), model.body_inertia[1]), qpos, qvel, 600)
        errors = [np.linalg.norm(p_ref - p) for (p_ref, _), (p, _) in zip(reference, predicted)]
        self.assertLess(max(errors), 0.01)
        self.assertLess(errors[-1], 0.005)
        self.assertLess(self.rotation_deg(reference[-1][1], predicted[-1][1]), 1.0)


class ContactHelperTests(unittest.TestCase):
    def test_solimp_matches_mujoco_sigmoid(self):
        value = solimp_impedance(torch.tensor([0.0, 0.0002, 0.002]))
        self.assertTrue(torch.allclose(value, torch.tensor([0.9, 0.904, 0.95]), atol=1e-4))

    def test_convex_contacts_flat_face_spread_and_rim_single(self):
        body = cylinder_spheres()
        normal = torch.tensor([0.0, 0.0, 1.0])
        centers = body.local_centers + torch.tensor([0.0, 0.0, HALF_HEIGHT - 0.001])
        phi = centers[:, 2] - body.radii
        flat = select_convex_plane_contacts(phi, centers - body.radii[:, None] * normal, normal)
        self.assertGreaterEqual(flat.numel(), 3)
        spread = (centers[flat, :2].unsqueeze(0) - centers[flat, :2].unsqueeze(1)).norm(dim=-1).max()
        self.assertGreater(float(spread), 1.5 * RADIUS)
        tilt = math.radians(10.0)
        rotation = torch.tensor([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]])
        tilted = body.local_centers @ rotation.T
        tilted = tilted + torch.tensor([0.0, 0.0, -float((tilted[:, 2] - body.radii).min()) - 0.0005])
        phi = tilted[:, 2] - body.radii
        rim = select_convex_plane_contacts(phi, tilted - body.radii[:, None] * normal, normal)
        self.assertLessEqual(rim.numel(), 4)
        # Only the rim arc that is actually below the plane may carry contacts:
        # 0.5 mm of penetration at 10 deg spans ~2 * 0.07 * acos(1 - 0.0005 / (0.07 sin 10deg)).
        arc = 2 * RADIUS * math.sin(math.acos(1 - 0.0005 / (RADIUS * math.sin(tilt))))
        spread = (tilted[rim, :2].unsqueeze(0) - tilted[rim, :2].unsqueeze(1)).norm(dim=-1).max()
        self.assertLess(float(spread), arc + 0.004)
        self.assertTrue(bool((phi[rim] < 0).all()))

    def test_solid_mass_properties_of_cylinder(self):
        body = cylinder_spheres()
        solid = solid_mass_properties(body.local_centers.numpy(), body.radii.numpy())
        expected = np.array([(3 * RADIUS ** 2 + (2 * HALF_HEIGHT) ** 2) / 12] * 2 + [RADIUS ** 2 / 2])
        np.testing.assert_allclose(solid.principal_inertia_per_mass(), expected, rtol=0.02)
        self.assertLess(np.abs(solid.center_of_mass).max(), 0.002)

    def test_plane_support_subset_keeps_extremes(self):
        body = cylinder_spheres()
        index = plane_support_indices(body.local_centers, body.radii)
        lowest = torch.argmin(body.local_centers[:, 2] - body.radii)
        self.assertLess(index.numel(), body.local_centers.shape[0] // 5)
        self.assertAlmostEqual(float((body.local_centers[index, 2] - body.radii[index]).min()),
                               float(body.local_centers[lowest, 2] - body.radii[lowest]), places=6)


if __name__ == "__main__":
    unittest.main()
