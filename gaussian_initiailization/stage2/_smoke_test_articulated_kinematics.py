"""Smoke test for articulated FK and multi-link kinematic contact."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_initiailization.stage2.articulated_kinematics import (
    ArticulatedLink,
    forward_kinematics,
    link_velocities_from_poses,
)
from gaussian_initiailization.stage2.differentiable_collision_detection import GaussianCollisionBody
from gaussian_initiailization.stage2.differentiable_complementarity_free_contact_dynamics import (
    MultiBodyGaussianImpedanceDynamics,
    MultiBodyImpedanceDynamicsConfig,
    RigidBodyState,
)


def main() -> None:
    links = [
        ArticulatedLink("palm", -1, "fixed", (0, 0, 1), (0.35, 0, 0), (1, 0, 0, 0)),
        ArticulatedLink("finger", 0, "revolute", (0, 0, 1), (-0.2, 0, 0), (1, 0, 0, 0)),
    ]
    joints = torch.tensor([[0.0, 0.0], [0.0, 0.2], [0.0, 0.4]], requires_grad=True)
    positions, quaternions = forward_kinematics(
        links,
        joints,
        base_position=torch.zeros(3),
        base_quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    linear, angular = link_velocities_from_poses(positions, quaternions, 0.1)
    body = GaussianCollisionBody(torch.zeros((1, 3)), torch.tensor([0.25]))
    dynamics = MultiBodyGaussianImpedanceDynamics(
        [body, body, body],
        stiffness=torch.tensor(100.0),
        damping=torch.tensor(10.0),
        names=["object", "palm", "finger"],
        config=MultiBodyImpedanceDynamicsConfig(
            gravity=(0, 0, 0),
            dynamic_flags=(True, False, False),
            kinematic_flags=(False, True, True),
            candidate_pair_mode="all",
            broad_phase_margin=0.1,
            contact_threshold=0.01,
        ),
    )
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    zero = torch.zeros(3)
    states = [
        RigidBodyState(zero, identity, zero, zero),
        RigidBodyState(positions[1, 0], quaternions[1, 0], linear[1, 0], angular[1, 0]),
        RigidBodyState(positions[1, 1], quaternions[1, 1], linear[1, 1], angular[1, 1]),
    ]
    next_states, diagnostics = dynamics.step(states)
    loss = next_states[0].position.square().sum() + positions.square().sum()
    loss.backward()
    assert joints.grad is not None and torch.isfinite(joints.grad).all()
    assert torch.allclose(next_states[1].linear_velocity, linear[1, 0])
    assert torch.allclose(next_states[2].angular_velocity, angular[1, 1])
    print({
        "active_edges": diagnostics["active_edges"],
        "joint_gradient_finite": True,
        "kinematic_velocities_preserved": True,
    })


if __name__ == "__main__":
    main()
