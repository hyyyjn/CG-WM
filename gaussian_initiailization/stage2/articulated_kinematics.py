"""Differentiable forward kinematics for articulated kinematic colliders."""
from __future__ import annotations

from dataclasses import dataclass

import torch


def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-12)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dim=-1)


def quaternion_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion(q)
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def axis_angle_quaternion(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.clamp(torch.linalg.norm(axis), min=1e-12)
    half = 0.5 * angle
    return normalize_quaternion(torch.cat((
        torch.cos(half).unsqueeze(-1),
        torch.sin(half).unsqueeze(-1) * axis.expand(angle.shape + (3,)),
    ), dim=-1))


@dataclass(frozen=True)
class ArticulatedLink:
    name: str
    parent: int
    joint_type: str
    joint_axis: tuple[float, float, float]
    origin_position: tuple[float, float, float]
    origin_quaternion_wxyz: tuple[float, float, float, float]
    joint_pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)


def forward_kinematics(
    links: list[ArticulatedLink],
    joint_positions: torch.Tensor,
    *,
    base_position: torch.Tensor,
    base_quaternion_wxyz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world link poses with shape ``(..., num_links, 3/4)``."""
    if joint_positions.shape[-1] != len(links):
        raise ValueError("joint_positions last dimension must equal number of links.")
    prefix = joint_positions.shape[:-1]
    positions: list[torch.Tensor] = []
    quaternions: list[torch.Tensor] = []
    base_position = base_position.expand(prefix + (3,))
    base_quaternion = normalize_quaternion(base_quaternion_wxyz.expand(prefix + (4,)))
    for index, link in enumerate(links):
        if link.parent >= index:
            raise ValueError(f"link {link.name}: parent must precede child.")
        parent_position = base_position if link.parent < 0 else positions[link.parent]
        parent_quaternion = base_quaternion if link.parent < 0 else quaternions[link.parent]
        origin_position = torch.tensor(
            link.origin_position, dtype=joint_positions.dtype, device=joint_positions.device
        ).expand(prefix + (3,))
        origin_quaternion = torch.tensor(
            link.origin_quaternion_wxyz,
            dtype=joint_positions.dtype,
            device=joint_positions.device,
        ).expand(prefix + (4,))
        position = parent_position + quaternion_rotate(parent_quaternion, origin_position)
        quaternion = normalize_quaternion(
            quaternion_multiply(parent_quaternion, origin_quaternion)
        )
        coordinate = joint_positions[..., index]
        axis = torch.tensor(
            link.joint_axis, dtype=joint_positions.dtype, device=joint_positions.device
        )
        if link.joint_type == "revolute":
            joint_quaternion = axis_angle_quaternion(axis, coordinate)
            pivot = torch.tensor(
                link.joint_pivot,
                dtype=joint_positions.dtype,
                device=joint_positions.device,
            ).expand(prefix + (3,))
            # MJCF joints may rotate about an anchor offset from the body
            # origin. Keep that anchor fixed while rotating the body frame.
            pivot_shift = pivot - quaternion_rotate(joint_quaternion, pivot)
            position = position + quaternion_rotate(quaternion, pivot_shift)
            quaternion = normalize_quaternion(
                quaternion_multiply(quaternion, joint_quaternion)
            )
        elif link.joint_type == "prismatic":
            offset = axis * coordinate.unsqueeze(-1)
            position = position + quaternion_rotate(quaternion, offset)
        elif link.joint_type != "fixed":
            raise ValueError(f"Unsupported joint type: {link.joint_type}")
        positions.append(position)
        quaternions.append(quaternion)
    return torch.stack(positions, dim=-2), torch.stack(quaternions, dim=-2)


def link_velocities_from_poses(
    positions: torch.Tensor,
    quaternions: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-difference world linear/angular velocities for link poses."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    linear = (positions[1:] - positions[:-1]) / float(dt)
    q0, q1 = quaternions[:-1], quaternions[1:]
    conjugate = torch.cat((q0[..., :1], -q0[..., 1:]), dim=-1)
    delta = normalize_quaternion(quaternion_multiply(q1, conjugate))
    sign = torch.where(delta[..., :1] < 0.0, -1.0, 1.0)
    delta = delta * sign
    vector = delta[..., 1:]
    vector_norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, torch.clamp(delta[..., :1], min=-1.0, max=1.0))
    angular = vector / torch.clamp(vector_norm, min=1e-12) * angle / float(dt)
    linear = torch.cat((linear, linear[-1:]), dim=0)
    angular = torch.cat((angular, angular[-1:]), dim=0)
    return linear, angular
