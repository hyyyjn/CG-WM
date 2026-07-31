"""Image-space initial pose and free-flight velocity pre-estimation."""
from __future__ import annotations

import torch


def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-12)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def constant_velocity_poses(
    position0: torch.Tensor,
    quaternion0: torch.Tensor,
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = position0.unsqueeze(0) + times.unsqueeze(-1) * linear_velocity.unsqueeze(0)
    rotation_vectors = times.unsqueeze(-1) * angular_velocity.unsqueeze(0)
    angle_sq = torch.sum(rotation_vectors**2, dim=-1)
    # Adding epsilon before sqrt keeps the derivative finite at exactly zero
    # angular velocity.  The Taylor branch retains the correct zero-angle limit.
    angles = torch.sqrt(angle_sq + 1e-12)
    half_angles = 0.5 * angles
    vector_scale = torch.where(
        angle_sq > 1e-8,
        torch.sin(half_angles) / angles,
        0.5 - angle_sq / 48.0,
    )
    scalar = torch.where(
        angle_sq > 1e-8,
        torch.cos(half_angles),
        1.0 - angle_sq / 8.0,
    )
    delta = torch.cat(
        (
            scalar.unsqueeze(-1),
            vector_scale.unsqueeze(-1) * rotation_vectors,
        ),
        dim=-1,
    )
    quaternions = normalize_quaternion(
        quaternion_multiply(delta, normalize_quaternion(quaternion0).expand_as(delta))
    )
    return positions, quaternions


def estimate_initial_state_from_images(
    render_loss,
    *,
    position_init: torch.Tensor,
    quaternion_init: torch.Tensor,
    times: torch.Tensor,
    pose_iters: int,
    velocity_iters: int,
    lr: float,
    geometry_centers: torch.Tensor | None = None,
    geometry_radii: torch.Tensor | None = None,
    velocity_l2: float = 1e-4,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Estimate pose first, then velocity while keeping the estimated pose fixed."""
    position = torch.nn.Parameter(position_init.detach().clone())
    quaternion_raw = torch.nn.Parameter(normalize_quaternion(quaternion_init.detach().clone()))
    pose_optimizer = torch.optim.Adam((position, quaternion_raw), lr=float(lr))
    pose_history = []
    best_pose = None
    for iteration in range(max(1, int(pose_iters))):
        pose_optimizer.zero_grad(set_to_none=True)
        quaternion = normalize_quaternion(quaternion_raw)
        loss, diagnostics = render_loss(
            position.reshape(1, 3),
            quaternion.reshape(1, 4),
            geometry_centers=geometry_centers,
            geometry_radii=geometry_radii,
        )
        loss.backward()
        value = float(loss.detach().cpu().item())
        pose_history.append({"iteration": iteration, "loss": value, **diagnostics})
        if best_pose is None or value < best_pose["loss"]:
            best_pose = {
                "loss": value,
                "position": position.detach().clone(),
                "quaternion": normalize_quaternion(quaternion_raw.detach().clone()),
            }
        pose_optimizer.step()

    position0 = best_pose["position"]
    quaternion0 = best_pose["quaternion"]
    linear_velocity = torch.nn.Parameter(torch.zeros_like(position0))
    angular_velocity = torch.nn.Parameter(torch.zeros_like(position0))
    velocity_optimizer = torch.optim.Adam((linear_velocity, angular_velocity), lr=float(lr))
    velocity_history = []
    best_velocity = None
    relative_times = times - times[0]
    for iteration in range(max(1, int(velocity_iters))):
        velocity_optimizer.zero_grad(set_to_none=True)
        positions, quaternions = constant_velocity_poses(
            position0, quaternion0, linear_velocity, angular_velocity, relative_times
        )
        image_loss, diagnostics = render_loss(
            positions,
            quaternions,
            geometry_centers=geometry_centers,
            geometry_radii=geometry_radii,
        )
        loss = image_loss + float(velocity_l2) * (
            torch.mean(linear_velocity**2) + 0.01 * torch.mean(angular_velocity**2)
        )
        loss.backward()
        value = float(loss.detach().cpu().item())
        velocity_history.append({"iteration": iteration, "loss": value, **diagnostics})
        if best_velocity is None or value < best_velocity["loss"]:
            best_velocity = {
                "loss": value,
                "linear_velocity": linear_velocity.detach().clone(),
                "angular_velocity": angular_velocity.detach().clone(),
            }
        velocity_optimizer.step()
    state = {
        "position": position0,
        "quaternion_wxyz": quaternion0,
        "linear_velocity": best_velocity["linear_velocity"],
        "angular_velocity": best_velocity["angular_velocity"],
    }
    return state, {
        "pose_history": pose_history,
        "velocity_history": velocity_history,
        "pose_best_loss": best_pose["loss"],
        "velocity_best_loss": best_velocity["loss"],
        "velocity_model": "constant world-frame linear/angular velocity before contact",
    }
