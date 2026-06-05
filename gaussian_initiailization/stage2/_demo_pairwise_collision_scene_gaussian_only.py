"""Render a Gaussian-only 3D collision scene.

Unlike `_demo_pairwise_collision_scene_3d.py`, this visualization does not draw
box faces or wireframes. Every visible body primitive is one of the spherical
Gaussian collision proxies used by the detector.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_initiailization.stage2.differentiable_collision_detection import (  # noqa: E402
    GaussianCollisionBody,
    make_box_surface_query_points,
)
from gaussian_initiailization.stage2.differentiable_complementarity_free_contact_dynamics import (  # noqa: E402
    PairwiseGaussianBodyImpedanceDynamics,
    PairwiseImpedanceDynamicsConfig,
    RigidBodyState,
)

OUT_DIR = REPO_ROOT / "gaussian_initiailization" / "stage2" / "_outputs" / "pairwise_collision_gaussian_only"


def make_gaussian_box_body(half_extent: float = 0.1, resolution: int = 5) -> GaussianCollisionBody:
    centers = make_box_surface_query_points([half_extent, half_extent, half_extent], grid_resolution=resolution)
    spacing = 2.0 * half_extent / float(max(resolution - 1, 1))
    radii = torch.full((centers.shape[0],), spacing * 0.32, dtype=torch.float32)
    return GaussianCollisionBody(centers, radii, centers)


def camera_basis(yaw: float, pitch: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    forward = torch.tensor([cy * cp, sy * cp, sp], dtype=torch.float32)
    right = torch.tensor([-sy, cy, 0.0], dtype=torch.float32)
    up = torch.cross(right, forward, dim=0)
    up = up / torch.clamp(torch.linalg.norm(up), min=1e-12)
    return right, up, forward


def project(point: torch.Tensor, basis, *, width: int, height: int, scale: float = 620.0) -> tuple[float, float, float, float]:
    right, up, forward = basis
    target = torch.tensor([0.02, 0.0, 0.0], dtype=torch.float32)
    rel = point.detach().cpu().float() - target
    x = torch.dot(rel, right)
    y = torch.dot(rel, up)
    z = torch.dot(rel, forward) + 1.35
    z_clamped = float(torch.clamp(z, min=0.2))
    perspective = scale / z_clamped
    return width * 0.5 + float(x) * perspective, height * 0.55 - float(y) * perspective, float(z), perspective


def draw_ground(draw: ImageDraw.ImageDraw, basis, *, width: int, height: int) -> None:
    color = (222, 222, 214, 255)
    for x in [i * 0.1 for i in range(-5, 6)]:
        a = project(torch.tensor([x, -0.32, -0.105]), basis, width=width, height=height)
        b = project(torch.tensor([x, 0.32, -0.105]), basis, width=width, height=height)
        draw.line((a[0], a[1], b[0], b[1]), fill=color)
    for y in [i * 0.1 for i in range(-3, 4)]:
        a = project(torch.tensor([-0.5, y, -0.105]), basis, width=width, height=height)
        b = project(torch.tensor([0.5, y, -0.105]), basis, width=width, height=height)
        draw.line((a[0], a[1], b[0], b[1]), fill=color)


def draw_gaussian_body(
    draw: ImageDraw.ImageDraw,
    body: GaussianCollisionBody,
    state: RigidBodyState,
    basis,
    *,
    fill,
    outline,
    width: int,
    height: int,
) -> None:
    centers = body.world_centers(state.position, quaternion_wxyz=state.quaternion_wxyz).detach().cpu()
    radii = body.radii.detach().cpu()
    projected = []
    for center, radius in zip(centers, radii):
        x, y, z, perspective = project(center, basis, width=width, height=height)
        projected_radius = max(2.0, float(radius) * perspective)
        projected.append((z, x, y, projected_radius))

    for _, x, y, radius in sorted(projected, key=lambda item: item[0], reverse=True):
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=fill,
            outline=outline,
            width=1,
        )


def draw_frame(frame_idx: int, body: GaussianCollisionBody, state_a: RigidBodyState, state_b: RigidBodyState, diagnostics) -> Image.Image:
    width, height = 960, 640
    img = Image.new("RGB", (width, height), (248, 247, 242))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle((0, 0, width, 48), fill=(28, 28, 30, 255))
    draw.text((16, 16), f"Gaussian-only contact scene | frame {frame_idx:03d}", fill=(255, 255, 245, 255))

    basis = camera_basis(math.radians(38.0 + 8.0 * math.sin(frame_idx / 28.0)), math.radians(19.0))
    draw_ground(draw, basis, width=width, height=height)
    draw_gaussian_body(draw, body, state_b, basis, fill=(86, 130, 230, 105), outline=(45, 75, 165, 210), width=width, height=height)
    draw_gaussian_body(draw, body, state_a, basis, fill=(235, 82, 62, 115), outline=(165, 45, 35, 225), width=width, height=height)

    contacts = diagnostics.get("contacts")
    lambdas = diagnostics.get("lambda")
    if contacts is not None:
        points = contacts.patch_points.detach().cpu()
        normals = contacts.patch_normals.detach().cpu()
        weights = diagnostics["patch_weights"].detach().cpu()
        for point, normal, weight in zip(points, normals, weights):
            if float(weight) <= 0.08:
                continue
            x, y, _, _ = project(point, basis, width=width, height=height)
            r = 5 + 9 * float(weight)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(62, 205, 108, 230), outline=(20, 105, 52, 255), width=2)
            end = point + normal * 0.055
            x2, y2, _, _ = project(end, basis, width=width, height=height)
            draw.line((x, y, x2, y2), fill=(20, 120, 55, 255), width=4)

    speed = float(torch.linalg.norm(state_a.linear_velocity).detach().cpu().item())
    max_lambda = float(torch.max(lambdas).detach().cpu().item()) if lambdas is not None else 0.0
    draw.rounded_rectangle((34, 566, 926, 620), radius=7, fill=(255, 255, 255, 220), outline=(190, 190, 184, 255))
    draw.text((50, 582), "red/blue circles: actual spherical Gaussian collision proxies | green: selected contact patches", fill=(35, 35, 35, 255))
    draw.text((50, 602), f"|v_A|={speed:.3f}, max lambda={max_lambda:.3f}", fill=(35, 35, 35, 255))
    return img.convert("RGB")


def make_montage(frames: list[Image.Image], path: Path) -> None:
    picks = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
    montage = Image.new("RGB", (960, 640), (248, 247, 242))
    for idx, frame_idx in enumerate(picks):
        montage.paste(frames[frame_idx].resize((480, 320)), (480 * (idx % 2), 320 * (idx // 2)))
    montage.save(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    body = make_gaussian_box_body()
    dynamics = PairwiseGaussianBodyImpedanceDynamics(
        body,
        body,
        stiffness=torch.tensor(180.0),
        damping=torch.tensor(14.0),
        config=PairwiseImpedanceDynamicsConfig(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, 0.0),
            dynamic_a=True,
            dynamic_b=False,
            mass_a=1.0,
            num_contact_patches=6,
            broad_phase_margin=0.03,
            linear_damping=0.02,
            angular_damping=0.02,
        ),
    )
    state_a = RigidBodyState(
        position=torch.tensor([-0.22, 0.0, 0.0]),
        quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        linear_velocity=torch.tensor([1.15, 0.06, 0.0]),
        angular_velocity=torch.zeros(3),
    )
    state_b = RigidBodyState(
        position=torch.tensor([0.10, 0.0, 0.0]),
        quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        linear_velocity=torch.zeros(3),
        angular_velocity=torch.zeros(3),
    )

    frames = []
    rows = []
    diagnostics = {"lambda": torch.zeros(1), "patch_weights": torch.zeros(1)}
    for frame_idx in range(72):
        frames.append(draw_frame(frame_idx, body, state_a, state_b, diagnostics))
        state_a, state_b, diagnostics = dynamics.step(state_a, state_b)
        rows.append(
            {
                "frame": frame_idx,
                "position_a": state_a.position.detach().cpu().tolist(),
                "velocity_a": state_a.linear_velocity.detach().cpu().tolist(),
                "max_lambda": float(torch.max(diagnostics["lambda"]).detach().cpu().item()),
                "max_patch_weight": float(torch.max(diagnostics["patch_weights"]).detach().cpu().item()),
                "broad_phase_overlaps": bool(diagnostics["broad_phase_overlaps"].detach().cpu().item()),
            }
        )

    gif_path = OUT_DIR / "pairwise_collision_gaussian_only.gif"
    montage_path = OUT_DIR / "pairwise_collision_gaussian_only_montage.png"
    summary_path = OUT_DIR / "pairwise_collision_gaussian_only_summary.json"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0)
    make_montage(frames, montage_path)
    summary_path.write_text(json.dumps({"frames": rows}, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "gif_path": str(gif_path),
                "montage_path": str(montage_path),
                "summary_path": str(summary_path),
                "num_gaussians_per_body": int(body.local_centers.shape[0]),
                "final_position_a": rows[-1]["position_a"],
                "final_velocity_a": rows[-1]["velocity_a"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
