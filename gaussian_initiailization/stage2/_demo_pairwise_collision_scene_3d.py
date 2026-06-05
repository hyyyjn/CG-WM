"""Render a 3D perspective GIF for pairwise Gaussian-body collision dynamics."""

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

OUT_DIR = REPO_ROOT / "gaussian_initiailization" / "stage2" / "_outputs" / "pairwise_collision_scene_3d"


def make_box_body(half_extent: float = 0.1, resolution: int = 5) -> GaussianCollisionBody:
    queries = make_box_surface_query_points([half_extent, half_extent, half_extent], grid_resolution=resolution)
    spacing = 2.0 * half_extent / float(max(resolution - 1, 1))
    radii = torch.full((queries.shape[0],), spacing * 0.32, dtype=torch.float32)
    return GaussianCollisionBody(queries, radii, queries)


def camera_basis(yaw: float, pitch: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    forward = torch.tensor([cy * cp, sy * cp, sp], dtype=torch.float32)
    right = torch.tensor([-sy, cy, 0.0], dtype=torch.float32)
    up = torch.cross(right, forward, dim=0)
    up = up / torch.clamp(torch.linalg.norm(up), min=1e-12)
    return right, up, forward


def project(point: torch.Tensor, basis, *, width: int, height: int, scale: float = 620.0) -> tuple[float, float, float]:
    right, up, forward = basis
    camera_target = torch.tensor([0.02, 0.0, 0.0], dtype=torch.float32)
    rel = point.detach().cpu().float() - camera_target
    x = torch.dot(rel, right)
    y = torch.dot(rel, up)
    z = torch.dot(rel, forward) + 1.35
    perspective = scale / float(torch.clamp(z, min=0.2))
    return width * 0.5 + float(x) * perspective, height * 0.55 - float(y) * perspective, float(z)


def cube_vertices(center: torch.Tensor, half: float = 0.1) -> list[torch.Tensor]:
    verts = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                verts.append(center.detach().cpu().float() + torch.tensor([sx * half, sy * half, sz * half]))
    return verts


FACES = [
    (0, 1, 3, 2),
    (4, 6, 7, 5),
    (0, 4, 5, 1),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 5, 7, 3),
]

EDGES = [
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
]


def draw_cube(draw: ImageDraw.ImageDraw, center: torch.Tensor, basis, *, fill, outline, width: int, height: int) -> None:
    verts = cube_vertices(center)
    projected = [project(v, basis, width=width, height=height) for v in verts]
    face_items = []
    for face in FACES:
        depth = sum(projected[idx][2] for idx in face) / len(face)
        pts = [(projected[idx][0], projected[idx][1]) for idx in face]
        face_items.append((depth, pts))
    for _, pts in sorted(face_items, key=lambda item: item[0], reverse=True):
        draw.polygon(pts, fill=fill, outline=outline)
    for a, b in EDGES:
        draw.line((projected[a][0], projected[a][1], projected[b][0], projected[b][1]), fill=outline, width=2)


def draw_ground(draw: ImageDraw.ImageDraw, basis, *, width: int, height: int) -> None:
    grid_color = (218, 218, 210)
    for x in [i * 0.1 for i in range(-5, 6)]:
        p0 = torch.tensor([x, -0.32, -0.105])
        p1 = torch.tensor([x, 0.32, -0.105])
        a = project(p0, basis, width=width, height=height)
        b = project(p1, basis, width=width, height=height)
        draw.line((a[0], a[1], b[0], b[1]), fill=grid_color)
    for y in [i * 0.1 for i in range(-3, 4)]:
        p0 = torch.tensor([-0.5, y, -0.105])
        p1 = torch.tensor([0.5, y, -0.105])
        a = project(p0, basis, width=width, height=height)
        b = project(p1, basis, width=width, height=height)
        draw.line((a[0], a[1], b[0], b[1]), fill=grid_color)


def draw_frame(frame_idx: int, state_a: RigidBodyState, state_b: RigidBodyState, diagnostics) -> Image.Image:
    width, height = 960, 640
    img = Image.new("RGB", (width, height), (248, 247, 242))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle((0, 0, width, 48), fill=(28, 28, 30, 255))
    draw.text((16, 16), f"3D Gaussian contact scene | frame {frame_idx:03d}", fill=(255, 255, 245, 255))

    yaw = math.radians(38.0 + 8.0 * math.sin(frame_idx / 28.0))
    pitch = math.radians(19.0)
    basis = camera_basis(yaw, pitch)
    draw_ground(draw, basis, width=width, height=height)
    draw_cube(draw, state_b.position, basis, fill=(114, 151, 225, 135), outline=(55, 86, 165, 255), width=width, height=height)
    draw_cube(draw, state_a.position, basis, fill=(235, 105, 82, 150), outline=(165, 58, 45, 255), width=width, height=height)

    contacts = diagnostics.get("contacts")
    lambdas = diagnostics.get("lambda")
    if contacts is not None:
        points = contacts.patch_points.detach().cpu()
        normals = contacts.patch_normals.detach().cpu()
        weights = diagnostics["patch_weights"].detach().cpu()
        for point, normal, weight in zip(points, normals, weights):
            if float(weight) <= 0.08:
                continue
            x, y, _ = project(point, basis, width=width, height=height)
            radius = 5 + 9 * float(weight)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(63, 190, 106, 220), outline=(20, 105, 52, 255), width=2)
            end = point + normal * 0.055
            x2, y2, _ = project(end, basis, width=width, height=height)
            draw.line((x, y, x2, y2), fill=(20, 120, 55, 255), width=4)

    speed = float(torch.linalg.norm(state_a.linear_velocity).detach().cpu().item())
    max_lambda = float(torch.max(lambdas).detach().cpu().item()) if lambdas is not None else 0.0
    draw.rounded_rectangle((34, 566, 926, 620), radius=7, fill=(255, 255, 255, 220), outline=(190, 190, 184, 255))
    draw.text((50, 582), "red: moving body A | blue: static body B | green: active contact patches and normals", fill=(35, 35, 35, 255))
    draw.text((50, 602), f"|v_A|={speed:.3f}, max lambda={max_lambda:.3f}", fill=(35, 35, 35, 255))
    return img.convert("RGB")


def make_montage(frames: list[Image.Image], path: Path) -> None:
    picks = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
    thumbs = [frames[idx].resize((480, 320)) for idx in picks]
    montage = Image.new("RGB", (960, 640), (248, 247, 242))
    for idx, thumb in enumerate(thumbs):
        montage.paste(thumb, (480 * (idx % 2), 320 * (idx // 2)))
    montage.save(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    body = make_box_body()
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
        angular_velocity=torch.tensor([0.0, 0.0, 0.0]),
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
        frames.append(draw_frame(frame_idx, state_a, state_b, diagnostics))
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

    gif_path = OUT_DIR / "pairwise_collision_scene_3d.gif"
    montage_path = OUT_DIR / "pairwise_collision_scene_3d_montage.png"
    summary_path = OUT_DIR / "pairwise_collision_scene_3d_summary.json"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0)
    make_montage(frames, montage_path)
    summary_path.write_text(json.dumps({"frames": rows}, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "gif_path": str(gif_path),
                "montage_path": str(montage_path),
                "summary_path": str(summary_path),
                "final_position_a": rows[-1]["position_a"],
                "final_velocity_a": rows[-1]["velocity_a"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
