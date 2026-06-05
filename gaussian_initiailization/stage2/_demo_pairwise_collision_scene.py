"""Render a small pairwise Gaussian-body collision scene as GIF/PNG."""

from __future__ import annotations

import json
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

OUT_DIR = REPO_ROOT / "gaussian_initiailization" / "stage2" / "_outputs" / "pairwise_collision_scene"


def make_box_body(half_extent: float = 0.1, resolution: int = 5) -> GaussianCollisionBody:
    queries = make_box_surface_query_points([half_extent, half_extent, half_extent], grid_resolution=resolution)
    spacing = 2.0 * half_extent / float(max(resolution - 1, 1))
    radii = torch.full((queries.shape[0],), spacing * 0.32, dtype=torch.float32)
    return GaussianCollisionBody(queries, radii, queries)


def project(point: torch.Tensor, axes: tuple[int, int], bounds: tuple[int, int, int, int]) -> tuple[float, float]:
    left, top, right, bottom = bounds
    x_axis, y_axis = axes
    xmin, xmax = -0.42, 0.46
    ymin, ymax = -0.26, 0.26
    if y_axis == 2:
        ymin, ymax = -0.18, 0.22
    x = left + (float(point[x_axis]) - xmin) / (xmax - xmin) * (right - left)
    y = bottom - (float(point[y_axis]) - ymin) / (ymax - ymin) * (bottom - top)
    return x, y


def draw_box(draw: ImageDraw.ImageDraw, center: torch.Tensor, axes, bounds, *, fill, outline) -> None:
    half = 0.1
    x_axis, y_axis = axes
    corners = []
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        point = center.clone()
        point[x_axis] += sx * half
        point[y_axis] += sy * half
        corners.append(project(point, axes, bounds))
    draw.polygon(corners, fill=fill, outline=outline)


def draw_frame(frame_idx: int, state_a: RigidBodyState, state_b: RigidBodyState, diagnostics) -> Image.Image:
    width, height = 980, 540
    img = Image.new("RGB", (width, height), (250, 249, 246))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width, 46), fill=(28, 28, 30))
    draw.text((14, 15), f"Pairwise collision scene | frame {frame_idx:03d}", fill=(255, 255, 245))

    panels = [
        ("top view: x/y", (0, 1), (54, 78, 450, 480)),
        ("side view: x/z", (0, 2), (530, 78, 926, 480)),
    ]
    contacts = diagnostics.get("contacts")
    lambdas = diagnostics.get("lambda")

    for title, axes, bounds in panels:
        left, top, right, bottom = bounds
        draw.rectangle(bounds, fill=(255, 255, 255), outline=(150, 150, 150))
        draw.text((left + 8, top + 8), title, fill=(35, 35, 35))
        draw_box(draw, state_b.position.detach().cpu(), axes, bounds, fill=(224, 235, 255), outline=(65, 105, 180))
        draw_box(draw, state_a.position.detach().cpu(), axes, bounds, fill=(255, 226, 214), outline=(190, 75, 55))

        if contacts is not None:
            patch_points = contacts.patch_points.detach().cpu()
            patch_normals = contacts.patch_normals.detach().cpu()
            patch_weights = diagnostics["patch_weights"].detach().cpu()
            for point, normal, weight in zip(patch_points, patch_normals, patch_weights):
                if float(weight) <= 0.05:
                    continue
                x, y = project(point, axes, bounds)
                radius = 3 + 8 * float(weight)
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(80, 180, 110), outline=(25, 110, 55))
                end = point + normal * 0.045
                x2, y2 = project(end, axes, bounds)
                draw.line((x, y, x2, y2), fill=(20, 120, 55), width=2)

    speed = float(torch.linalg.norm(state_a.linear_velocity).detach().cpu().item())
    max_lambda = float(torch.max(lambdas).detach().cpu().item()) if lambdas is not None else 0.0
    draw.rectangle((54, 490, 926, 526), fill=(255, 255, 255), outline=(190, 190, 190))
    draw.text(
        (66, 501),
        f"red: moving body A | blue: static body B | green: active contact patches | |v_A|={speed:.3f}, max lambda={max_lambda:.3f}",
        fill=(45, 45, 45),
    )
    return img


def make_montage(frames: list[Image.Image], path: Path) -> None:
    picks = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
    thumbs = [frames[idx].resize((490, 270)) for idx in picks]
    montage = Image.new("RGB", (980, 540), (250, 249, 246))
    for idx, thumb in enumerate(thumbs):
        x = 490 * (idx % 2)
        y = 270 * (idx // 2)
        montage.paste(thumb, (x, y))
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

    gif_path = OUT_DIR / "pairwise_collision_scene.gif"
    montage_path = OUT_DIR / "pairwise_collision_scene_montage.png"
    summary_path = OUT_DIR / "pairwise_collision_scene_summary.json"
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
