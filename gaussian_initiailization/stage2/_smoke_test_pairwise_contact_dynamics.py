"""Smoke test for pairwise Gaussian-body multi-contact dynamics."""

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

OUT_DIR = REPO_ROOT / "gaussian_initiailization" / "stage2" / "_outputs" / "pairwise_contact_dynamics"


def make_box_body(half_extent: float = 0.1, resolution: int = 3) -> GaussianCollisionBody:
    query_points = make_box_surface_query_points(
        [half_extent, half_extent, half_extent],
        grid_resolution=resolution,
        dtype=torch.float32,
    )
    spacing = 2.0 * float(half_extent) / float(max(resolution - 1, 1))
    radii = torch.full((query_points.shape[0],), spacing * 0.25, dtype=torch.float32)
    return GaussianCollisionBody(query_points, radii, query_points)


def draw_pairwise_contact_plot(state_a, state_b, next_a, diagnostics, path: Path) -> None:
    width, height = 980, 520
    img = Image.new("RGB", (width, height), (250, 249, 246))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width, 44), fill=(28, 28, 30))
    draw.text((14, 14), "Pairwise Gaussian contact dynamics smoke test", fill=(255, 255, 245))

    patch_points = diagnostics["contacts"].patch_points.detach().cpu()
    patch_normals = diagnostics["contacts"].patch_normals.detach().cpu()
    patch_weights = diagnostics["patch_weights"].detach().cpu()
    pos_a = state_a.position.detach().cpu()
    pos_b = state_b.position.detach().cpu()
    next_pos_a = next_a.position.detach().cpu()

    panels = [
        ("top view: x/y", (0, 1), (54, 74, 450, 470)),
        ("side view: x/z", (0, 2), (530, 74, 926, 470)),
    ]
    box_half = 0.1

    def project(point, axes, bounds):
        left, top, right, bottom = bounds
        x_axis, y_axis = axes
        xmin, xmax = -0.16, 0.31
        ymin, ymax = -0.18, 0.18
        if y_axis == 2:
            ymin, ymax = -0.15, 0.15
        x = left + (float(point[x_axis]) - xmin) / (xmax - xmin) * (right - left)
        y = bottom - (float(point[y_axis]) - ymin) / (ymax - ymin) * (bottom - top)
        return x, y

    def draw_box(center, axes, bounds, color, outline):
        x_axis, y_axis = axes
        corners = []
        for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            point = center.clone()
            point[x_axis] += sx * box_half
            point[y_axis] += sy * box_half
            corners.append(project(point, axes, bounds))
        draw.polygon(corners, fill=color, outline=outline)

    for title, axes, bounds in panels:
        left, top, right, bottom = bounds
        draw.rectangle(bounds, fill=(255, 255, 255), outline=(150, 150, 150))
        draw.text((left + 8, top + 8), title, fill=(35, 35, 35))
        draw.line((left + 18, bottom - 28, right - 18, bottom - 28), fill=(220, 220, 220))
        draw_box(pos_b, axes, bounds, (225, 235, 255), (70, 110, 185))
        draw_box(pos_a, axes, bounds, (255, 226, 216), (190, 80, 55))
        draw_box(next_pos_a, axes, bounds, (255, 245, 206), (190, 150, 55))

        for point, normal, weight in zip(patch_points, patch_normals, patch_weights):
            x, y = project(point, axes, bounds)
            radius = 4 + 8 * float(weight)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(80, 180, 110), outline=(25, 110, 55))
            normal_end = point + normal * 0.045
            x2, y2 = project(normal_end, axes, bounds)
            draw.line((x, y, x2, y2), fill=(20, 120, 55), width=2)

    draw.rectangle((54, 476, 926, 504), fill=(255, 255, 255), outline=(190, 190, 190))
    draw.text(
        (66, 484),
        "red: body A before step, yellow: body A after step, blue: static body B, green: contact patches/normals",
        fill=(45, 45, 45),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def draw_rollout_plot(rows: list[dict], path: Path) -> None:
    width, height = 980, 560
    margin_l, margin_r = 82, 28
    margin_t, margin_b = 64, 44
    panel_gap = 28
    panel_h = (height - margin_t - margin_b - panel_gap) // 2
    img = Image.new("RGB", (width, height), (250, 249, 246))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width, 44), fill=(28, 28, 30))
    draw.text((14, 14), "Pairwise Gaussian contact rollout", fill=(255, 255, 245))

    frames = [row["frame"] for row in rows]
    x_values = [row["position_a"][0] for row in rows]
    vx_values = [row["velocity_a"][0] for row in rows]
    weights = [row["max_patch_weight"] for row in rows]
    lambdas = [row["max_lambda"] for row in rows]

    def draw_panel(idx: int, title: str, series: list[tuple[list[float], tuple[int, int, int]]], y_label: str):
        y_top = margin_t + idx * (panel_h + panel_gap)
        y_bot = y_top + panel_h
        x_left = margin_l
        x_right = width - margin_r
        draw.rectangle((x_left, y_top, x_right, y_bot), fill=(255, 255, 255), outline=(155, 155, 155))
        draw.text((x_left + 8, y_top + 6), title, fill=(30, 30, 30))
        draw.text((14, y_top + panel_h // 2 - 6), y_label, fill=(70, 70, 70))
        values = [float(value) for vals, _ in series for value in vals]
        ymin = min(values)
        ymax = max(values)
        pad = max(1e-4, 0.08 * (ymax - ymin + 1e-6))
        ymin -= pad
        ymax += pad
        if ymin <= 0.0 <= ymax:
            zy = int(y_bot - (0.0 - ymin) / max(ymax - ymin, 1e-6) * (y_bot - y_top))
            draw.line((x_left, zy, x_right, zy), fill=(220, 220, 220), width=1)
        for tick_idx in range(5):
            x = x_left + int(tick_idx / 4 * (x_right - x_left))
            draw.line((x, y_bot, x, y_bot + 4), fill=(120, 120, 120))
            frame = frames[0] + int((frames[-1] - frames[0]) * tick_idx / 4)
            draw.text((x - 10, y_bot + 8), str(frame), fill=(80, 80, 80))
        for vals, color in series:
            pts = []
            for point_idx, value in enumerate(vals):
                x = x_left + int(point_idx / max(len(vals) - 1, 1) * (x_right - x_left))
                y = int(y_bot - (float(value) - ymin) / max(ymax - ymin, 1e-6) * (y_bot - y_top))
                pts.append((x, y))
            if len(pts) > 1:
                draw.line(pts, fill=color, width=2)

    draw_panel(
        0,
        "body A x-position (blue) and x-velocity (orange)",
        [(x_values, (35, 90, 190)), (vx_values, (220, 125, 45))],
        "x/vx",
    )
    draw_panel(
        1,
        "max contact patch weight (green) and max lambda (purple)",
        [(weights, (35, 145, 80)), (lambdas, (125, 75, 180))],
        "contact",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def rollout_pairwise(dynamics, state_a: RigidBodyState, state_b: RigidBodyState, num_steps: int):
    rows = []
    diagnostics = None
    lambda_terms = []
    for frame_idx in range(int(num_steps)):
        state_a, state_b, diagnostics = dynamics.step(state_a, state_b)
        lambda_terms.append(diagnostics["lambda"].sum())
        rows.append(
            {
                "frame": frame_idx + 1,
                "position_a": state_a.position.detach().cpu().tolist(),
                "quaternion_a": state_a.quaternion_wxyz.detach().cpu().tolist(),
                "velocity_a": state_a.linear_velocity.detach().cpu().tolist(),
                "angular_velocity_a": state_a.angular_velocity.detach().cpu().tolist(),
                "position_b": state_b.position.detach().cpu().tolist(),
                "max_lambda": float(torch.max(diagnostics["lambda"]).detach().cpu().item()),
                "mean_lambda": float(torch.mean(diagnostics["lambda"]).detach().cpu().item()),
                "max_patch_weight": float(torch.max(diagnostics["patch_weights"]).detach().cpu().item()),
                "max_penetration": float(torch.max(diagnostics["patch_penetrations"]).detach().cpu().item()),
                "max_friction_force": float(torch.linalg.norm(diagnostics["friction_force"], dim=-1).max().detach().cpu().item()),
                "max_tangential_speed": float(torch.linalg.norm(diagnostics["tangential_velocity"], dim=-1).max().detach().cpu().item()),
                "broad_phase_overlaps": bool(diagnostics["broad_phase_overlaps"].detach().cpu().item()),
            }
        )
    return state_a, state_b, diagnostics, rows, torch.stack(lambda_terms)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    body = make_box_body()
    dynamics = PairwiseGaussianBodyImpedanceDynamics(
        body,
        body,
        stiffness=torch.tensor(250.0),
        damping=torch.tensor(10.0),
        config=PairwiseImpedanceDynamicsConfig(
            gravity=(0.0, 0.0, 0.0),
            dynamic_a=True,
            dynamic_b=False,
            num_contact_patches=4,
            broad_phase_margin=0.02,
            friction_coefficient=0.35,
            tangential_damping=4.0,
            inertia_matrix_a=((1.0, 0.08, 0.0), (0.08, 1.25, 0.03), (0.0, 0.03, 0.90)),
        ),
    )
    initial_state_a = RigidBodyState(
        position=torch.tensor([-0.14, 0.0, 0.0], requires_grad=True),
        quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        linear_velocity=torch.tensor([1.0, 0.0, 0.0]),
        angular_velocity=torch.tensor([0.0, 0.0, 0.0]),
    )
    initial_state_b = RigidBodyState(
        position=torch.tensor([0.12, 0.0, 0.0]),
        quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        linear_velocity=torch.tensor([0.0, 0.0, 0.0]),
        angular_velocity=torch.tensor([0.0, 0.0, 0.0]),
    )

    one_step_a, next_b, one_step_diagnostics = dynamics.step(initial_state_a, initial_state_b)
    final_a, final_b, diagnostics, rollout_rows, lambda_terms = rollout_pairwise(
        dynamics,
        initial_state_a,
        initial_state_b,
        num_steps=72,
    )
    loss = final_a.position.sum() + final_a.linear_velocity.sum() + lambda_terms.sum()
    loss.backward()
    contact_frames = [row["frame"] for row in rollout_rows if row["max_patch_weight"] > 0.5]

    summary = {
        "num_rollout_steps": len(rollout_rows),
        "first_contact_frame": int(contact_frames[0]) if contact_frames else None,
        "max_rollout_patch_weight": max(row["max_patch_weight"] for row in rollout_rows),
        "max_rollout_lambda": max(row["max_lambda"] for row in rollout_rows),
        "max_rollout_friction_force": max(row["max_friction_force"] for row in rollout_rows),
        "max_rollout_tangential_speed": max(row["max_tangential_speed"] for row in rollout_rows),
        "broad_phase_overlaps_final": bool(diagnostics["broad_phase_overlaps"].detach().cpu().item()),
        "lambda_final": diagnostics["lambda"].detach().cpu().tolist(),
        "patch_weights_final": diagnostics["patch_weights"].detach().cpu().tolist(),
        "friction_force_final": diagnostics["friction_force"].detach().cpu().tolist(),
        "tangent_velocity_components_final": diagnostics["tangent_velocity_components"].detach().cpu().tolist(),
        "angular_delta_final": diagnostics["angular_delta_a"].detach().cpu().tolist(),
        "one_step_a_position": one_step_a.position.detach().cpu().tolist(),
        "one_step_a_linear_velocity": one_step_a.linear_velocity.detach().cpu().tolist(),
        "final_a_position": final_a.position.detach().cpu().tolist(),
        "final_a_linear_velocity": final_a.linear_velocity.detach().cpu().tolist(),
        "final_a_angular_velocity": final_a.angular_velocity.detach().cpu().tolist(),
        "next_b_position": next_b.position.detach().cpu().tolist(),
        "gradient_is_finite": bool(torch.isfinite(initial_state_a.position.grad).all().detach().cpu().item()),
    }
    output_path = OUT_DIR / "pairwise_contact_dynamics_summary.json"
    frames_path = OUT_DIR / "pairwise_contact_dynamics_frames.json"
    plot_path = OUT_DIR / "pairwise_contact_dynamics_plot.png"
    rollout_plot_path = OUT_DIR / "pairwise_contact_dynamics_rollout_plot.png"
    draw_pairwise_contact_plot(initial_state_a, initial_state_b, one_step_a, one_step_diagnostics, plot_path)
    draw_rollout_plot(rollout_rows, rollout_plot_path)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    frames_path.write_text(json.dumps({"frames": rollout_rows}, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                **summary,
                "output_path": str(output_path),
                "frames_path": str(frames_path),
                "plot_path": str(plot_path),
                "rollout_plot_path": str(rollout_plot_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
