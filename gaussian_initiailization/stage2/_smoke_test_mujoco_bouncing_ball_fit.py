"""Generate a MuJoCo bouncing ball trajectory and fit Stage 2 floor dynamics."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "gaussian_initiailization" / "stage2" / "_outputs" / "mujoco_bouncing_ball_fit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo bouncing-ball GT vs Stage 2 restitution fit.")
    parser.add_argument("--mode", default="all", choices=("generate_gt", "fit", "all"))
    parser.add_argument("--output_dir", default=OUT_DIR, type=Path)
    parser.add_argument("--gt_json", default=None, type=Path)
    parser.add_argument("--frames", default=120, type=int)
    parser.add_argument("--steps_per_frame", default=4, type=int)
    parser.add_argument("--mujoco_timestep", default=0.005, type=float)
    parser.add_argument("--sphere_radius", default=0.08, type=float)
    parser.add_argument("--initial_height", default=0.95, type=float)
    parser.add_argument("--initial_velocity_z", default=0.0, type=float)
    parser.add_argument("--fit_iters", default=220, type=int)
    parser.add_argument("--lr", default=0.04, type=float)
    parser.add_argument("--query_radius_scale", default=1.1, type=float)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_mjcf(args: argparse.Namespace) -> str:
    radius = float(args.sphere_radius)
    return f"""
<mujoco model="cgwm_bouncing_ball">
  <option timestep="{float(args.mujoco_timestep)}" gravity="0 0 -9.81" integrator="Euler"/>
  <default>
    <geom solref="0.015 0.35" solimp="0.90 0.98 0.001" friction="0.6 0.01 0.001" condim="3"/>
  </default>
  <worldbody>
    <light name="key" pos="1 -1 2"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.92 0.92 0.92 1"/>
    <body name="ball" pos="0 0 {float(args.initial_height)}">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="{radius}" mass="1.0" rgba="0.2 0.45 0.85 1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


def generate_gt(args: argparse.Namespace) -> Path:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    xml = build_mjcf(args)
    write_json(output_dir / "bouncing_ball_scene_manifest.json", {"mjcf": xml})
    (output_dir / "bouncing_ball_scene.xml").write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qvel[2] = float(args.initial_velocity_z)
    mujoco.mj_forward(model, data)

    states = []
    frame_count = int(args.frames)
    steps_per_frame = int(args.steps_per_frame)
    for frame_idx in range(frame_count):
        states.append(
            {
                "frame_index": frame_idx,
                "time": float(data.time),
                "position": data.qpos[:3].tolist(),
                "quaternion_wxyz": data.qpos[3:7].tolist(),
                "linear_velocity": data.qvel[:3].tolist(),
                "angular_velocity": data.qvel[3:6].tolist(),
            }
        )
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

    positions = np.asarray([state["position"] for state in states], dtype=np.float64)
    z = positions[:, 2]
    min_idx = int(np.argmin(z))
    post_bounce_peak = float(np.max(z[min(min_idx + 1, len(z) - 1):]))
    summary = {
        "generator": "_smoke_test_mujoco_bouncing_ball_fit.py",
        "frames": frame_count,
        "dt": float(args.mujoco_timestep) * steps_per_frame,
        "sphere_radius": float(args.sphere_radius),
        "min_z": float(np.min(z)),
        "min_z_frame": min_idx,
        "post_bounce_peak_z": post_bounce_peak,
        "bounce_detected": bool(np.min(z) <= float(args.sphere_radius) + 0.02 and post_bounce_peak > float(args.sphere_radius) + 0.04),
    }
    payload = {"summary": summary, "states": states}
    gt_path = output_dir / "mujoco_bouncing_ball_trajectory.json"
    write_json(gt_path, payload)
    print(json.dumps({**summary, "gt_json": str(gt_path)}, indent=2), flush=True)
    return gt_path


def draw_fit_plot(rows: list[dict], path: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 980, 480
    margin_l, margin_r = 72, 28
    margin_t, margin_b = 58, 44
    img = Image.new("RGB", (width, height), (250, 249, 246))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width, 42), fill=(28, 28, 30))
    draw.text((14, 13), "MuJoCo bouncing ball GT vs Stage 2 fit", fill=(255, 255, 245))
    x_left, x_right = margin_l, width - margin_r
    y_top, y_bot = margin_t, height - margin_b
    draw.rectangle((x_left, y_top, x_right, y_bot), fill=(255, 255, 255), outline=(155, 155, 155))
    gt_z = [row["target_position"][2] for row in rows]
    pred_z = [row["predicted_position"][2] for row in rows]
    values = gt_z + pred_z
    ymin = min(values) - 0.04
    ymax = max(values) + 0.04
    for value, color in ((0.0, (210, 210, 210)), (rows[0]["sphere_radius"], (190, 190, 190))):
        y = int(y_bot - (value - ymin) / max(ymax - ymin, 1e-6) * (y_bot - y_top))
        draw.line((x_left, y, x_right, y), fill=color, width=1)
    for series, color in ((gt_z, (35, 90, 190)), (pred_z, (220, 110, 55))):
        points = []
        for idx, value in enumerate(series):
            x = x_left + int(idx / max(len(series) - 1, 1) * (x_right - x_left))
            y = int(y_bot - (float(value) - ymin) / max(ymax - ymin, 1e-6) * (y_bot - y_top))
            points.append((x, y))
        draw.line(points, fill=color, width=2)
    draw.text((x_left + 10, y_top + 8), "blue: MuJoCo z, orange: Stage 2 z", fill=(40, 40, 40))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def draw_fit_gif(rows: list[dict], path: Path, *, fps: int = 24) -> None:
    from PIL import Image, ImageDraw

    width, height = 860, 560
    horizon_y = 178
    floor_y = height - 78
    center_x = width // 2
    target_z = [float(row["target_position"][2]) for row in rows]
    pred_z = [float(row["predicted_position"][2]) for row in rows]
    radius_world = float(rows[0]["sphere_radius"])
    z_min = min(0.0, min(target_z), min(pred_z)) - 0.04
    z_max = max(max(target_z), max(pred_z), radius_world * 3.0) + 0.06

    def z_to_y(z_value: float) -> float:
        t = (float(z_value) - z_min) / max(z_max - z_min, 1e-6)
        return floor_y - t * (floor_y - 76)

    radius_px = max(10, int(radius_world / max(z_max - z_min, 1e-6) * (floor_y - 58)))

    def project_ground(x_world: float, y_world: float) -> tuple[float, float]:
        depth = (float(y_world) + 1.05) / 2.35
        depth = max(0.02, min(1.0, depth))
        scale = 0.28 + 0.92 * depth
        x = center_x + float(x_world) * 260.0 * scale
        y = horizon_y + depth * (floor_y - horizon_y)
        return x, y

    def draw_floor(draw: ImageDraw.ImageDraw) -> None:
        draw.polygon(
            [(88, floor_y), (width - 88, floor_y), (width - 262, horizon_y), (262, horizon_y)],
            fill=(236, 235, 226),
            outline=(188, 185, 172),
        )

    def draw_ball(draw: ImageDraw.ImageDraw, xy: tuple[float, float], z_value: float, color: tuple[int, int, int], outline: tuple[int, int, int]) -> None:
        ground_x, ground_y = xy
        cy = z_to_y(z_value)
        cx = ground_x
        draw.ellipse((cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px), fill=outline)
        inner = radius_px - 3
        draw.ellipse((cx - inner, cy - inner, cx + inner, cy + inner), fill=color)

    frames = []
    for idx, row in enumerate(rows):
        img = Image.new("RGB", (width, height), (242, 244, 246))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, width, 48), fill=(28, 28, 30))
        draw.text((14, 16), f"MuJoCo bouncing ball vs Stage 2 | frame {idx:03d}", fill=(255, 255, 245))
        draw_floor(draw)

        gate = float(row["contact_gate"])
        pred_fill = (238, 132, 61) if gate < 0.5 else (220, 72, 48)
        gt_ground = project_ground(-0.32, -0.10)
        pred_ground = project_ground(0.32, 0.10)
        draw_ball(draw, gt_ground, row["target_position"][2], (65, 138, 217), (24, 76, 135))
        draw_ball(draw, pred_ground, row["predicted_position"][2], pred_fill, (130, 62, 30))
        draw.text((int(gt_ground[0]) - 42, floor_y + 24), "MuJoCo", fill=(24, 76, 135))
        draw.text((int(pred_ground[0]) - 38, floor_y + 24), "Stage 2", fill=(130, 62, 30))
        draw.text(
            (18, height - 30),
            f"gt_z={row['target_position'][2]:.3f} pred_z={row['predicted_position'][2]:.3f} contact_gate={gate:.3f}",
            fill=(30, 30, 28),
        )
        frames.append(img)

    duration_ms = max(1, int(round(1000.0 / float(fps))))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)


def fit_stage2(args: argparse.Namespace, gt_path: Path) -> Path:
    import torch

    from gaussian_initiailization.stage2.differentiable_collision_detection import make_floor_disk_query_points
    from gaussian_initiailization.tools.run_stage2_mujoco_stage1_fit import simulate

    payload = json.loads(gt_path.read_text(encoding="utf-8"))
    states = payload["states"]
    target_positions = torch.tensor([state["position"] for state in states], dtype=torch.float32, device=args.device)
    times = torch.tensor([state["time"] for state in states], dtype=torch.float32, device=args.device)
    dt = float(torch.median(times[1:] - times[:-1]).detach().cpu().item())
    radius = float(args.sphere_radius)
    local_centers = torch.zeros((1, 3), dtype=torch.float32, device=args.device)
    radii = torch.tensor([radius], dtype=torch.float32, device=args.device)
    floor_query_offsets_xy = make_floor_disk_query_points(
        radius * float(args.query_radius_scale),
        num_rings=5,
        num_angles=32,
        dtype=torch.float32,
        device=args.device,
    )
    initial_position = target_positions[0].detach()
    initial_velocity = torch.nn.Parameter(((target_positions[1] - target_positions[0]) / dt).detach().clone())
    gravity_z = torch.nn.Parameter(torch.tensor(-9.81, dtype=torch.float32, device=args.device))
    raw_restitution = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=args.device))
    optimizer = torch.optim.Adam([initial_velocity, gravity_z, raw_restitution], lr=float(args.lr))
    initial_loss = None
    final_loss = None
    for iteration in range(max(1, int(args.fit_iters))):
        optimizer.zero_grad(set_to_none=True)
        predicted, gates = simulate(
            initial_position,
            initial_velocity,
            gravity_z,
            torch.sigmoid(raw_restitution),
            local_centers,
            radii,
            floor_query_offsets_xy,
            steps=target_positions.shape[0],
            dt=dt,
            contact_softness=2e-3,
            smooth_max_temperature=1e-2,
            inside_penalty=0.02,
            inside_sharpness=50.0,
        )
        position_loss = torch.mean((predicted - target_positions) ** 2)
        loss = position_loss + 1e-4 * torch.mean(initial_velocity * initial_velocity)
        loss.backward()
        optimizer.step()
        if iteration == 0:
            initial_loss = float(position_loss.detach().cpu().item())
        final_loss = float(position_loss.detach().cpu().item())

    with torch.no_grad():
        predicted, gates = simulate(
            initial_position,
            initial_velocity,
            gravity_z,
            torch.sigmoid(raw_restitution),
            local_centers,
            radii,
            floor_query_offsets_xy,
            steps=target_positions.shape[0],
            dt=dt,
            contact_softness=2e-3,
            smooth_max_temperature=1e-2,
            inside_penalty=0.02,
            inside_sharpness=50.0,
        )
    target_cpu = target_positions.detach().cpu()
    pred_cpu = predicted.detach().cpu()
    gates_cpu = gates.detach().cpu()
    contact_indices = torch.nonzero(gates_cpu > 0.5).flatten()
    first_stage2_contact_frame = int(contact_indices[0].item() + 1) if contact_indices.numel() else None
    target_z = target_cpu[:, 2].numpy()
    pred_z = pred_cpu[:, 2].numpy()
    target_min_frame = int(np.argmin(target_z))
    pred_min_frame = int(np.argmin(pred_z))
    rows = []
    for idx in range(pred_cpu.shape[0]):
        rows.append(
            {
                "frame_index": int(states[idx]["frame_index"]),
                "time": float(states[idx]["time"]),
                "sphere_radius": radius,
                "target_position": target_cpu[idx].tolist(),
                "predicted_position": pred_cpu[idx].tolist(),
                "contact_gate": float(gates_cpu[idx - 1].item()) if idx > 0 and idx - 1 < gates_cpu.numel() else 0.0,
            }
        )
    summary = {
        "frames": len(rows),
        "dt": dt,
        "sphere_radius": radius,
        "initial_position_loss": initial_loss,
        "final_position_loss": final_loss,
        "position_rmse": float(torch.sqrt(torch.mean((predicted - target_positions) ** 2)).detach().cpu().item()),
        "z_rmse": float(np.sqrt(np.mean((pred_z - target_z) ** 2))),
        "learned_initial_velocity": initial_velocity.detach().cpu().tolist(),
        "learned_gravity_z": float(gravity_z.detach().cpu().item()),
        "learned_restitution": float(torch.sigmoid(raw_restitution).detach().cpu().item()),
        "target_min_z": float(np.min(target_z)),
        "predicted_min_z": float(np.min(pred_z)),
        "target_min_frame": target_min_frame,
        "predicted_min_frame": pred_min_frame,
        "stage2_first_contact_frame": first_stage2_contact_frame,
        "target_bounce_detected": bool(payload["summary"].get("bounce_detected", False)),
        "fit_contact_detected": first_stage2_contact_frame is not None,
    }
    output_dir = args.output_dir.resolve()
    write_json(output_dir / "stage2_bouncing_ball_fit_summary.json", summary)
    write_json(output_dir / "stage2_bouncing_ball_fit_frames.json", {"frames": rows})
    plot_path = output_dir / "stage2_bouncing_ball_fit_plot.png"
    gif_path = output_dir / "stage2_bouncing_ball_fit.gif"
    draw_fit_plot(rows, plot_path)
    draw_fit_gif(rows, gif_path)
    print(json.dumps({**summary, "plot_path": str(plot_path), "gif_path": str(gif_path)}, indent=2), flush=True)
    return output_dir / "stage2_bouncing_ball_fit_summary.json"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_path = args.gt_json.resolve() if args.gt_json is not None else args.output_dir.resolve() / "mujoco_bouncing_ball_trajectory.json"
    if args.mode in ("generate_gt", "all"):
        gt_path = generate_gt(args)
    if args.mode in ("fit", "all"):
        fit_stage2(args, gt_path)


if __name__ == "__main__":
    main()
