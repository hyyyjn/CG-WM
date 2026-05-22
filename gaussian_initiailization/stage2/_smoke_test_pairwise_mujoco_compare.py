"""Compare pairwise Stage 2 dynamics against a MuJoCo object-object trajectory.

The script is split into modes because local environments often keep MuJoCo and
PyTorch in separate conda envs:

  1. ``--mode generate_gt`` writes a MuJoCo trajectory and only imports mujoco.
  2. ``--mode compare`` rolls out Stage 2 and only imports torch.
  3. ``--mode all`` does both when one environment has both dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "gaussian_initiailization" / "stage2" / "_outputs" / "pairwise_mujoco_compare"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Stage 2 pairwise dynamics with MuJoCo GT.")
    parser.add_argument("--mode", default="all", choices=("all", "generate_gt", "compare"))
    parser.add_argument("--output_dir", default=OUT_DIR, type=Path)
    parser.add_argument("--gt_json", default=None, type=Path)
    parser.add_argument("--frames", default=90, type=int)
    parser.add_argument("--fps", default=60.0, type=float)
    parser.add_argument("--mujoco_timestep", default=0.001, type=float)
    parser.add_argument("--half_extent", default=0.10, type=float)
    parser.add_argument("--body_a_initial_position", default="-0.22,0,0", type=str)
    parser.add_argument("--body_b_position", default="0.12,0,0", type=str)
    parser.add_argument("--body_a_initial_velocity", default="1.20,0.03,0", type=str)
    parser.add_argument("--stage2_stiffness", default=250.0, type=float)
    parser.add_argument("--stage2_damping", default=10.0, type=float)
    parser.add_argument("--stage2_proxy_resolution", default=3, type=int)
    parser.add_argument("--stage2_radius_scale", default=1.0, type=float)
    parser.add_argument("--stage2_num_contact_patches", default=4, type=int)
    parser.add_argument("--stage2_broad_phase_margin", default=0.03, type=float)
    parser.add_argument("--stage2_friction_coefficient", default=0.35, type=float)
    parser.add_argument("--stage2_tangential_damping", default=4.0, type=float)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def parse_vec3(value: str) -> np.ndarray:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected x,y,z, got {value!r}")
    return np.asarray([float(part) for part in parts], dtype=np.float64)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def build_pairwise_mjcf(args: argparse.Namespace) -> str:
    half_extent = float(args.half_extent)
    timestep = float(args.mujoco_timestep)
    body_b_position = parse_vec3(args.body_b_position)
    return f"""
<mujoco model="pairwise_box_compare">
  <option timestep="{timestep}" gravity="0 0 0" integrator="Euler"/>
  <default>
    <geom type="box" size="{half_extent} {half_extent} {half_extent}" density="1000"
          friction="0.02 0.001 0.0001" solref="-800 0"/>
  </default>
  <worldbody>
    <body name="box_a" pos="0 0 0">
      <joint name="root_free" type="free" damping="0"/>
      <geom name="box_a_geom" rgba="0.9 0.25 0.18 1"/>
    </body>
    <body name="box_b" pos="{body_b_position[0]} {body_b_position[1]} {body_b_position[2]}">
      <geom name="box_b_geom" rgba="0.15 0.35 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


def generate_mujoco_gt(args: argparse.Namespace) -> Path:
    import mujoco

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(build_pairwise_mjcf(args))
    data = mujoco.MjData(model)
    body_a_initial_position = parse_vec3(args.body_a_initial_position)
    body_a_initial_velocity = parse_vec3(args.body_a_initial_velocity)
    steps_per_frame = max(1, int(round(1.0 / (float(args.fps) * float(args.mujoco_timestep)))))

    data.qpos[:3] = body_a_initial_position
    data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qvel[:] = 0.0
    data.qvel[:3] = body_a_initial_velocity
    mujoco.mj_forward(model, data)

    states = []
    for frame_idx in range(int(args.frames)):
        contacts = []
        for contact_idx in range(data.ncon):
            contact = data.contact[contact_idx]
            contacts.append(
                {
                    "geom1": int(contact.geom1),
                    "geom2": int(contact.geom2),
                    "distance": float(contact.dist),
                    "position": contact.pos.tolist(),
                    "frame": contact.frame.tolist(),
                }
            )
        states.append(
            {
                "frame_index": frame_idx,
                "time": float(data.time),
                "position": data.qpos[:3].tolist(),
                "quaternion_wxyz": data.qpos[3:7].tolist(),
                "linear_velocity": data.qvel[:3].tolist(),
                "angular_velocity": data.qvel[3:6].tolist(),
                "num_contacts": int(data.ncon),
                "contacts": contacts,
            }
        )
        if frame_idx < int(args.frames) - 1:
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

    gt_path = output_dir / "mujoco_pairwise_gt_trajectory.json"
    write_json(
        gt_path,
        {
            "metadata": {
                "generator": "_smoke_test_pairwise_mujoco_compare.py",
                "half_extent": float(args.half_extent),
                "fps": float(args.fps),
                "mujoco_timestep": float(args.mujoco_timestep),
                "steps_per_frame": int(steps_per_frame),
                "body_b_position": parse_vec3(args.body_b_position).tolist(),
                "body_a_initial_position": body_a_initial_position.tolist(),
                "body_a_initial_velocity": body_a_initial_velocity.tolist(),
            },
            "states": states,
        },
    )
    return gt_path


def make_stage2_box_body(half_extent: float, resolution: int, radius_scale: float, *, torch, device):
    from gaussian_initiailization.stage2.differentiable_collision_detection import (
        GaussianCollisionBody,
        make_box_surface_query_points,
    )

    query_points = make_box_surface_query_points(
        [half_extent, half_extent, half_extent],
        grid_resolution=int(resolution),
        dtype=torch.float32,
        device=device,
    )
    spacing = 2.0 * float(half_extent) / float(max(int(resolution) - 1, 1))
    radii = torch.full((query_points.shape[0],), spacing * 0.25 * float(radius_scale), dtype=torch.float32, device=device)
    return GaussianCollisionBody(query_points, radii, query_points)


def quaternion_angle_error_deg(q_pred: np.ndarray, q_gt: np.ndarray) -> float:
    q_pred = q_pred / max(float(np.linalg.norm(q_pred)), 1e-12)
    q_gt = q_gt / max(float(np.linalg.norm(q_gt)), 1e-12)
    dot = abs(float(np.dot(q_pred, q_gt)))
    dot = min(max(dot, -1.0), 1.0)
    return math.degrees(2.0 * math.acos(dot))


def compare_stage2_to_gt(args: argparse.Namespace, gt_path: Path) -> dict:
    import torch

    from gaussian_initiailization.stage2.differentiable_complementarity_free_contact_dynamics import (
        PairwiseGaussianBodyImpedanceDynamics,
        PairwiseImpedanceDynamicsConfig,
        RigidBodyState,
    )

    payload = read_json(gt_path)
    states = payload["states"]
    if len(states) < 2:
        raise ValueError("Need at least two GT states for comparison.")
    device = torch.device(args.device)
    dt = float(states[1]["time"] - states[0]["time"])
    body = make_stage2_box_body(
        float(args.half_extent),
        int(args.stage2_proxy_resolution),
        float(args.stage2_radius_scale),
        torch=torch,
        device=device,
    )
    dynamics = PairwiseGaussianBodyImpedanceDynamics(
        body,
        body,
        stiffness=torch.tensor(float(args.stage2_stiffness), dtype=torch.float32, device=device),
        damping=torch.tensor(float(args.stage2_damping), dtype=torch.float32, device=device),
        config=PairwiseImpedanceDynamicsConfig(
            dt=dt,
            gravity=(0.0, 0.0, 0.0),
            dynamic_a=True,
            dynamic_b=False,
            mass_a=1.0,
            num_contact_patches=int(args.stage2_num_contact_patches),
            broad_phase_margin=float(args.stage2_broad_phase_margin),
            friction_coefficient=float(args.stage2_friction_coefficient),
            tangential_damping=float(args.stage2_tangential_damping),
        ),
    )
    state_a = RigidBodyState(
        position=torch.tensor(states[0]["position"], dtype=torch.float32, device=device),
        quaternion_wxyz=torch.tensor(states[0]["quaternion_wxyz"], dtype=torch.float32, device=device),
        linear_velocity=torch.tensor(states[0]["linear_velocity"], dtype=torch.float32, device=device),
        angular_velocity=torch.tensor(states[0]["angular_velocity"], dtype=torch.float32, device=device),
    )
    state_b = RigidBodyState(
        position=torch.tensor(parse_vec3(args.body_b_position), dtype=torch.float32, device=device),
        quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device),
        linear_velocity=torch.zeros(3, dtype=torch.float32, device=device),
        angular_velocity=torch.zeros(3, dtype=torch.float32, device=device),
    )

    rows = []
    pred_positions = [state_a.position.detach().cpu().numpy()]
    pred_quats = [state_a.quaternion_wxyz.detach().cpu().numpy()]
    contact_gates = [0.0]
    lambdas = [0.0]
    for frame_idx in range(1, len(states)):
        state_a, state_b, diagnostics = dynamics.step(state_a, state_b)
        pred_positions.append(state_a.position.detach().cpu().numpy())
        pred_quats.append(state_a.quaternion_wxyz.detach().cpu().numpy())
        contact_gates.append(float(torch.max(diagnostics["patch_weights"]).detach().cpu().item()))
        lambdas.append(float(torch.max(diagnostics["lambda"]).detach().cpu().item()))

    gt_positions = np.asarray([state["position"] for state in states], dtype=np.float64)
    gt_quats = np.asarray([state["quaternion_wxyz"] for state in states], dtype=np.float64)
    pred_positions_np = np.asarray(pred_positions, dtype=np.float64)
    pred_quats_np = np.asarray(pred_quats, dtype=np.float64)
    position_errors = np.linalg.norm(pred_positions_np - gt_positions, axis=-1)
    quat_errors = np.asarray(
        [quaternion_angle_error_deg(pred_quats_np[idx], gt_quats[idx]) for idx in range(len(states))],
        dtype=np.float64,
    )
    gt_contact_frames = [int(state["frame_index"]) for state in states if int(state.get("num_contacts", 0)) > 0]
    pred_contact_frames = [idx for idx, gate in enumerate(contact_gates) if gate > 0.5]
    for idx, state in enumerate(states):
        rows.append(
            {
                "frame_index": int(state["frame_index"]),
                "time": float(state["time"]),
                "gt_position": gt_positions[idx].tolist(),
                "pred_position": pred_positions_np[idx].tolist(),
                "position_error_m": float(position_errors[idx]),
                "gt_quaternion_wxyz": gt_quats[idx].tolist(),
                "pred_quaternion_wxyz": pred_quats_np[idx].tolist(),
                "quaternion_error_deg": float(quat_errors[idx]),
                "gt_num_contacts": int(state.get("num_contacts", 0)),
                "stage2_contact_gate": float(contact_gates[idx]),
                "stage2_max_lambda": float(lambdas[idx]),
            }
        )

    summary = {
        "frames": len(states),
        "dt": dt,
        "mean_translation_error_m": float(position_errors.mean()),
        "median_translation_error_m": float(np.median(position_errors)),
        "final_translation_error_m": float(position_errors[-1]),
        "max_translation_error_m": float(position_errors.max()),
        "mean_quaternion_error_deg": float(quat_errors.mean()),
        "final_quaternion_error_deg": float(quat_errors[-1]),
        "gt_first_contact_frame": int(gt_contact_frames[0]) if gt_contact_frames else None,
        "stage2_first_contact_frame": int(pred_contact_frames[0]) if pred_contact_frames else None,
        "max_stage2_contact_gate": float(max(contact_gates)),
        "max_stage2_lambda": float(max(lambdas)),
        "stage2_config": {
            "stiffness": float(args.stage2_stiffness),
            "damping": float(args.stage2_damping),
            "proxy_resolution": int(args.stage2_proxy_resolution),
            "radius_scale": float(args.stage2_radius_scale),
            "num_contact_patches": int(args.stage2_num_contact_patches),
            "broad_phase_margin": float(args.stage2_broad_phase_margin),
            "friction_coefficient": float(args.stage2_friction_coefficient),
            "tangential_damping": float(args.stage2_tangential_damping),
        },
    }

    output_dir = args.output_dir.resolve()
    write_json(output_dir / "pairwise_mujoco_compare_summary.json", summary)
    write_json(output_dir / "pairwise_mujoco_compare_frames.json", {"frames": rows})
    draw_compare_plot(rows, output_dir / "pairwise_mujoco_compare_plot.png")
    return {**summary, "output_dir": str(output_dir), "gt_json": str(gt_path)}


def draw_compare_plot(rows: list[dict], path: Path) -> None:
    width, height = 980, 580
    margin_l, margin_r = 82, 28
    margin_t, margin_b = 64, 42
    panel_gap = 28
    panel_h = (height - margin_t - margin_b - 2 * panel_gap) // 3
    img = Image.new("RGB", (width, height), (250, 249, 246))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width, 44), fill=(28, 28, 30))
    draw.text((14, 14), "Pairwise MuJoCo GT vs Stage 2", fill=(255, 255, 245))

    gt_x = [row["gt_position"][0] for row in rows]
    pred_x = [row["pred_position"][0] for row in rows]
    pos_err = [row["position_error_m"] for row in rows]
    gt_contacts = [1.0 if row["gt_num_contacts"] > 0 else 0.0 for row in rows]
    gates = [row["stage2_contact_gate"] for row in rows]

    def draw_panel(idx, title, series, y_label):
        y_top = margin_t + idx * (panel_h + panel_gap)
        y_bot = y_top + panel_h
        x_left, x_right = margin_l, width - margin_r
        draw.rectangle((x_left, y_top, x_right, y_bot), fill=(255, 255, 255), outline=(155, 155, 155))
        draw.text((x_left + 8, y_top + 5), title, fill=(30, 30, 30))
        draw.text((14, y_top + panel_h // 2 - 6), y_label, fill=(70, 70, 70))
        values = [float(v) for vals, _ in series for v in vals]
        ymin, ymax = min(values), max(values)
        pad = max(1e-4, 0.08 * (ymax - ymin + 1e-6))
        ymin -= pad
        ymax += pad
        for vals, color in series:
            pts = []
            for i, value in enumerate(vals):
                x = x_left + int(i / max(len(vals) - 1, 1) * (x_right - x_left))
                y = int(y_bot - (float(value) - ymin) / max(ymax - ymin, 1e-6) * (y_bot - y_top))
                pts.append((x, y))
            if len(pts) > 1:
                draw.line(pts, fill=color, width=2)

    draw_panel(0, "x position: MuJoCo GT (blue), Stage 2 (orange)", [(gt_x, (35, 90, 190)), (pred_x, (220, 125, 45))], "x")
    draw_panel(1, "translation error", [(pos_err, (190, 70, 70))], "m")
    draw_panel(2, "contact: MuJoCo binary (green), Stage 2 gate (purple)", [(gt_contacts, (35, 145, 80)), (gates, (125, 75, 180))], "contact")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_path = args.gt_json.resolve() if args.gt_json else args.output_dir.resolve() / "mujoco_pairwise_gt_trajectory.json"
    result = {}
    if args.mode in ("all", "generate_gt"):
        gt_path = generate_mujoco_gt(args)
        result["gt_json"] = str(gt_path)
    if args.mode in ("all", "compare"):
        result.update(compare_stage2_to_gt(args, gt_path))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
