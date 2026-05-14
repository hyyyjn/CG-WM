"""End-to-end Stage 2 sphere-on-floor test.

Pipeline:
  1. Synthesise a stage1-like PLY of the sphere (Fibonacci lattice of
     spherical Gaussians on a radius-0.10 surface). This stands in for the
     real Stage 1 SG-GS output; for a uniformly textured sphere the actual
     trained representation should converge to a similar distribution.
  2. Lay out a Stage 2 dataset (objects/sphere, fall_and_rebound train+test).
  3. Run ``generate_mujoco_fall_dataset.py`` to render real MuJoCo trajectories
     and RGB frames for the sphere.
  4. Run ``run_stage2_mujoco_stage1_fit.py`` (impedance, log-K/D) on one train
     episode to learn (v0, g, K, D).
  5. Re-roll the learned dynamics on every test episode and compute paper-style
     translation error vs MuJoCo ground truth.
  6. Plot a comparison figure for one test episode.

Outputs land in ``gaussian_initiailization/tools/_stage2_e2e/``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OUT_ROOT = REPO_ROOT / "gaussian_initiailization" / "stage2" / "_outputs" / "e2e"
DATASET_ROOT = OUT_ROOT / "dataset"
STAGE1_DIR = OUT_ROOT / "stage1_sphere"
FIT_DIR = OUT_ROOT / "fit"

SPHERE_RADIUS = 0.10
NUM_STAGE1_PRIMS = 1        # single primitive at origin: perfectly axisymmetric
STAGE1_PRIM_SCALE = 0.05    # exp(log_scale) ≈ 0.05 → r_i = 2s ≈ 0.10 = sphere radius
FPS = 60
FRAMES_PER_EPISODE = 180   # 3.0 s of motion: lets the settled phase dominate the loss
TRAIN_EPISODES = 1
TEST_EPISODES = 3
IMAGE_W, IMAGE_H = 640, 480
GENERATOR = REPO_ROOT / "gaussian_initiailization" / "tools" / "generate_mujoco_fall_dataset.py"
FITTER = REPO_ROOT / "gaussian_initiailization" / "tools" / "run_stage2_mujoco_stage1_fit.py"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as h:
        json.dump(payload, h, indent=2)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as h:
        return json.load(h)


def fibonacci_sphere(n: int) -> np.ndarray:
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    indices = np.arange(n, dtype=np.float64)
    golden = 2.399963229728653
    z = 1.0 - (2.0 * indices + 1.0) / float(n)
    radial = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    theta = golden * indices
    return np.stack((radial * np.cos(theta), radial * np.sin(theta), z), axis=-1)


def synthesize_stage1_sphere_ply() -> Path:
    """Produce a minimal stage1 PLY that the fit script can read directly."""
    ply_path = STAGE1_DIR / "point_cloud" / "iteration_0" / "point_cloud.ply"
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    if NUM_STAGE1_PRIMS == 1:
        centers = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    else:
        centers = fibonacci_sphere(NUM_STAGE1_PRIMS) * SPHERE_RADIUS
    log_scale = float(np.log(STAGE1_PRIM_SCALE))
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {centers.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "end_header",
    ]
    for c in centers:
        lines.append(
            f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f} {log_scale:.6f} {log_scale:.6f} {log_scale:.6f}"
        )
    with open(ply_path, "w", encoding="utf-8") as h:
        h.write("\n".join(lines) + "\n")
    return ply_path


def setup_dataset_layout() -> None:
    obj_root = DATASET_ROOT / "objects" / "sphere"
    obj_root.mkdir(parents=True, exist_ok=True)
    write_json(
        obj_root / "object_manifest.json",
        {
            "object_name": "sphere",
            "physics_shape": "sphere",
            "mesh_path": "",
            "stage1_dataset_path": str(STAGE1_DIR),
            "stage1_points_ply": "",
            "normalization": {
                "bbox_min": [-SPHERE_RADIUS, -SPHERE_RADIUS, -SPHERE_RADIUS],
                "bbox_max": [SPHERE_RADIUS, SPHERE_RADIUS, SPHERE_RADIUS],
                "scale": 1.0,
            },
        },
    )

    scenario_root = DATASET_ROOT / "stage2" / "fall_and_rebound"
    scenario_root.mkdir(parents=True, exist_ok=True)
    write_json(
        scenario_root / "scenario_manifest.json",
        {
            "scenario": "fall_and_rebound",
            "image_size": [IMAGE_W, IMAGE_H],
            "frames_per_episode": FRAMES_PER_EPISODE,
            "fps": FPS,
        },
    )

    for split, count in (("train", TRAIN_EPISODES), ("test", TEST_EPISODES)):
        split_root = scenario_root / split / "sphere"
        split_root.mkdir(parents=True, exist_ok=True)
        for ep in range(count):
            ep_root = split_root / f"episode_{ep:03d}"
            ep_root.mkdir(parents=True, exist_ok=True)
            write_json(
                ep_root / "episode_manifest.json",
                {
                    "episode_id": f"{split}_{ep:03d}",
                    "scenario": "fall_and_rebound",
                    "split": split,
                    "object_name": "sphere",
                    "frames_per_episode": FRAMES_PER_EPISODE,
                    "fps": FPS,
                    "image_size": [IMAGE_W, IMAGE_H],
                    "physics_shape": "sphere",
                    "normalization": {
                        "bbox_min": [-SPHERE_RADIUS, -SPHERE_RADIUS, -SPHERE_RADIUS],
                        "bbox_max": [SPHERE_RADIUS, SPHERE_RADIUS, SPHERE_RADIUS],
                        "scale": 1.0,
                    },
                },
            )


def run_mujoco_generator(split: str, seed: int) -> None:
    cmd = [
        sys.executable,
        str(GENERATOR),
        "--dataset_root", str(DATASET_ROOT),
        "--object_name", "sphere",
        "--split", split,
        "--fps", str(FPS),
        "--seed", str(seed),
        "--ground_size", "2.0",
        "--camera_distance", "1.4",
        "--camera_height", "0.8",
        "--drop_height_train", "0.6",
        "--drop_height_test", "0.7",
        "--xy_range_train", "0.0",
        "--xy_range_test", "0.05",
        "--max_tilt_deg_train", "0.0",
        "--max_tilt_deg_test", "0.0",
        "--planar_speed_train", "0.0",
        "--planar_speed_test", "0.05",
        "--spin_speed_train", "0.0",
        "--spin_speed_test", "0.0",
        "--sphere_solref", "-2000 -50",
        "--sphere_friction", "0.02 0.001 0.0001",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"[mujoco {split}] returncode={result.returncode}")
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"MuJoCo generator failed for split={split}")


def run_fit(episode_root: Path, ply_path: Path) -> dict:
    cmd = [
        sys.executable,
        str(FITTER),
        "--episode_root", str(episode_root),
        "--stage1_ply", str(ply_path),
        "--output_dir", str(FIT_DIR),
        "--max_frames", str(FRAMES_PER_EPISODE),
        "--fit_iters", "300",
        "--lr", "0.05",
        "--device", "cpu",
        "--dynamics", "impedance",
        "--init_stiffness", "2000.0",
        "--init_damping", "55.0",
        "--query_rings", "4",
        "--query_angles", "24",
        "--contact_softness", "8e-4",
        "--smooth_max_temperature", "8e-3",
        "--inside_penalty", "0.025",
        "--inside_sharpness", "100.0",
        "--query_radius_scale", "1.20",
    ]
    print("[fit] running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"[fit] returncode={result.returncode}")
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Fit script failed")
    return read_json(FIT_DIR / "fit_summary.json")


def rollout_impedance(
    initial_position: torch.Tensor,
    initial_velocity: torch.Tensor,
    gravity_z: float,
    K: float,
    D: float,
    local_centers: torch.Tensor,
    radii: torch.Tensor,
    floor_offsets: torch.Tensor,
    steps: int,
    dt: float,
):
    """Pure rollout (no grad) using the learned impedance parameters."""
    from gaussian_initiailization.stage2.differentiable_collision_detection import (
        PlaneCollider,
        detect_gaussian_union_contacts,
    )
    from gaussian_initiailization.stage2.differentiable_complementarity_free_contact_dynamics import (
        _smooth_min,
    )
    import torch.nn.functional as F

    collider = PlaneCollider.floor(dtype=initial_position.dtype, device=initial_position.device)
    position = initial_position.clone()
    velocity = initial_velocity.clone()
    gravity = torch.tensor([0.0, 0.0, gravity_z], dtype=position.dtype, device=position.device)
    K_t = torch.tensor(K, dtype=position.dtype, device=position.device)
    D_t = torch.tensor(D, dtype=position.dtype, device=position.device)
    positions = [position.clone()]
    for _ in range(steps - 1):
        b = velocity + dt * gravity
        floor_points = torch.cat(
            (
                position[:2].unsqueeze(0) + floor_offsets,
                torch.zeros(
                    (floor_offsets.shape[0], 1),
                    dtype=position.dtype,
                    device=position.device,
                ),
            ),
            dim=-1,
        )
        gaussian_centers = local_centers + position.unsqueeze(0)
        contacts = detect_gaussian_union_contacts(
            floor_points,
            gaussian_centers,
            radii,
            collider.normal.to(dtype=position.dtype, device=position.device),
            softness=8e-4,
            smooth_min_temperature=8e-3,
            inside_penalty=0.025,
            inside_sharpness=100.0,
        )
        phi_agg = _smooth_min(contacts.signed_distances, 8e-3)
        normal = contacts.collider_normal.to(dtype=position.dtype, device=position.device)
        Jb = torch.sum(b * normal)
        lam = F.softplus(-K_t * (dt * Jb + phi_agg) - D_t * Jb)
        velocity = b + (dt / 1.0) * lam * normal
        position = position + dt * velocity
        positions.append(position.clone())
    return torch.stack(positions, dim=0)


def evaluate_on_test(ply_path: Path, fit_summary: dict) -> dict:
    from gaussian_initiailization.stage2.differentiable_collision_detection import (
        load_gaussian_collision_primitives_from_ply,
        make_floor_disk_query_points,
    )

    device = torch.device("cpu")
    local_centers, radii = load_gaussian_collision_primitives_from_ply(
        ply_path,
        radius_scale=1.0,
        recenter=True,
        dtype=torch.float32,
        device=device,
    )
    xy_extent = torch.linalg.norm(local_centers[:, :2], dim=-1) + radii
    query_radius = float(torch.quantile(xy_extent, 0.98).item()) * 1.20
    floor_offsets = make_floor_disk_query_points(
        query_radius, num_rings=4, num_angles=24, dtype=torch.float32, device=device
    )

    g_learned = fit_summary["learned_gravity_z"]
    K_learned = fit_summary["learned_stiffness"]
    D_learned = fit_summary["learned_damping"]

    per_episode = []
    test_root = DATASET_ROOT / "stage2" / "fall_and_rebound" / "test" / "sphere"
    episodes = sorted(p for p in test_root.iterdir() if p.is_dir())
    for ep in episodes:
        traj = read_json(ep / "state" / "trajectory.json")
        states = traj["states"][:FRAMES_PER_EPISODE]
        target_positions = torch.tensor(
            [s["position"] for s in states], dtype=torch.float32
        )
        times = torch.tensor([float(s["time"]) for s in states], dtype=torch.float32)
        dt = float(torch.median(times[1:] - times[:-1]).item())

        initial_position = target_positions[0]
        initial_velocity_gt = torch.tensor(states[0]["linear_velocity"], dtype=torch.float32)
        # Use the GT first-frame velocity as the rollout's initial velocity so
        # we measure dynamics fidelity (paper Sec IV-A: open-loop with first
        # frame state provided).
        predicted = rollout_impedance(
            initial_position,
            initial_velocity_gt,
            g_learned,
            K_learned,
            D_learned,
            local_centers,
            radii,
            floor_offsets,
            steps=target_positions.shape[0],
            dt=dt,
        )

        diffs = predicted - target_positions
        per_frame_err = torch.linalg.norm(diffs, dim=-1)
        cumulative_translation_err = float(per_frame_err.mean().item())
        final_err = float(per_frame_err[-1].item())
        max_err = float(per_frame_err.max().item())
        # Bounce timing: find frames where GT z is at local minimum.
        gt_z = target_positions[:, 2].numpy()
        pred_z = predicted[:, 2].numpy()
        per_episode.append(
            {
                "episode": ep.name,
                "frames": int(target_positions.shape[0]),
                "dt": dt,
                "mean_translation_error_m": cumulative_translation_err,
                "final_frame_error_m": final_err,
                "max_frame_error_m": max_err,
                "gt_settled_z_last10_mean": float(gt_z[-10:].mean()),
                "pred_settled_z_last10_mean": float(pred_z[-10:].mean()),
                "gt_first_contact_frame": int(np.argmin(gt_z[:30])),
                "pred_first_contact_frame": int(np.argmin(pred_z[:30])),
                "predicted_trajectory": predicted.numpy().tolist(),
                "target_trajectory": target_positions.numpy().tolist(),
            }
        )

    aggregate = {
        "mean_translation_error_m": float(
            np.mean([e["mean_translation_error_m"] for e in per_episode])
        ),
        "median_translation_error_m": float(
            np.median([e["mean_translation_error_m"] for e in per_episode])
        ),
        "max_translation_error_m": float(
            np.max([e["max_frame_error_m"] for e in per_episode])
        ),
        "settled_z_abs_diff_mean_m": float(
            np.mean(
                [
                    abs(e["gt_settled_z_last10_mean"] - e["pred_settled_z_last10_mean"])
                    for e in per_episode
                ]
            )
        ),
        "per_episode": per_episode,
    }
    return aggregate


def plot_trajectory_comparison(eval_result: dict, path: Path) -> None:
    from PIL import Image, ImageDraw

    ep = eval_result["per_episode"][0]
    pred = np.asarray(ep["predicted_trajectory"])
    gt = np.asarray(ep["target_trajectory"])
    dt = ep["dt"]
    n = pred.shape[0]
    t = np.arange(n) * dt

    width, height = 940, 760
    margin_l, margin_r, margin_t, margin_b = 78, 30, 92, 40
    panel_gap = 22
    panel_h = (height - margin_t - margin_b - 2 * panel_gap) // 3

    img = Image.new("RGB", (width, height), (250, 249, 246))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width, 60), fill=(28, 28, 30))
    draw.text((14, 14), f"Sphere fall fit: test/{ep['episode']}", fill=(255, 255, 245))
    draw.text(
        (14, 30),
        f"mean_translation_err = {ep['mean_translation_error_m']*1000:.2f} mm  "
        f"final_err = {ep['final_frame_error_m']*1000:.2f} mm  "
        f"max_err = {ep['max_frame_error_m']*1000:.2f} mm",
        fill=(220, 220, 220),
    )
    draw.text(
        (14, 44),
        f"GT settled z (last 10 frames) = {ep['gt_settled_z_last10_mean']:.4f} m  | "
        f"pred = {ep['pred_settled_z_last10_mean']:.4f} m  | target r = 0.100 m",
        fill=(220, 220, 220),
    )

    def panel(idx, title, series, labels, colors, y_label):
        y_top = margin_t + idx * (panel_h + panel_gap)
        y_bot = y_top + panel_h
        draw.rectangle((margin_l, y_top, width - margin_r, y_bot), fill=(255, 255, 255), outline=(160, 160, 160))
        draw.text((margin_l + 4, y_top + 4), title, fill=(40, 40, 40))
        draw.text((6, y_top + panel_h // 2 - 6), y_label, fill=(60, 60, 60))

        all_values = np.concatenate(series)
        ymin = float(all_values.min())
        ymax = float(all_values.max())
        if ymax - ymin < 1e-6:
            ymax = ymin + 1.0
        span = ymax - ymin
        if ymin <= 0.0 <= ymax:
            zy = int(y_bot - (0.0 - ymin) / span * (y_bot - y_top))
            draw.line((margin_l, zy, width - margin_r, zy), fill=(220, 220, 220), width=1)
        for k in range(5):
            tx = margin_l + int(k / 4 * (width - margin_l - margin_r))
            draw.line((tx, y_bot, tx, y_bot + 4), fill=(120, 120, 120))
            draw.text((tx - 14, y_bot + 6), f"{t[-1] * k / 4:.2f}s", fill=(80, 80, 80))
        for k in range(4):
            yv = ymin + k / 3 * span
            yy = int(y_bot - (yv - ymin) / span * (y_bot - y_top))
            draw.line((margin_l - 4, yy, margin_l, yy), fill=(120, 120, 120))
            draw.text((margin_l - 70, yy - 6), f"{yv:+.3f}", fill=(80, 80, 80))
        for color, vals, label in zip(colors, series, labels):
            n_v = vals.shape[0]
            pts = []
            for i in range(n_v):
                px = margin_l + int(i / max(n - 1, 1) * (width - margin_l - margin_r))
                py = int(y_bot - (vals[i] - ymin) / span * (y_bot - y_top))
                pts.append((px, py))
            if len(pts) > 1:
                draw.line(pts, fill=color, width=2)
        # legend
        lx = width - margin_r - 140
        for k, (label, color) in enumerate(zip(labels, colors)):
            ly = y_top + 6 + k * 14
            draw.line((lx, ly + 4, lx + 18, ly + 4), fill=color, width=2)
            draw.text((lx + 22, ly), label, fill=(40, 40, 40))

    panel(
        0,
        "z [m]",
        [gt[:, 2], pred[:, 2]],
        ["mujoco GT", "stage2 pred"],
        [(200, 70, 40), (34, 87, 184)],
        "z",
    )
    err = np.linalg.norm(pred - gt, axis=-1) * 1000.0
    panel(
        1,
        "per-frame translation error [mm]",
        [err],
        ["err mm"],
        [(150, 60, 180)],
        "mm",
    )
    panel(
        2,
        "xy [m] (GT solid, pred drawn at same scale)",
        [gt[:, 0], gt[:, 1], pred[:, 0], pred[:, 1]],
        ["GT x", "GT y", "pred x", "pred y"],
        [(200, 70, 40), (220, 130, 30), (34, 87, 184), (60, 160, 100)],
        "xy",
    )
    img.save(path)


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("[1/6] synth stage1 sphere PLY")
    ply_path = synthesize_stage1_sphere_ply()
    print(f"   PLY: {ply_path}  ({NUM_STAGE1_PRIMS} primitives)")

    print("[2/6] setup dataset layout")
    setup_dataset_layout()

    print("[3/6] generate MuJoCo trajectories (train)")
    run_mujoco_generator("train", seed=0)
    print("[4/6] generate MuJoCo trajectories (test)")
    run_mujoco_generator("test", seed=7)

    print("[5/6] run fit on train/episode_000")
    train_episode = DATASET_ROOT / "stage2" / "fall_and_rebound" / "train" / "sphere" / "episode_000"
    fit_summary = run_fit(train_episode, ply_path)
    print("    fit summary:")
    print(json.dumps(fit_summary, indent=2))

    print("[6/6] evaluate on test episodes")
    eval_result = evaluate_on_test(ply_path, fit_summary)

    # Strip the embedded full trajectories from the saved JSON to keep it small;
    # we only need them for the plot.
    eval_for_save = {
        "mean_translation_error_m": eval_result["mean_translation_error_m"],
        "median_translation_error_m": eval_result["median_translation_error_m"],
        "max_translation_error_m": eval_result["max_translation_error_m"],
        "settled_z_abs_diff_mean_m": eval_result["settled_z_abs_diff_mean_m"],
        "per_episode_summary": [
            {k: v for k, v in e.items() if k not in ("predicted_trajectory", "target_trajectory")}
            for e in eval_result["per_episode"]
        ],
    }
    write_json(OUT_ROOT / "eval_summary.json", eval_for_save)

    plot_trajectory_comparison(eval_result, OUT_ROOT / "test_episode_000_compare.png")

    print(json.dumps(
        {
            "fit_summary": fit_summary,
            "eval_summary": eval_for_save,
            "outputs": {
                "fit_dir": str(FIT_DIR),
                "eval_summary": str(OUT_ROOT / "eval_summary.json"),
                "comparison_plot": str(OUT_ROOT / "test_episode_000_compare.png"),
            },
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
