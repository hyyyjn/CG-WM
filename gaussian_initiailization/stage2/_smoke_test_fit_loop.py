"""Drive the Stage 2 fit script end-to-end with a synthetic episode.

We need this because no MuJoCo episode or Stage 1 PLY exists on disk yet, but
the fit loop is the actual workhorse the team will use. The script fabricates:

  * a "stage1" PLY (a tiny cluster of spherical Gaussians approximating a ball)
  * a MuJoCo-style episode trajectory.json from an analytic bouncing ball
  * the manifest files the fit script reads

then runs `run_stage2_mujoco_stage1_fit.py` with a small fit budget so we can
see whether the optimisation converges for floor and pairwise dynamics modes.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "gaussian_initiailization" / "stage2" / "_outputs" / "smoke"
EPISODE_ROOT = OUT_ROOT / "synthetic_episode"
STAGE1_DIR = OUT_ROOT / "synthetic_stage1"


def analytic_bounce(z0, vz0, g, e, r, dt, steps):
    z = z0
    vz = vz0
    floor = r
    out = [z]
    for _ in range(steps - 1):
        vz = vz + g * dt
        z = z + vz * dt
        if z < floor:
            z = floor + (floor - z)
            vz = -e * vz
            if abs(vz) < 0.05 and abs(z - floor) < 5e-3:
                vz = 0.0
                z = floor
        out.append(z)
    return np.array(out)


def write_synthetic_trajectory():
    EPISODE_ROOT.mkdir(parents=True, exist_ok=True)
    state_dir = EPISODE_ROOT / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    dt = 1.0 / 60.0
    steps = 120
    z = analytic_bounce(1.0, 0.0, -9.81, 0.5, 0.10, dt, steps)
    states = [
        {
            "frame_index": idx,
            "time": float(idx * dt),
            "position": [0.0, 0.0, float(z[idx])],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "linear_velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
        }
        for idx in range(steps)
    ]
    payload = {"states": states}
    with open(state_dir / "trajectory.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_synthetic_stage1_ply():
    STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    ply_path = STAGE1_DIR / "point_cloud" / "iteration_0" / "point_cloud.ply"
    ply_path.parent.mkdir(parents=True, exist_ok=True)

    # tiny cluster approximating a ball of radius ~0.10
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [-0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, -0.05, 0.0],
            [0.0, 0.0, 0.05],
            [0.0, 0.0, -0.05],
        ],
        dtype=np.float64,
    )
    log_scale = float(np.log(0.025))  # so exp(scale)≈0.025 → primitive radius ~0.025 m
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
        lines.append(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f} {log_scale:.6f} {log_scale:.6f} {log_scale:.6f}")
    with open(ply_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return ply_path


def main():
    if OUT_ROOT.exists():
        for p in OUT_ROOT.glob("synthetic_*"):
            shutil.rmtree(p, ignore_errors=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_synthetic_trajectory()
    ply_path = write_synthetic_stage1_ply()

    for dynamics_mode in ("impedance", "restitution", "pairwise_impedance"):
        fit_output = OUT_ROOT / f"synthetic_fit_{dynamics_mode}"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "gaussian_initiailization" / "tools" / "run_stage2_mujoco_stage1_fit.py"),
            "--episode_root", str(EPISODE_ROOT),
            "--stage1_ply", str(ply_path),
            "--output_dir", str(fit_output),
            "--max_frames", "120",
            "--fit_iters", "200",
            "--lr", "0.05",
            "--device", "cpu",
            "--gif_fps", "24",
            "--query_rings", "3",
            "--query_angles", "16",
            "--dynamics", dynamics_mode,
        ]
        if dynamics_mode == "pairwise_impedance":
            cmd.extend(["--pairwise_static_position", "0,0,0"])
        print(f"\n[smoke] === {dynamics_mode} ===")
        print("[smoke] running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("[smoke] stdout:")
        print(result.stdout)
        if result.stderr:
            print("[smoke] stderr:")
            print(result.stderr)
        print("[smoke] returncode:", result.returncode)


if __name__ == "__main__":
    main()
