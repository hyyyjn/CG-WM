"""Run Stage 2 fitting over many episodes and summarize trajectory errors."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 fit generalization across episodes.")
    parser.add_argument("--episodes_root", required=True, type=Path)
    parser.add_argument("--stage1_model_path", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--fit_iters", default=120, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_episodes", default=0, type=int)
    parser.add_argument("--max_primitives", default=2000, type=int)
    parser.add_argument("--foreground_threshold", default=0.55, type=float)
    parser.add_argument("--opacity_threshold", default=0.02, type=float)
    parser.add_argument("--radius_scale", default=0.1, type=float)
    parser.add_argument("--floor_tangential_damping", default=80.0, type=float)
    parser.add_argument("--collision_bbox_margin_z_ratio", default=0.22, type=float)
    parser.add_argument("--lr", default=0.015, type=float)
    parser.add_argument("--dynamics", default="restitution", choices=("restitution", "impedance", "pairwise_impedance"))
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def summarize_prediction(path: Path) -> dict:
    payload = load_json(path)
    predicted = np.asarray([state["predicted_position"] for state in payload["states"]], dtype=np.float64)
    target = np.asarray([state["target_position"] for state in payload["states"]], dtype=np.float64)
    diff = predicted - target
    return {
        "component_rmse": float(np.sqrt(np.mean(diff * diff))),
        "vector_rmse": float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))),
        "mean_abs_xyz": np.mean(np.abs(diff), axis=0).tolist(),
        "final_error_xyz": diff[-1].tolist(),
        "frames": int(diff.shape[0]),
    }


def main() -> None:
    args = parse_args()
    episodes_root = args.episodes_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    episode_roots = sorted(path for path in episodes_root.iterdir() if path.is_dir())
    if args.max_episodes > 0:
        episode_roots = episode_roots[: int(args.max_episodes)]
    if not episode_roots:
        raise FileNotFoundError(f"No episode directories found under {episodes_root}")

    fit_script = Path(__file__).resolve().parent / "run_stage2_mujoco_stage1_fit.py"
    results = []
    for episode_root in episode_roots:
        episode_output = output_root / episode_root.name
        command = [
            sys.executable,
            str(fit_script),
            "--episode_root",
            str(episode_root),
            "--stage1_model_path",
            str(args.stage1_model_path.resolve()),
            "--output_dir",
            str(episode_output),
            "--dynamics",
            str(args.dynamics),
            "--fit_iters",
            str(args.fit_iters),
            "--lr",
            str(args.lr),
            "--device",
            str(args.device),
            "--max_primitives",
            str(args.max_primitives),
            "--foreground_threshold",
            str(args.foreground_threshold),
            "--opacity_threshold",
            str(args.opacity_threshold),
            "--radius_scale",
            str(args.radius_scale),
            "--initial_velocity_source",
            "trajectory",
            "--floor_tangential_damping",
            str(args.floor_tangential_damping),
            "--collision_bbox_margin_z_ratio",
            str(args.collision_bbox_margin_z_ratio),
        ]
        print(f"[EVAL] {episode_root.name}", flush=True)
        subprocess.run(command, check=True)

        fit_summary = load_json(episode_output / "fit_summary.json")
        prediction_summary = summarize_prediction(episode_output / "predicted_trajectory.json")
        row = {
            "episode": episode_root.name,
            "output_dir": str(episode_output),
            **prediction_summary,
            "fit_position_rmse": float(fit_summary["position_rmse"]),
            "learned_gravity_z": float(fit_summary["learned_gravity_z"]),
            "learned_restitution": float(fit_summary.get("learned_restitution", 0.0)),
            "first_contact_frame": fit_summary.get("first_contact_frame"),
        }
        results.append(row)

    component_rmses = np.asarray([row["component_rmse"] for row in results], dtype=np.float64)
    vector_rmses = np.asarray([row["vector_rmse"] for row in results], dtype=np.float64)
    report = {
        "episodes_root": str(episodes_root),
        "stage1_model_path": str(args.stage1_model_path.resolve()),
        "settings": {
            "fit_iters": int(args.fit_iters),
            "dynamics": str(args.dynamics),
            "max_primitives": int(args.max_primitives),
            "foreground_threshold": float(args.foreground_threshold),
            "opacity_threshold": float(args.opacity_threshold),
            "radius_scale": float(args.radius_scale),
            "floor_tangential_damping": float(args.floor_tangential_damping),
            "collision_bbox_margin_z_ratio": float(args.collision_bbox_margin_z_ratio),
        },
        "aggregate": {
            "episode_count": len(results),
            "component_rmse_mean": float(np.mean(component_rmses)),
            "component_rmse_max": float(np.max(component_rmses)),
            "vector_rmse_mean": float(np.mean(vector_rmses)),
            "vector_rmse_max": float(np.max(vector_rmses)),
        },
        "episodes": results,
    }
    (output_root / "generalization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
