"""End-to-end smoke for Stage2 variant evaluation and best-param reuse.

This creates a tiny synthetic two-dice trajectory plus a minimal Stage1-style
Gaussian PLY, runs the variant evaluator with a soft/dual friction sweep, then
verifies that the top-ranked refined params can be loaded by the rollout runner.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "gaussian_initiailization" / "stage2" / "_outputs" / "stage2_variant_e2e"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_synthetic_trajectory(path: Path) -> None:
    half_extent = 0.05
    frames = []
    for frame_idx in range(3):
        time = frame_idx / 30.0
        offset = 0.015 * frame_idx
        frames.append(
            {
                "frame_index": frame_idx,
                "time": time,
                "dice": [
                    {
                        "die": 0,
                        "position": [-0.08 + offset, 0.0, 0.08],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "linear_velocity": [0.45, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                    },
                    {
                        "die": 1,
                        "position": [0.08 - offset, 0.0, 0.08],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "linear_velocity": [-0.45, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                    },
                ],
            }
        )
    write_json(path, {"dice_count": 2, "half_extent": half_extent, "states": frames})


def write_synthetic_stage1_ply(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    log_scale = float(np.log(0.018))
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
    for center in centers:
        lines.append(
            f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f} "
            f"{log_scale:.6f} {log_scale:.6f} {log_scale:.6f}"
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_command(command: list[str]) -> subprocess.CompletedProcess:
    print("[smoke] running:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.stdout:
        print("[smoke] stdout:")
        print(result.stdout)
    if result.stderr:
        print("[smoke] stderr:")
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}: {' '.join(command)}")
    return result


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    data_root = OUT_ROOT / "synthetic_data"
    trajectory = data_root / "trajectory.json"
    stage1_ply = data_root / "stage1" / "point_cloud" / "iteration_0" / "point_cloud.ply"
    eval_root = OUT_ROOT / "eval"
    reload_root = OUT_ROOT / "reload"
    write_synthetic_trajectory(trajectory)
    write_synthetic_stage1_ply(stage1_ply)

    evaluator = REPO_ROOT / "gaussian_initiailization" / "tools" / "evaluate_multi_dice_stage2_variants.py"
    run_command(
        [
            sys.executable,
            str(evaluator),
            "--trajectory",
            str(trajectory),
            "--stage1_ply",
            str(stage1_ply),
            "--output_root",
            str(eval_root),
            "--variants",
            "stage2",
            "--friction_model_sweep",
            "soft_projection",
            "dual_cone",
            "--max_frames",
            "3",
            "--max_primitives",
            "16",
            "--substeps",
            "1",
            "--stage2_patch_selection",
            "soft",
            "--stage2_normal_mode",
            "signed_distance",
            "--device",
            "cpu",
            "--skip_render",
        ]
    )

    report_path = eval_root / "multi_dice_stage2_variant_report.json"
    best_params_path = eval_root / "best_refined_params.json"
    manifest_path = eval_root / "best_refined_params_manifest.json"
    report = read_json(report_path)
    if not best_params_path.exists():
        raise AssertionError(f"Missing best refined params: {best_params_path}")
    if not manifest_path.exists():
        raise AssertionError(f"Missing best refined params manifest: {manifest_path}")
    comparison = report.get("friction_model_comparison") or []
    ranking = report.get("ranking") or {}
    if not comparison:
        raise AssertionError("Expected non-empty friction_model_comparison.")
    if not ranking.get("best_run_variant"):
        raise AssertionError("Expected ranking.best_run_variant.")

    runner = REPO_ROOT / "gaussian_initiailization" / "tools" / "run_stage2_multi_dice_rollout_comparison.py"
    run_command(
        [
            sys.executable,
            str(runner),
            "--trajectory",
            str(trajectory),
            "--stage1_ply",
            str(stage1_ply),
            "--output_dir",
            str(reload_root),
            "--max_frames",
            "3",
            "--max_primitives",
            "16",
            "--substeps",
            "1",
            "--dynamics_backend",
            "stage2_impedance",
            "--stage2_patch_selection",
            "soft",
            "--stage2_normal_mode",
            "signed_distance",
            "--load_refined_params",
            str(best_params_path),
            "--device",
            "cpu",
            "--skip_render",
        ]
    )

    reload_summary = read_json(reload_root / "stage2_rollout_summary.json")
    if reload_summary.get("refined_params", {}).get("loaded_from") != str(best_params_path.resolve()):
        raise AssertionError("Reload summary did not record the expected best_refined_params path.")
    if "stage2_contact_diagnostics" not in reload_summary.get("metrics", {}):
        raise AssertionError("Reload rollout did not emit Stage2 contact diagnostics.")

    smoke_summary = {
        "report_path": str(report_path),
        "best_refined_params": str(best_params_path),
        "manifest_path": str(manifest_path),
        "best_run_variant": ranking.get("best_run_variant"),
        "best_friction_model": ranking.get("best_friction_model"),
        "reload_summary": str(reload_root / "stage2_rollout_summary.json"),
        "comparison_rows": len(comparison),
    }
    write_json(OUT_ROOT / "stage2_variant_e2e_smoke_summary.json", smoke_summary)
    print(json.dumps(smoke_summary, indent=2))


if __name__ == "__main__":
    main()
