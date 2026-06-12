from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


VARIANT_CHOICES = (
    "impulse",
    "stage2",
    "velocity_fit",
    "physics_fit",
    "mask_fit",
    "gaussian_rgb_fit",
    "full_image_fit",
)
DEFAULT_VARIANTS = ("impulse", "stage2", "velocity_fit", "physics_fit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-dice Stage2 rollout variants and summarize trajectory/contact/image-space metrics."
    )
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--stage1_ply", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--gt_rgb_dir", default=None, type=Path)
    parser.add_argument("--gt_mask_dir", default=None, type=Path)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS), choices=VARIANT_CHOICES)
    parser.add_argument("--max_frames", default=100, type=int)
    parser.add_argument("--max_primitives", default=256, type=int)
    parser.add_argument("--substeps", default=4, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fit_iters", default=40, type=int)
    parser.add_argument("--fit_lr", default=0.03, type=float)
    parser.add_argument("--fit_physics_iters", default=40, type=int)
    parser.add_argument("--fit_physics_lr", default=0.02, type=float)
    parser.add_argument("--fit_horizon_frames", default=36, type=int)
    parser.add_argument("--mask_loss_weight", default=0.1, type=float)
    parser.add_argument("--mask_loss_resolution", default=64, type=int)
    parser.add_argument("--gaussian_rgb_loss_weight", default=0.05, type=float)
    parser.add_argument("--gaussian_render_width", default=160, type=int)
    parser.add_argument("--gaussian_render_height", default=120, type=int)
    parser.add_argument("--gaussian_render_stride", default=4, type=int)
    parser.add_argument("--gaussian_render_loss", default="l1", choices=("l1", "mse"))
    parser.add_argument("--gaussian_render_white_background", action="store_true")
    parser.add_argument("--fit_geometry_radii", action="store_true")
    parser.add_argument("--fit_geometry_centers", action="store_true")
    parser.add_argument("--fit_geometry_radius_l2", default=1e-3, type=float)
    parser.add_argument("--fit_geometry_center_l2", default=1e-2, type=float)
    parser.add_argument("--fit_geometry_max_log_radius_offset", default=0.7, type=float)
    parser.add_argument("--fit_geometry_max_center_offset", default=0.015, type=float)
    parser.add_argument("--stage2_static_friction", default=0.0, type=float)
    parser.add_argument("--stage2_friction_transition_velocity", default=1e-3, type=float)
    parser.add_argument("--skip_render", action="store_true", default=True)
    parser.add_argument("--render", dest="skip_render", action="store_false")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def geometry_options(args: argparse.Namespace) -> list[str]:
    options = []
    if bool(args.fit_geometry_radii):
        options.append("--fit_geometry_radii")
    if bool(args.fit_geometry_centers):
        options.append("--fit_geometry_centers")
    if options:
        options.extend(
            [
                "--fit_geometry_radius_l2",
                str(args.fit_geometry_radius_l2),
                "--fit_geometry_center_l2",
                str(args.fit_geometry_center_l2),
                "--fit_geometry_max_log_radius_offset",
                str(args.fit_geometry_max_log_radius_offset),
                "--fit_geometry_max_center_offset",
                str(args.fit_geometry_max_center_offset),
            ]
        )
    return options


def friction_options(args: argparse.Namespace) -> list[str]:
    options = []
    if float(args.stage2_static_friction) > 0.0:
        options.extend(["--stage2_static_friction", str(args.stage2_static_friction)])
    options.extend(["--stage2_friction_transition_velocity", str(args.stage2_friction_transition_velocity)])
    return options


def variant_options(name: str, args: argparse.Namespace) -> list[str]:
    if name == "impulse":
        return ["--dynamics_backend", "impulse"]
    if name == "stage2":
        return ["--dynamics_backend", "stage2_impedance"] + friction_options(args)
    if name == "velocity_fit":
        return [
            "--dynamics_backend",
            "stage2_impedance",
            "--fit_iters",
            str(args.fit_iters),
            "--fit_lr",
            str(args.fit_lr),
        ] + friction_options(args)
    if name == "physics_fit":
        return [
            "--dynamics_backend",
            "stage2_impedance",
            "--fit_iters",
            str(args.fit_iters),
            "--fit_lr",
            str(args.fit_lr),
            "--fit_physics_iters",
            str(args.fit_physics_iters),
            "--fit_physics_lr",
            str(args.fit_physics_lr),
        ] + friction_options(args) + geometry_options(args)
    if name == "mask_fit":
        if args.gt_mask_dir is None:
            raise ValueError("mask_fit requires --gt_mask_dir.")
        return [
            "--dynamics_backend",
            "stage2_impedance",
            "--fit_iters",
            str(args.fit_iters),
            "--fit_lr",
            str(args.fit_lr),
            "--fit_physics_iters",
            str(args.fit_physics_iters),
            "--fit_physics_lr",
            str(args.fit_physics_lr),
            "--gt_mask_dir",
            str(args.gt_mask_dir.resolve()),
            "--mask_loss_weight",
            str(args.mask_loss_weight),
            "--mask_loss_resolution",
            str(args.mask_loss_resolution),
        ] + friction_options(args) + geometry_options(args)
    if name == "gaussian_rgb_fit":
        if args.gt_rgb_dir is None:
            raise ValueError("gaussian_rgb_fit requires --gt_rgb_dir.")
        if not str(args.device).startswith("cuda"):
            raise ValueError("gaussian_rgb_fit requires --device cuda.")
        options = [
            "--dynamics_backend",
            "stage2_impedance",
            "--fit_iters",
            str(args.fit_iters),
            "--fit_lr",
            str(args.fit_lr),
            "--fit_physics_iters",
            str(args.fit_physics_iters),
            "--fit_physics_lr",
            str(args.fit_physics_lr),
            "--gaussian_rgb_loss_weight",
            str(args.gaussian_rgb_loss_weight),
            "--gaussian_render_width",
            str(args.gaussian_render_width),
            "--gaussian_render_height",
            str(args.gaussian_render_height),
            "--gaussian_render_stride",
            str(args.gaussian_render_stride),
            "--gaussian_render_loss",
            str(args.gaussian_render_loss),
        ]
        if bool(args.gaussian_render_white_background):
            options.append("--gaussian_render_white_background")
        return options + friction_options(args) + geometry_options(args)
    if name == "full_image_fit":
        if args.gt_mask_dir is None:
            raise ValueError("full_image_fit requires --gt_mask_dir.")
        if args.gt_rgb_dir is None:
            raise ValueError("full_image_fit requires --gt_rgb_dir.")
        if not str(args.device).startswith("cuda"):
            raise ValueError("full_image_fit requires --device cuda.")
        options = [
            "--dynamics_backend",
            "stage2_impedance",
            "--fit_iters",
            str(args.fit_iters),
            "--fit_lr",
            str(args.fit_lr),
            "--fit_physics_iters",
            str(args.fit_physics_iters),
            "--fit_physics_lr",
            str(args.fit_physics_lr),
            "--gt_mask_dir",
            str(args.gt_mask_dir.resolve()),
            "--mask_loss_weight",
            str(args.mask_loss_weight),
            "--mask_loss_resolution",
            str(args.mask_loss_resolution),
            "--gaussian_rgb_loss_weight",
            str(args.gaussian_rgb_loss_weight),
            "--gaussian_render_width",
            str(args.gaussian_render_width),
            "--gaussian_render_height",
            str(args.gaussian_render_height),
            "--gaussian_render_stride",
            str(args.gaussian_render_stride),
            "--gaussian_render_loss",
            str(args.gaussian_render_loss),
        ]
        if bool(args.gaussian_render_white_background):
            options.append("--gaussian_render_white_background")
        return options + friction_options(args) + geometry_options(args)
    raise ValueError(f"Unknown variant: {name}")


def flatten_summary(variant: str, output_dir: Path, summary: dict) -> dict:
    metrics = summary.get("metrics", {})
    physics_fit = summary.get("physics_fit") or {}
    image_space = physics_fit.get("image_space_supervision") or {}
    learned_physics = physics_fit.get("learned_physics") or {}
    learned_geometry = physics_fit.get("learned_geometry") or {}
    history = physics_fit.get("history") or []
    last_fit = history[-1] if history else {}
    row = {
        "variant": variant,
        "output_dir": str(output_dir),
        "dynamics_backend": summary.get("dynamics_backend"),
        "position_rmse": metrics.get("position_rmse"),
        "mean_center_error": metrics.get("mean_center_error"),
        "final_mean_center_error": metrics.get("final_mean_center_error"),
        "max_center_error": metrics.get("max_center_error"),
        "mean_rotation_error_deg": metrics.get("mean_rotation_error_deg"),
        "final_mean_rotation_error_deg": metrics.get("final_mean_rotation_error_deg"),
        "active_pair_substeps": metrics.get("stage2_active_pair_substeps"),
        "fit_best_loss": None if summary.get("fit") is None else summary["fit"].get("best_loss"),
        "physics_best_loss": physics_fit.get("best_loss"),
        "last_mask_bce": last_fit.get("mask_bce"),
        "last_gaussian_rgb_loss": last_fit.get("gaussian_rgb_loss"),
        "image_space_enabled": image_space.get("enabled"),
        "mask_loss_enabled": image_space.get("gt_mask_dir") is not None and float(image_space.get("mask_loss_weight") or 0.0) > 0.0,
        "gaussian_rgb_enabled": image_space.get("gaussian_rgb_enabled"),
        "gaussian_render_scale_multiplier": image_space.get("gaussian_render_scale_multiplier"),
    }
    if learned_geometry.get("radius_multipliers") is not None:
        radius_multipliers = np.asarray(learned_geometry["radius_multipliers"], dtype=np.float64)
        row["geometry_radius_multiplier_mean"] = float(np.mean(radius_multipliers))
        row["geometry_radius_multiplier_min"] = float(np.min(radius_multipliers))
        row["geometry_radius_multiplier_max"] = float(np.max(radius_multipliers))
        row["geometry_radius_count"] = int(radius_multipliers.size)
    if learned_geometry.get("center_offsets") is not None:
        center_offsets = np.asarray(learned_geometry["center_offsets"], dtype=np.float64)
        norms = np.linalg.norm(center_offsets.reshape(-1, 3), axis=-1)
        row["geometry_center_offset_mean"] = float(np.mean(norms))
        row["geometry_center_offset_max"] = float(np.max(norms))
        row["geometry_center_count"] = int(norms.size)
    for key, value in learned_physics.items():
        row[f"learned_{key}"] = value
    return row


def aggregate_rows(rows: list[dict]) -> dict:
    numeric_keys = [
        "position_rmse",
        "mean_center_error",
        "final_mean_center_error",
        "mean_rotation_error_deg",
        "final_mean_rotation_error_deg",
        "last_mask_bce",
        "last_gaussian_rgb_loss",
        "geometry_radius_multiplier_mean",
        "geometry_center_offset_mean",
    ]
    aggregate = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            aggregate[f"{key}_mean"] = float(np.mean(values))
            aggregate[f"{key}_min"] = float(np.min(values))
            aggregate[f"{key}_max"] = float(np.max(values))
    if rows:
        best = min(
            (row for row in rows if row.get("position_rmse") is not None),
            key=lambda row: float(row["position_rmse"]),
            default=None,
        )
        aggregate["best_position_rmse_variant"] = None if best is None else best["variant"]
    return aggregate


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).resolve().parent / "run_stage2_multi_dice_rollout_comparison.py"
    rows = []
    commands = {}

    for variant in args.variants:
        output_dir = output_root / variant
        command = [
            args.python,
            str(runner),
            "--trajectory",
            str(args.trajectory.resolve()),
            "--stage1_ply",
            str(args.stage1_ply.resolve()),
            "--output_dir",
            str(output_dir),
            "--max_frames",
            str(args.max_frames),
            "--max_primitives",
            str(args.max_primitives),
            "--substeps",
            str(args.substeps),
            "--device",
            str(args.device),
            "--fit_horizon_frames",
            str(args.fit_horizon_frames),
        ]
        if args.gt_rgb_dir is not None:
            command.extend(["--gt_rgb_dir", str(args.gt_rgb_dir.resolve())])
        if args.skip_render:
            command.append("--skip_render")
        command.extend(variant_options(variant, args))
        commands[variant] = command
        print(f"[EVAL] running {variant}", flush=True)
        subprocess.run(command, check=True)
        summary = read_json(output_dir / "stage2_rollout_summary.json")
        rows.append(flatten_summary(variant, output_dir, summary))

    report = {
        "trajectory": str(args.trajectory.resolve()),
        "stage1_ply": str(args.stage1_ply.resolve()),
        "settings": {
            "variants": list(args.variants),
            "max_frames": int(args.max_frames),
            "max_primitives": int(args.max_primitives),
            "substeps": int(args.substeps),
            "fit_iters": int(args.fit_iters),
            "fit_physics_iters": int(args.fit_physics_iters),
            "fit_horizon_frames": int(args.fit_horizon_frames),
            "mask_loss_weight": float(args.mask_loss_weight),
            "mask_loss_resolution": int(args.mask_loss_resolution),
            "gaussian_rgb_loss_weight": float(args.gaussian_rgb_loss_weight),
            "gaussian_render_width": int(args.gaussian_render_width),
            "gaussian_render_height": int(args.gaussian_render_height),
            "gaussian_render_stride": int(args.gaussian_render_stride),
            "gaussian_render_loss": str(args.gaussian_render_loss),
            "gaussian_render_white_background": bool(args.gaussian_render_white_background),
            "fit_geometry_radii": bool(args.fit_geometry_radii),
            "fit_geometry_centers": bool(args.fit_geometry_centers),
            "fit_geometry_radius_l2": float(args.fit_geometry_radius_l2),
            "fit_geometry_center_l2": float(args.fit_geometry_center_l2),
            "fit_geometry_max_log_radius_offset": float(args.fit_geometry_max_log_radius_offset),
            "fit_geometry_max_center_offset": float(args.fit_geometry_max_center_offset),
            "stage2_static_friction": float(args.stage2_static_friction),
            "stage2_friction_transition_velocity": float(args.stage2_friction_transition_velocity),
            "skip_render": bool(args.skip_render),
        },
        "aggregate": aggregate_rows(rows),
        "rows": rows,
        "commands": {name: " ".join(command) for name, command in commands.items()},
    }
    write_json(output_root / "multi_dice_stage2_variant_report.json", report)
    write_csv(output_root / "multi_dice_stage2_variant_results.csv", rows)
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
