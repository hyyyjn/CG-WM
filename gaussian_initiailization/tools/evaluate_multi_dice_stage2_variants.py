from __future__ import annotations

import argparse
import csv
import json
import shutil
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
    parser.add_argument("--stage2_friction_model", default="soft_projection", choices=("soft_projection", "dual_cone"))
    parser.add_argument(
        "--friction_model_sweep",
        nargs="+",
        default=None,
        choices=("soft_projection", "dual_cone"),
        help="Run each Stage2 variant once per listed friction model and emit comparison tables.",
    )
    parser.add_argument("--stage2_friction_num_directions", default=8, type=int)
    parser.add_argument("--stage2_patch_selection", default="spatial", choices=("spatial", "topk", "soft"))
    parser.add_argument("--stage2_normal_mode", default="phi_soft", choices=("phi_soft", "signed_distance", "autograd"))
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


def friction_options(args: argparse.Namespace, *, friction_model: str | None = None) -> list[str]:
    options = []
    if float(args.stage2_static_friction) > 0.0:
        options.extend(["--stage2_static_friction", str(args.stage2_static_friction)])
    options.extend(["--stage2_friction_transition_velocity", str(args.stage2_friction_transition_velocity)])
    options.extend(["--stage2_friction_model", str(friction_model or args.stage2_friction_model)])
    options.extend(["--stage2_friction_num_directions", str(args.stage2_friction_num_directions)])
    options.extend(["--stage2_patch_selection", str(args.stage2_patch_selection)])
    options.extend(["--stage2_normal_mode", str(args.stage2_normal_mode)])
    return options


def variant_options(name: str, args: argparse.Namespace, *, friction_model: str | None = None) -> list[str]:
    if name == "impulse":
        return ["--dynamics_backend", "impulse_baseline"]
    if name == "stage2":
        return ["--dynamics_backend", "stage2_impedance"] + friction_options(args, friction_model=friction_model)
    if name == "velocity_fit":
        return [
            "--dynamics_backend",
            "stage2_impedance",
            "--fit_iters",
            str(args.fit_iters),
            "--fit_lr",
            str(args.fit_lr),
        ] + friction_options(args, friction_model=friction_model)
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
        ] + friction_options(args, friction_model=friction_model) + geometry_options(args)
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
        ] + friction_options(args, friction_model=friction_model) + geometry_options(args)
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
        return options + friction_options(args, friction_model=friction_model) + geometry_options(args)
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
        return options + friction_options(args, friction_model=friction_model) + geometry_options(args)
    raise ValueError(f"Unknown variant: {name}")


def flatten_summary(variant: str, output_dir: Path, summary: dict, *, run_variant: str | None = None) -> dict:
    metrics = summary.get("metrics", {})
    contact_diagnostics = metrics.get("stage2_contact_diagnostics") or {}
    friction_cone = summary.get("friction_cone") or {}
    physics_fit = summary.get("physics_fit") or {}
    image_space = physics_fit.get("image_space_supervision") or {}
    learned_physics = physics_fit.get("learned_physics") or {}
    learned_geometry = physics_fit.get("learned_geometry") or {}
    history = physics_fit.get("history") or []
    last_fit = history[-1] if history else {}
    row = {
        "variant": variant,
        "run_variant": run_variant or variant,
        "output_dir": str(output_dir),
        "dynamics_backend": summary.get("dynamics_backend"),
        "refined_params_saved_to": (summary.get("refined_params") or {}).get("saved_to"),
        "refined_params_loaded_from": (summary.get("refined_params") or {}).get("loaded_from"),
        "friction_model": friction_cone.get("model"),
        "friction_num_directions": friction_cone.get("num_directions"),
        "position_rmse": metrics.get("position_rmse"),
        "mean_center_error": metrics.get("mean_center_error"),
        "final_mean_center_error": metrics.get("final_mean_center_error"),
        "max_center_error": metrics.get("max_center_error"),
        "mean_rotation_error_deg": metrics.get("mean_rotation_error_deg"),
        "final_mean_rotation_error_deg": metrics.get("final_mean_rotation_error_deg"),
        "active_pair_substeps": metrics.get("stage2_active_pair_substeps"),
        "contact_total_candidate_edges": contact_diagnostics.get("total_candidate_edges"),
        "contact_total_active_edges": contact_diagnostics.get("total_active_edges"),
        "contact_total_edges_with_gate": contact_diagnostics.get("total_contact_edges_with_gate"),
        "contact_max_edge_gate": contact_diagnostics.get("max_edge_gate"),
        "contact_max_lambda": contact_diagnostics.get("max_lambda"),
        "contact_max_friction_force": contact_diagnostics.get("max_friction_force"),
        "contact_max_raw_friction": contact_diagnostics.get("max_raw_friction"),
        "contact_max_friction_cone_violation": contact_diagnostics.get("max_friction_cone_violation"),
        "contact_max_friction_force_to_cone_radius_ratio": contact_diagnostics.get(
            "max_friction_force_to_cone_radius_ratio"
        ),
        "contact_mean_slip_speed": contact_diagnostics.get("mean_slip_speed"),
        "contact_max_friction_facet_budget": contact_diagnostics.get("max_friction_facet_budget"),
        "contact_max_friction_facet_reconstruction_error": contact_diagnostics.get(
            "max_friction_facet_reconstruction_error"
        ),
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
        "contact_max_friction_cone_violation",
        "contact_max_friction_force_to_cone_radius_ratio",
        "contact_max_friction_facet_budget",
        "contact_max_friction_facet_reconstruction_error",
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


def friction_model_comparison_rows(rows: list[dict]) -> list[dict]:
    metrics = (
        "position_rmse",
        "final_mean_center_error",
        "mean_rotation_error_deg",
        "contact_total_active_edges",
        "contact_max_lambda",
        "contact_max_friction_force",
        "contact_max_raw_friction",
        "contact_max_friction_cone_violation",
        "contact_max_friction_force_to_cone_radius_ratio",
        "contact_max_friction_facet_budget",
        "contact_max_friction_facet_reconstruction_error",
    )
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        if row.get("dynamics_backend") != "stage2_impedance":
            continue
        model = row.get("friction_model")
        if model is None:
            continue
        grouped.setdefault(str(row["variant"]), {})[str(model)] = row

    comparison = []
    for variant, by_model in sorted(grouped.items()):
        if "soft_projection" not in by_model or "dual_cone" not in by_model:
            continue
        soft = by_model["soft_projection"]
        dual = by_model["dual_cone"]
        item = {
            "variant": variant,
            "soft_output_dir": soft.get("output_dir"),
            "dual_output_dir": dual.get("output_dir"),
        }
        for metric in metrics:
            soft_value = soft.get(metric)
            dual_value = dual.get(metric)
            item[f"soft_{metric}"] = soft_value
            item[f"dual_{metric}"] = dual_value
            if soft_value is not None and dual_value is not None:
                item[f"delta_{metric}"] = float(dual_value) - float(soft_value)
        comparison.append(item)
    return comparison


RANKING_METRIC_WEIGHTS = {
    "position_rmse": 0.35,
    "final_mean_center_error": 0.20,
    "mean_rotation_error_deg": 0.15,
    "contact_max_friction_cone_violation": 0.10,
    "contact_max_friction_force_to_cone_radius_ratio": 0.08,
    "contact_max_lambda": 0.07,
    "contact_max_friction_facet_reconstruction_error": 0.05,
}


def _normalized_lower_is_better(value: float | int | None, values: list[float]) -> float | None:
    if value is None or not values:
        return None
    finite_values = [float(v) for v in values if np.isfinite(float(v))]
    if not finite_values:
        return None
    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    if abs(maximum - minimum) < 1e-12:
        return 0.0
    return float((float(value) - minimum) / (maximum - minimum))


def rank_variant_rows(rows: list[dict]) -> dict:
    """Rank rollout variants using trajectory accuracy plus contact stability."""

    candidates = [row for row in rows if row.get("position_rmse") is not None]
    metric_values = {
        metric: [float(row[metric]) for row in candidates if row.get(metric) is not None]
        for metric in RANKING_METRIC_WEIGHTS
    }
    ranked = []
    for row in candidates:
        score_terms = {}
        total_weight = 0.0
        score = 0.0
        missing_metrics = []
        for metric, weight in RANKING_METRIC_WEIGHTS.items():
            normalized = _normalized_lower_is_better(row.get(metric), metric_values.get(metric, []))
            if normalized is None:
                missing_metrics.append(metric)
                continue
            weighted = float(weight) * normalized
            score += weighted
            total_weight += float(weight)
            score_terms[metric] = {
                "raw": row.get(metric),
                "normalized": normalized,
                "weight": float(weight),
                "weighted": weighted,
            }
        normalized_score = None if total_weight <= 0.0 else float(score / total_weight)
        ranked.append(
            {
                "rank": 0,
                "run_variant": row.get("run_variant", row.get("variant")),
                "variant": row.get("variant"),
                "friction_model": row.get("friction_model"),
                "dynamics_backend": row.get("dynamics_backend"),
                "output_dir": row.get("output_dir"),
                "refined_params_saved_to": row.get("refined_params_saved_to"),
                "score": normalized_score,
                "available_weight": float(total_weight),
                "missing_metrics": missing_metrics,
                "score_terms": score_terms,
            }
        )
    ranked.sort(
        key=lambda item: (
            float("inf") if item["score"] is None else float(item["score"]),
            str(item.get("run_variant")),
        )
    )
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    best = ranked[0] if ranked else None
    return {
        "lower_score_is_better": True,
        "metric_weights": dict(RANKING_METRIC_WEIGHTS),
        "best_run_variant": None if best is None else best.get("run_variant"),
        "best_variant": None if best is None else best.get("variant"),
        "best_friction_model": None if best is None else best.get("friction_model"),
        "rows": ranked,
    }


def flatten_ranking_for_csv(ranking: dict) -> list[dict]:
    rows = []
    for item in ranking.get("rows", []):
        row = {
            "rank": item.get("rank"),
            "run_variant": item.get("run_variant"),
            "variant": item.get("variant"),
            "friction_model": item.get("friction_model"),
            "dynamics_backend": item.get("dynamics_backend"),
            "score": item.get("score"),
            "available_weight": item.get("available_weight"),
            "missing_metrics": ",".join(item.get("missing_metrics") or []),
            "output_dir": item.get("output_dir"),
            "refined_params_saved_to": item.get("refined_params_saved_to"),
        }
        for metric, term in (item.get("score_terms") or {}).items():
            row[f"{metric}_raw"] = term.get("raw")
            row[f"{metric}_normalized"] = term.get("normalized")
            row[f"{metric}_weighted"] = term.get("weighted")
        rows.append(row)
    return rows


def export_best_refined_params(
    *,
    output_root: Path,
    ranking: dict,
    trajectory: Path,
    stage1_ply: Path,
) -> dict:
    """Copy the top-ranked available refined-params file to a stable path."""

    for ranked_row in ranking.get("rows", []):
        source = ranked_row.get("refined_params_saved_to")
        if not source:
            continue
        source_path = Path(source)
        if not source_path.exists():
            continue
        target_path = output_root / "best_refined_params.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)
        manifest = {
            "source": str(source_path.resolve()),
            "target": str(target_path.resolve()),
            "rank": ranked_row.get("rank"),
            "run_variant": ranked_row.get("run_variant"),
            "variant": ranked_row.get("variant"),
            "friction_model": ranked_row.get("friction_model"),
            "score": ranked_row.get("score"),
            "load_command_args": [
                "--trajectory",
                str(trajectory.resolve()),
                "--stage1_ply",
                str(stage1_ply.resolve()),
                "--load_refined_params",
                str(target_path.resolve()),
            ],
        }
        write_json(output_root / "best_refined_params_manifest.json", manifest)
        return manifest
    return {
        "source": None,
        "target": None,
        "reason": "No ranked run produced an existing refined_params file.",
    }


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
    friction_models = list(args.friction_model_sweep or [args.stage2_friction_model])

    for variant in args.variants:
        variant_friction_models = [None] if variant == "impulse" else friction_models
        for friction_model in variant_friction_models:
            run_variant = variant if friction_model is None or len(variant_friction_models) == 1 else f"{variant}_{friction_model}"
            output_dir = output_root / run_variant
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
                "--save_refined_params",
                str((output_dir / "refined_params.json").resolve()),
            ]
            if args.gt_rgb_dir is not None:
                command.extend(["--gt_rgb_dir", str(args.gt_rgb_dir.resolve())])
            if args.skip_render:
                command.append("--skip_render")
            command.extend(variant_options(variant, args, friction_model=friction_model))
            commands[run_variant] = command
            print(f"[EVAL] running {run_variant}", flush=True)
            subprocess.run(command, check=True)
            summary = read_json(output_dir / "stage2_rollout_summary.json")
            rows.append(flatten_summary(variant, output_dir, summary, run_variant=run_variant))

    friction_comparison = friction_model_comparison_rows(rows)
    ranking = rank_variant_rows(rows)
    ranking_rows = flatten_ranking_for_csv(ranking)
    best_refined_params = export_best_refined_params(
        output_root=output_root,
        ranking=ranking,
        trajectory=args.trajectory,
        stage1_ply=args.stage1_ply,
    )

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
            "stage2_friction_model": str(args.stage2_friction_model),
            "friction_model_sweep": friction_models,
            "stage2_friction_num_directions": int(args.stage2_friction_num_directions),
            "stage2_patch_selection": str(args.stage2_patch_selection),
            "stage2_normal_mode": str(args.stage2_normal_mode),
            "skip_render": bool(args.skip_render),
        },
        "aggregate": aggregate_rows(rows),
        "ranking": ranking,
        "best_refined_params": best_refined_params,
        "friction_model_comparison": friction_comparison,
        "rows": rows,
        "commands": {name: " ".join(command) for name, command in commands.items()},
    }
    write_json(output_root / "multi_dice_stage2_variant_report.json", report)
    write_csv(output_root / "multi_dice_stage2_variant_results.csv", rows)
    if ranking_rows:
        write_csv(output_root / "multi_dice_stage2_variant_ranking.csv", ranking_rows)
    if friction_comparison:
        write_csv(output_root / "multi_dice_stage2_friction_model_comparison.csv", friction_comparison)
    print(json.dumps({"aggregate": report["aggregate"], "ranking": ranking.get("best_run_variant")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
