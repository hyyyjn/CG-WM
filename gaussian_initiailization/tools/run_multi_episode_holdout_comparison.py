#!/usr/bin/env python
"""Multi-initialization, multi-episode train/holdout query comparison.

The dataset manifest is JSON:

{
  "episodes": [
    {"name": "train_0", "split": "train", "episode_root": "...", "stage1_ply": "..."},
    {"name": "test_0",  "split": "holdout", "episode_root": "...", "stage1_ply": "..."}
  ]
}

For every initialization and train episode the regular comparison fit is run.
The best initialization is selected independently per query scheme using mean
train score. Learned K/D are median-aggregated across train episodes. Holdout
episodes then use those fixed parameters with ``--eval_only``.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


COMPARE = Path(__file__).resolve().parent / "compare_query_modes.py"
DEFAULT_VARIANTS = ["floor_disk", "axis6", "fib26", "analytic"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_manifest", required=True, type=Path)
    p.add_argument("--output_root", required=True, type=Path)
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    p.add_argument(
        "--initializations",
        nargs="+",
        default=["400:15", "800:30", "1600:60"],
        help="Stiffness:damping initial values.",
    )
    p.add_argument("--fit_iters", default=250, type=int)
    p.add_argument("--max_frames", default=80, type=int)
    p.add_argument("--body_lowest_k", default=32, type=int)
    p.add_argument("--pairwise_friction_coefficient", default=0.1, type=float)
    p.add_argument(
        "--pairwise_friction_mode",
        default="learned",
        choices=("off", "fixed", "learned"),
    )
    p.add_argument(
        "--pairwise_contact_model",
        default="dual_cone",
        choices=("dual_cone", "projected"),
    )
    p.add_argument("--pairwise_dual_cone_directions", default=4, type=int)
    p.add_argument("--mass", default=1.0, type=float)
    p.add_argument("--pairwise_mass_mode", default="learned", choices=("fixed", "learned"))
    p.add_argument("--pairwise_inertia_diag", default="1,1,1")
    p.add_argument("--pairwise_inertia_mode", default="learned", choices=("fixed", "learned"))
    p.add_argument("--orientation_score_weight", default=0.1, type=float)
    p.add_argument("--geometry_loss_weight", default=0.0, type=float)
    p.add_argument("--gaussian_rgb_loss_weight", default=0.0, type=float)
    p.add_argument("--gaussian_render_stride", default=10, type=int)
    p.add_argument("--image_only_objective", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def load_dataset(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    base = path.resolve().parent
    episodes = payload.get("episodes", [])
    if not episodes:
        raise ValueError("dataset manifest must contain a non-empty 'episodes' list")
    names = set()
    resolved = []
    for item in episodes:
        missing = {"name", "split", "episode_root", "stage1_ply"} - set(item)
        if missing:
            raise ValueError(f"episode is missing fields: {sorted(missing)}")
        if item["name"] in names:
            raise ValueError(f"duplicate episode name: {item['name']}")
        if item["split"] not in ("train", "holdout"):
            raise ValueError(f"{item['name']}: split must be train or holdout")
        names.add(item["name"])
        copy = dict(item)
        for key in (
            "episode_root", "stage1_ply", "pairwise_body_b_ply",
            "rgb_dir", "mask_dir", "views_manifest", "actions_json",
            "pairwise_body_b_trajectory_json",
        ):
            if copy.get(key):
                candidate = Path(copy[key])
                copy[key] = str((base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())
        resolved.append(copy)
    if not any(e["split"] == "train" for e in resolved):
        raise ValueError("at least one train episode is required")
    if not any(e["split"] == "holdout" for e in resolved):
        raise ValueError("at least one holdout episode is required")
    return resolved


def parse_initializations(values: list[str]) -> list[dict]:
    parsed = []
    for index, value in enumerate(values):
        try:
            stiffness, damping = (float(v) for v in value.split(":", 1))
        except Exception as exc:
            raise ValueError(f"invalid initialization '{value}', expected stiffness:damping") from exc
        if stiffness <= 0 or damping <= 0:
            raise ValueError("stiffness and damping initializations must be positive")
        parsed.append({"id": f"init_{index:02d}", "stiffness": stiffness, "damping": damping})
    return parsed


def metric_score(metric: dict, orientation_weight: float) -> float:
    rotation_radians = math.radians(float(metric.get("orientation_rmse_degrees") or 0.0))
    return float(metric["total"]) + float(orientation_weight) * rotation_radians


def run_comparison(
    args: argparse.Namespace,
    episode: dict,
    output_dir: Path,
    variants: list[str],
    stiffness: float,
    damping: float,
    friction: float,
    mass: float,
    inertia_diag: str,
    *,
    eval_only: bool,
) -> dict | None:
    metrics_path = output_dir / "query_mode_comparison_metrics.json"
    if args.resume and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    cmd = [
        sys.executable, str(COMPARE),
        "--episode_root", episode["episode_root"],
        "--stage1_ply", episode["stage1_ply"],
        "--output_root", str(output_dir),
        "--variants", *variants,
        "--init_stiffness", str(stiffness),
        "--init_damping", str(damping),
        "--pairwise_friction_coefficient", str(friction),
        "--pairwise_friction_mode", (
            "fixed" if eval_only and args.pairwise_friction_mode == "learned"
            else args.pairwise_friction_mode
        ),
        "--pairwise_contact_model", args.pairwise_contact_model,
        "--pairwise_dual_cone_directions", str(args.pairwise_dual_cone_directions),
        "--mass", str(mass),
        "--pairwise_mass_mode", (
            "fixed" if eval_only and args.pairwise_mass_mode == "learned"
            else args.pairwise_mass_mode
        ),
        "--pairwise_inertia_diag", inertia_diag,
        "--pairwise_inertia_mode", (
            "fixed" if eval_only and args.pairwise_inertia_mode == "learned"
            else args.pairwise_inertia_mode
        ),
        "--fit_iters", str(args.fit_iters),
        "--max_frames", str(args.max_frames),
        "--body_lowest_k", str(args.body_lowest_k),
        "--pairwise_num_contact_patches", str(args.body_lowest_k),
        "--geometry_loss_weight", str(args.geometry_loss_weight),
        "--gaussian_rgb_loss_weight", str(args.gaussian_rgb_loss_weight),
        "--gaussian_render_stride", str(args.gaussian_render_stride),
        "--skip_render",
    ]
    if episode.get("pairwise_body_b_ply"):
        cmd += ["--pairwise_body_b_ply", episode["pairwise_body_b_ply"]]
    if episode.get("rgb_dir"):
        cmd += ["--gaussian_rgb_dir", episode["rgb_dir"]]
    if episode.get("mask_dir"):
        cmd += ["--gaussian_mask_dir", episode["mask_dir"]]
    if episode.get("views_manifest"):
        cmd += ["--gaussian_views_manifest", episode["views_manifest"]]
    if episode.get("actions_json"):
        cmd += ["--actions_json", episode["actions_json"]]
    if episode.get("pairwise_body_b_trajectory_json"):
        cmd += [
            "--pairwise_body_b_trajectory_json",
            episode["pairwise_body_b_trajectory_json"],
        ]
    if args.image_only_objective:
        cmd.append("--image_only_objective")
    if eval_only:
        cmd.append("--eval_only")
    print("$ " + " ".join(cmd), flush=True)
    if args.dry_run:
        return None
    subprocess.run(cmd, check=True)
    return json.loads(metrics_path.read_text(encoding="utf-8-sig"))


def main() -> None:
    args = parse_args()
    episodes = load_dataset(args.dataset_manifest)
    initializations = parse_initializations(args.initializations)
    args.output_root.mkdir(parents=True, exist_ok=True)
    train_episodes = [e for e in episodes if e["split"] == "train"]
    holdout_episodes = [e for e in episodes if e["split"] == "holdout"]

    train_runs = []
    for init in initializations:
        for episode in train_episodes:
            out = args.output_root / "train" / init["id"] / episode["name"]
            payload = run_comparison(
                args, episode, out, args.variants, init["stiffness"], init["damping"],
                args.pairwise_friction_coefficient, args.mass, args.pairwise_inertia_diag,
                eval_only=False
            )
            record = {"initialization": init, "episode": episode["name"], "output_dir": str(out.resolve())}
            if payload is not None:
                record["metrics"] = payload["variants"]
            train_runs.append(record)

    selections = {}
    if not args.dry_run:
        for variant in args.variants:
            candidates = []
            for init in initializations:
                matching = [r for r in train_runs if r["initialization"]["id"] == init["id"]]
                scores = [metric_score(r["metrics"][variant], args.orientation_score_weight) for r in matching]
                learned_k = [float(r["metrics"][variant]["stiffness"]) for r in matching]
                learned_d = [float(r["metrics"][variant]["damping"]) for r in matching]
                learned_mu = [float(r["metrics"][variant]["mu"]) for r in matching]
                learned_mass = [float(r["metrics"][variant]["mass"]) for r in matching]
                learned_inertia = [
                    [float(value) for value in r["metrics"][variant]["inertia_diag"]]
                    for r in matching
                ]
                candidates.append({
                    "initialization": init,
                    "mean_train_score": statistics.mean(scores),
                    "train_scores": scores,
                    "aggregated_stiffness": statistics.median(learned_k),
                    "aggregated_damping": statistics.median(learned_d),
                    "aggregated_friction": statistics.median(learned_mu),
                    "aggregated_mass": statistics.median(learned_mass),
                    "aggregated_inertia_diag": [
                        statistics.median(values)
                        for values in zip(*learned_inertia)
                    ],
                })
            selections[variant] = min(candidates, key=lambda item: item["mean_train_score"])

    holdout_runs = []
    if not args.dry_run:
        for variant, selection in selections.items():
            for episode in holdout_episodes:
                out = args.output_root / "holdout" / variant / episode["name"]
                payload = run_comparison(
                    args,
                    episode,
                    out,
                    [variant],
                    selection["aggregated_stiffness"],
                    selection["aggregated_damping"],
                    selection["aggregated_friction"],
                    selection["aggregated_mass"],
                    ",".join(str(value) for value in selection["aggregated_inertia_diag"]),
                    eval_only=True,
                )
                metric = payload["variants"][variant]
                holdout_runs.append({
                    "variant": variant,
                    "episode": episode["name"],
                    "fixed_stiffness": selection["aggregated_stiffness"],
                    "fixed_damping": selection["aggregated_damping"],
                    "fixed_friction": selection["aggregated_friction"],
                    "fixed_mass": selection["aggregated_mass"],
                    "fixed_inertia_diag": selection["aggregated_inertia_diag"],
                    "score": metric_score(metric, args.orientation_score_weight),
                    "metrics": metric,
                    "output_dir": str(out.resolve()),
                })

    report = {
        "protocol": {
            "selection": "lowest mean train score per variant and initialization",
            "parameter_aggregation": "median learned K/D/mu/mass/inertia across train episodes",
            "holdout": "fixed K/D/mu/mass/inertia, optimizer disabled with --eval_only",
            "score": "position_total_rmse + orientation_score_weight * orientation_rmse_radians",
            "orientation_score_weight": args.orientation_score_weight,
        },
        "dataset_manifest": str(args.dataset_manifest.resolve()),
        "episodes": episodes,
        "initializations": initializations,
        "variants": args.variants,
        "train_runs": train_runs,
        "selections": selections,
        "holdout_runs": holdout_runs,
        "dry_run": args.dry_run,
    }
    report_path = args.output_root / "multi_episode_holdout_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[report] {report_path}")


if __name__ == "__main__":
    main()
