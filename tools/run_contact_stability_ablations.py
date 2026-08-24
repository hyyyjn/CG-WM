#!/usr/bin/env python
"""Run the 50-iteration stabilization, contact-LR and silhouette ablations."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2.scene_manifest import load_scene_manifest
from tools.run_native_multibody_manifest import fit_native_image_only
from tools.run_paper_protocol_benchmark import contact_parameters, load_protocol, score


@dataclass(frozen=True)
class Experiment:
    name: str
    group: str
    stiffness_lr: float
    damping_lr: float
    friction_lr: float
    silhouette_weight: float
    physics_warmup_fraction: float = 0.2
    contact_parameter_l2: float = 1e-4
    trajectory_stability_weight: float = 1e-4
    contact_curriculum_frames: int = 8
    freeze_initial_state_after_warmup: bool = True


def build_experiments() -> list[Experiment]:
    # Keep the before/after comparison controlled: only stabilization switches
    # differ. Learning-rate changes belong exclusively to the LR sweep below.
    stable = Experiment("stabilized", "stabilization", 0.002, 0.002, 0.002, 0.005)
    experiments = [
        replace(
            stable, name="legacy", silhouette_weight=0.0,
            physics_warmup_fraction=0.0, contact_parameter_l2=0.0,
            trajectory_stability_weight=0.0, contact_curriculum_frames=0,
            freeze_initial_state_after_warmup=False,
            stiffness_lr=0.002, damping_lr=0.002, friction_lr=0.002,
        ),
        stable,
    ]
    for name, k_lr, d_lr, mu_lr in (
        ("lr_equal_002", 0.002, 0.002, 0.002),
        ("lr_kd005_mu002", 0.005, 0.005, 0.002),
        ("lr_kd010_mu002", 0.010, 0.010, 0.002),
        ("lr_k010_d005_mu001", 0.010, 0.005, 0.001),
    ):
        experiments.append(replace(
            stable, name=name, group="lr", stiffness_lr=k_lr,
            damping_lr=d_lr, friction_lr=mu_lr,
        ))
    for weight in (0.0, 0.002, 0.005, 0.01, 0.02):
        suffix = str(weight).replace(".", "p")
        experiments.append(replace(
            stable, name=f"silhouette_{suffix}", group="silhouette",
            silhouette_weight=weight,
        ))
    experiments.append(replace(
        stable, name="gradient_attribution_recommended", group="attribution",
    ))
    return experiments


def _gradient_summary(history: list[dict]) -> dict:
    keys = ("log_stiffness", "log_damping", "log_friction")
    result = {}
    for key in keys:
        values = [float(row.get("physics_gradient_norms", {}).get(key, 0.0)) for row in history]
        result[key] = {
            "mean": statistics.fmean(values) if values else 0.0,
            "max": max(values, default=0.0),
            "nonzero_iterations": sum(value > 0.0 for value in values),
        }
    return result


def _loss_attribution_summary(history: list[dict]) -> dict:
    collected: dict[str, dict[str, list[float]]] = {}
    for row in history:
        for loss_name, groups in (row.get("loss_gradient_attribution") or {}).items():
            for group_name, value in groups.items():
                collected.setdefault(loss_name, {}).setdefault(group_name, []).append(float(value))
    return {
        loss_name: {
            group_name: {
                "mean": statistics.fmean(values), "max": max(values),
                "nonzero_iterations": sum(value > 0.0 for value in values),
            }
            for group_name, values in groups.items()
        }
        for loss_name, groups in collected.items()
    }


def _run_fit(manifest_path: Path, args, experiment: Experiment, *, fit_iters: int,
             stride: int, max_frames: int, physics_override: dict | None = None) -> dict:
    learn = physics_override is None and fit_iters > 0
    result = fit_native_image_only(
        load_scene_manifest(manifest_path), fit_iters=fit_iters, lr=args.lr,
        stride=stride, max_frames=max_frames, width=args.width, height=args.height,
        image_loss=args.image_loss, device=torch.device(args.device),
        pipeline_mode=args.pipeline_mode, learn_mass_inertia=learn,
        learn_contact_parameters=learn, evaluation_only=not learn,
        physics_override=physics_override, refine_geometry=args.refine_geometry and learn,
        physics_warmup_fraction=experiment.physics_warmup_fraction,
        freeze_initial_state_after_warmup=experiment.freeze_initial_state_after_warmup,
        contact_parameter_l2=experiment.contact_parameter_l2,
        trajectory_stability_weight=experiment.trajectory_stability_weight,
        max_dynamic_displacement=args.max_dynamic_displacement,
        stiffness_lr=experiment.stiffness_lr, damping_lr=experiment.damping_lr,
        friction_lr=experiment.friction_lr,
        contact_curriculum_frames=experiment.contact_curriculum_frames,
        silhouette_weight=experiment.silhouette_weight,
        silhouette_fp_weight=args.silhouette_fp_weight,
        silhouette_fn_weight=args.silhouette_fn_weight,
        gradient_attribution=args.gradient_attribution,
        gradient_attribution_interval=args.gradient_attribution_interval,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _compact(result: dict) -> dict:
    return {
        "evaluation": result["evaluation"],
        "fit_iterations": result["fit_iterations"],
        "frame_indices": result["frame_indices"],
        "learned_contact_pairs": result["learned_contact_pairs"],
        "optimization_stability": result["optimization_stability"],
        "gradient_summary": _gradient_summary(result.get("loss_history", [])),
        "loss_gradient_attribution_summary": _loss_attribution_summary(
            result.get("loss_history", [])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--groups", nargs="+",
        choices=("stabilization", "lr", "silhouette", "attribution"),
        default=("stabilization", "lr", "silhouette"),
    )
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--pipeline_mode", default="image_only", choices=("image_only", "paper_compatible"))
    parser.add_argument("--fit_iters", default=50, type=int)
    parser.add_argument("--train_frames", default=15, type=int)
    parser.add_argument("--train_stride", default=5, type=int)
    parser.add_argument("--holdout_stride", default=1, type=int)
    parser.add_argument("--width", default=160, type=int)
    parser.add_argument("--height", default=120, type=int)
    parser.add_argument("--image_loss", default="l1_ssim", choices=("l1", "l1_ssim", "l1_loftr"))
    parser.add_argument("--lr", default=0.002, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--refine_geometry", action="store_true")
    parser.add_argument("--max_dynamic_displacement", default=3.0, type=float)
    parser.add_argument("--silhouette_fp_weight", default=2.0, type=float)
    parser.add_argument("--silhouette_fn_weight", default=1.0, type=float)
    parser.add_argument("--gradient_attribution", action="store_true")
    parser.add_argument("--gradient_attribution_interval", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.fit_iters != 50:
        parser.error("this benchmark fixes --fit_iters to 50")
    if args.train_frames != 15:
        parser.error("the paper protocol requires exactly 15 training frames")
    if "attribution" in args.groups and not args.gradient_attribution:
        parser.error("the attribution group requires --gradient_attribution")

    protocol = load_protocol(args.protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [item for item in build_experiments() if item.group in args.groups]
    summaries = []
    for index, experiment in enumerate(selected, 1):
        run_path = args.output / f"{experiment.name}.json"
        if args.resume and run_path.exists():
            summaries.append(json.loads(run_path.read_text(encoding="utf-8")))
            continue
        print(f"[{index}/{len(selected)}] {experiment.name}", flush=True)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        started = time.monotonic()
        train = _run_fit(
            protocol["train_manifest"], args, experiment, fit_iters=50,
            stride=args.train_stride, max_frames=15,
        )
        physics = contact_parameters(train)
        holdouts = []
        for path in protocol["holdout_manifests"]:
            evaluated = _run_fit(
                path, args, experiment, fit_iters=0, stride=args.holdout_stride,
                max_frames=0, physics_override=physics,
            )
            holdouts.append({"manifest": str(path), **_compact(evaluated), "score": score(evaluated)})
        summary = {
            "experiment": asdict(experiment), "elapsed_seconds": time.monotonic() - started,
            "train": _compact(train), "train_score": score(train), "holdouts": holdouts,
            "holdout_score_mean": statistics.fmean(item["score"] for item in holdouts),
        }
        run_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)

    # Build the report from every completed run in the output directory so a
    # resumed group does not discard results produced by earlier groups.
    completed = []
    for experiment in build_experiments():
        run_path = args.output / f"{experiment.name}.json"
        if run_path.exists():
            completed.append(json.loads(run_path.read_text(encoding="utf-8")))
    available_groups = list(dict.fromkeys(
        item["experiment"]["group"] for item in completed
    ))
    best_by_group = {
        group: min(
            (item for item in completed if item["experiment"]["group"] == group),
            key=lambda item: item["holdout_score_mean"], default=None,
        )
        for group in available_groups
    }
    best_lr = best_by_group.get("lr")
    best_silhouette = best_by_group.get("silhouette")
    recommended_settings = None
    if best_lr is not None and best_silhouette is not None:
        lr_config = best_lr["experiment"]
        silhouette_config = best_silhouette["experiment"]
        recommended_settings = {
            "stiffness_lr": lr_config["stiffness_lr"],
            "damping_lr": lr_config["damping_lr"],
            "friction_lr": lr_config["friction_lr"],
            "silhouette_weight": silhouette_config["silhouette_weight"],
            "physics_warmup_fraction": silhouette_config["physics_warmup_fraction"],
            "contact_parameter_l2": silhouette_config["contact_parameter_l2"],
            "trajectory_stability_weight": silhouette_config["trajectory_stability_weight"],
            "contact_curriculum_frames": silhouette_config["contact_curriculum_frames"],
            "freeze_initial_state_after_warmup": silhouette_config[
                "freeze_initial_state_after_warmup"
            ],
            "selection_metric": "holdout_score_mean",
            "lr_source": best_lr["experiment"]["name"],
            "silhouette_source": best_silhouette["experiment"]["name"],
        }
    report = {
        "protocol": str(args.protocol.resolve()), "fit_iterations": 50,
        "train_frames": 15, "seed": args.seed, "executed_groups": list(args.groups),
        "available_groups": available_groups, "experiments": completed,
        "best_by_group": best_by_group,
        "recommended_settings": recommended_settings,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(args.output / "report.json"),
        "best": {key: None if value is None else value["experiment"]["name"]
                 for key, value in report["best_by_group"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
