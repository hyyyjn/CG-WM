#!/usr/bin/env python
"""15-frame train/unseen holdout benchmark with learned, no-opt and CEM baselines."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2.scene_manifest import load_scene_manifest
from tools.run_native_multibody_manifest import fit_native_image_only


def load_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if "train_manifest" not in payload or not payload.get("holdout_manifests"):
        raise ValueError("protocol requires train_manifest and non-empty holdout_manifests")
    base = path.resolve().parent
    resolve = lambda value: (base / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    return {
        "train_manifest": resolve(payload["train_manifest"]),
        "holdout_manifests": [resolve(value) for value in payload["holdout_manifests"]],
    }


def contact_parameters(result: dict) -> dict:
    return {"contact_pairs": result["learned_contact_pairs"]}


def geometric_candidates(initial: dict, count: int, seed: int, log_radius: float) -> list[dict]:
    rng = random.Random(seed)
    candidates = [initial]
    for _ in range(max(0, count - 1)):
        pairs = []
        for pair in initial["contact_pairs"]:
            varied = dict(pair)
            for key in ("stiffness", "damping", "friction"):
                varied[key] = float(pair[key]) * math.exp(rng.uniform(-log_radius, log_radius))
            pairs.append(varied)
        candidates.append({"contact_pairs": pairs})
    return candidates


def run_fit(manifest_path: Path, args, *, fit_iters: int, max_frames: int,
            physics_override: dict | None = None, learn_contact: bool = True,
            stride: int | None = None) -> dict:
    manifest = load_scene_manifest(manifest_path)
    result = fit_native_image_only(
        manifest, fit_iters=fit_iters, lr=args.lr,
        stride=args.stride if stride is None else stride,
        max_frames=max_frames, width=args.width, height=args.height,
        image_loss=args.image_loss, device=torch.device(args.device),
        pipeline_mode=args.pipeline_mode, prefit_initial_state=args.prefit_initial_state,
        learn_mass_inertia=learn_contact, refine_geometry=args.refine_geometry and learn_contact,
        physics_override=physics_override, learn_contact_parameters=learn_contact,
        evaluation_only=not learn_contact,
        physics_warmup_fraction=args.physics_warmup_fraction,
        freeze_initial_state_after_warmup=args.freeze_initial_state_after_warmup,
        contact_parameter_l2=args.contact_parameter_l2,
        trajectory_stability_weight=args.trajectory_stability_weight,
        max_dynamic_displacement=args.max_dynamic_displacement,
        stiffness_lr=args.stiffness_lr, damping_lr=args.damping_lr,
        friction_lr=args.friction_lr,
        contact_curriculum_frames=args.contact_curriculum_frames,
        silhouette_weight=args.silhouette_weight,
        silhouette_fp_weight=args.silhouette_fp_weight,
        silhouette_fn_weight=args.silhouette_fn_weight,
        gradient_attribution=args.gradient_attribution,
        gradient_attribution_interval=args.gradient_attribution_interval,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def score(result: dict, *, prefer_trajectory: bool = True) -> float:
    trajectory = result["evaluation"].get("trajectory")
    if prefer_trajectory and trajectory is not None:
        return float(trajectory["translation_error_mean_m"]) + 0.1 * float(
            trajectory["rotation_error_mean_rad"]
        )
    return float(result["evaluation"]["rgb_l1"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--pipeline_mode", default="image_only", choices=("image_only", "paper_compatible"))
    parser.add_argument("--fit_iters", default=250, type=int)
    parser.add_argument("--train_frames", default=15, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--holdout_stride", default=1, type=int)
    parser.add_argument("--width", default=160, type=int)
    parser.add_argument("--height", default=120, type=int)
    parser.add_argument("--image_loss", default="l1_ssim", choices=("l1", "l1_ssim", "l1_loftr"))
    parser.add_argument("--lr", default=0.002, type=float)
    parser.add_argument("--prefit_initial_state", action="store_true")
    parser.add_argument("--refine_geometry", action="store_true")
    parser.add_argument("--cem_candidates", default=100, type=int)
    parser.add_argument("--cem_rounds", default=5, type=int)
    parser.add_argument("--cem_elite_fraction", default=0.2, type=float)
    parser.add_argument("--cem_seed", default=0, type=int)
    parser.add_argument("--cem_log_radius", default=1.5, type=float)
    parser.add_argument("--physics_warmup_fraction", default=0.2, type=float)
    parser.add_argument(
        "--keep_initial_state_trainable_after_warmup", action="store_false",
        dest="freeze_initial_state_after_warmup",
    )
    parser.set_defaults(freeze_initial_state_after_warmup=True)
    parser.add_argument("--contact_parameter_l2", default=1e-4, type=float)
    parser.add_argument("--trajectory_stability_weight", default=1e-4, type=float)
    parser.add_argument("--max_dynamic_displacement", default=3.0, type=float)
    parser.add_argument("--stiffness_lr", default=None, type=float)
    parser.add_argument("--damping_lr", default=None, type=float)
    parser.add_argument("--friction_lr", default=None, type=float)
    parser.add_argument("--contact_curriculum_frames", default=8, type=int)
    parser.add_argument("--silhouette_weight", default=0.005, type=float)
    parser.add_argument("--silhouette_fp_weight", default=2.0, type=float)
    parser.add_argument("--silhouette_fn_weight", default=1.0, type=float)
    parser.add_argument("--gradient_attribution", action="store_true")
    parser.add_argument("--gradient_attribution_interval", default=1, type=int)
    args = parser.parse_args()
    if args.train_frames != 15:
        parser.error("paper protocol requires --train_frames 15")

    protocol = load_protocol(args.protocol)
    learned_train = run_fit(
        protocol["train_manifest"], args, fit_iters=args.fit_iters, max_frames=15
    )
    learned_physics = contact_parameters(learned_train)

    no_opt_train = run_fit(
        protocol["train_manifest"], args, fit_iters=0, max_frames=15,
        learn_contact=False,
    )
    initial_physics = contact_parameters(no_opt_train)
    if args.cem_rounds < 1 or args.cem_candidates < args.cem_rounds:
        parser.error("CEM requires candidates >= rounds >= 1")
    if not 0.0 < args.cem_elite_fraction <= 1.0:
        parser.error("--cem_elite_fraction must be in (0, 1]")
    template_pairs = initial_physics["contact_pairs"]
    mean = [math.log(float(pair[key])) for pair in template_pairs
            for key in ("stiffness", "damping", "friction")]
    std = [float(args.cem_log_radius)] * len(mean)
    rng = random.Random(args.cem_seed)
    cem_trials = []
    remaining = int(args.cem_candidates)
    for round_index in range(int(args.cem_rounds)):
        round_count = remaining // (int(args.cem_rounds) - round_index)
        remaining -= round_count
        vectors = [mean] if round_index == 0 else []
        while len(vectors) < round_count:
            vectors.append([rng.gauss(mu, sigma) for mu, sigma in zip(mean, std)])
        round_trials = []
        for vector in vectors:
            cursor, pairs = 0, []
            for pair in template_pairs:
                varied = dict(pair)
                for key in ("stiffness", "damping", "friction"):
                    varied[key] = math.exp(vector[cursor])
                    cursor += 1
                pairs.append(varied)
            candidate = {"contact_pairs": pairs}
            result = run_fit(
                protocol["train_manifest"], args, fit_iters=0, max_frames=15,
                physics_override=candidate, learn_contact=False,
            )
            trial = {
                "index": len(cem_trials), "round": round_index,
                "score": score(result, prefer_trajectory=False),
                "physics": candidate, "log_vector": vector,
            }
            cem_trials.append(trial)
            round_trials.append(trial)
        elite_count = max(1, int(math.ceil(len(round_trials) * args.cem_elite_fraction)))
        elites = sorted(round_trials, key=lambda item: item["score"])[:elite_count]
        mean = [sum(item["log_vector"][i] for item in elites) / elite_count for i in range(len(mean))]
        std = [max(0.05, math.sqrt(sum(
            (item["log_vector"][i] - mean[i]) ** 2 for item in elites
        ) / elite_count)) for i in range(len(std))]
    best_cem = min(cem_trials, key=lambda item: item["score"])

    holdouts = []
    for manifest_path in protocol["holdout_manifests"]:
        methods = {}
        for name, physics in (
            ("contactgaussian", learned_physics), ("no_opt", initial_physics),
            ("cem", best_cem["physics"]),
        ):
            result = run_fit(
                manifest_path, args, fit_iters=0, max_frames=0,
                physics_override=physics, learn_contact=False,
                stride=args.holdout_stride,
            )
            methods[name] = {
                "evaluation": result["evaluation"],
                "learned_contact_pairs": result["learned_contact_pairs"],
            }
        holdouts.append({"manifest": str(manifest_path), "methods": methods})

    report = {
        "protocol": {
            "train_frames": 15, "holdout": "unseen manifests, full open-loop sequence",
            "metric": "mean translation + 0.1*mean rotation when pose labels exist; RGB L1 otherwise",
            "cem_candidates": args.cem_candidates,
            "cem_seed": args.cem_seed,
            "cem_rounds": args.cem_rounds,
            "cem_elite_fraction": args.cem_elite_fraction,
        },
        "train_manifest": str(protocol["train_manifest"]),
        "learned_train": {
            "evaluation": learned_train["evaluation"], "physics": learned_physics,
            "optimization_stability": learned_train["optimization_stability"],
        },
        "no_opt_train": {"evaluation": no_opt_train["evaluation"], "physics": initial_physics},
        "cem": {"best": best_cem, "trials": cem_trials},
        "holdouts": holdouts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "holdouts": len(holdouts)}, indent=2))


if __name__ == "__main__":
    main()
