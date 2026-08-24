#!/usr/bin/env python
"""Evaluate fixed learned contact physics on test_001 and additional episodes."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2.scene_manifest import load_scene_manifest
from tools.run_native_multibody_manifest import fit_native_image_only
from tools.run_paper_protocol_benchmark import score


def load_multi_episode_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    base = path.resolve().parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()

    episodes = payload.get("episodes") or []
    if len(episodes) < 2:
        raise ValueError("multi-episode evaluation requires at least two episodes")
    ids = [str(item["id"]) for item in episodes]
    if "test_001" not in ids:
        raise ValueError("multi-episode evaluation must include test_001")
    if len(ids) != len(set(ids)):
        raise ValueError("episode ids must be unique")
    return {
        "template_manifest": resolve(payload["template_manifest"]),
        "episodes": [
            {"id": str(item["id"]), "episode_root": resolve(item["episode_root"])}
            for item in episodes
        ],
    }


def extract_contact_parameters(payload: dict, experiment: str | None = None) -> dict:
    if payload.get("learned_contact_pairs"):
        return {"contact_pairs": payload["learned_contact_pairs"]}
    if payload.get("learned_train", {}).get("physics", {}).get("contact_pairs"):
        return payload["learned_train"]["physics"]
    experiments = payload.get("experiments") or []
    if experiments:
        candidates = [
            item for item in experiments
            if experiment is None or item.get("experiment", {}).get("name") == experiment
        ]
        if len(candidates) != 1:
            raise ValueError("select exactly one result with --experiment")
        return {"contact_pairs": candidates[0]["train"]["learned_contact_pairs"]}
    raise ValueError("physics report does not contain learned contact pairs")


def materialize_episode_manifest(template_path: Path, episode: dict, output_dir: Path) -> Path:
    root = episode["episode_root"]
    episode_manifest_path = root / "episode_manifest.json"
    payload = json.loads(episode_manifest_path.read_text(encoding="utf-8-sig"))
    template = json.loads(template_path.read_text(encoding="utf-8-sig"))
    template_base = template_path.resolve().parent
    for body in [*template.get("bodies", []), *template.get("environment", [])]:
        for section, key in (("render", "gaussian_ply"), ("collision", "gaussian_ply"),
                             ("initialization", "state_json")):
            value = body.get(section, {}).get(key)
            if value and not Path(value).is_absolute():
                body[section][key] = str((template_base / value).resolve())
    for required in (root / "rgb", root / "masks", root / "state" / "trajectory.json"):
        if not required.exists():
            raise FileNotFoundError(required)

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{episode['id']}_initial_state.json"
    state_path.write_text(json.dumps(payload["initial_state"], indent=2), encoding="utf-8")
    template["scene_id"] = str(episode["id"])
    dynamic_bodies = [body for body in template.get("bodies", []) if body.get("role") == "dynamic"]
    if len(dynamic_bodies) != 1:
        raise ValueError("episode adapter currently requires exactly one dynamic body")
    dynamic_bodies[0]["initialization"] = {"state_json": str(state_path.resolve())}
    template["observations"] = {
        "rgb_dir": str((root / "rgb").resolve()),
        "instance_mask_dir": str((root / "masks").resolve()),
        "camera_manifest": str(episode_manifest_path.resolve()),
        "fps": float(payload.get("fps", 30.0)),
    }
    template["evaluation"] = {
        "trajectory": str((root / "state" / "trajectory.json").resolve())
    }
    template.pop("actions", None)
    scene_path = output_dir / f"{episode['id']}_scene_manifest.json"
    scene_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return scene_path


def aggregate_episode_metrics(episodes: list[dict]) -> dict:
    metric_names = ("score", "rgb_l1", "rgb_psnr", "translation_error_mean_m", "rotation_error_mean_rad")
    aggregate = {}
    for name in metric_names:
        values = [float(item["metrics"][name]) for item in episodes if item["metrics"].get(name) is not None]
        if not values:
            continue
        candidates = [item for item in episodes if item["metrics"].get(name) is not None]
        worst = min(candidates, key=lambda item: float(item["metrics"][name])) if name == "rgb_psnr" else max(
            candidates, key=lambda item: float(item["metrics"][name])
        )
        aggregate[name] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "min": min(values), "max": max(values),
            "worst_episode": worst["id"],
            "count": len(values),
        }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--physics_report", required=True, type=Path)
    parser.add_argument("--experiment")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--pipeline_mode", default="image_only", choices=("image_only", "paper_compatible"))
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--width", default=160, type=int)
    parser.add_argument("--height", default=120, type=int)
    parser.add_argument("--image_loss", default="l1_ssim", choices=("l1", "l1_ssim", "l1_loftr"))
    args = parser.parse_args()

    protocol = load_multi_episode_protocol(args.protocol)
    physics = extract_contact_parameters(
        json.loads(args.physics_report.read_text(encoding="utf-8-sig")), args.experiment
    )
    args.output.mkdir(parents=True, exist_ok=True)
    generated_dir = args.output / "generated_manifests"
    episode_results = []
    for episode in protocol["episodes"]:
        started = time.monotonic()
        scene_path = materialize_episode_manifest(
            protocol["template_manifest"], episode, generated_dir
        )
        result = fit_native_image_only(
            load_scene_manifest(scene_path), fit_iters=0, lr=0.002,
            stride=args.stride, max_frames=0, width=args.width, height=args.height,
            image_loss=args.image_loss, device=torch.device(args.device),
            pipeline_mode=args.pipeline_mode, physics_override=physics,
            learn_contact_parameters=False, learn_mass_inertia=False, evaluation_only=True,
        )
        trajectory = result["evaluation"].get("trajectory") or {}
        metrics = {
            "score": score(result), "rgb_l1": result["evaluation"]["rgb_l1"],
            "rgb_psnr": result["evaluation"]["rgb_psnr"],
            "translation_error_mean_m": trajectory.get("translation_error_mean_m"),
            "rotation_error_mean_rad": trajectory.get("rotation_error_mean_rad"),
        }
        item = {
            "id": episode["id"], "episode_root": str(episode["episode_root"]),
            "scene_manifest": str(scene_path), "frame_count": len(result["frame_indices"]),
            "metrics": metrics, "elapsed_seconds": time.monotonic() - started,
        }
        (args.output / f"{episode['id']}.json").write_text(
            json.dumps(item, indent=2), encoding="utf-8"
        )
        episode_results.append(item)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "protocol": str(args.protocol.resolve()), "physics_report": str(args.physics_report.resolve()),
        "experiment": args.experiment, "physics": physics, "episodes": episode_results,
        "aggregate": aggregate_episode_metrics(episode_results),
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "episodes": len(episode_results)}, indent=2))


if __name__ == "__main__":
    main()
