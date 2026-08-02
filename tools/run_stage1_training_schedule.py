from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESETS = REPO_ROOT / "stage1_training_presets.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible Stage1 dataset/train/render/metrics schedule.")
    parser.add_argument("--preset", default="dice_smoke")
    parser.add_argument("--presets_json", default=DEFAULT_PRESETS, type=Path)
    parser.add_argument("--data_root", default=Path("actual_dice_stage1_data"), type=Path)
    parser.add_argument("--output_root", default=Path("actual_dice_stage1_output"), type=Path)
    parser.add_argument("--scene_name", default="")
    parser.add_argument("--model_name", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mujoco_python", default=sys.executable)
    parser.add_argument("--skip_dataset", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--skip_export_physics", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--print_json", action="store_true")
    return parser.parse_args()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def add_flag(command: list[str], name: str, value) -> None:
    if isinstance(value, bool):
        if value:
            command.append(f"--{name}")
        return
    if isinstance(value, (list, tuple)):
        command.append(f"--{name}")
        command.extend(str(item) for item in value)
        return
    if value is None:
        return
    command.extend([f"--{name}", str(value)])


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def validate_schedule(preset_name: str, preset: dict, dataset_dir: Path, *, require_existing_assets: bool) -> None:
    training = preset.get("training", {})
    if training.get("stage1_preset") == "contactwm":
        sam_features = str(training.get("sam_features", ""))
        if not sam_features:
            raise ValueError(f"{preset_name} requires training.sam_features.")
        if not require_existing_assets:
            return
        sam_dir = dataset_dir / sam_features
        if not sam_dir.exists():
            raise FileNotFoundError(
                f"{preset_name} requires SAM features at {sam_dir}. "
                "Generate them first or use dice_smoke/dice_full."
            )


def build_commands(args: argparse.Namespace, preset_name: str, preset: dict) -> tuple[list[dict], dict]:
    scene_name = args.scene_name or f"{preset_name}_dataset"
    model_name = args.model_name or f"{preset_name}_stage1"
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    dataset_dir = data_root / scene_name
    model_dir = output_root / model_name
    iteration = int((preset.get("training") or {}).get("iterations", 30000))

    commands: list[dict] = []
    if not args.skip_dataset:
        dataset_cmd = [
            args.mujoco_python,
            "-m",
            "stage1.generate_mujoco_synthetic_dataset",
            "--output_root",
            str(data_root),
            "--scene_name",
            scene_name,
        ]
        for key, value in (preset.get("dataset") or {}).items():
            add_flag(dataset_cmd, key, value)
        commands.append({"name": "dataset", "command": dataset_cmd})

    validate_schedule(
        preset_name,
        preset,
        dataset_dir,
        require_existing_assets=(not args.dry_run and not args.skip_train),
    )

    if not args.skip_train:
        train_cmd = [
            args.python,
            "-m",
            "stage1.train",
            "--source_path",
            str(dataset_dir),
            "--model_path",
            str(model_dir),
            "--masks_dir",
            str(dataset_dir / "masks"),
        ]
        for key, value in (preset.get("training") or {}).items():
            add_flag(train_cmd, key, value)
        commands.append({"name": "train", "command": train_cmd})

    render_cfg = preset.get("render") or {}
    if bool(render_cfg.get("enabled", False)) and not args.skip_render:
        render_cmd = [
            args.python,
            "-m",
            "stage1.render",
            "--source_path",
            str(dataset_dir),
            "--model_path",
            str(model_dir),
            "--masks_dir",
            str(dataset_dir / "masks"),
            "--iteration",
            str(int(render_cfg.get("iteration", iteration))),
        ]
        for key, value in render_cfg.items():
            if key in ("enabled", "iteration"):
                continue
            add_flag(render_cmd, key, value)
        commands.append({"name": "render", "command": render_cmd})

    metrics_cfg = preset.get("metrics") or {}
    if bool(metrics_cfg.get("enabled", False)) and not args.skip_metrics:
        commands.append(
            {
                "name": "metrics",
                "command": [
                    args.python,
                    "-m",
                    "stage1.metrics",
                    "--model_paths",
                    str(model_dir),
                ],
            }
        )

    export_cfg = preset.get("export_physics") or {}
    if bool(export_cfg.get("enabled", False)) and not args.skip_export_physics:
        export_cmd = [
            args.python,
            "-m",
            "stage1.export_physics_scene",
            "--source_path",
            str(dataset_dir),
            "--model_path",
            str(model_dir),
            "--masks_dir",
            str(dataset_dir / "masks"),
            "--iteration",
            str(int(export_cfg.get("iteration", iteration))),
            "--output_dir",
            str(model_dir / "physics_export" / f"iteration_{int(export_cfg.get('iteration', iteration))}"),
        ]
        for key, value in export_cfg.items():
            if key in ("enabled", "iteration"):
                continue
            add_flag(export_cmd, key, value)
        commands.append({"name": "export_physics", "command": export_cmd})

    manifest = {
        "preset": preset_name,
        "description": preset.get("description", ""),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "dataset_dir": str(dataset_dir),
        "model_dir": str(model_dir),
        "iteration": iteration,
        "commands": [{"name": item["name"], "command": item["command"]} for item in commands],
    }
    return commands, manifest


def main() -> None:
    args = parse_args()
    presets = read_json(args.presets_json.resolve())
    if args.preset not in presets:
        raise ValueError(f"Unknown preset {args.preset!r}. Available: {sorted(presets)}")
    commands, manifest = build_commands(args, args.preset, presets[args.preset])
    if args.print_json:
        print(json.dumps(manifest, indent=2))
    else:
        for item in commands:
            print(f"[{item['name']}] {shell_join(item['command'])}", flush=True)
    if args.dry_run:
        return
    Path(manifest["model_dir"]).mkdir(parents=True, exist_ok=True)
    write_json(Path(manifest["model_dir"]) / "stage1_schedule_manifest.json", manifest)
    for item in commands:
        print(f"[RUN] {item['name']}", flush=True)
        subprocess.run(item["command"], check=True)
    write_json(Path(manifest["model_dir"]) / "stage1_schedule_manifest.json", manifest)


if __name__ == "__main__":
    main()
