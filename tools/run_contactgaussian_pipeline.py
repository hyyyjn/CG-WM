"""Compile a generic scene manifest into the current single-body Stage-II CLI."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2.scene_manifest import BodySpec, SceneManifest, load_scene_manifest, manifest_summary


def _resolve_manifest_value(manifest: SceneManifest, value: str | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else manifest.path.parent / path).resolve()


def _sequence_arg(value: Any, *, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _single_body_contract(manifest: SceneManifest) -> tuple[BodySpec, BodySpec, Any]:
    dynamic = [body for body in manifest.all_bodies if body.role == "dynamic"]
    if len(dynamic) != 1:
        raise ValueError(
            f"The current adapter supports exactly one dynamic body, got {[body.id for body in dynamic]}. "
            "Native multi-body manifest execution is a later step."
        )
    dynamic_body = dynamic[0]
    relevant_pairs = [
        pair for pair in manifest.contact_pairs if dynamic_body.id in {pair.body_a, pair.body_b}
    ]
    if len(relevant_pairs) != 1:
        raise ValueError(f"Dynamic body {dynamic_body.id!r} must have exactly one contact pair.")
    pair = relevant_pairs[0]
    environment_id = pair.body_b if pair.body_a == dynamic_body.id else pair.body_a
    environment = manifest.body(environment_id)
    if environment.role != "static" or environment.collision.type != "plane":
        raise ValueError("The current adapter supports one static analytic plane contact target.")
    normal = [float(value) for value in environment.collision.parameters.get("normal", [0, 0, 1])]
    height = float(environment.collision.parameters.get("height", 0.0))
    if any(abs(value - expected) > 1e-6 for value, expected in zip(normal, [0.0, 0.0, 1.0])) or abs(height) > 1e-6:
        raise ValueError("The current PlaneCollider adapter requires normal=[0,0,1] and height=0.")
    if dynamic_body.render is None:
        raise ValueError("The dynamic body requires render.gaussian_ply.")
    collision_ply = dynamic_body.collision.gaussian_ply or dynamic_body.render.gaussian_ply
    if collision_ply != dynamic_body.render.gaussian_ply:
        raise ValueError("The current unified adapter requires render and collision to share one Gaussian PLY.")
    return dynamic_body, environment, pair


def compile_manifest_command(
    manifest: SceneManifest,
    *,
    output_dir: Path,
    python_executable: str,
    compatibility_root: Path,
    fit_iters_override: int | None = None,
    max_frames_override: int | None = None,
    device_override: str | None = None,
    image_loss_override: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    dynamic, environment, pair = _single_body_contract(manifest)
    training = manifest.training
    if str(training.get("supervision", "image_only")) != "image_only":
        raise ValueError("The generic adapter currently requires training.supervision='image_only'.")

    collision = dynamic.collision.parameters
    render = dynamic.render
    physics = dynamic.physics
    mass = physics.get("mass", {})
    mass_initial = float(mass.get("initial", 1.0)) if isinstance(mass, dict) else float(mass)
    initialization = dynamic.initialization
    initial_state_json = _resolve_manifest_value(manifest, initialization.get("state_json"))
    evaluation = manifest.evaluation_trajectory
    actions_path = _resolve_manifest_value(manifest, manifest.actions.get("trajectory"))
    pair_parameters = pair.parameters
    stiffness = pair_parameters.get("stiffness", {})
    damping = pair_parameters.get("damping", {})
    stiffness_initial = float(stiffness.get("initial", 800.0)) if isinstance(stiffness, dict) else float(stiffness)
    damping_initial = float(damping.get("initial", 30.0)) if isinstance(damping, dict) else float(damping)

    command = [
        python_executable,
        str(REPO_ROOT / "tools" / "run_stage2_mujoco_stage1_fit.py"),
        "--episode_root", str(compatibility_root),
        "--source_scene_manifest", str(manifest.path),
        "--stage1_ply", str(render.gaussian_ply),
        "--output_dir", str(output_dir),
        "--image_only_objective",
        "--gaussian_rgb_dir", str(manifest.observations.rgb_dir),
        "--gaussian_rgb_loss_weight", str(float(training.get("gaussian_rgb_loss_weight", 1.0))),
        "--gaussian_render_loss", str(
            training.get("image_loss", "l1_loftr") if image_loss_override is None else image_loss_override
        ),
        "--fit_iters", str(int(training.get("fit_iterations", 500)) if fit_iters_override is None else fit_iters_override),
        "--lr", str(float(training.get("lr", 0.02))),
        "--dynamics", str(training.get("dynamics", "impedance")),
        "--init_stiffness", str(stiffness_initial),
        "--init_damping", str(damping_initial),
        "--mass", str(mass_initial),
        "--gaussian_radius_convention", str(collision.get("radius_convention", "paper_r2s")),
        "--max_primitives", str(int(collision.get("max_primitives", 512))),
        "--gaussian_render_stride", str(int(training.get("render_stride", 1))),
        "--gaussian_render_max_frames", str(int(training.get("render_max_frames", 0))),
        "--gaussian_render_width", str(int(training.get("render_width", 160))),
        "--gaussian_render_height", str(int(training.get("render_height", 120))),
        "--device", str(training.get("device", "cuda") if device_override is None else device_override),
        "--disable_collision_bbox_calibration",
        "--disable_floor_clip",
    ]
    if max_frames_override is not None:
        command.extend(["--max_frames", str(int(max_frames_override))])
    if manifest.observations.instance_mask_dir is not None:
        command.extend(["--gaussian_mask_dir", str(manifest.observations.instance_mask_dir)])
    if manifest.observations.camera_manifest is not None:
        command.extend(["--gaussian_views_manifest", str(manifest.observations.camera_manifest)])
    if render.foreground_threshold is not None:
        command.extend(["--foreground_threshold", str(render.foreground_threshold)])
    if render.opacity_threshold is not None:
        command.extend(["--opacity_threshold", str(render.opacity_threshold)])
    if initial_state_json is not None:
        command.extend(["--initial_state_json", str(initial_state_json)])
    else:
        command.extend(
            [
                "--prefit_initial_state",
                "--prefit_position_init", _sequence_arg(initialization.get("position_guess"), default="0,0,0"),
                "--prefit_quaternion_init", _sequence_arg(
                    initialization.get("quaternion_guess_wxyz"), default="1,0,0,0"
                ),
                "--prefit_velocity_frames", str(int(initialization.get("velocity_frames", 3))),
            ]
        )
    if evaluation is not None:
        command.extend(["--evaluation_trajectory", str(evaluation)])
    else:
        command.extend(["--evaluation_trajectory", str(compatibility_root / "no_evaluation_trajectory.json")])
    if actions_path is not None:
        command.extend(["--actions_json", str(actions_path)])
    if bool(training.get("freeze_gravity", True)):
        command.append("--freeze_gravity")

    compatibility_manifest = {
        "schema_version": 1,
        "source_scene_manifest": str(manifest.path),
        "fps": manifest.observations.fps or 30.0,
        "stage1_gaussian_body": {"coordinate_frame": str(collision.get("coordinate_frame", "object_local"))},
        "adapter": {
            "dynamic_body_id": dynamic.id,
            "environment_body_id": environment.id,
            "contact_model": pair.model,
            "uses_object_shape_presets": False,
        },
    }
    return command, compatibility_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable, dest="python_executable")
    parser.add_argument("--fit_iters", default=None, type=int, help="Smoke/ablation override for training.fit_iterations.")
    parser.add_argument("--max_frames", default=None, type=int, help="Limit observation frames for smoke tests.")
    parser.add_argument("--device", default=None, choices=("cpu", "cuda"))
    parser.add_argument("--image_loss", default=None, choices=("l1", "mse", "l1_ssim", "l1_loftr"))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    manifest = load_scene_manifest(args.manifest)
    output_dir = args.output_dir.resolve()
    compatibility_root = output_dir / "manifest_adapter_input"
    command, compatibility_manifest = compile_manifest_command(
        manifest,
        output_dir=output_dir,
        python_executable=args.python_executable,
        compatibility_root=compatibility_root,
        fit_iters_override=args.fit_iters,
        max_frames_override=args.max_frames,
        device_override=args.device,
        image_loss_override=args.image_loss,
    )
    report = {
        "manifest": manifest_summary(manifest),
        "compatibility_manifest": compatibility_manifest,
        "command": command,
        "command_shell_display": shlex.join(command),
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.dry_run:
        return
    compatibility_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (compatibility_root / "episode_manifest.json").write_text(
        json.dumps(compatibility_manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "compiled_manifest_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
