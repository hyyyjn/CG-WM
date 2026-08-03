"""Generate the complete MuJoCo can/floor dataset used by CG-WM stages 1 and 2.

This is an orchestration entry point: it creates a calibrated static multi-view
scene with instance masks, then creates randomized can/floor contact rollouts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--train_views", type=int, default=48)
    parser.add_argument("--test_views", type=int, default=12)
    parser.add_argument("--train_episodes", type=int, default=8)
    parser.add_argument("--test_episodes", type=int, default=2)
    parser.add_argument("--frames_per_episode", type=int, default=90)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mujoco_gl", choices=("auto", "egl", "osmesa", "glfw"), default="auto")
    parser.add_argument("--skip_render", action="store_true", help="Physics/layout smoke test with placeholder frames.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command: list[str], cwd: Path):
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_gif(rgb_dir: Path, output_path: Path, fps: int):
    paths = sorted(rgb_dir.glob("*.png"))
    if not paths:
        return
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB").copy())
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, round(1000 / fps)),
        loop=0,
    )


def main():
    args = parse_args()
    # Must be set before a child imports mujoco; otherwise its renderer may
    # initialize GLFW even though the Stage 1 command requested EGL.
    os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl") if args.mujoco_gl == "auto" else args.mujoco_gl)
    repo = Path(__file__).resolve().parents[1]
    root = Path(args.output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty dataset: {root}; pass --overwrite")
    root.mkdir(parents=True, exist_ok=True)

    stage1_root = root / "stage1"
    stage1_scene = stage1_root / "can_floor"
    stage1_command = [
        sys.executable, "-m", "stage1.generate_mujoco_synthetic_dataset",
        "--output_root", str(stage1_root), "--scene_name", "can_floor",
        "--object_type", "cola_can", "--alpha_subject", "scene",
        "--train_views", str(args.train_views), "--test_views", str(args.test_views),
        "--width", str(args.width), "--height", str(args.height),
        "--camera_radius", "0.75", "--elevation_deg", "28",
        "--mujoco_gl", args.mujoco_gl, "--seed", str(args.seed),
    ]
    run(stage1_command, repo)

    # Stage 1 training will create this PLY. Keeping the intended path in the
    # asset manifest makes the Stage 2 dataset immediately reusable afterward.
    expected_ply = root / "stage1_model" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    asset_path = root / "can_asset.json"
    write_json(asset_path, {
        "object_name": "cola_can",
        "object_type": "cola_can",
        "visual_model": "cola_can",
        "physics_shape": "cylinder",
        "mesh_path": str(stage1_scene / "scene.xml"),
        "stage1_dataset_path": str(stage1_scene),
        "stage1_points_ply": str(expected_ply),
        "stage1_gaussian_body": {"coordinate_frame": "object_local", "object_id": 1},
        "normalization": {"bbox_min": [-0.07, -0.07, -0.10], "bbox_max": [0.07, 0.07, 0.10], "scale": 1.0},
        "physics_prior": {"mass_kg": 0.35, "friction": 0.55},
    })

    stage2_root = root / "contactwm"
    run([
        sys.executable, "tools/create_contactwm_stage2_layout.py",
        "--dataset_root", str(stage2_root), "--object_asset", str(asset_path),
        "--scenario", "fall_and_rebound", "--train_episodes", str(args.train_episodes),
        "--test_episodes", str(args.test_episodes), "--frames_per_episode", str(args.frames_per_episode),
        "--fps", str(args.fps), "--image_width", str(args.width), "--image_height", str(args.height),
        "--overwrite",
    ], repo)
    rollout_command = [
        sys.executable, "tools/generate_mujoco_fall_dataset.py",
        "--dataset_root", str(stage2_root), "--object_name", "cola_can", "--split", "all",
        "--fps", str(args.fps), "--seed", str(args.seed),
        "--camera_distance", "1.25", "--camera_height", "0.65", "--camera_target_z", "0.65",
        "--drop_height_train", "1.1", "--drop_height_test", "1.35",
    ]
    if args.skip_render:
        rollout_command.append("--skip_render")
    run(rollout_command, repo)

    episode_dirs = sorted((stage2_root / "stage2" / "fall_and_rebound").glob("*/*/episode_*"))
    for episode_dir in episode_dirs:
        make_gif(episode_dir / "rgb", episode_dir / "rollout.gif", args.fps)

    write_json(root / "dataset_summary.json", {
        "stage1_dataset": str(stage1_scene),
        "stage1_instance_labels": {"0": "background", "1": "cola_can", "2": "floor"},
        "stage1_expected_trained_ply": str(expected_ply),
        "stage2_dataset": str(stage2_root / "stage2" / "fall_and_rebound"),
        "stage2_episodes": len(episode_dirs),
        "fps": args.fps,
        "seed": args.seed,
    })
    print(f"[DONE] complete can/floor dataset: {root}")


if __name__ == "__main__":
    main()
