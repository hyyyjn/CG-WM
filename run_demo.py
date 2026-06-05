"""ContactGaussian-WM Demo Pipeline

conda env 안에서 실행:
    python run_demo.py
    python run_demo.py --object_type sphere --stage1_iters 3000
    python run_demo.py --skip_stage1          # Stage 1 PLY 재사용
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

REPO = Path(__file__).parent
SCRIPTS = REPO / "gaussian_initiailization"


def run(cmd: list, **kwargs):
    pretty = " ".join(str(c) for c in cmd)
    print(f"\n[RUN] {pretty}", flush=True)
    result = subprocess.run([str(c) for c in cmd], **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)


def py(script: Path, args: list):
    run([sys.executable, script] + args)


# ──────────────────────────────────────────────
# geometry presets
# ──────────────────────────────────────────────

GEOM = {
    "box":      {"bmin": [-0.08, -0.08, -0.08], "bmax": [0.08, 0.08, 0.08], "cz": 0.08},
    "sphere":   {"bmin": [-0.09, -0.09, -0.09], "bmax": [0.09, 0.09, 0.09], "cz": 0.09},
    "cylinder": {"bmin": [-0.07, -0.07, -0.10], "bmax": [0.07, 0.07, 0.10], "cz": 0.10},
}

FALL_PARAMS = {
    "box": dict(
        drop_train="0.75", drop_test="0.95",
        xy_train="0.12",   xy_test="0.25",
        plan_train="0.8",  plan_test="1.1",
        spin_train="8.0",  spin_test="12.0",
        tilt_train="18.0", tilt_test="38.0",
    ),
    "sphere": dict(
        drop_train="0.80", drop_test="1.20",
        xy_train="0.02",   xy_test="0.03",
        plan_train="0.05", plan_test="0.05",
        spin_train="0.1",  spin_test="0.1",
        tilt_train="1.0",  tilt_test="1.0",
    ),
    "cylinder": dict(
        drop_train="0.75", drop_test="0.95",
        xy_train="0.10",   xy_test="0.20",
        plan_train="0.5",  plan_test="0.8",
        spin_train="5.0",  spin_test="8.0",
        tilt_train="10.0", tilt_test="25.0",
    ),
}


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="ContactGaussian-WM demo pipeline")
    p.add_argument("--output_root",          default=str(REPO / "demo_data"))
    p.add_argument("--model_root",           default=str(REPO / "demo_output"))
    p.add_argument("--scene_name",           default="sphere_demo")
    p.add_argument("--object_type",          default="sphere", choices=["box", "sphere", "cylinder"])
    p.add_argument("--stage1_iters",         type=int, default=10000)
    p.add_argument("--stage2_fit_iters",     type=int, default=500)
    p.add_argument("--foreground_threshold", type=float, default=0.50)
    p.add_argument("--sphere_solref",        default="0.02 0.2")
    p.add_argument("--mujoco_gl",            default="egl", choices=["auto", "egl", "osmesa", "glfw"])
    p.add_argument("--skip_stage1",          action="store_true")
    args = p.parse_args()

    output_root  = Path(args.output_root)
    model_root   = Path(args.model_root)
    scene        = args.scene_name
    obj          = args.object_type
    geom         = GEOM.get(obj, GEOM["box"])
    fall         = FALL_PARAMS.get(obj, FALL_PARAMS["box"])
    cz           = geom["cz"]

    dataset_dir   = output_root / scene
    model_dir     = model_root / f"{scene}_stage1"
    asset_json    = output_root / f"{scene}_asset.json"
    fit_output    = model_root / f"{scene}_stage2_fit"
    episode_root  = output_root / "stage2" / "fall_and_rebound" / "test" / scene / "episode_000"

    radius_scale  = "0.8" if obj == "sphere" else "0.1"
    tang_damp     = "5.0" if obj == "sphere" else "80.0"

    print("=== ContactGaussian-WM Demo ===")
    print(f"Dataset : {dataset_dir}")
    print(f"Model   : {model_dir}")
    print(f"Stage2  : {fit_output}")
    print(f"Python  : {sys.executable}")

    # ── STEP 1  multi-view dataset ──────────────
    print("\n[STEP 1] Generating MuJoCo synthetic dataset...")
    py(SCRIPTS / "generate_mujoco_synthetic_dataset.py", [
        "--output_root", str(output_root),
        "--scene_name",  scene,
        "--object_type", obj,
        "--train_views", "32",
        "--test_views",  "8",
        "--width",       "512",
        "--height",      "512",
        "--mujoco_gl",   args.mujoco_gl,
    ])

    # ── STEP 2  Stage 1 training ────────────────
    if not args.skip_stage1:
        print(f"\n[STEP 2] Stage 1 Gaussian training ({args.stage1_iters} iters)...")
        py(SCRIPTS / "train.py", [
            "--source_path",            str(dataset_dir),
            "--model_path",             str(model_dir),
            "--iterations",             str(args.stage1_iters),
            "--save_iterations",        str(args.stage1_iters),
            "--test_iterations",        str(args.stage1_iters),
            "--masks_dir",              str(dataset_dir / "masks"),
            "--object_mask_weight",     "1.0",
            "--object_mask_bce_weight", "2.0",
            "--sam_feature_weight",     "0.0",
            "--disable_viewer",
            "--quiet",
            "--eval",
        ])
    else:
        print("\n[STEP 2] Skipping Stage 1 (--skip_stage1).")

    # ── STEP 3  asset JSON ──────────────────────
    print("\n[STEP 3] Writing object asset JSON...")

    # PLY 경로: 요청한 iteration 없으면 최신 checkpoint 자동 탐색
    ply_path = model_dir / "point_cloud" / f"iteration_{args.stage1_iters}" / "point_cloud.ply"
    if not ply_path.exists():
        candidates = sorted(
            (model_dir / "point_cloud").glob("iteration_*/point_cloud.ply"),
            key=lambda f: int(f.parent.name.replace("iteration_", "")),
        )
        if candidates:
            ply_path = candidates[-1]
    if not ply_path.exists():
        print(f"[ERROR] Stage 1 PLY not found: {ply_path}", file=sys.stderr)
        sys.exit(1)

    asset = {
        "object_name": scene,
        "mesh_path": "",
        "stage1_dataset_path": dataset_dir.as_posix(),
        "stage1_points_ply": ply_path.as_posix(),
        "physics_shape": obj,
        "normalization": {
            "bbox_min": geom["bmin"],
            "bbox_max": geom["bmax"],
            "scale": 1.0,
        },
        "stage1_gaussian_body": {
            "coordinate_frame": "world",
            "world_pose": {
                "translation": [0.0, 0.0, cz],
                "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    asset_json.write_text(json.dumps(asset, indent=2))
    print(f"  Wrote: {asset_json}")

    print("\n[STEP 3b] Auditing Stage 1 PLY (warn-only)...")
    try:
        py(SCRIPTS / "tools" / "audit_stage1_ply.py", [
            "--stage1_ply",           str(ply_path),
            "--object_asset",         str(asset_json),
            "--foreground_threshold", str(args.foreground_threshold),
            "--max_extent_ratio",     "3.0",
        ])
    except SystemExit:
        print("[WARN] audit_stage1_ply failed (expected for low-iter runs). Continuing.")

    # ── STEP 4  Stage 2 layout ──────────────────
    print("\n[STEP 4] Creating Stage 2 dataset layout...")
    py(SCRIPTS / "tools" / "create_contactwm_stage2_layout.py", [
        "--dataset_root",      str(output_root),
        "--object_asset",      str(asset_json),
        "--scenario",          "fall_and_rebound",
        "--train_episodes",    "2",
        "--test_episodes",     "4",
        "--frames_per_episode","120",
        "--fps",               "30",
        "--overwrite",
    ])

    # ── STEP 5  fall trajectories ───────────────
    print("\n[STEP 5] Generating MuJoCo fall trajectories...")
    py(SCRIPTS / "tools" / "generate_mujoco_fall_dataset.py", [
        "--dataset_root",       str(output_root),
        "--object_name",        scene,
        "--split",              "all",
        "--camera_distance",    "1.35",
        "--camera_height",      "0.75",
        "--ground_size",        "2.0",
        "--drop_height_train",  fall["drop_train"],
        "--drop_height_test",   fall["drop_test"],
        "--xy_range_train",     fall["xy_train"],
        "--xy_range_test",      fall["xy_test"],
        "--planar_speed_train", fall["plan_train"],
        "--planar_speed_test",  fall["plan_test"],
        "--spin_speed_train",   fall["spin_train"],
        "--spin_speed_test",    fall["spin_test"],
        "--max_tilt_deg_train", fall["tilt_train"],
        "--max_tilt_deg_test",  fall["tilt_test"],
        "--sphere_solref",      args.sphere_solref,
    ])

    # ── STEP 6  Stage 2 fitting ─────────────────
    print("\n[STEP 6] Fitting Stage 2 contact dynamics...")
    py(SCRIPTS / "tools" / "run_stage2_mujoco_stage1_fit.py", [
        "--episode_root",              str(episode_root),
        "--stage1_model_path",         str(model_dir),
        "--output_dir",                str(fit_output),
        "--dynamics",                  "restitution",
        "--fit_iters",                 str(args.stage2_fit_iters),
        "--lr",                        "0.015",
        "--device",                    "cuda",
        "--gif_fps",                   "20",
        "--max_primitives",            "2000",
        "--foreground_threshold",      str(args.foreground_threshold),
        "--opacity_threshold",         "0.02",
        "--radius_scale",              radius_scale,
        "--initial_velocity_source",   "trajectory",
        "--freeze_initial_velocity",
        "--floor_tangential_damping",  tang_damp,
    ])

    # ── STEP 7  GT GIF ──────────────────────────
    print("\n[STEP 7] Exporting GT episode GIF...")
    gt_gif = fit_output / "gt_episode.gif"
    py(SCRIPTS / "tools" / "export_episode_gif.py", [
        "--rgb_dir",      str(episode_root / "rgb"),
        "--output_gif",   str(gt_gif),
        "--fps",          "20",
        "--resize_width", "480",
    ])

    # ── STEP 8  comparison GIF ──────────────────
    print("\n[STEP 8] Rendering GT vs 3DGS comparison GIF...")
    comparison_gif = fit_output / "comparison_gt_vs_3dgs.gif"
    py(SCRIPTS / "tools" / "render_trajectory_comparison.py", [
        "--episode_root",          str(episode_root),
        "--predicted_trajectory",  str(fit_output / "predicted_trajectory.json"),
        "--stage1_model_path",     str(model_dir),
        "--output_gif",            str(comparison_gif),
        "--stage1_centroid",       f"0,0,{cz}",
        "--cam_distance",          "1.35",
        "--cam_height",            "0.75",
        "--cam_fovy_deg",          "45.0",
        "--image_width",           "640",
        "--image_height",          "480",
        "--panel_width",           "480",
        "--fps",                   "20",
        "--foreground_threshold",  str(args.foreground_threshold),
        "--device",                "cuda",
    ])

    print("\n=== DONE ===")
    print(f"  [GIF 1] GT video     : {gt_gif}")
    print(f"  [GIF 2] Traj compare : {fit_output / 'stage2_fit_follow_view.gif'}")
    print(f"  [GIF 3] 3DGS compare : {comparison_gif}  <-- MAIN")
    print(f"  [JSON]  Fit stats    : {fit_output / 'fit_summary.json'}")


if __name__ == "__main__":
    main()
