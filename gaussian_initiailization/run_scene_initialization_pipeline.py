import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the ContactGaussian-WM-style scene initialization pipeline end-to-end."
    )
    parser.add_argument("--source_path", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--masks_dir", default="", type=str)
    parser.add_argument("--images", default="images", type=str)
    parser.add_argument("--sam_features", default="sam_features_sam2", type=str)
    parser.add_argument("--visual_hull_ply", default="", type=str)
    parser.add_argument("--gaussian_env", default="gaussian_splatting", type=str)
    parser.add_argument("--sam2_env", default="sam2cpu", type=str)
    parser.add_argument("--sam_checkpoint", default="", type=str)
    parser.add_argument("--sam_config", default="configs/sam2.1/sam2.1_hiera_t.yaml", type=str)
    parser.add_argument("--sam_output_channels", default=9, type=int)
    parser.add_argument("--sam_feature_source", default="high_res0", type=str)
    parser.add_argument("--grid_resolution", default=128, type=int)
    parser.add_argument("--vh_max_points", default=200000, type=int)
    parser.add_argument("--iterations", default=10000, type=int)
    parser.add_argument("--resolution", default=8, type=int)
    parser.add_argument("--sam_feature_weight", default=0.1, type=float)
    parser.add_argument("--geometry_feature_dim", default=9, type=int)
    parser.add_argument("--skip_mask_extraction", action="store_true")
    parser.add_argument("--skip_masked_colmap", action="store_true")
    parser.add_argument("--skip_visual_hull", action="store_true")
    parser.add_argument("--skip_sam2", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--mask_method", default="auto", choices=["auto", "alpha", "bg_subtract"])
    parser.add_argument("--mask_background_mode", default="white", choices=["white", "black", "keep"])
    parser.add_argument("--mask_dilate", default=5, type=int)
    parser.add_argument("--no_gpu_colmap", action="store_true")
    parser.add_argument("--joint_optimization", action="store_true")
    parser.add_argument("--alternating_optimization", action="store_true")
    parser.add_argument("--geometry_iters", default=1, type=int)
    parser.add_argument("--appearance_iters", default=1, type=int)
    parser.add_argument("--disable_viewer", action="store_true", default=True)
    parser.add_argument("--eval", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true", default=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def repo_script(script_name: str) -> str:
    return str(Path(__file__).resolve().parent / script_name)


def run_command(command, dry_run=False):
    pretty = " ".join(shlex.quote(part) for part in command)
    print(f"[RUN] {pretty}")
    if dry_run:
        return
    subprocess.run(command, check=True)


def conda_python(env_name: str):
    return ["conda", "run", "-n", env_name, "python"]


def ensure_masks_dir(args):
    if args.masks_dir:
        return Path(args.masks_dir).expanduser().resolve()
    return Path(args.source_path).expanduser().resolve() / "auto_masks"


def ensure_visual_hull_path(args):
    if args.visual_hull_ply:
        return Path(args.visual_hull_ply).expanduser().resolve()
    return Path(args.source_path).expanduser().resolve() / "visual_hull" / "visual_hull.ply"


def stage_extract_masks(args, masks_dir: Path):
    command = conda_python(args.gaussian_env) + [
        repo_script("extract_object_masks.py"),
        "--source_path", str(Path(args.source_path).expanduser().resolve()),
        "--output_masks_dir", str(masks_dir),
        "--method", args.mask_method,
        "--overwrite",
    ]
    run_command(command, dry_run=args.dry_run)


def stage_masked_colmap(args, masks_dir: Path):
    command = conda_python(args.gaussian_env) + [
        repo_script("estimate_masked_colmap.py"),
        "--source_path", str(Path(args.source_path).expanduser().resolve()),
        "--masks_dir", str(masks_dir),
        "--background_mode", args.mask_background_mode,
        "--mask_dilate", str(args.mask_dilate),
        "--overwrite",
    ]
    if args.images:
        command.extend(["--images", args.images])
    if args.no_gpu_colmap:
        command.append("--no_gpu")
    run_command(command, dry_run=args.dry_run)


def stage_visual_hull(args, masks_dir: Path, visual_hull_ply: Path):
    command = conda_python(args.gaussian_env) + [
        repo_script("build_visual_hull.py"),
        "--source_path", str(Path(args.source_path).expanduser().resolve()),
        "--masks_dir", str(masks_dir),
        "--grid_resolution", str(args.grid_resolution),
        "--max_points", str(args.vh_max_points),
        "--output_ply", str(visual_hull_ply),
    ]
    if args.images:
        command.extend(["--images", args.images])
    run_command(command, dry_run=args.dry_run)


def stage_sam2(args):
    command = conda_python(args.sam2_env) + [
        repo_script("extract_sam2_features.py"),
        "--source_path", str(Path(args.source_path).expanduser().resolve()),
        "--output_dir", args.sam_features,
        "--output_channels", str(args.sam_output_channels),
        "--feature_source", args.sam_feature_source,
    ]
    if args.sam_checkpoint:
        command.extend(["--checkpoint", args.sam_checkpoint])
    if args.sam_config:
        command.extend(["--config", args.sam_config])
    run_command(command, dry_run=args.dry_run)


def stage_train(args, visual_hull_ply: Path):
    command = conda_python(args.gaussian_env) + [
        repo_script("train.py"),
        "--source_path", str(Path(args.source_path).expanduser().resolve()),
        "--model_path", str(Path(args.model_path).expanduser().resolve()),
        "--iterations", str(args.iterations),
        "--resolution", str(args.resolution),
        "--sam_features", args.sam_features,
        "--sam_feature_weight", str(args.sam_feature_weight),
        "--geometry_feature_dim", str(args.geometry_feature_dim),
        "--init_mode", "visual_hull",
        "--init_ply_path", str(visual_hull_ply),
    ]

    if args.eval:
        command.append("--eval")
    if args.disable_viewer:
        command.append("--disable_viewer")
    if args.quiet:
        command.append("--quiet")
    if args.joint_optimization:
        command.append("--joint_optimization")
    if args.alternating_optimization:
        command.extend(
            [
                "--alternating_optimization",
                "--geometry_iters", str(args.geometry_iters),
                "--appearance_iters", str(args.appearance_iters),
            ]
        )

    run_command(command, dry_run=args.dry_run)


def main():
    args = parse_args()

    if args.joint_optimization and args.alternating_optimization:
        raise ValueError("Choose either --joint_optimization or --alternating_optimization, not both.")

    masks_dir = ensure_masks_dir(args)
    visual_hull_ply = ensure_visual_hull_path(args)

    print("[INFO] Pipeline configuration")
    print(f"[INFO] source_path: {Path(args.source_path).expanduser().resolve()}")
    print(f"[INFO] model_path: {Path(args.model_path).expanduser().resolve()}")
    print(f"[INFO] masks_dir: {masks_dir}")
    print(f"[INFO] visual_hull_ply: {visual_hull_ply}")
    print(f"[INFO] sam_features: {args.sam_features}")

    if not args.skip_mask_extraction and not args.masks_dir:
        stage_extract_masks(args, masks_dir)

    if not args.skip_masked_colmap:
        stage_masked_colmap(args, masks_dir)

    if not args.skip_visual_hull:
        stage_visual_hull(args, masks_dir, visual_hull_ply)

    if not args.skip_sam2:
        stage_sam2(args)

    if not args.skip_train:
        stage_train(args, visual_hull_ply)

    print("[DONE] Scene initialization pipeline finished.")


if __name__ == "__main__":
    main()
