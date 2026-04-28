import argparse
import json
from pathlib import Path


SCENARIO_DEFAULTS = {
    "fall_and_rebound": {
        "description": "MuJoCo free-fall / bounce trajectory dataset.",
        "action_type": "initial_pose_and_release",
        "motion_type": "high_dynamic_discrete_contact",
    },
    "push_slide_settle": {
        "description": "MuJoCo push / slide / settle trajectory dataset.",
        "action_type": "end_effector_velocity",
        "motion_type": "quasi_dynamic_continuous_contact",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a ContactGaussian-WM-style Stage 2 dataset layout that reuses Stage 1 object assets."
    )
    parser.add_argument("--dataset_root", required=True, type=str)
    parser.add_argument(
        "--object_asset",
        required=True,
        nargs="+",
        help="One or more object_asset.json files generated from Stage 1 object preparation.",
    )
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=["fall_and_rebound", "push_slide_settle"],
        choices=tuple(SCENARIO_DEFAULTS.keys()),
    )
    parser.add_argument("--train_episodes", default=1, type=int)
    parser.add_argument("--test_episodes", default=4, type=int)
    parser.add_argument("--frames_per_episode", default=120, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--camera_count", default=1, type=int)
    parser.add_argument("--image_width", default=640, type=int)
    parser.add_argument("--image_height", default=480, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_object_asset(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    required = ["object_name", "mesh_path", "stage1_dataset_path", "stage1_points_ply"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    return payload


def ensure_empty_or_create(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} already exists and is not empty. Pass --overwrite to continue.")
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_episode_manifest(object_asset, scenario_name, split_name, episode_index, args):
    return {
        "episode_id": f"{split_name}_{episode_index:03d}",
        "scenario": scenario_name,
        "split": split_name,
        "object_name": object_asset["object_name"],
        "object_asset_path": object_asset["asset_manifest_path"],
        "mesh_path": object_asset["mesh_path"],
        "stage1_dataset_path": object_asset["stage1_dataset_path"],
        "stage1_points_ply": object_asset["stage1_points_ply"],
        "frames_per_episode": int(args.frames_per_episode),
        "fps": int(args.fps),
        "camera_count": int(args.camera_count),
        "image_size": [int(args.image_width), int(args.image_height)],
        "physics_prior": object_asset.get("physics_prior", {}),
        "physics_shape": object_asset.get("physics_shape", "box"),
        "normalization": object_asset.get("normalization", {}),
        "action_type": SCENARIO_DEFAULTS[scenario_name]["action_type"],
        "motion_type": SCENARIO_DEFAULTS[scenario_name]["motion_type"],
        "notes": "Populate rgb/, masks/, state/, and actions/ from your MuJoCo rollout/export pipeline.",
    }


def create_episode_dirs(base_dir: Path, manifest):
    for subdir in ("rgb", "masks", "state", "actions"):
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)
    write_json(base_dir / "episode_manifest.json", manifest)


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    objects_root = dataset_root / "objects"
    stage2_root = dataset_root / "stage2"

    ensure_empty_or_create(dataset_root, args.overwrite)
    objects_root.mkdir(parents=True, exist_ok=True)
    stage2_root.mkdir(parents=True, exist_ok=True)

    object_assets = []
    for asset_path_str in args.object_asset:
        asset_path = Path(asset_path_str).expanduser().resolve()
        asset = load_object_asset(asset_path)
        asset["asset_manifest_path"] = str(asset_path)
        object_assets.append(asset)

        object_dir = objects_root / asset["object_name"]
        object_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            object_dir / "object_manifest.json",
            {
                "object_name": asset["object_name"],
                "physics_shape": asset.get("physics_shape", "box"),
                "mesh_path": asset["mesh_path"],
                "stage1_dataset_path": asset["stage1_dataset_path"],
                "stage1_points_ply": asset["stage1_points_ply"],
                "stage1_summary_path": asset.get("stage1_summary_path", ""),
                "source_asset_manifest_path": asset["asset_manifest_path"],
                "normalization": asset.get("normalization", {}),
                "physics_prior": asset.get("physics_prior", {}),
            },
        )

    top_manifest = {
        "dataset_root": str(dataset_root),
        "paper_reference": "ContactGaussian-WM arXiv:2602.11021",
        "objects": [asset["object_name"] for asset in object_assets],
        "scenarios": args.scenario,
        "train_episodes_per_object": int(args.train_episodes),
        "test_episodes_per_object": int(args.test_episodes),
        "frames_per_episode": int(args.frames_per_episode),
        "fps": int(args.fps),
        "camera_count": int(args.camera_count),
        "image_size": [int(args.image_width), int(args.image_height)],
        "design_note": (
            "Stage 1 owns object-centric canonical assets; Stage 2 reuses them as dynamics rollout objects "
            "for MuJoCo-style push-slide-settle and fall-and-rebound sequences."
        ),
    }
    write_json(dataset_root / "dataset_manifest.json", top_manifest)

    for scenario_name in args.scenario:
        scenario_root = stage2_root / scenario_name
        scenario_root.mkdir(parents=True, exist_ok=True)
        write_json(
            scenario_root / "scenario_manifest.json",
            {
                "scenario": scenario_name,
                **SCENARIO_DEFAULTS[scenario_name],
                "train_episodes_per_object": int(args.train_episodes),
                "test_episodes_per_object": int(args.test_episodes),
                "frames_per_episode": int(args.frames_per_episode),
                "fps": int(args.fps),
                "camera_count": int(args.camera_count),
                "image_size": [int(args.image_width), int(args.image_height)],
            },
        )

        for split_name, episode_count in (("train", args.train_episodes), ("test", args.test_episodes)):
            split_root = scenario_root / split_name
            split_root.mkdir(parents=True, exist_ok=True)
            split_entries = []
            for asset in object_assets:
                object_split_root = split_root / asset["object_name"]
                object_split_root.mkdir(parents=True, exist_ok=True)
                for episode_index in range(int(episode_count)):
                    episode_root = object_split_root / f"episode_{episode_index:03d}"
                    episode_manifest = build_episode_manifest(asset, scenario_name, split_name, episode_index, args)
                    create_episode_dirs(episode_root, episode_manifest)
                    split_entries.append(str(episode_root))
            write_json(split_root / "split_index.json", {"episodes": split_entries})

    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "objects": [asset["object_name"] for asset in object_assets],
                "scenarios": args.scenario,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
