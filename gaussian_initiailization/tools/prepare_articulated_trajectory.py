#!/usr/bin/env python
"""Convert URDF/MJCF joint trajectories into per-link Stage-2 pose trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_initiailization.stage2.articulated_model_loader import (  # noqa: E402
    load_articulated_model,
    load_joint_trajectory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--joint_trajectory", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--dt", default=None, type=float)
    args = parser.parse_args()

    model = load_articulated_model(args.model.resolve())
    trajectory = load_joint_trajectory(
        model, args.joint_trajectory.resolve(), dt=args.dt
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    link_entries = []
    for link_index, link in enumerate(model.links):
        states = []
        for frame in range(trajectory["positions"].shape[0]):
            states.append({
                "frame_index": frame,
                "time": frame * trajectory["dt"],
                "position": trajectory["positions"][frame, link_index].tolist(),
                "quaternion_wxyz": trajectory["quaternions"][frame, link_index].tolist(),
                "linear_velocity": trajectory["linear_velocities"][frame, link_index].tolist(),
                "angular_velocity": trajectory["angular_velocities"][frame, link_index].tolist(),
            })
        path = output_dir / f"{link_index:03d}_{link.name}.json"
        path.write_text(json.dumps({"states": states}, indent=2), encoding="utf-8")
        link_entries.append({
            "index": link_index,
            "name": link.name,
            "joint_name": model.joint_names[link_index],
            "joint_type": link.joint_type,
            "trajectory_json": str(path),
        })
    manifest = {
        "source_model": str(model.source_path),
        "source_format": model.source_format,
        "source_joint_trajectory": str(args.joint_trajectory.resolve()),
        "dt": trajectory["dt"],
        "links": link_entries,
    }
    manifest_path = output_dir / "articulated_trajectory_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
