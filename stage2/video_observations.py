"""Typed Stage-II video observations and optional evaluation trajectories.

Training observations deliberately contain no object pose.  Ground-truth poses
live in ``EvaluationTrajectory`` so later image-only fitting cannot accidentally
consume evaluation labels as supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import re

import torch


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _frame_index(path: Path, fallback: int) -> int:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else int(fallback)


def _image_sequence(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        (path.resolve() for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: (_frame_index(path, 0), path.name),
    )


@dataclass(frozen=True)
class VideoObservations:
    rgb_paths: tuple[Path, ...]
    mask_paths: tuple[Path | None, ...]
    frame_indices: tuple[int, ...]
    times: torch.Tensor
    camera: dict[str, Any]
    fps: float
    source: dict[str, Any]

    @property
    def num_frames(self) -> int:
        return len(self.rgb_paths)


@dataclass(frozen=True)
class EvaluationTrajectory:
    positions: torch.Tensor
    quaternions_wxyz: torch.Tensor
    times: torch.Tensor
    frame_indices: tuple[int, ...]
    states: tuple[dict[str, Any], ...]
    path: Path

    @property
    def num_frames(self) -> int:
        return len(self.states)


def load_video_observations(
    episode_root: Path,
    *,
    max_frames: int = 0,
    rgb_dir: Path | None = None,
    mask_dir: Path | None = None,
    views_manifest: Path | None = None,
    camera_defaults: dict[str, Any] | None = None,
) -> VideoObservations:
    """Load image-space training inputs without reading ``state/trajectory.json``."""
    episode_root = episode_root.resolve()
    manifest_path = episode_root / "episode_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    resolved_rgb_dir = (rgb_dir or (episode_root / "rgb")).resolve()
    resolved_mask_dir = (mask_dir or (episode_root / "masks")).resolve()
    rgb_paths = _image_sequence(resolved_rgb_dir)
    if max_frames > 0:
        rgb_paths = rgb_paths[: int(max_frames)]
    if not rgb_paths:
        raise FileNotFoundError(f"No RGB observation frames found in {resolved_rgb_dir}")

    masks_by_index = {
        _frame_index(path, index): path
        for index, path in enumerate(_image_sequence(resolved_mask_dir))
    }
    frame_indices = tuple(_frame_index(path, index) for index, path in enumerate(rgb_paths))
    mask_paths = tuple(masks_by_index.get(frame_index) for frame_index in frame_indices)

    fps = float(manifest.get("fps", 0.0))
    if fps <= 0.0:
        fps = 30.0
    times = torch.tensor([frame_index / fps for frame_index in frame_indices], dtype=torch.float32)

    camera: dict[str, Any]
    if views_manifest is not None:
        resolved_views = views_manifest.resolve()
        if not resolved_views.exists():
            raise FileNotFoundError(resolved_views)
        camera = {
            "source": "views_manifest",
            "path": str(resolved_views),
            "payload": _read_json(resolved_views),
        }
    elif isinstance(manifest.get("camera"), dict):
        camera = {"source": "episode_manifest", **manifest["camera"]}
    else:
        camera = {"source": "cli_defaults", **(camera_defaults or {})}

    return VideoObservations(
        rgb_paths=tuple(rgb_paths),
        mask_paths=mask_paths,
        frame_indices=frame_indices,
        times=times,
        camera=camera,
        fps=fps,
        source={
            "episode_root": str(episode_root),
            "episode_manifest": str(manifest_path) if manifest_path.exists() else None,
            "rgb_dir": str(resolved_rgb_dir),
            "mask_dir": str(resolved_mask_dir) if resolved_mask_dir.exists() else None,
            "trajectory_read": False,
        },
    )


def load_optional_evaluation_trajectory(
    path: Path | None,
    *,
    max_frames: int = 0,
) -> EvaluationTrajectory | None:
    """Load pose labels for metrics/legacy adaptation, or return ``None`` when absent."""
    if path is None:
        return None
    path = path.resolve()
    if not path.exists():
        return None
    payload = _read_json(path)
    states = payload.get("states")
    if not isinstance(states, list):
        raise ValueError(f"{path} must contain a 'states' list")
    if max_frames > 0:
        states = states[: int(max_frames)]
    if len(states) < 3:
        raise ValueError("Need at least 3 evaluation trajectory states.")
    positions = torch.tensor([state["position"] for state in states], dtype=torch.float32)
    quaternions = torch.tensor(
        [state.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]) for state in states],
        dtype=torch.float32,
    )
    quaternions = torch.nn.functional.normalize(quaternions, dim=-1)
    times = torch.tensor([float(state.get("time", index)) for index, state in enumerate(states)])
    frame_indices = tuple(int(state.get("frame_index", index)) for index, state in enumerate(states))
    return EvaluationTrajectory(
        positions=positions,
        quaternions_wxyz=quaternions,
        times=times,
        frame_indices=frame_indices,
        states=tuple(states),
        path=path,
    )


def observation_summary(observations: VideoObservations) -> dict[str, Any]:
    return {
        "num_frames": observations.num_frames,
        "frame_indices": list(observations.frame_indices),
        "fps": observations.fps,
        "rgb_paths": [str(path) for path in observations.rgb_paths],
        "mask_paths": [None if path is None else str(path) for path in observations.mask_paths],
        "camera": observations.camera,
        "source": observations.source,
    }
