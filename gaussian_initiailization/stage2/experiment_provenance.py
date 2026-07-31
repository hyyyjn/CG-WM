"""Reproducibility bundle helpers for Stage 2 experiments."""
from __future__ import annotations

import hashlib
import json
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Namespace):
        return {key: _jsonable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str, binary: bool = False):
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=not binary,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def collect_git_state(repo_root: Path) -> dict:
    diff = _git(repo_root, "diff", "--binary", "HEAD", binary=True)
    status = _git(repo_root, "status", "--short") or ""
    return {
        "commit": (_git(repo_root, "rev-parse", "HEAD") or "").strip() or None,
        "branch": (_git(repo_root, "branch", "--show-current") or "").strip() or None,
        "dirty": bool(status.strip()),
        "status_short": status.splitlines(),
        "tracked_diff_sha256": None if diff is None else hashlib.sha256(diff).hexdigest(),
    }


def write_experiment_bundle(
    *,
    output_dir: Path,
    repo_root: Path,
    args: Namespace,
    episode_manifest_path: Path,
    input_paths: dict[str, Path | None],
    result_paths: dict[str, Path],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(episode_manifest_path.read_text(encoding="utf-8-sig"))
    config = _jsonable(args)
    inputs = {}
    for name, path in input_paths.items():
        if path is None:
            inputs[name] = None
            continue
        resolved = path.resolve()
        inputs[name] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
        }
    results = {}
    for name, path in result_paths.items():
        resolved = path.resolve()
        results[name] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
        }
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "input_episode_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    bundle = {
        "schema_version": 1,
        "git": collect_git_state(repo_root),
        "config": config,
        "episode_manifest": manifest,
        "inputs": inputs,
        "results": results,
    }
    path = output_dir / "experiment_bundle.json"
    path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return path
