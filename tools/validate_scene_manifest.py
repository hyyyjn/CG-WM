from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2.scene_manifest import load_scene_manifest, manifest_summary, validate_scene_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an object-name-independent Stage-II scene manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--skip_path_checks", action="store_true")
    args = parser.parse_args()
    manifest = load_scene_manifest(args.manifest, validate=False)
    errors = validate_scene_manifest(manifest, check_paths=not args.skip_path_checks)
    result = {"valid": not errors, "errors": errors, "summary": manifest_summary(manifest)}
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
