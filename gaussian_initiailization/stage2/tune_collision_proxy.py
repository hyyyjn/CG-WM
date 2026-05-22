"""Sweep spherical-Gaussian collision proxy settings on analytic cube/floor SDF.

The cube smoke test already compares Gaussian-union minimum SDF against an
analytic box SDF. This script runs that evaluator over a grid of proxy
resolutions and radius scales so Stage 2 tuning has a repeatable artifact rather
than a one-off manual run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import copy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_initiailization.stage2._smoke_test_cube_floor_collision import (  # noqa: E402
    evaluate_collision_over_trajectory,
    load_trajectory,
    synthesize_falling_cube_trajectory,
)

OUT_DIR = REPO_ROOT / "gaussian_initiailization" / "stage2" / "_outputs" / "collision_proxy_tuning"


def parse_csv_numbers(value: str, cast):
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    if not out:
        raise ValueError(f"Expected at least one comma-separated value, got {value!r}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Stage 2 spherical-Gaussian collision proxy settings.")
    parser.add_argument("--trajectory_json", default=None, type=Path)
    parser.add_argument("--output_dir", default=OUT_DIR, type=Path)
    parser.add_argument("--frames", default=120, type=int)
    parser.add_argument("--fps", default=60.0, type=float)
    parser.add_argument("--half_extent", default=0.10, type=float)
    parser.add_argument("--drop_height", default=0.65, type=float)
    parser.add_argument("--gravity", default=-9.81, type=float)
    parser.add_argument("--tilt_deg", default="18 -11 23", type=str)
    parser.add_argument("--proxy_resolutions", default="3,4,5,6,7", type=str)
    parser.add_argument("--radius_scales", default="0.70,0.85,1.00,1.15,1.30", type=str)
    parser.add_argument("--query_resolution", default=17, type=int)
    parser.add_argument("--query_scale", default=1.45, type=float)
    parser.add_argument("--smooth_min_temperature", default=1.5e-2, type=float)
    parser.add_argument("--contact_softness", default=1.5e-3, type=float)
    parser.add_argument("--inside_penalty", default=0.03, type=float)
    parser.add_argument("--inside_sharpness", default=80.0, type=float)
    parser.add_argument("--num_contact_patches", default=4, type=int)
    return parser.parse_args()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "proxy_resolution",
        "radius_scale",
        "num_gaussian_primitives",
        "mean_min_sdf_abs_error",
        "max_min_sdf_abs_error",
        "first_contact_frame",
        "max_contact_weight",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main() -> None:
    args = parse_args()
    proxy_resolutions = parse_csv_numbers(args.proxy_resolutions, int)
    radius_scales = parse_csv_numbers(args.radius_scales, float)
    states = load_trajectory(args.trajectory_json) if args.trajectory_json else synthesize_falling_cube_trajectory(args)

    results = []
    for proxy_resolution in proxy_resolutions:
        for radius_scale in radius_scales:
            eval_args = copy(args)
            eval_args.proxy_resolution = int(proxy_resolution)
            eval_args.radius_scale = float(radius_scale)
            _, summary = evaluate_collision_over_trajectory(states, eval_args)
            results.append(summary)

    results.sort(key=lambda item: (item["mean_min_sdf_abs_error"], item["max_min_sdf_abs_error"]))
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank

    payload = {
        "metric": "min signed-distance absolute error vs analytic box SDF",
        "num_candidates": len(results),
        "best": results[0] if results else None,
        "results": results,
    }

    output_dir = args.output_dir.resolve()
    write_json(output_dir / "collision_proxy_tuning_summary.json", payload)
    write_csv(output_dir / "collision_proxy_tuning_results.csv", results)
    print(json.dumps({**payload, "output_dir": str(output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
