#!/usr/bin/env python
"""Run separate primitive-budget and raw query-point-budget ablations.

Primitive-budget experiments hold the number of moving-body Gaussian
primitives fixed. Point-budget experiments instead choose the primitive cap per
scheme so that ``primitives * directions_per_primitive`` is held fixed. The
post-SDF contact patch budget (lowest-K) is independent and shared by all runs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parent
COMPARE = TOOLS / "compare_query_modes.py"
DIRECTIONS = {"axis6": 6, "fib26": 26, "analytic": 1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode_root", required=True, type=Path)
    p.add_argument("--stage1_ply", required=True, type=Path)
    p.add_argument("--output_root", required=True, type=Path)
    p.add_argument("--variants", nargs="+", default=list(DIRECTIONS), choices=sorted(DIRECTIONS))
    p.add_argument("--suites", nargs="+", default=["primitive", "point"], choices=("primitive", "point"))
    p.add_argument("--primitive_budgets", nargs="+", default=[60, 120, 180], type=int)
    p.add_argument("--point_budgets", nargs="+", default=[180, 360, 720], type=int)
    p.add_argument("--body_lowest_k", default=32, type=int)
    p.add_argument("--max_frames", default=80, type=int)
    p.add_argument("--fit_iters", default=250, type=int)
    p.add_argument("--lr", default=0.02, type=float)
    p.add_argument("--pairwise_body_b_ply", default=None, type=Path)
    p.add_argument("--render", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def run_one(args: argparse.Namespace, suite: str, budget: int, variant: str) -> tuple[Path, int]:
    directions = DIRECTIONS[variant]
    primitive_cap = budget if suite == "primitive" else max(1, budget // directions)
    out = args.output_root / f"{suite}_budget" / f"budget_{budget}" / variant
    cmd = [
        sys.executable, str(COMPARE),
        "--episode_root", str(args.episode_root),
        "--stage1_ply", str(args.stage1_ply),
        "--output_root", str(out),
        "--variants", variant,
        "--body_lowest_k", str(args.body_lowest_k),
        "--max_primitives", str(primitive_cap),
        "--max_frames", str(args.max_frames),
        "--fit_iters", str(args.fit_iters),
        "--lr", str(args.lr),
    ]
    if args.pairwise_body_b_ply is not None:
        cmd += ["--pairwise_body_b_ply", str(args.pairwise_body_b_ply)]
    if not args.render:
        cmd.append("--skip_render")
    print("$ " + " ".join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True)
    return out, primitive_cap


def main() -> None:
    args = parse_args()
    if any(v <= 0 for v in args.primitive_budgets + args.point_budgets):
        raise ValueError("all budgets must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for suite in args.suites:
        budgets = args.primitive_budgets if suite == "primitive" else args.point_budgets
        for budget in budgets:
            for variant in args.variants:
                out, primitive_cap = run_one(args, suite, budget, variant)
                record = {
                    "suite": suite,
                    "requested_budget": budget,
                    "budget_unit": "moving_body_primitives" if suite == "primitive" else "moving_body_raw_query_candidates",
                    "variant": variant,
                    "directions_per_primitive": DIRECTIONS[variant],
                    "requested_primitive_cap": primitive_cap,
                    "output_dir": str(out.resolve()),
                }
                metrics_path = out / "query_mode_comparison_metrics.json"
                if metrics_path.exists():
                    payload = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
                    record["metrics"] = payload["variants"][variant]
                records.append(record)
    payload = {
        "definition": {
            "primitive": "same moving-body primitive cap; raw query count may differ by scheme",
            "point": "same requested moving-body raw query budget; primitive cap=floor(point_budget/directions)",
            "contact_patch_budget": args.body_lowest_k,
            "static_body_budget": "unchanged across schemes and excluded from the moving-body point budget",
        },
        "config": {
            "episode_root": str(args.episode_root.resolve()),
            "stage1_ply": str(args.stage1_ply.resolve()),
            "variants": args.variants,
            "primitive_budgets": args.primitive_budgets,
            "point_budgets": args.point_budgets,
            "max_frames": args.max_frames,
            "fit_iters": args.fit_iters,
            "lr": args.lr,
        },
        "dry_run": args.dry_run,
        "runs": records,
    }
    report = args.output_root / "query_budget_ablation_report.json"
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[report] {report}")


if __name__ == "__main__":
    main()
