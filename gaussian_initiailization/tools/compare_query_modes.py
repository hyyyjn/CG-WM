#!/usr/bin/env python
"""Compare object-side contact-query variants on one MuJoCo episode.

Every variant keeps its query points on the object (the Gaussian primitives) --
the floor-side `floor_disk` scheme is no longer part of the comparison because
object-side queries were shown to track the bounce far better. What changes
between variants is *how the points are sampled per primitive* and *how many of
them survive as the contact patch*:

  axis6      6 local axis directions per primitive, lowest-K patch (baseline)
  fib26      Fibonacci-lattice directions per primitive, lowest-K patch
  analytic   the exact lowest point of each sphere in world frame, lowest-K patch
  surface    6 local axis directions, NO patch restriction (over-friction reference)

Why `analytic` should win: a sphere's lowest point against the floor is
`c_world - r*n` at *any* orientation. axis6/fib26 sample directions in the body
frame, so those points rotate with the can and miss the true lowest point by up
to `r*(1-cos t)` when it tilts -- exactly the rim-landing case that matters.

Outputs into --output_root:
  * ``query_mode_comparison.gif``           GT + one panel per variant, labelled
  * ``query_mode_comparison_metrics.json``  per-axis / total RMSE, apex, e, mu
  * a console table

Run with the CUDA-enabled ``gs`` env python (rendering needs the rasterizer):
    python gaussian_initiailization/tools/compare_query_modes.py --episode_root ... --stage1_ply ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
FIT_SCRIPT = TOOLS / "run_stage2_mujoco_stage1_fit.py"
RENDER_SCRIPT = TOOLS / "render_stage2_gaussian_trajectory.py"

# label -> (query_mode, body_query_scheme, dirs-per-primitive or None)
VARIANTS = {
    "axis6": ("body_lowest_k", "axis6", None),
    "fib26": ("body_lowest_k", "fibonacci", 26),
    "analytic": ("body_lowest_k", "analytic", None),
    "surface": ("body_surface", "axis6", None),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode_root", required=True, type=Path)
    p.add_argument("--stage1_ply", required=True, type=Path)
    p.add_argument("--output_root", required=True, type=Path)
    p.add_argument(
        "--variants",
        nargs="+",
        default=["axis6", "fib26", "analytic"],
        choices=sorted(VARIANTS),
        help="Query-point sampling variants to compare (order = panel order after GT).",
    )
    p.add_argument("--body_lowest_k", default=32, type=int)
    p.add_argument("--max_frames", default=80, type=int)
    p.add_argument("--fit_iters", default=250, type=int)
    p.add_argument("--lr", default=0.02, type=float)
    p.add_argument("--init_restitution", default=0.55, type=float)
    p.add_argument("--floor_friction_init", default=0.10, type=float)
    p.add_argument("--substeps", default=6, type=int)
    p.add_argument("--fps", default=12, type=int)
    # Close follow camera: the object stays centered and large instead of being a
    # speck at a fixed wide shot.
    p.add_argument("--cam_distance", default=0.95, type=float)
    p.add_argument("--cam_height", default=0.35, type=float)
    p.add_argument("--cam_fovy_deg", default=45.0, type=float)
    p.add_argument("--follow_offset", default="0,0,0", type=str)
    p.add_argument("--skip_fit", action="store_true", help="Reuse existing fit dirs.")
    p.add_argument("--skip_render", action="store_true", help="Reuse existing render dirs.")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def fit_dir(root: Path, label: str) -> Path:
    return root / f"stage2_fit_{label}"


def render_dir(root: Path, label: str) -> Path:
    return root / f"gaussian_render_{label}"


def run_fit(args: argparse.Namespace, label: str) -> Path:
    query_mode, scheme, dirs = VARIANTS[label]
    out = fit_dir(args.output_root, label)
    cmd = [
        sys.executable, FIT_SCRIPT,
        "--episode_root", args.episode_root,
        "--stage1_ply", args.stage1_ply,
        "--output_dir", out,
        "--dynamics", "restitution",
        "--query_mode", query_mode,
        "--body_query_scheme", scheme,
        "--floor_friction_mode", "learned",
        "--floor_friction_init", args.floor_friction_init,
        "--init_restitution", args.init_restitution,
        "--freeze_gravity",
        "--substeps", args.substeps,
        "--floor_tangential_damping", 0.0,
        "--max_frames", args.max_frames,
        "--fit_iters", args.fit_iters,
        "--lr", args.lr,
        "--foreground_threshold", 0.99,
        "--opacity_threshold", 0.02,
        "--max_primitives", 180,
        "--radius_scale", 0.1,
        "--initial_velocity_source", "trajectory",
        "--freeze_initial_velocity",
    ]
    if query_mode == "body_lowest_k":
        cmd += ["--body_lowest_k", args.body_lowest_k]
    if dirs is not None:
        cmd += ["--body_query_dirs", dirs]
    run(cmd)
    return out


def render_trajectory(args: argparse.Namespace, trajectory: Path, out: Path) -> Path:
    run([
        sys.executable, RENDER_SCRIPT,
        "--stage1_ply", args.stage1_ply,
        "--trajectory", trajectory,
        "--output_dir", out,
        "--max_frames", args.max_frames,
        "--foreground_threshold", 0.99,
        "--opacity_threshold", 0.02,
        "--fps", args.fps,
        # orbit_camera with 0 degrees = fixed yaw, but it is what enables the
        # per-frame follow target, which keeps the can centered and close.
        "--orbit_camera", "--orbit_degrees", 0.0,
        "--follow_object", f"--follow_target_offset={args.follow_offset}",
        "--cam_distance", args.cam_distance,
        "--cam_height", args.cam_height,
        "--cam_fovy_deg", args.cam_fovy_deg,
        "--white_background",
    ])
    return out


def write_gt_trajectory(any_fit: Path, dst: Path) -> Path:
    """Build a GT trajectory json (predicted := target) for the reference panel."""
    payload = json.loads((any_fit / "predicted_trajectory.json").read_text(encoding="utf-8-sig"))
    for s in payload["states"]:
        s["predicted_position"] = s["target_position"]
        if "target_quaternion_wxyz" in s:
            s["predicted_quaternion_wxyz"] = s["target_quaternion_wxyz"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst


def axis_rmse(fit: Path) -> dict:
    payload = json.loads((fit / "predicted_trajectory.json").read_text(encoding="utf-8-sig"))
    states = payload["states"]
    pred = np.array([s["predicted_position"] for s in states], dtype=np.float64)
    tgt = np.array([s["target_position"] for s in states], dtype=np.float64)
    gate = np.array([s.get("contact_gate", 0.0) for s in states], dtype=np.float64)
    err = pred - tgt
    per_axis = np.sqrt((err ** 2).mean(axis=0))
    total = float(np.sqrt((err ** 2).sum(axis=1).mean()))
    # Rebound apex = highest z AFTER first contact; the initial drop height is
    # identical for every variant so only the post-bounce peak discriminates.
    contact_idx = int(np.argmax(gate > 0.5)) if (gate > 0.5).any() else 0
    post = slice(contact_idx + 1, None)
    apex_pred = float(pred[post, 2].max()) if pred[post].size else float(pred[:, 2].max())
    apex_gt = float(tgt[post, 2].max()) if tgt[post].size else float(tgt[:, 2].max())
    return {
        "x": float(per_axis[0]), "y": float(per_axis[1]), "z": float(per_axis[2]),
        "total": total,
        "bounce_apex_pred": apex_pred, "bounce_apex_gt": apex_gt,
        "first_contact_frame": contact_idx,
        "final_xy_err": float(np.linalg.norm(err[-1, :2])),
    }


def fit_scalars(fit: Path) -> dict:
    d = json.loads((fit / "fit_summary.json").read_text(encoding="utf-8-sig"))
    return {
        "e": d.get("learned_restitution"),
        "mu": d.get("learned_friction"),
        "num_query_points": d.get("num_query_points"),
        "scheme": d.get("body_query_scheme"),
    }


def build_comparison_gif(args, panels: list[tuple[str, Path]], out_gif: Path, metrics: dict) -> None:
    """Tile per-frame PNGs across panels with labels into one gif."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    panel_frames = [(label, sorted((rd / "gaussian_rgb").glob("*.png"))) for label, rd in panels]
    n = min(len(f) for _, f in panel_frames)
    if n == 0:
        raise RuntimeError("no rendered frames found for at least one panel")

    label_h = 26
    out_frames = []
    for i in range(n):
        tiles = []
        for label, files in panel_frames:
            img = Image.open(files[i]).convert("RGB")
            w, h = img.size
            canvas = Image.new("RGB", (w, h + label_h), (255, 255, 255))
            canvas.paste(img, (0, label_h))
            caption = label
            if label in metrics:
                caption = f"{label}  RMSE {metrics[label]['total']:.3f}"
            ImageDraw.Draw(canvas).text((6, 6), caption, fill=(0, 0, 0))
            tiles.append(canvas)
        row = Image.new("RGB", (sum(t.size[0] for t in tiles), max(t.size[1] for t in tiles)), (255, 255, 255))
        x = 0
        for t in tiles:
            row.paste(t, (x, 0))
            x += t.size[0]
        out_frames.append(np.asarray(row))
    imageio.mimsave(out_gif, out_frames, fps=max(1, int(args.fps)))
    print(f"[gif] wrote {out_gif} ({n} frames, {len(panels)} panels)")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    for label in args.variants:
        if args.skip_fit and (fit_dir(args.output_root, label) / "fit_summary.json").exists():
            print(f"[skip fit] {label}")
            continue
        run_fit(args, label)

    metrics = {}
    for label in args.variants:
        fd = fit_dir(args.output_root, label)
        metrics[label] = {**axis_rmse(fd), **fit_scalars(fd)}

    gt_traj = write_gt_trajectory(fit_dir(args.output_root, args.variants[0]), args.output_root / "_gt_trajectory.json")
    if not (args.skip_render and (render_dir(args.output_root, "gt") / "gaussian_rgb").exists()):
        render_trajectory(args, gt_traj, render_dir(args.output_root, "gt"))
    for label in args.variants:
        if args.skip_render and (render_dir(args.output_root, label) / "gaussian_rgb").exists():
            print(f"[skip render] {label}")
            continue
        render_trajectory(args, fit_dir(args.output_root, label) / "predicted_trajectory.json",
                          render_dir(args.output_root, label))

    panels = [("GT", render_dir(args.output_root, "gt"))]
    panels += [(label, render_dir(args.output_root, label)) for label in args.variants]
    build_comparison_gif(args, panels, args.output_root / "query_mode_comparison.gif", metrics)

    (args.output_root / "query_mode_comparison_metrics.json").write_text(
        json.dumps({"gt_bounce_apex": metrics[args.variants[0]]["bounce_apex_gt"], "variants": metrics}, indent=2),
        encoding="utf-8",
    )
    print("\n=== Query-point sampling comparison (RMSE vs GT, meters) ===")
    hdr = f"{'variant':<12}{'x':>8}{'y':>8}{'z':>8}{'total':>9}{'apex':>8}{'finXY':>8}{'e':>7}{'mu':>7}{'Npts':>7}"
    print(hdr)
    print("-" * len(hdr))
    for label in args.variants:
        m = metrics[label]
        print(f"{label:<12}{m['x']:>8.3f}{m['y']:>8.3f}{m['z']:>8.3f}{m['total']:>9.3f}"
              f"{m['bounce_apex_pred']:>8.3f}{m['final_xy_err']:>8.3f}"
              f"{(m['e'] or 0):>7.2f}{(m['mu'] or 0):>7.2f}{int(m['num_query_points'] or 0):>7d}")
    print(f"{'GT apex':<12}{'':>8}{'':>8}{metrics[args.variants[0]]['bounce_apex_gt']:>8.3f}")


if __name__ == "__main__":
    main()
