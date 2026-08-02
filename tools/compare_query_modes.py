#!/usr/bin/env python
"""Compare object-side contact-query variants with pairwise impedance dynamics.

Every variant keeps its query points on the object (the Gaussian primitives) --
the floor-side `floor_disk` scheme is no longer part of the comparison because
object-side queries were shown to track the bounce far better. What changes
between variants is *how the points are sampled per primitive* and *how many of
them survive as the contact patch*:

  axis6      6 local axis directions per primitive, lowest-K patch (baseline)
  fib26      Fibonacci-lattice directions per primitive, lowest-K patch
  analytic   support point toward the closest target primitive, lowest-K patch
  surface    6 local axis directions, fixed pairwise patch count

The moving Stage-I Gaussian body is evaluated against a static Gaussian floor
body, so only query-point construction changes between variants.

Outputs into --output_root:
  * ``query_mode_comparison.gif``           GT + one panel per variant, labelled
  * ``query_mode_comparison_metrics.json``  position/orientation RMSE and K/D
  * a console table

Run with the CUDA-enabled ``gs`` env python (rendering needs the rasterizer):
    python  tools/compare_query_modes.py --episode_root ... --stage1_ply ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
FIT_SCRIPT = TOOLS / "run_stage2_mujoco_stage1_fit.py"
RENDER_SCRIPT = TOOLS / "render_stage2_gaussian_trajectory.py"

# label -> (query_mode, body_query_scheme, dirs-per-primitive or None)
VARIANTS = {
    "floor_disk": ("floor_disk", "axis6", None),
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
        default=["floor_disk", "axis6", "fib26", "analytic"],
        choices=sorted(VARIANTS),
        help="Query-point sampling variants to compare (order = panel order after GT).",
    )
    p.add_argument("--body_lowest_k", default=32, type=int)
    p.add_argument("--max_primitives", default=180, type=int)
    p.add_argument("--pairwise_body_b_max_primitives", default=None, type=int)
    p.add_argument("--max_frames", default=80, type=int)
    p.add_argument("--fit_iters", default=250, type=int)
    p.add_argument("--lr", default=0.02, type=float)
    p.add_argument("--init_stiffness", default=800.0, type=float)
    p.add_argument("--init_damping", default=30.0, type=float)
    p.add_argument("--mass", default=1.0, type=float)
    p.add_argument("--pairwise_mass_mode", default="fixed", choices=("fixed", "learned"))
    p.add_argument("--pairwise_inertia_diag", default="1,1,1")
    p.add_argument("--pairwise_inertia_mode", default="fixed", choices=("fixed", "learned"))
    p.add_argument("--pairwise_mass_l2_weight", default=1e-4, type=float)
    p.add_argument("--pairwise_inertia_l2_weight", default=1e-4, type=float)
    p.add_argument("--actions_json", default=None, type=Path)
    p.add_argument("--action_force_scale", default=1.0, type=float)
    p.add_argument("--action_torque_scale", default=1.0, type=float)
    p.add_argument("--pairwise_body_b_trajectory_json", default=None, type=Path)
    p.add_argument("--pairwise_friction_coefficient", default=0.10, type=float)
    p.add_argument(
        "--pairwise_friction_mode",
        default="fixed",
        choices=("off", "fixed", "learned"),
    )
    p.add_argument(
        "--pairwise_contact_model",
        default="dual_cone",
        choices=("dual_cone", "projected"),
    )
    p.add_argument("--pairwise_dual_cone_directions", default=4, type=int)
    p.add_argument("--pairwise_tangential_damping", default=0.0, type=float)
    p.add_argument("--orientation_loss_weight", default=1.0, type=float)
    p.add_argument("--position_loss_weight", default=1.0, type=float)
    p.add_argument("--image_only_objective", action="store_true")
    p.add_argument("--geometry_loss_weight", default=0.0, type=float)
    p.add_argument("--geometry_loss_stride", default=1, type=int)
    p.add_argument("--refine_geometry", action="store_true")
    p.add_argument(
        "--geometry_gradient_route",
        default="collision_only",
        choices=("collision_only", "collision_and_render"),
    )
    p.add_argument("--geometry_center_l2_weight", default=1e-3, type=float)
    p.add_argument("--geometry_radius_l2_weight", default=1e-3, type=float)
    p.add_argument("--geometry_max_center_offset", default=0.02, type=float)
    p.add_argument("--geometry_max_log_radius_offset", default=0.35, type=float)
    p.add_argument("--gaussian_rgb_loss_weight", default=0.0, type=float)
    p.add_argument("--gaussian_rgb_dir", default=None, type=Path)
    p.add_argument("--gaussian_mask_dir", default=None, type=Path)
    p.add_argument("--gaussian_views_manifest", default=None, type=Path)
    p.add_argument("--gaussian_render_stride", default=10, type=int)
    p.add_argument("--gaussian_render_max_frames", default=0, type=int)
    p.add_argument("--gaussian_render_width", default=160, type=int)
    p.add_argument("--gaussian_render_height", default=120, type=int)
    p.add_argument(
        "--gaussian_render_loss",
        default="l1_ssim",
        choices=("l1", "mse", "l1_ssim", "l1_loftr"),
    )
    p.add_argument("--gaussian_render_ssim_weight", default=0.2, type=float)
    p.add_argument("--gaussian_render_loftr_weight", default=0.1, type=float)
    p.add_argument("--loftr_pretrained", default="outdoor", choices=("outdoor", "indoor"))
    p.add_argument("--loftr_confidence_threshold", default=0.2, type=float)
    p.add_argument("--loftr_max_matches", default=1024, type=int)
    p.add_argument("--loftr_min_matches", default=8, type=int)
    p.add_argument("--loftr_patch_radius", default=2, type=int)
    p.add_argument("--gaussian_cam_distance", default=1.12, type=float)
    p.add_argument("--gaussian_cam_height", default=0.66, type=float)
    p.add_argument("--gaussian_cam_fovy_deg", default=40.0, type=float)
    p.add_argument("--pairwise_num_contact_patches", default=32, type=int)
    p.add_argument("--query_radius_scale", default=1.10, type=float)
    p.add_argument("--query_rings", default=5, type=int)
    p.add_argument("--query_angles", default=32, type=int)
    p.add_argument("--pairwise_body_b_ply", default=None, type=Path)
    p.add_argument("--floor_proxy_radius", default=0.10, type=float)
    p.add_argument("--floor_proxy_spacing", default=0.10, type=float)
    p.add_argument("--floor_proxy_margin", default=0.30, type=float)
    p.add_argument("--radius_scale", default=0.1, type=float)
    p.add_argument(
        "--gaussian_radius_convention",
        default="paper_r2s",
        choices=("paper_r2s", "legacy_r_equals_s"),
    )
    p.add_argument(
        "--gaussian_scale_reduction",
        default="mean",
        choices=("strict", "mean", "max"),
    )
    p.add_argument("--gaussian_isotropic_tolerance", default=1e-4, type=float)
    p.add_argument("--fps", default=12, type=int)
    # Close follow camera: the object stays centered and large instead of being a
    # speck at a fixed wide shot.
    p.add_argument("--cam_distance", default=0.95, type=float)
    p.add_argument("--cam_height", default=0.35, type=float)
    p.add_argument("--cam_fovy_deg", default=45.0, type=float)
    p.add_argument("--follow_offset", default="0,0,0", type=str)
    p.add_argument("--skip_fit", action="store_true", help="Reuse existing fit dirs.")
    p.add_argument("--skip_render", action="store_true", help="Skip rendering/GIF creation and write metrics only.")
    p.add_argument("--eval_only", action="store_true", help="Do not optimize; evaluate the supplied K/D on this episode.")
    p.add_argument("--initial_state_json", default=None, type=Path)
    p.add_argument("--prefit_initial_state", action="store_true")
    p.add_argument("--prefit_pose_iters", default=100, type=int)
    p.add_argument("--prefit_velocity_iters", default=100, type=int)
    p.add_argument("--prefit_velocity_frames", default=3, type=int)
    p.add_argument("--prefit_lr", default=0.01, type=float)
    p.add_argument("--prefit_velocity_l2", default=1e-4, type=float)
    p.add_argument("--prefit_position_init", default="0,0,0")
    p.add_argument("--prefit_quaternion_init", default="1,0,0,0")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def fit_dir(root: Path, label: str) -> Path:
    return root / f"stage2_fit_{label}"


def render_dir(root: Path, label: str) -> Path:
    return root / f"gaussian_render_{label}"


def make_floor_proxy(args: argparse.Namespace) -> Path:
    """Write overlapping spherical Gaussians whose upper envelope is near z=0."""
    if args.floor_proxy_radius <= 0 or args.floor_proxy_spacing <= 0 or args.radius_scale <= 0:
        raise ValueError("floor proxy radius, spacing, and radius_scale must be positive")
    trajectory = json.loads(
        (args.episode_root / "state" / "trajectory.json").read_text(encoding="utf-8-sig")
    )
    states = trajectory["states"] if isinstance(trajectory, dict) else trajectory
    positions = np.asarray([s.get("position", s.get("translation")) for s in states], dtype=np.float64)
    lo = positions[:, :2].min(axis=0) - args.floor_proxy_margin
    hi = positions[:, :2].max(axis=0) + args.floor_proxy_margin
    centre = 0.5 * (lo + hi)
    half_min = max(args.floor_proxy_margin, 2.0 * args.floor_proxy_radius)
    lo, hi = np.minimum(lo, centre - half_min), np.maximum(hi, centre + half_min)
    xs = np.arange(lo[0], hi[0] + 0.5 * args.floor_proxy_spacing, args.floor_proxy_spacing)
    ys = np.arange(lo[1], hi[1] + 0.5 * args.floor_proxy_spacing, args.floor_proxy_spacing)
    points = [(float(x), float(y), -args.floor_proxy_radius) for x in xs for y in ys]
    # The shared Stage-I loader subsequently multiplies exp(scale) by radius_scale.
    log_r = float(np.log(args.floor_proxy_radius / args.radius_scale))
    path = args.output_root / "_generated_floor_gaussians.ply"
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(points)}",
        "property float x", "property float y", "property float z",
        "property float scale_0", "property float scale_1", "property float scale_2",
        "end_header",
    ]
    lines.extend(f"{x:.9g} {y:.9g} {z:.9g} {log_r:.9g} {log_r:.9g} {log_r:.9g}" for x, y, z in points)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"[floor proxy] wrote {path} ({len(points)} Gaussian spheres)")
    return path


def run_fit(args: argparse.Namespace, label: str, body_b_ply: Path) -> Path:
    query_mode, scheme, dirs = VARIANTS[label]
    out = fit_dir(args.output_root, label)
    cmd = [
        sys.executable, FIT_SCRIPT,
        "--episode_root", args.episode_root,
        "--stage1_ply", args.stage1_ply,
        "--output_dir", out,
        "--dynamics", "pairwise_impedance",
        "--query_mode", query_mode,
        "--body_query_scheme", scheme,
        "--pairwise_body_b_ply", body_b_ply,
        "--pairwise_num_contact_patches", args.pairwise_num_contact_patches,
        "--query_radius_scale", args.query_radius_scale,
        "--query_rings", args.query_rings,
        "--query_angles", args.query_angles,
        "--pairwise_friction_coefficient", args.pairwise_friction_coefficient,
        "--pairwise_friction_mode", args.pairwise_friction_mode,
        "--pairwise_contact_model", args.pairwise_contact_model,
        "--pairwise_dual_cone_directions", args.pairwise_dual_cone_directions,
        "--pairwise_tangential_damping", args.pairwise_tangential_damping,
        "--init_stiffness", args.init_stiffness,
        "--init_damping", args.init_damping,
        "--mass", args.mass,
        "--pairwise_mass_mode", args.pairwise_mass_mode,
        "--pairwise_inertia_diag", args.pairwise_inertia_diag,
        "--pairwise_inertia_mode", args.pairwise_inertia_mode,
        "--pairwise_mass_l2_weight", args.pairwise_mass_l2_weight,
        "--pairwise_inertia_l2_weight", args.pairwise_inertia_l2_weight,
        "--action_force_scale", args.action_force_scale,
        "--action_torque_scale", args.action_torque_scale,
        "--orientation_loss_weight", args.orientation_loss_weight,
        "--position_loss_weight", args.position_loss_weight,
        "--geometry_loss_weight", args.geometry_loss_weight,
        "--geometry_loss_stride", args.geometry_loss_stride,
        "--geometry_center_l2_weight", args.geometry_center_l2_weight,
        "--geometry_radius_l2_weight", args.geometry_radius_l2_weight,
        "--geometry_max_center_offset", args.geometry_max_center_offset,
        "--geometry_max_log_radius_offset", args.geometry_max_log_radius_offset,
        "--geometry_gradient_route", args.geometry_gradient_route,
        "--gaussian_rgb_loss_weight", args.gaussian_rgb_loss_weight,
        "--gaussian_render_stride", args.gaussian_render_stride,
        "--gaussian_render_max_frames", args.gaussian_render_max_frames,
        "--gaussian_render_width", args.gaussian_render_width,
        "--gaussian_render_height", args.gaussian_render_height,
        "--gaussian_render_loss", args.gaussian_render_loss,
        "--gaussian_render_ssim_weight", args.gaussian_render_ssim_weight,
        "--gaussian_render_loftr_weight", args.gaussian_render_loftr_weight,
        "--loftr_pretrained", args.loftr_pretrained,
        "--loftr_confidence_threshold", args.loftr_confidence_threshold,
        "--loftr_max_matches", args.loftr_max_matches,
        "--loftr_min_matches", args.loftr_min_matches,
        "--loftr_patch_radius", args.loftr_patch_radius,
        "--gaussian_cam_distance", args.gaussian_cam_distance,
        "--gaussian_cam_height", args.gaussian_cam_height,
        "--gaussian_cam_fovy_deg", args.gaussian_cam_fovy_deg,
        "--fit_initial_angular_velocity",
        "--max_frames", args.max_frames,
        "--fit_iters", args.fit_iters,
        "--lr", args.lr,
        "--foreground_threshold", 0.99,
        "--opacity_threshold", 0.02,
        "--max_primitives", args.max_primitives,
        "--radius_scale", args.radius_scale,
        "--gaussian_radius_convention", args.gaussian_radius_convention,
        "--gaussian_scale_reduction", args.gaussian_scale_reduction,
        "--gaussian_isotropic_tolerance", args.gaussian_isotropic_tolerance,
        "--initial_velocity_source", "trajectory",
        "--freeze_initial_velocity",
    ]
    if args.pairwise_body_b_max_primitives is not None:
        cmd += ["--pairwise_body_b_max_primitives", args.pairwise_body_b_max_primitives]
    if args.eval_only:
        cmd.append("--eval_only")
    if args.initial_state_json is not None:
        cmd += ["--initial_state_json", args.initial_state_json]
    if args.actions_json is not None:
        cmd += ["--actions_json", args.actions_json]
    if args.pairwise_body_b_trajectory_json is not None:
        cmd += ["--pairwise_body_b_trajectory_json", args.pairwise_body_b_trajectory_json]
    if args.prefit_initial_state:
        cmd += [
            "--prefit_initial_state",
            "--prefit_pose_iters", args.prefit_pose_iters,
            "--prefit_velocity_iters", args.prefit_velocity_iters,
            "--prefit_velocity_frames", args.prefit_velocity_frames,
            "--prefit_lr", args.prefit_lr,
            "--prefit_velocity_l2", args.prefit_velocity_l2,
            "--prefit_position_init", args.prefit_position_init,
            "--prefit_quaternion_init", args.prefit_quaternion_init,
        ]
    if args.refine_geometry:
        cmd.append("--refine_geometry")
    if args.image_only_objective:
        cmd.append("--image_only_objective")
    if args.gaussian_rgb_dir is not None:
        cmd += ["--gaussian_rgb_dir", args.gaussian_rgb_dir]
    if args.gaussian_mask_dir is not None:
        cmd += ["--gaussian_mask_dir", args.gaussian_mask_dir]
    if args.gaussian_views_manifest is not None:
        cmd += ["--gaussian_views_manifest", args.gaussian_views_manifest]
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
    orientation_rmse_degrees = None
    if states and all("predicted_quaternion_wxyz" in s and "target_quaternion_wxyz" in s for s in states):
        pred_q = np.asarray([s["predicted_quaternion_wxyz"] for s in states], dtype=np.float64)
        tgt_q = np.asarray([s["target_quaternion_wxyz"] for s in states], dtype=np.float64)
        pred_q /= np.maximum(np.linalg.norm(pred_q, axis=1, keepdims=True), 1e-12)
        tgt_q /= np.maximum(np.linalg.norm(tgt_q, axis=1, keepdims=True), 1e-12)
        dots = np.clip(np.abs((pred_q * tgt_q).sum(axis=1)), 0.0, 1.0)
        orientation_rmse_degrees = float(np.degrees(np.sqrt(np.mean((2.0 * np.arccos(dots)) ** 2))))
    return {
        "x": float(per_axis[0]), "y": float(per_axis[1]), "z": float(per_axis[2]),
        "total": total,
        "bounce_apex_pred": apex_pred, "bounce_apex_gt": apex_gt,
        "first_contact_frame": contact_idx,
        "final_xy_err": float(np.linalg.norm(err[-1, :2])),
        "orientation_rmse_degrees": orientation_rmse_degrees,
    }


def fit_scalars(fit: Path) -> dict:
    d = json.loads((fit / "fit_summary.json").read_text(encoding="utf-8-sig"))
    return {
        "stiffness": d.get("learned_stiffness"),
        "damping": d.get("learned_damping"),
        "mu": d.get("pairwise_friction_coefficient"),
        "friction_mode": d.get("pairwise_friction_mode"),
        "contact_model": d.get("pairwise_contact_model"),
        "dual_cone_directions": d.get("pairwise_dual_cone_directions"),
        "mass": d.get("learned_mass"),
        "inertia_diag": d.get("learned_inertia_diag"),
        "mass_mode": d.get("pairwise_mass_mode"),
        "inertia_mode": d.get("pairwise_inertia_mode"),
        "geometry_gradient_route": d.get("geometry_gradient_route"),
        "num_query_points": d.get("num_query_points"),
        "num_primitives_a": d.get("num_primitives_a"),
        "num_primitives_b": d.get("num_primitives_b"),
        "raw_query_candidates_a": d.get("raw_query_candidates_a"),
        "raw_query_candidates_b": d.get("raw_query_candidates_b"),
        "raw_query_candidates_total": d.get("raw_query_candidates_total"),
        "scheme": d.get("pairwise_body_query_scheme", d.get("body_query_scheme")),
        "num_contact_patches": d.get("pairwise_num_contact_patches"),
        "geometry_loss": d.get("geometry_loss"),
        "geometry_loss_weight": d.get("geometry_loss_weight"),
        "geometry_refinement_enabled": d.get("geometry_refinement_enabled"),
        "geometry_center_offset_rms": d.get("geometry_center_offset_rms"),
        "geometry_log_radius_offset_rms": d.get("geometry_log_radius_offset_rms"),
        "refined_geometry": d.get("refined_geometry"),
        "gaussian_rgb_loss": d.get("gaussian_rgb_loss"),
        "gaussian_rgb_loss_weight": d.get("gaussian_rgb_loss_weight"),
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
    body_b_ply = (args.pairwise_body_b_ply or make_floor_proxy(args)).resolve()

    for label in args.variants:
        if args.skip_fit and (fit_dir(args.output_root, label) / "fit_summary.json").exists():
            print(f"[skip fit] {label}")
            continue
        run_fit(args, label, body_b_ply)

    metrics = {}
    for label in args.variants:
        fd = fit_dir(args.output_root, label)
        metrics[label] = {**axis_rmse(fd), **fit_scalars(fd)}

    if not args.skip_render:
        gt_traj = write_gt_trajectory(
            fit_dir(args.output_root, args.variants[0]), args.output_root / "_gt_trajectory.json"
        )
        render_trajectory(args, gt_traj, render_dir(args.output_root, "gt"))
        for label in args.variants:
            render_trajectory(args, fit_dir(args.output_root, label) / "predicted_trajectory.json",
                              render_dir(args.output_root, label))
        panels = [("GT", render_dir(args.output_root, "gt"))]
        panels += [(label, render_dir(args.output_root, label)) for label in args.variants]
        build_comparison_gif(args, panels, args.output_root / "query_mode_comparison.gif", metrics)

    (args.output_root / "query_mode_comparison_metrics.json").write_text(
        json.dumps({
            "dynamics": "pairwise_impedance",
            "static_body_b_ply": str(body_b_ply),
            "gt_bounce_apex": metrics[args.variants[0]]["bounce_apex_gt"],
            "variants": metrics,
        }, indent=2),
        encoding="utf-8",
    )
    print("\n=== Query-point sampling comparison (RMSE vs GT, meters) ===")
    hdr = f"{'variant':<12}{'x':>8}{'y':>8}{'z':>8}{'total':>9}{'rot°':>8}{'apex':>8}{'K':>9}{'D':>8}{'Npts':>7}"
    print(hdr)
    print("-" * len(hdr))
    for label in args.variants:
        m = metrics[label]
        rot = m["orientation_rmse_degrees"]
        print(f"{label:<12}{m['x']:>8.3f}{m['y']:>8.3f}{m['z']:>8.3f}{m['total']:>9.3f}"
              f"{(rot if rot is not None else float('nan')):>8.2f}{m['bounce_apex_pred']:>8.3f}"
              f"{(m['stiffness'] or 0):>9.1f}{(m['damping'] or 0):>8.2f}{int(m['num_query_points'] or 0):>7d}")
    print(f"{'GT apex':<12}{'':>8}{'':>8}{metrics[args.variants[0]]['bounce_apex_gt']:>8.3f}")


if __name__ == "__main__":
    main()
