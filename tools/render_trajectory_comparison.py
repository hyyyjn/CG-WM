"""Render a side-by-side comparison GIF: MuJoCo ground truth vs. Stage 2 predicted
trajectory rendered with the Stage 1 Gaussian model.

For each trajectory frame the script:
  1. Computes the delta from the Stage 1 training centroid to the predicted position.
  2. Moves the effective camera so the Gaussians appear at the predicted position
     (equivalent to translating the object without touching model state).
  3. Renders with the 3DGS rasterizer.
  4. Places the render next to the MuJoCo ground truth frame.
  5. Assembles everything into a side-by-side comparison GIF.

Camera convention: generate_mujoco_fall_dataset.py uses
  pos="0 -<cam_distance> <cam_height>" xyaxes="1 0 0 0 0.5 0.8660254" fovy=45°
Pass --cam_distance / --cam_height / --cam_fovy_deg if you changed the defaults.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = REPO_ROOT
for _p in (str(REPO_ROOT), str(GS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage1.scene.cameras import MiniCam  # noqa: E402
from stage1.gaussian_renderer import GaussianModel  # noqa: E402
from stage1.gaussian_renderer import render as gs_render  # noqa: E402
from stage1.utils.graphics_utils import getWorld2View2, getProjectionMatrix  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Side-by-side trajectory comparison GIF (GT vs. 3DGS predicted)"
    )
    p.add_argument("--episode_root", required=True, type=Path,
                   help="Episode directory (must contain rgb/ subfolder).")
    p.add_argument("--predicted_trajectory", required=True, type=Path,
                   help="predicted_trajectory.json from run_stage2_mujoco_stage1_fit.py.")
    p.add_argument("--output_gif", required=True, type=Path)

    # Stage 1 source
    p.add_argument("--stage1_ply", default=None, type=Path,
                   help="Explicit path to point_cloud.ply. Overrides --stage1_model_path.")
    p.add_argument("--stage1_model_path", default=None, type=Path,
                   help="Model directory; picks the highest-iteration PLY automatically.")

    # Coordinate alignment: object centroid in Stage 1 training scene
    p.add_argument("--stage1_centroid", default="0,0,0.08", type=str,
                   help="Object centroid in Stage 1 world coordinates (x,y,z). "
                        "Default 0,0,0.08 for a box settled on the floor.")

    # MuJoCo cam0 parameters — must match generate_mujoco_fall_dataset.py defaults
    p.add_argument("--cam_distance", default=1.35, type=float,
                   help="MuJoCo camera distance from scene origin (--camera_distance).")
    p.add_argument("--cam_height", default=0.75, type=float,
                   help="MuJoCo camera height (--camera_height).")
    p.add_argument("--cam_fovy_deg", default=45.0, type=float,
                   help="Camera vertical FOV in degrees.")
    p.add_argument("--image_width", default=640, type=int)
    p.add_argument("--image_height", default=480, type=int)

    # Filtering
    p.add_argument("--foreground_threshold", default=None, type=float,
                   help="Only render Gaussians with foreground_score >= threshold. "
                        "Useful when Stage 1 was trained with --object_mask_weight > 0.")
    p.add_argument("--orientation_source", default="target", choices=("target", "predicted", "none"),
                   help="Orientation used for the 3DGS render. 'target' reads MuJoCo quaternions from the episode.")
    p.add_argument("--sh_degree", default=3, type=int)
    p.add_argument("--geometry_feature_dim", default=3, type=int)

    # Output
    p.add_argument("--fps", default=20, type=int)
    p.add_argument("--max_frames", default=0, type=int, help="0 = all frames.")
    p.add_argument("--panel_width", default=480, type=int,
                   help="Width of each panel in the output GIF (both panels will be this wide).")
    p.add_argument("--white_background", action="store_true", default=False)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Camera helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mujoco_cam0_c2w_fov(cam_distance: float, cam_height: float, fovy_deg: float,
                         width: int, height: int):
    """Return R (camera-to-world, shape 3×3), T_base (camera-space translation),
    fovx and fovy in radians for the standard MuJoCo cam0.

    MuJoCo xyaxes = "1 0 0  0 0.5 0.8660254":
      camera_x (right) = (1, 0, 0)
      camera_y (up)    = (0, 0.5, 0.866)  → ~60° elevation tilt
      camera_z (out)   = cross(x, y) = (0, -0.866, 0.5)

    Camera position in world = (0, -cam_distance, cam_height).
    """
    x_cam = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y_cam = np.array([0.0, 0.5, 0.8660254], dtype=np.float64)
    z_cam = np.cross(x_cam, y_cam)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)
    cam_center = np.array([0.0, -cam_distance, cam_height], dtype=np.float32)
    c2w[:3, 3] = cam_center

    fovy = math.radians(fovy_deg)
    fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * float(width) / float(height))
    return c2w, fovx, fovy


def _c2w_to_3dgs_R_T(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c2w_3dgs = np.array(c2w, dtype=np.float32, copy=True)
    c2w_3dgs[:3, 1:3] *= -1.0
    w2c = np.linalg.inv(c2w_3dgs)
    return np.transpose(w2c[:3, :3]).astype(np.float32), w2c[:3, 3].astype(np.float32)


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _make_mini_cam(R: np.ndarray, T: np.ndarray,
                   fovx: float, fovy: float,
                   width: int, height: int,
                   device: str,
                   znear: float = 0.01, zfar: float = 200.0) -> MiniCam:
    W2V = getWorld2View2(R, T)
    W2V_t = torch.tensor(W2V, dtype=torch.float32).T.to(device)
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy)
    proj_t = proj.T.to(device)
    full_proj = W2V_t.unsqueeze(0).bmm(proj_t.unsqueeze(0)).squeeze(0)
    return MiniCam(width, height, fovy, fovx, znear, zfar, W2V_t, full_proj)


def _cam_at_predicted_pos(c2w_base: np.ndarray,
                           fovx: float, fovy: float,
                           width: int, height: int,
                           device: str,
                           predicted_world: np.ndarray,
                           stage1_centroid: np.ndarray,
                           object_rotation: np.ndarray | None = None) -> MiniCam:
    """Build a MiniCam that renders Stage 1 Gaussians as if the object is at
    predicted_world.

    Moving the object by delta = predicted - centroid is equivalent to moving the
    camera by -delta (keeping Gaussians fixed in memory).

    The renderer uses the same axis conversion as the normal NeRF-synthetic
    dataset loader before constructing the MiniCam.
    """
    if object_rotation is None:
        delta = (predicted_world - stage1_centroid).astype(np.float32)
        c2w_shifted = c2w_base.copy()
        c2w_shifted[:3, 3] = c2w_shifted[:3, 3] - delta
    else:
        # Desired object transform: x' = R_obj (x - c_stage1) + p_pred.
        # Keep Gaussians fixed and pull the camera through the inverse transform.
        R_obj_inv = np.asarray(object_rotation, dtype=np.float32).T
        c2w_shifted = c2w_base.copy()
        c2w_shifted[:3, :3] = R_obj_inv @ c2w_base[:3, :3]
        c2w_shifted[:3, 3] = R_obj_inv @ (c2w_base[:3, 3] - predicted_world) + stage1_centroid
    R, T_shifted = _c2w_to_3dgs_R_T(c2w_shifted)
    return _make_mini_cam(R, T_shifted, fovx, fovy, width, height, device)


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian model
# ──────────────────────────────────────────────────────────────────────────────

class _PipelineParams:
    """Minimal stand-in for PipelineParams accepted by gaussian_renderer.render."""
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


def _resolve_ply(stage1_ply: Path | None, stage1_model_path: Path | None) -> Path:
    if stage1_ply is not None:
        p = stage1_ply.resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    if stage1_model_path is None:
        raise ValueError("Pass --stage1_ply or --stage1_model_path.")
    model_path = stage1_model_path.resolve()
    candidates = list(model_path.glob("point_cloud/iteration_*/point_cloud.ply"))
    if not candidates:
        raise FileNotFoundError(f"No point_cloud/iteration_*/point_cloud.ply under {model_path}")
    def _iter(p: Path) -> int:
        m = re.search(r"iteration_(\d+)", str(p))
        return int(m.group(1)) if m else -1
    return max(candidates, key=_iter)


def _load_gaussians(ply_path: Path, sh_degree: int,
                    geometry_feature_dim: int, device: str) -> GaussianModel:
    print(f"[INFO] Loading Stage 1 PLY: {ply_path}")
    g = GaussianModel(sh_degree, geometry_feature_dim=geometry_feature_dim)
    g.load_ply(str(ply_path))
    for attr in ("_xyz", "_features_dc", "_features_rest", "_scaling",
                 "_rotation", "_opacity", "_geometry_feature", "_foreground_feature",
                 "_object_ids"):
        if hasattr(g, attr) and isinstance(getattr(g, attr), torch.Tensor):
            setattr(g, attr, getattr(g, attr).to(device))
    print(f"[INFO] Loaded {g._xyz.shape[0]:,} Gaussians.")
    return g


def _build_gaussian_mask(gaussians: GaussianModel,
                          foreground_threshold: float | None) -> torch.Tensor | None:
    if foreground_threshold is None:
        return None
    mask = gaussians.get_foreground_scores[:, 0] >= float(foreground_threshold)
    if not mask.any():
        print("[WARN] foreground_threshold filtered ALL Gaussians — rendering without filter.")
        return None
    kept = int(mask.sum().item())
    total = mask.shape[0]
    print(f"[INFO] foreground_threshold={foreground_threshold}: keeping {kept}/{total} Gaussians.")
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

def _render_to_numpy(cam: MiniCam, gaussians: GaussianModel,
                     background: torch.Tensor,
                     gaussian_mask: torch.Tensor | None) -> np.ndarray:
    with torch.no_grad():
        out = gs_render(cam, gaussians, _PipelineParams(), background,
                        separate_sh=False, gaussian_mask=gaussian_mask)["render"]
    return (out.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _resize_to(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return np.array(Image.fromarray(frame).resize((width, height), Image.BILINEAR))


# ──────────────────────────────────────────────────────────────────────────────
# Composite frame builder
# ──────────────────────────────────────────────────────────────────────────────

_HDR_H = 52   # header bar height in pixels
_SEP_W = 4    # separator width

def _make_comparison_frame(gt: np.ndarray, pred: np.ndarray,
                            frame_idx: int, total: int,
                            target_z: float, pred_z: float,
                            contact_gate: float) -> Image.Image:
    ph, pw = gt.shape[:2]
    canvas_w = pw * 2 + _SEP_W
    canvas_h = ph + _HDR_H
    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))

    # Header bars
    canvas.paste(Image.new("RGB", (pw, _HDR_H), (25, 55, 105)), (0, 0))
    canvas.paste(Image.new("RGB", (pw, _HDR_H), (90, 40, 20)),  (pw + _SEP_W, 0))

    # Frames
    canvas.paste(Image.fromarray(gt),   (0, _HDR_H))
    canvas.paste(Image.fromarray(pred), (pw + _SEP_W, _HDR_H))

    draw = ImageDraw.Draw(canvas)

    # GT header text
    draw.text((8, 6),  "Ground Truth  (MuJoCo)", fill=(190, 215, 255))
    draw.text((8, 28), f"z = {target_z:+.4f} m", fill=(150, 195, 255))

    # Predicted header text
    contact_color = (255, 80, 60) if contact_gate > 0.5 else (200, 170, 130)
    draw.text((pw + _SEP_W + 8, 6),
              f"Stage 2 Predicted  (3DGS)   frame {frame_idx:03d}/{total - 1:03d}",
              fill=(255, 195, 145))
    draw.text((pw + _SEP_W + 8, 28),
              f"z = {pred_z:+.4f} m   Δz = {pred_z - target_z:+.4f} m",
              fill=contact_color)

    # Contact indicator dot
    if contact_gate > 0.1:
        dot_r = min(18, max(4, int(contact_gate * 8)))
        cx = pw + _SEP_W + pw - 20
        cy = _HDR_H - 16
        draw.ellipse((cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r),
                     fill=(255, 60, 40) if contact_gate > 0.5 else (255, 180, 50))

    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = args.device

    episode_root = args.episode_root.resolve()
    traj_data = json.loads(args.predicted_trajectory.read_text(encoding="utf-8"))
    states = traj_data["states"]
    if args.max_frames > 0:
        states = states[: int(args.max_frames)]
    total = len(states)
    print(f"[INFO] Trajectory: {total} frames")
    target_quaternions = None
    if args.orientation_source == "target":
        trajectory_path = episode_root / "state" / "trajectory.json"
        if not trajectory_path.exists():
            trajectory_path = episode_root / "trajectory.json"
        episode_trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        target_quaternions = [
            np.array(frame["quaternion_wxyz"], dtype=np.float32)
            for frame in episode_trajectory["states"][:total]
        ]

    stage1_centroid = np.array(
        [float(v.strip()) for v in args.stage1_centroid.split(",")], dtype=np.float32
    )
    print(f"[INFO] Stage 1 centroid: {stage1_centroid}")

    ply_path = _resolve_ply(args.stage1_ply, args.stage1_model_path)
    gaussians = _load_gaussians(ply_path, args.sh_degree, args.geometry_feature_dim, device)
    gaussian_mask = _build_gaussian_mask(gaussians, args.foreground_threshold)

    c2w_base, fovx, fovy = _mujoco_cam0_c2w_fov(
        args.cam_distance, args.cam_height, args.cam_fovy_deg,
        args.image_width, args.image_height,
    )
    print(f"[INFO] Camera fovx={math.degrees(fovx):.1f}° fovy={math.degrees(fovy):.1f}°")

    bg = torch.tensor(
        [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32, device=device,
    )

    rgb_dir = episode_root / "rgb"
    gt_files = sorted(rgb_dir.glob("*.png"))
    if not gt_files:
        raise FileNotFoundError(f"No PNG frames found in {rgb_dir}")
    print(f"[INFO] Ground truth frames: {len(gt_files)} in {rgb_dir}")

    panel_h = int(args.image_height * args.panel_width / args.image_width)

    frames_out: list[Image.Image] = []
    for idx, state in enumerate(states):
        predicted_pos = np.array(state["predicted_position"], dtype=np.float32)
        target_pos    = np.array(state["target_position"],    dtype=np.float32)
        contact_gate  = float(state.get("contact_gate", 0.0))
        object_rotation = None
        if args.orientation_source == "predicted" and "predicted_quaternion_wxyz" in state:
            object_rotation = _quat_wxyz_to_matrix(np.array(state["predicted_quaternion_wxyz"], dtype=np.float32))
        elif args.orientation_source == "target" and target_quaternions is not None and idx < len(target_quaternions):
            object_rotation = _quat_wxyz_to_matrix(target_quaternions[idx])

        # Ground truth frame
        gt_path = gt_files[idx] if idx < len(gt_files) else gt_files[-1]
        gt_frame = _resize_to(
            np.array(Image.open(gt_path).convert("RGB")),
            args.panel_width, panel_h
        )

        # Predicted render
        cam = _cam_at_predicted_pos(
            c2w_base, fovx, fovy,
            args.image_width, args.image_height,
            device, predicted_pos, stage1_centroid,
            object_rotation=object_rotation,
        )
        pred_raw = _render_to_numpy(cam, gaussians, bg, gaussian_mask)
        pred_frame = _resize_to(pred_raw, args.panel_width, panel_h)

        frame_img = _make_comparison_frame(
            gt_frame, pred_frame,
            idx, total,
            float(target_pos[2]), float(predicted_pos[2]),
            contact_gate,
        )
        frames_out.append(frame_img)

        if idx % 20 == 0 or idx == total - 1:
            print(f"  frame {idx:03d}/{total - 1:03d}  "
                  f"target_z={target_pos[2]:.3f}  pred_z={predicted_pos[2]:.3f}", flush=True)

    output_gif = args.output_gif.resolve()
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / float(args.fps))))
    frames_out[0].save(
        output_gif,
        save_all=True,
        append_images=frames_out[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"\n[DONE] {output_gif}")
    print(f"       {total} frames  ·  {args.fps} fps  ·  "
          f"{frames_out[0].width}×{frames_out[0].height} px per frame")


if __name__ == "__main__":
    main()
