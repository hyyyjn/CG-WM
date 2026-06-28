from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
GS_ROOT = REPO_ROOT / "gaussian_initiailization"
for path in (str(REPO_ROOT), str(GS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from gaussian_initiailization.stage2.renderable_gaussian_asset import (  # noqa: E402
    RenderableGaussianAsset,
    copy_asset_to_gaussian_model,
    instantiate_rigid_gaussian_scene,
    load_renderable_gaussian_asset,
)
from gaussian_initiailization.stage2.differentiable_collision_detection import (  # noqa: E402
    load_gaussian_collision_primitives_from_ply,
)
from gaussian_renderer import GaussianModel  # noqa: E402
from gaussian_renderer import render as gs_render  # noqa: E402
from scene.cameras import MiniCam  # noqa: E402
from utils.graphics_utils import getProjectionMatrix, getWorld2View2  # noqa: E402


class PipelineParams:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Stage2 pose trajectory with a rigid Stage1 Gaussian asset.")
    parser.add_argument("--stage1_ply", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_frames", default=0, type=int)
    parser.add_argument("--frame_stride", default=1, type=int)
    parser.add_argument("--image_width", default=640, type=int)
    parser.add_argument("--image_height", default=480, type=int)
    parser.add_argument("--cam_distance", default=1.12, type=float)
    parser.add_argument("--cam_height", default=0.66, type=float)
    parser.add_argument("--cam_fovy_deg", default=40.0, type=float)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--scale_multiplier", default=1.0, type=float)
    parser.add_argument("--foreground_threshold", default=None, type=float)
    parser.add_argument("--opacity_threshold", default=None, type=float)
    parser.add_argument("--recenter_asset", action="store_true")
    parser.add_argument(
        "--auto_scale_to_trajectory_half_extent",
        action="store_true",
        help="Multiply render geometry by the same Stage1-to-MuJoCo scale used by the collision proxy.",
    )
    parser.add_argument("--fps", default=12, type=int)
    parser.add_argument("--allow_cpu_skip", action="store_true")
    return parser.parse_args()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def mujoco_cam0_c2w_fov(cam_distance: float, cam_height: float, fovy_deg: float, width: int, height: int):
    x_cam = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y_cam = np.array([0.0, 0.48, 0.8772685], dtype=np.float64)
    z_cam = np.cross(x_cam, y_cam)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)
    c2w[:3, 3] = np.array([0.0, -cam_distance, cam_height], dtype=np.float32)
    fovy = math.radians(float(fovy_deg))
    fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * float(width) / float(height))
    return c2w, fovx, fovy


def c2w_to_3dgs_rt(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c2w_3dgs = np.array(c2w, dtype=np.float32, copy=True)
    c2w_3dgs[:3, 1:3] *= -1.0
    w2c = np.linalg.inv(c2w_3dgs)
    return np.transpose(w2c[:3, :3]).astype(np.float32), w2c[:3, 3].astype(np.float32)


def make_camera(args: argparse.Namespace, *, device: str) -> MiniCam:
    c2w, fovx, fovy = mujoco_cam0_c2w_fov(
        float(args.cam_distance),
        float(args.cam_height),
        float(args.cam_fovy_deg),
        int(args.image_width),
        int(args.image_height),
    )
    R, T = c2w_to_3dgs_rt(c2w)
    world_view = torch.tensor(getWorld2View2(R, T), dtype=torch.float32).T.to(device)
    proj = getProjectionMatrix(znear=0.01, zfar=200.0, fovX=fovx, fovY=fovy).T.to(device)
    full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return MiniCam(int(args.image_width), int(args.image_height), fovy, fovx, 0.01, 200.0, world_view, full_proj)


def select_frames(payload: dict, *, max_frames: int, frame_stride: int) -> list[dict]:
    states = payload.get("states", [])
    if not states:
        raise ValueError("trajectory JSON must contain a non-empty states list.")
    stride = max(1, int(frame_stride))
    states = states[::stride]
    if max_frames > 0:
        states = states[: int(max_frames)]
    return states


def frame_pose_tensors(frame: dict, *, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    dice = frame.get("dice", [])
    if not dice:
        raise ValueError("trajectory frame is missing dice states.")
    positions = torch.tensor([die["position"] for die in dice], dtype=torch.float32, device=device)
    quaternions = torch.tensor([die["quaternion_wxyz"] for die in dice], dtype=torch.float32, device=device)
    return positions, quaternions


def filter_asset(
    asset: RenderableGaussianAsset,
    *,
    foreground_threshold: float | None,
    opacity_threshold: float | None,
    recenter: bool,
) -> tuple[RenderableGaussianAsset, dict]:
    mask = torch.ones((asset.num_gaussians,), dtype=torch.bool, device=asset.xyz.device)
    if foreground_threshold is not None:
        foreground_score = torch.sigmoid(asset.foreground_logit[:, 0])
        mask &= foreground_score >= float(foreground_threshold)
    if opacity_threshold is not None:
        mask &= asset.activated_opacity[:, 0] >= float(opacity_threshold)
    if not bool(mask.any()):
        raise ValueError("Gaussian filtering removed every primitive.")

    xyz = asset.xyz[mask]
    bbox_min = torch.min(xyz, dim=0).values
    bbox_max = torch.max(xyz, dim=0).values
    center = 0.5 * (bbox_min + bbox_max)
    if recenter:
        xyz = xyz - center

    filtered = RenderableGaussianAsset(
        xyz=xyz,
        features_dc=asset.features_dc[mask],
        features_rest=asset.features_rest[mask],
        opacity=asset.opacity[mask],
        scaling=asset.scaling[mask],
        rotation=asset.rotation[mask],
        features_geo=asset.features_geo[mask],
        foreground_logit=asset.foreground_logit[mask],
        object_ids=asset.object_ids[mask],
        sh_degree=asset.sh_degree,
    )
    info = {
        "kept_gaussians": int(mask.sum().detach().cpu().item()),
        "total_gaussians": int(asset.num_gaussians),
        "bbox_min": [float(v) for v in bbox_min.detach().cpu().tolist()],
        "bbox_max": [float(v) for v in bbox_max.detach().cpu().tolist()],
        "bbox_center": [float(v) for v in center.detach().cpu().tolist()],
        "recenter_asset": bool(recenter),
    }
    return filtered, info


def infer_scale_from_asset(asset: RenderableGaussianAsset, *, half_extent: float) -> float:
    bbox_min = torch.min(asset.xyz, dim=0).values
    bbox_max = torch.max(asset.xyz, dim=0).values
    diameter = torch.max(bbox_max - bbox_min)
    return float((float(half_extent) * 2.0 / torch.clamp(diameter, min=1e-12)).detach().cpu().item())


def infer_stage1_to_mujoco_scale(stage1_ply: Path, *, half_extent: float, device: str) -> float:
    centers, radii = load_gaussian_collision_primitives_from_ply(
        stage1_ply,
        radius_scale=1.0,
        recenter=True,
        dtype=torch.float32,
        device=torch.device(device),
    )
    bbox_min = torch.min(centers - radii.unsqueeze(-1), dim=0).values
    bbox_max = torch.max(centers + radii.unsqueeze(-1), dim=0).values
    diameter = torch.max(bbox_max - bbox_min)
    return float((float(half_extent) * 2.0 / torch.clamp(diameter, min=1e-12)).detach().cpu().item())


def render_frame(base_asset, frame: dict, camera: MiniCam, background: torch.Tensor, scale_multiplier: float) -> np.ndarray:
    positions, quaternions = frame_pose_tensors(frame, device=str(base_asset.xyz.device))
    scene_asset = instantiate_rigid_gaussian_scene(
        base_asset,
        positions,
        quaternions,
        scale_multiplier=float(scale_multiplier),
    )
    gaussians = GaussianModel(scene_asset.sh_degree)
    copy_asset_to_gaussian_model(scene_asset, gaussians)
    output = gs_render(camera, gaussians, PipelineParams(), background, separate_sh=False)["render"]
    return (output.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        message = "Gaussian trajectory renderer requires CUDA because gaussian_renderer.render allocates CUDA tensors."
        if args.allow_cpu_skip:
            print(f"[SKIP] {message}")
            return
        raise RuntimeError(message)

    device = "cuda"
    import imageio.v2 as imageio

    output_dir = args.output_dir.resolve()
    rgb_dir = output_dir / "gaussian_rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    payload = read_json(args.trajectory.resolve())
    frames = select_frames(payload, max_frames=int(args.max_frames), frame_stride=int(args.frame_stride))
    base_asset = load_renderable_gaussian_asset(args.stage1_ply.resolve(), device=device)
    base_asset, filter_info = filter_asset(
        base_asset,
        foreground_threshold=args.foreground_threshold,
        opacity_threshold=args.opacity_threshold,
        recenter=bool(args.recenter_asset),
    )
    scale_multiplier = float(args.scale_multiplier)
    if bool(args.auto_scale_to_trajectory_half_extent):
        scale_multiplier *= infer_scale_from_asset(base_asset, half_extent=float(payload.get("half_extent", 0.055)))
    camera = make_camera(args, device=device)
    bg_value = 1.0 if args.white_background else 0.0
    background = torch.full((3,), bg_value, dtype=torch.float32, device=device)
    rendered_frames = []
    frame_records = []
    for local_idx, frame in enumerate(frames):
        image = render_frame(base_asset, frame, camera, background, scale_multiplier)
        frame_index = int(frame.get("frame_index", local_idx))
        path = rgb_dir / f"{frame_index:06d}.png"
        Image.fromarray(image).save(path)
        rendered_frames.append(image)
        frame_records.append({"frame_index": frame_index, "path": str(path)})
    gif_path = output_dir / "stage2_gaussian_trajectory.gif"
    imageio.mimsave(gif_path, rendered_frames, fps=max(1, int(args.fps)))
    manifest = {
        "stage1_ply": str(args.stage1_ply.resolve()),
        "trajectory": str(args.trajectory.resolve()),
        "rgb_dir": str(rgb_dir),
        "gif": str(gif_path),
        "frames": frame_records,
        "num_frames": len(frame_records),
        "num_base_gaussians": int(base_asset.num_gaussians),
        "filter": filter_info,
        "scale_multiplier": float(scale_multiplier),
        "requested_scale_multiplier": float(args.scale_multiplier),
        "auto_scale_to_trajectory_half_extent": bool(args.auto_scale_to_trajectory_half_extent),
    }
    write_json(output_dir / "stage2_gaussian_trajectory_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
