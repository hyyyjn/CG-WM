from __future__ import annotations

import argparse
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
    copy_asset_to_gaussian_model,
    instantiate_rigid_gaussian_scene,
    load_renderable_gaussian_asset,
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
    parser = argparse.ArgumentParser(description="Render one rigid Stage2 pose with a Stage1 Gaussian asset.")
    parser.add_argument("--stage1_ply", required=True, type=Path)
    parser.add_argument("--output_png", required=True, type=Path)
    parser.add_argument("--position", default="0,0,0.08")
    parser.add_argument("--quaternion_wxyz", default="1,0,0,0")
    parser.add_argument("--image_width", default=640, type=int)
    parser.add_argument("--image_height", default=480, type=int)
    parser.add_argument("--cam_distance", default=1.12, type=float)
    parser.add_argument("--cam_height", default=0.66, type=float)
    parser.add_argument("--cam_fovy_deg", default=40.0, type=float)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--scale_multiplier", default=1.0, type=float)
    parser.add_argument("--allow_cpu_skip", action="store_true")
    return parser.parse_args()


def parse_vec(text: str, *, length: int, label: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != length:
        raise ValueError(f"{label} must contain {length} comma-separated values, got {text!r}.")
    return values


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


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        message = "Gaussian rasterizer smoke requires CUDA because gaussian_renderer.render allocates CUDA tensors."
        if args.allow_cpu_skip:
            print(f"[SKIP] {message}")
            return
        raise RuntimeError(message)

    device = "cuda"
    position = torch.tensor(parse_vec(args.position, length=3, label="--position"), dtype=torch.float32, device=device)
    quaternion = torch.tensor(
        parse_vec(args.quaternion_wxyz, length=4, label="--quaternion_wxyz"),
        dtype=torch.float32,
        device=device,
    )
    base_asset = load_renderable_gaussian_asset(args.stage1_ply.resolve(), device=device)
    scene_asset = instantiate_rigid_gaussian_scene(
        base_asset,
        position.reshape(1, 3),
        quaternion.reshape(1, 4),
        scale_multiplier=float(args.scale_multiplier),
    )
    gaussians = GaussianModel(scene_asset.sh_degree)
    copy_asset_to_gaussian_model(scene_asset, gaussians)

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
    camera = MiniCam(int(args.image_width), int(args.image_height), fovy, fovx, 0.01, 200.0, world_view, full_proj)
    bg_value = 1.0 if args.white_background else 0.0
    background = torch.full((3,), bg_value, dtype=torch.float32, device=device)
    output = gs_render(camera, gaussians, PipelineParams(), background, separate_sh=False)["render"]
    image = (output.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.output_png)
    print(
        {
            "output_png": str(args.output_png.resolve()),
            "num_gaussians": int(scene_asset.num_gaussians),
            "sh_degree": int(scene_asset.sh_degree),
            "image_size": [int(args.image_width), int(args.image_height)],
        }
    )


if __name__ == "__main__":
    main()
