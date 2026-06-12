"""Differentiable Stage 2 image-space losses through the Gaussian renderer."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


@dataclass(frozen=True)
class GaussianRenderLossConfig:
    image_width: int = 160
    image_height: int = 120
    cam_distance: float = 1.12
    cam_height: float = 0.66
    cam_fovy_deg: float = 40.0
    white_background: bool = False
    scale_multiplier: float = 1.0
    loss: str = "l1"


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


def make_mujoco_cam0_minicam(config: GaussianRenderLossConfig, *, device: torch.device | str) -> MiniCam:
    c2w, fovx, fovy = mujoco_cam0_c2w_fov(
        float(config.cam_distance),
        float(config.cam_height),
        float(config.cam_fovy_deg),
        int(config.image_width),
        int(config.image_height),
    )
    R, T = c2w_to_3dgs_rt(c2w)
    world_view = torch.tensor(getWorld2View2(R, T), dtype=torch.float32).T.to(device)
    proj = getProjectionMatrix(znear=0.01, zfar=200.0, fovX=fovx, fovY=fovy).T.to(device)
    full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return MiniCam(
        int(config.image_width),
        int(config.image_height),
        fovy,
        fovx,
        0.01,
        200.0,
        world_view,
        full_proj,
    )


def load_rgb_sequence(
    rgb_dir: Path,
    frame_indices: list[int],
    *,
    width: int,
    height: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    frames = []
    if not frame_indices:
        return torch.empty((0, 3, int(height), int(width)), dtype=dtype, device=device)
    for frame_index in frame_indices:
        path = rgb_dir / f"{int(frame_index):06d}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing RGB supervision frame: {path}")
        image = Image.open(path).convert("RGB").resize((int(width), int(height)), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        frames.append(torch.as_tensor(array, dtype=dtype, device=device).permute(2, 0, 1))
    return torch.stack(frames, dim=0)


class Stage2GaussianRenderLoss:
    """Render Stage 2 rigid poses and compare them to RGB supervision frames."""

    def __init__(
        self,
        *,
        stage1_ply: Path,
        gt_rgb_dir: Path,
        frame_indices: list[int],
        config: GaussianRenderLossConfig,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cuda",
    ) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("Stage2GaussianRenderLoss requires a CUDA device.")
        if not torch.cuda.is_available():
            raise RuntimeError("Stage2GaussianRenderLoss requires CUDA because gaussian_renderer.render uses CUDA tensors.")
        self.config = config
        self.device = device
        self.base_asset = load_renderable_gaussian_asset(stage1_ply, dtype=dtype, device=device)
        self.camera = make_mujoco_cam0_minicam(config, device=device)
        bg_value = 1.0 if bool(config.white_background) else 0.0
        self.background = torch.full((3,), bg_value, dtype=dtype, device=device)
        self.targets = load_rgb_sequence(
            gt_rgb_dir,
            frame_indices,
            width=int(config.image_width),
            height=int(config.image_height),
            dtype=dtype,
            device=device,
        )

    def render_frame(self, positions: torch.Tensor, quaternions_wxyz: torch.Tensor) -> torch.Tensor:
        scene_asset = instantiate_rigid_gaussian_scene(
            self.base_asset,
            positions,
            quaternions_wxyz,
            scale_multiplier=float(self.config.scale_multiplier),
        )
        gaussians = GaussianModel(scene_asset.sh_degree)
        copy_asset_to_gaussian_model(scene_asset, gaussians)
        return gs_render(self.camera, gaussians, PipelineParams(), self.background, separate_sh=False)["render"]

    def render_sequence(self, positions: torch.Tensor, quaternions_wxyz: torch.Tensor) -> torch.Tensor:
        frames = [
            self.render_frame(positions[idx], quaternions_wxyz[idx])
            for idx in range(int(positions.shape[0]))
        ]
        return torch.stack(frames, dim=0)

    def __call__(self, positions: torch.Tensor, quaternions_wxyz: torch.Tensor) -> tuple[torch.Tensor, dict]:
        rendered = self.render_sequence(positions, quaternions_wxyz)
        targets = self.targets[: rendered.shape[0]]
        if self.config.loss == "mse":
            loss = F.mse_loss(rendered, targets)
        elif self.config.loss == "l1":
            loss = F.l1_loss(rendered, targets)
        else:
            raise ValueError(f"Unsupported Gaussian render loss: {self.config.loss!r}")
        diagnostics = {
            "gaussian_render_loss": float(loss.detach().cpu().item()),
            "gaussian_render_frames": int(rendered.shape[0]),
            "gaussian_render_width": int(self.config.image_width),
            "gaussian_render_height": int(self.config.image_height),
            "gaussian_render_scale_multiplier": float(self.config.scale_multiplier),
        }
        return loss, diagnostics
