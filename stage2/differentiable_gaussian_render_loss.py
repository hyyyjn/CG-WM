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

REPO_ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = REPO_ROOT
for path in (str(REPO_ROOT), str(GS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from stage2.renderable_gaussian_asset import (  # noqa: E402
    copy_asset_to_gaussian_model,
    instantiate_rigid_gaussian_scene,
    load_renderable_gaussian_asset,
)
from stage1.gaussian_renderer import GaussianModel  # noqa: E402
from stage1.gaussian_renderer import render as gs_render  # noqa: E402
from stage1.scene.cameras import MiniCam  # noqa: E402
from stage1.utils.graphics_utils import getProjectionMatrix, getWorld2View2  # noqa: E402
from stage1.utils.loss_utils import ssim  # noqa: E402


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
    collision_radius_to_gaussian_scale: float = 0.5
    loss: str = "l1"
    ssim_weight: float = 0.2
    loftr_weight: float = 0.1
    loftr_pretrained: str = "outdoor"
    loftr_confidence_threshold: float = 0.2
    loftr_max_matches: int = 1024
    loftr_min_matches: int = 8
    loftr_patch_radius: int = 2


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


def load_mask_sequence(
    mask_dir: Path,
    frame_indices: list[int],
    *,
    width: int,
    height: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    frames = []
    for frame_index in frame_indices:
        path = mask_dir / f"{int(frame_index):06d}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing mask supervision frame: {path}")
        image = Image.open(path).convert("L").resize((int(width), int(height)), Image.Resampling.NEAREST)
        frames.append(torch.as_tensor(np.asarray(image, dtype=np.float32) / 255.0, dtype=dtype, device=device))
    return torch.stack(frames, dim=0).unsqueeze(1)


def gaussian_image_loss(
    rendered: torch.Tensor,
    targets: torch.Tensor,
    *,
    config: GaussianRenderLossConfig,
    masks: torch.Tensor | None = None,
    background: torch.Tensor | None = None,
    loftr_loss=None,
) -> tuple[torch.Tensor, dict]:
    if masks is not None:
        if background is None:
            raise ValueError("masked image loss requires a background color")
        bg = background.reshape(1, 3, 1, 1)
        rendered = rendered * masks + bg * (1.0 - masks)
        targets = targets * masks + bg * (1.0 - masks)
    l1_loss = F.l1_loss(rendered, targets)
    ssim_loss = 1.0 - ssim(rendered, targets)
    if config.loss == "mse":
        loss = F.mse_loss(rendered, targets)
    elif config.loss == "l1":
        loss = l1_loss
    elif config.loss == "l1_ssim":
        weight = min(max(float(config.ssim_weight), 0.0), 1.0)
        loss = (1.0 - weight) * l1_loss + weight * ssim_loss
    elif config.loss == "l1_loftr":
        if loftr_loss is None:
            raise ValueError("l1_loftr requires an initialized LoFTR loss module.")
        feature_loss, feature_diagnostics = loftr_loss(rendered, targets, masks=masks)
        loss = l1_loss + float(config.loftr_weight) * feature_loss
    else:
        raise ValueError(f"Unsupported Gaussian render loss: {config.loss!r}")
    diagnostics = {
        "gaussian_render_loss": float(loss.detach().cpu().item()),
        "gaussian_render_l1": float(l1_loss.detach().cpu().item()),
        "gaussian_render_ssim": float((1.0 - ssim_loss).detach().cpu().item()),
        "gaussian_render_masked": masks is not None,
    }
    if config.loss == "l1_loftr":
        diagnostics.update(feature_diagnostics)
        diagnostics["loftr_weight"] = float(config.loftr_weight)
    return loss, diagnostics


class Stage2GaussianRenderLoss:
    """Render Stage 2 rigid poses and compare them to RGB supervision frames."""

    def __init__(
        self,
        *,
        stage1_ply: Path,
        gt_rgb_dir: Path,
        frame_indices: list[int],
        config: GaussianRenderLossConfig,
        gaussian_indices: list[int] | None = None,
        gt_mask_dir: Path | None = None,
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
        if gaussian_indices is not None:
            self.base_asset = self.base_asset.index_select(
                torch.as_tensor(gaussian_indices, dtype=torch.long, device=device)
            )
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
        self.masks = (
            None
            if gt_mask_dir is None
            else load_mask_sequence(
                gt_mask_dir,
                frame_indices,
                width=int(config.image_width),
                height=int(config.image_height),
                dtype=dtype,
                device=device,
            )
        )
        self.loftr_loss = None
        if config.loss == "l1_loftr":
            from stage2.differentiable_loftr_loss import (
                LoFTRCorrespondenceLoss,
            )
            self.loftr_loss = LoFTRCorrespondenceLoss(
                pretrained=str(config.loftr_pretrained),
                confidence_threshold=float(config.loftr_confidence_threshold),
                max_matches=int(config.loftr_max_matches),
                min_matches=int(config.loftr_min_matches),
                patch_radius=int(config.loftr_patch_radius),
            ).to(device)

    def render_frame(
        self,
        positions: torch.Tensor,
        quaternions_wxyz: torch.Tensor,
        *,
        geometry_centers: torch.Tensor | None = None,
        geometry_radii: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if positions.ndim == 1:
            positions = positions.reshape(1, 3)
        if quaternions_wxyz.ndim == 1:
            quaternions_wxyz = quaternions_wxyz.reshape(1, 4)
        asset = self.base_asset
        if geometry_centers is not None or geometry_radii is not None:
            if geometry_centers is None or geometry_radii is None:
                raise ValueError("geometry_centers and geometry_radii must be provided together")
            asset = asset.with_spherical_geometry(
                geometry_centers,
                geometry_radii,
                radius_to_scale=float(self.config.collision_radius_to_gaussian_scale),
            )
        scene_asset = instantiate_rigid_gaussian_scene(
            asset,
            positions,
            quaternions_wxyz,
            scale_multiplier=float(self.config.scale_multiplier),
        )
        gaussians = GaussianModel(scene_asset.sh_degree)
        copy_asset_to_gaussian_model(scene_asset, gaussians)
        return gs_render(self.camera, gaussians, PipelineParams(), self.background, separate_sh=False)["render"]

    def render_sequence(
        self,
        positions: torch.Tensor,
        quaternions_wxyz: torch.Tensor,
        *,
        geometry_centers: torch.Tensor | None = None,
        geometry_radii: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frames = [
            self.render_frame(
                positions[idx],
                quaternions_wxyz[idx],
                geometry_centers=geometry_centers,
                geometry_radii=geometry_radii,
            )
            for idx in range(int(positions.shape[0]))
        ]
        return torch.stack(frames, dim=0)

    def __call__(
        self,
        positions: torch.Tensor,
        quaternions_wxyz: torch.Tensor,
        *,
        geometry_centers: torch.Tensor | None = None,
        geometry_radii: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        rendered = self.render_sequence(
            positions,
            quaternions_wxyz,
            geometry_centers=geometry_centers,
            geometry_radii=geometry_radii,
        )
        targets = self.targets[: rendered.shape[0]]
        masks = None if self.masks is None else self.masks[: rendered.shape[0]]
        loss, image_diagnostics = gaussian_image_loss(
            rendered,
            targets,
            config=self.config,
            masks=masks,
            background=self.background,
            loftr_loss=self.loftr_loss,
        )
        diagnostics = {
            **image_diagnostics,
            "gaussian_render_frames": int(rendered.shape[0]),
            "gaussian_render_width": int(self.config.image_width),
            "gaussian_render_height": int(self.config.image_height),
            "gaussian_render_scale_multiplier": float(self.config.scale_multiplier),
        }
        return loss, diagnostics


class MultiViewStage2GaussianRenderLoss:
    """Average image supervision from multiple calibrated fixed-camera views."""

    def __init__(self, views: list[Stage2GaussianRenderLoss]) -> None:
        if not views:
            raise ValueError("at least one Gaussian render-loss view is required")
        self.views = views

    def __call__(
        self,
        positions: torch.Tensor,
        quaternions_wxyz: torch.Tensor,
        *,
        geometry_centers: torch.Tensor | None = None,
        geometry_radii: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        losses, view_diagnostics = [], []
        for view in self.views:
            loss, diagnostics = view(
                positions,
                quaternions_wxyz,
                geometry_centers=geometry_centers,
                geometry_radii=geometry_radii,
            )
            losses.append(loss)
            view_diagnostics.append(diagnostics)
        loss = torch.stack(losses).mean()
        return loss, {
            "gaussian_render_loss": float(loss.detach().cpu().item()),
            "gaussian_render_num_views": len(self.views),
            "gaussian_render_views": view_diagnostics,
        }
