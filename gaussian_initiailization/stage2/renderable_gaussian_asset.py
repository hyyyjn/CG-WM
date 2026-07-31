"""Renderable Stage 1 Gaussian assets for Stage 2 rigid rollout rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData


@dataclass(frozen=True)
class RenderableGaussianAsset:
    """Raw Gaussian Splatting tensors loaded from a Stage 1 point_cloud.ply.

    The tensors intentionally mirror the fields used by ``GaussianModel`` while
    staying lightweight and device-agnostic, so Stage 2 code can rigid-transform
    them before handing them to a renderer.
    """

    xyz: torch.Tensor
    features_dc: torch.Tensor
    features_rest: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    features_geo: torch.Tensor
    foreground_logit: torch.Tensor
    object_ids: torch.Tensor
    sh_degree: int

    def to(self, *, dtype=None, device=None) -> "RenderableGaussianAsset":
        return RenderableGaussianAsset(
            xyz=self.xyz.to(dtype=dtype, device=device),
            features_dc=self.features_dc.to(dtype=dtype, device=device),
            features_rest=self.features_rest.to(dtype=dtype, device=device),
            opacity=self.opacity.to(dtype=dtype, device=device),
            scaling=self.scaling.to(dtype=dtype, device=device),
            rotation=self.rotation.to(dtype=dtype, device=device),
            features_geo=self.features_geo.to(dtype=dtype, device=device),
            foreground_logit=self.foreground_logit.to(dtype=dtype, device=device),
            object_ids=self.object_ids.to(device=device),
            sh_degree=int(self.sh_degree),
        )

    @property
    def num_gaussians(self) -> int:
        return int(self.xyz.shape[0])

    def index_select(self, indices: torch.Tensor) -> "RenderableGaussianAsset":
        indices = indices.to(dtype=torch.long, device=self.xyz.device)
        return RenderableGaussianAsset(
            xyz=self.xyz[indices],
            features_dc=self.features_dc[indices],
            features_rest=self.features_rest[indices],
            opacity=self.opacity[indices],
            scaling=self.scaling[indices],
            rotation=self.rotation[indices],
            features_geo=self.features_geo[indices],
            foreground_logit=self.foreground_logit[indices],
            object_ids=self.object_ids[indices],
            sh_degree=self.sh_degree,
        )

    def with_spherical_geometry(
        self,
        centers: torch.Tensor,
        collision_radii: torch.Tensor,
        *,
        radius_to_scale: float = 0.5,
    ) -> "RenderableGaussianAsset":
        """Replace geometry using paper spherical primitives.

        ContactGaussian-WM defines collision radius ``r = 2s`` for Gaussian
        scale ``s``. ``radius_to_scale=1`` reproduces legacy checkpoints where
        collision radius and renderer scale were treated as equal.
        """
        if centers.shape != self.xyz.shape or collision_radii.shape != (self.num_gaussians,):
            raise ValueError("geometry override must match the selected renderable Gaussian count")
        gaussian_scales = collision_radii * float(radius_to_scale)
        log_scales = torch.log(torch.clamp(gaussian_scales, min=1e-8)).unsqueeze(-1).expand(-1, 3)
        identity = torch.zeros_like(self.rotation)
        identity[:, 0] = 1.0
        return RenderableGaussianAsset(
            xyz=centers,
            features_dc=self.features_dc,
            features_rest=self.features_rest,
            opacity=self.opacity,
            scaling=log_scales,
            rotation=identity,
            features_geo=self.features_geo,
            foreground_logit=self.foreground_logit,
            object_ids=self.object_ids,
            sh_degree=self.sh_degree,
        )

    @property
    def activated_scaling(self) -> torch.Tensor:
        return torch.exp(self.scaling)

    @property
    def activated_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity)

    @property
    def normalized_rotation(self) -> torch.Tensor:
        return F.normalize(self.rotation, dim=-1)


def _property_names(vertex) -> set[str]:
    return set(vertex.data.dtype.names or ())


def _stack_properties(vertex, names: list[str], *, default_width: int = 0) -> np.ndarray:
    if not names:
        return np.zeros((len(vertex.data), default_width), dtype=np.float32)
    return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=1)


def infer_sh_degree(feature_rest_width: int) -> int:
    if feature_rest_width == 0:
        return 0
    coeffs_per_channel = feature_rest_width // 3
    degree_plus_one = int(round((coeffs_per_channel + 1) ** 0.5))
    if 3 * (degree_plus_one * degree_plus_one - 1) != feature_rest_width:
        raise ValueError(f"Cannot infer SH degree from {feature_rest_width} f_rest values.")
    return degree_plus_one - 1


def load_renderable_gaussian_asset(
    path: str | Path,
    *,
    dtype=torch.float32,
    device=None,
) -> RenderableGaussianAsset:
    path = Path(path)
    plydata = PlyData.read(path)
    vertex = plydata.elements[0]
    names = _property_names(vertex)
    required = {"x", "y", "z"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{path} is missing required Gaussian PLY properties: {missing}")

    xyz_np = np.stack(
        (
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ),
        axis=1,
    )
    f_dc_names = [f"f_dc_{idx}" for idx in range(3)]
    if all(name in names for name in f_dc_names):
        features_dc_np = _stack_properties(vertex, f_dc_names).reshape(-1, 1, 3)
    else:
        features_dc_np = np.zeros((xyz_np.shape[0], 1, 3), dtype=np.float32)

    f_rest_names = sorted(
        [name for name in names if name.startswith("f_rest_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    features_rest_flat = _stack_properties(vertex, f_rest_names)
    sh_degree = infer_sh_degree(features_rest_flat.shape[1])
    if features_rest_flat.shape[1] > 0:
        features_rest_np = features_rest_flat.reshape(xyz_np.shape[0], -1, 3)
    else:
        features_rest_np = np.zeros((xyz_np.shape[0], 0, 3), dtype=np.float32)

    f_geo_names = sorted(
        [name for name in names if name.startswith("f_geo_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    features_geo_np = _stack_properties(vertex, f_geo_names)
    foreground_np = (
        np.asarray(vertex["foreground_logit"], dtype=np.float32)[:, None]
        if "foreground_logit" in names
        else np.zeros((xyz_np.shape[0], 1), dtype=np.float32)
    )
    object_ids_np = (
        np.asarray(vertex["object_id"], dtype=np.int32)
        if "object_id" in names
        else np.zeros((xyz_np.shape[0],), dtype=np.int32)
    )

    scale_names = sorted(
        [name for name in names if name.startswith("scale_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    scaling_np = _stack_properties(vertex, scale_names)
    if scaling_np.shape[1] == 0:
        scaling_np = np.full((xyz_np.shape[0], 3), np.log(0.01), dtype=np.float32)
    elif scaling_np.shape[1] == 1:
        scaling_np = np.repeat(scaling_np, 3, axis=1)

    rot_names = sorted(
        [name for name in names if name.startswith("rot")],
        key=lambda name: int(name.split("_")[-1]),
    )
    rotation_np = _stack_properties(vertex, rot_names)
    if rotation_np.shape[1] == 0:
        rotation_np = np.zeros((xyz_np.shape[0], 4), dtype=np.float32)
        rotation_np[:, 0] = 1.0

    asset = RenderableGaussianAsset(
        xyz=torch.as_tensor(xyz_np, dtype=dtype, device=device),
        features_dc=torch.as_tensor(features_dc_np, dtype=dtype, device=device),
        features_rest=torch.as_tensor(features_rest_np, dtype=dtype, device=device),
        opacity=torch.as_tensor(
            (
                np.asarray(vertex["opacity"], dtype=np.float32)[:, None]
                if "opacity" in names
                else np.full((xyz_np.shape[0], 1), 4.0, dtype=np.float32)
            ),
            dtype=dtype,
            device=device,
        ),
        scaling=torch.as_tensor(scaling_np, dtype=dtype, device=device),
        rotation=torch.as_tensor(rotation_np, dtype=dtype, device=device),
        features_geo=torch.as_tensor(features_geo_np, dtype=dtype, device=device),
        foreground_logit=torch.as_tensor(foreground_np, dtype=dtype, device=device),
        object_ids=torch.as_tensor(object_ids_np, dtype=torch.int32, device=device),
        sh_degree=sh_degree,
    )
    return asset


def quat_mul_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = lhs.unbind(dim=-1)
    w2, x2, y2, z2 = rhs.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_to_matrix_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = F.normalize(quaternion, dim=-1)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)), dim=-1),
            torch.stack((2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)), dim=-1),
            torch.stack((2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)), dim=-1),
        ),
        dim=-2,
    )


def rigid_transform_asset(
    asset: RenderableGaussianAsset,
    position: torch.Tensor,
    quaternion_wxyz: torch.Tensor,
    *,
    scale_multiplier: torch.Tensor | float = 1.0,
) -> RenderableGaussianAsset:
    rotation_matrix = quat_to_matrix_wxyz(quaternion_wxyz)
    scale = torch.as_tensor(scale_multiplier, dtype=asset.scaling.dtype, device=asset.scaling.device)
    scaled_xyz = asset.xyz * scale
    world_xyz = scaled_xyz @ rotation_matrix.transpose(-1, -2) + position.unsqueeze(-2)
    pose_quat = F.normalize(quaternion_wxyz, dim=-1)
    while pose_quat.ndim < asset.rotation.ndim:
        pose_quat = pose_quat.unsqueeze(-2)
    world_rotation = quat_mul_wxyz(pose_quat.expand_as(asset.rotation), asset.normalized_rotation)
    transformed_scaling = asset.scaling + torch.log(torch.clamp(scale, min=1e-8))
    return RenderableGaussianAsset(
        xyz=world_xyz,
        features_dc=asset.features_dc,
        features_rest=asset.features_rest,
        opacity=asset.opacity,
        scaling=transformed_scaling,
        rotation=world_rotation,
        features_geo=asset.features_geo,
        foreground_logit=asset.foreground_logit,
        object_ids=asset.object_ids,
        sh_degree=asset.sh_degree,
    )


def instantiate_rigid_gaussian_scene(
    asset: RenderableGaussianAsset,
    positions: torch.Tensor,
    quaternions_wxyz: torch.Tensor,
    *,
    scale_multiplier: torch.Tensor | float = 1.0,
) -> RenderableGaussianAsset:
    instances = [
        rigid_transform_asset(asset, positions[idx], quaternions_wxyz[idx], scale_multiplier=scale_multiplier)
        for idx in range(int(positions.shape[0]))
    ]
    return RenderableGaussianAsset(
        xyz=torch.cat([item.xyz for item in instances], dim=0),
        features_dc=torch.cat([item.features_dc for item in instances], dim=0),
        features_rest=torch.cat([item.features_rest for item in instances], dim=0),
        opacity=torch.cat([item.opacity for item in instances], dim=0),
        scaling=torch.cat([item.scaling for item in instances], dim=0),
        rotation=torch.cat([item.rotation for item in instances], dim=0),
        features_geo=torch.cat([item.features_geo for item in instances], dim=0),
        foreground_logit=torch.cat([item.foreground_logit for item in instances], dim=0),
        object_ids=torch.cat([torch.full_like(item.object_ids, idx) for idx, item in enumerate(instances)], dim=0),
        sh_degree=asset.sh_degree,
    )


def copy_asset_to_gaussian_model(asset: RenderableGaussianAsset, gaussian_model) -> object:
    """Populate a GaussianModel-like object from a renderable asset.

    This adapter is intentionally tiny and avoids constructing optimizer state.
    It is meant for render-only smoke tests and later differentiable render loss
    wrappers.
    """

    device = asset.xyz.device
    gaussian_model.active_sh_degree = int(asset.sh_degree)
    gaussian_model.max_sh_degree = int(asset.sh_degree)
    gaussian_model._xyz = asset.xyz
    gaussian_model._features_dc = asset.features_dc.contiguous()
    gaussian_model._features_rest = asset.features_rest.contiguous()
    gaussian_model._opacity = asset.opacity
    gaussian_model._scaling = asset.scaling
    gaussian_model._rotation = asset.rotation
    gaussian_model._features_geo = asset.features_geo
    gaussian_model._foreground_logit = asset.foreground_logit
    gaussian_model._object_ids = asset.object_ids.to(device=device)
    gaussian_model.max_radii2D = torch.zeros((asset.num_gaussians,), dtype=asset.xyz.dtype, device=device)
    return gaussian_model
