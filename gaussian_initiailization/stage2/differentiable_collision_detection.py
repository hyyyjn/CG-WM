"""Differentiable collision detection module for Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PlaneCollider:
    """A differentiable infinite plane collider.

    Signed distance is `dot(point, normal) - height`. A floor at z=0 therefore
    uses normal=(0, 0, 1), height=0.
    """

    normal: torch.Tensor
    height: float = 0.0

    @classmethod
    def floor(cls, height: float = 0.0, *, dtype=torch.float32, device=None) -> "PlaneCollider":
        return cls(torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device), float(height))

    def on_like(self, tensor: torch.Tensor) -> "PlaneCollider":
        return PlaneCollider(self.normal.to(dtype=tensor.dtype, device=tensor.device), self.height)

    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        collider = self.on_like(points)
        normal = collider.normal / torch.clamp(torch.linalg.norm(collider.normal), min=1e-12)
        return torch.sum(points * normal, dim=-1) - collider.height


@dataclass(frozen=True)
class ContactSamples:
    world_points: torch.Tensor
    signed_distances: torch.Tensor
    penetrations: torch.Tensor
    contact_weights: torch.Tensor
    collider_normal: torch.Tensor

    @property
    def min_signed_distance(self) -> torch.Tensor:
        return torch.min(self.signed_distances)

    @property
    def max_penetration(self) -> torch.Tensor:
        return torch.max(self.penetrations)


@dataclass(frozen=True)
class FloorQuerySphereContacts:
    floor_points: torch.Tensor
    signed_distances: torch.Tensor
    penetrations: torch.Tensor
    contact_weights: torch.Tensor
    contact_point: torch.Tensor
    collider_normal: torch.Tensor

    @property
    def min_signed_distance(self) -> torch.Tensor:
        return torch.min(self.signed_distances)

    @property
    def max_penetration(self) -> torch.Tensor:
        return torch.max(self.penetrations)


@dataclass(frozen=True)
class GaussianUnionContacts:
    """Result of evaluating query points against a union of spherical Gaussians.

    `signed_distances` follows the paper's ϕ(p): a LogSumExp smooth min over
    primitives with a sigmoid-blended inside-object penalty so deep penetration
    is clamped to ≈ -inside_penalty instead of leaking back toward zero.

    `surface_normals` is the analytic ∇ϕ_soft(p) / ‖∇ϕ_soft‖ per query point
    (a softmax-weighted blend of per-primitive outward directions). The
    `collider_normal` field is the contact-weight-aggregated single normal used
    by downstream rigid-body dynamics, oriented to match the environment's
    plane normal (paper III-D-1).
    """

    query_points: torch.Tensor
    signed_distances: torch.Tensor
    penetrations: torch.Tensor
    contact_weights: torch.Tensor
    surface_normals: torch.Tensor
    contact_point: torch.Tensor
    collider_normal: torch.Tensor

    @property
    def min_signed_distance(self) -> torch.Tensor:
        return torch.min(self.signed_distances)

    @property
    def max_penetration(self) -> torch.Tensor:
        return torch.max(self.penetrations)


def _as_3_tensor(values: Iterable[float], *, dtype=torch.float32, device=None) -> torch.Tensor:
    tensor = torch.as_tensor(list(values), dtype=dtype, device=device)
    if tensor.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got shape {tuple(tensor.shape)}.")
    return tensor


def make_box_query_points(
    half_extents: Iterable[float],
    *,
    bottom_only: bool = True,
    grid_resolution: int = 2,
    dtype=torch.float32,
    device=None,
) -> torch.Tensor:
    """Return local-space query points for a box collision proxy.

    For a floor collision smoke test, bottom-face query points are enough and
    make the contact interpretation easy to debug.
    """

    if grid_resolution < 2:
        raise ValueError("grid_resolution must be at least 2.")
    half_extents_tensor = _as_3_tensor(half_extents, dtype=dtype, device=device)
    hx, hy, hz = [float(v) for v in half_extents_tensor.detach().cpu().tolist()]
    xs = torch.linspace(-hx, hx, grid_resolution, dtype=dtype, device=device)
    ys = torch.linspace(-hy, hy, grid_resolution, dtype=dtype, device=device)
    zs = torch.tensor([-hz], dtype=dtype, device=device)
    if not bottom_only:
        zs = torch.linspace(-hz, hz, grid_resolution, dtype=dtype, device=device)

    points = []
    for x, y, z in product(xs, ys, zs):
        points.append(torch.stack((x, y, z)))
    return torch.stack(points, dim=0)


def make_sphere_query_points(
    radius: float,
    *,
    num_points: int = 64,
    include_center: bool = False,
    dtype=torch.float32,
    device=None,
) -> torch.Tensor:
    """Return local-space query points on a sphere using a Fibonacci lattice."""

    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if num_points < 4:
        raise ValueError("num_points must be at least 4.")

    indices = torch.arange(num_points, dtype=dtype, device=device)
    golden_angle = torch.tensor(2.399963229728653, dtype=dtype, device=device)
    z = 1.0 - (2.0 * indices + 1.0) / float(num_points)
    radial = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = golden_angle * indices
    points = torch.stack(
        (
            radial * torch.cos(theta),
            radial * torch.sin(theta),
            z,
        ),
        dim=-1,
    ) * float(radius)
    if include_center:
        center = torch.zeros((1, 3), dtype=dtype, device=device)
        points = torch.cat((center, points), dim=0)
    return points


def make_floor_disk_query_points(
    radius: float,
    *,
    num_rings: int = 5,
    num_angles: int = 24,
    dtype=torch.float32,
    device=None,
) -> torch.Tensor:
    """Return XY offsets for floor-side contact queries below a round object."""

    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if num_rings < 1:
        raise ValueError("num_rings must be at least 1.")
    if num_angles < 3:
        raise ValueError("num_angles must be at least 3.")

    offsets = [torch.zeros(2, dtype=dtype, device=device)]
    angles = torch.linspace(0.0, 2.0 * torch.pi, num_angles + 1, dtype=dtype, device=device)[:-1]
    for ring_idx in range(1, num_rings + 1):
        ring_radius = float(radius) * float(ring_idx) / float(num_rings)
        ring = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1) * ring_radius
        offsets.extend(point for point in ring)
    return torch.stack(offsets, dim=0)


def detect_sphere_floor_contacts(
    position: torch.Tensor,
    floor_query_offsets_xy: torch.Tensor,
    collider: PlaneCollider,
    *,
    radius: float,
    softness: float = 1e-3,
) -> FloorQuerySphereContacts:
    """Detect sphere/floor overlap using query points that live on the floor.

    The query points are sampled on the environment side. They are centered
    below the sphere's XY projection and evaluated against the sphere SDF.
    """

    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if softness <= 0.0:
        raise ValueError("softness must be positive.")
    if position.shape[-1] != 3:
        raise ValueError(f"position must end with 3 coordinates, got {tuple(position.shape)}.")
    if floor_query_offsets_xy.ndim != 2 or floor_query_offsets_xy.shape[-1] != 2:
        raise ValueError("floor_query_offsets_xy must have shape (N, 2).")

    offsets = floor_query_offsets_xy.to(dtype=position.dtype, device=position.device)
    collider = collider.on_like(position)
    normal = collider.normal / torch.clamp(torch.linalg.norm(collider.normal), min=1e-12)
    floor_z = collider.height
    floor_points = torch.cat(
        (
            position[..., :2].unsqueeze(-2) + offsets,
            torch.full((*offsets.shape[:-1], 1), float(floor_z), dtype=position.dtype, device=position.device),
        ),
        dim=-1,
    )
    signed_distances = torch.linalg.norm(floor_points - position.unsqueeze(-2), dim=-1) - float(radius)
    penetrations = F.softplus(-signed_distances / softness) * softness
    contact_weights = torch.sigmoid(-signed_distances / softness)
    weight_sum = torch.clamp(torch.sum(contact_weights, dim=-1, keepdim=True), min=1e-12)
    contact_point = torch.sum(floor_points * contact_weights.unsqueeze(-1), dim=-2) / weight_sum
    return FloorQuerySphereContacts(
        floor_points=floor_points,
        signed_distances=signed_distances,
        penetrations=penetrations,
        contact_weights=contact_weights,
        contact_point=contact_point,
        collider_normal=normal,
    )


def detect_gaussian_union_contacts(
    query_points: torch.Tensor,
    gaussian_centers: torch.Tensor,
    gaussian_radii: torch.Tensor,
    collider_normal: torch.Tensor,
    *,
    softness: float = 1e-3,
    smooth_min_temperature: float = 2e-2,
    inside_penalty: float = 0.02,
    inside_sharpness: float = 50.0,
) -> GaussianUnionContacts:
    """ContactGaussian-WM differentiable union-of-spheres SDF (paper III-D-1).

    Implements three stages from the paper:

    1. ``ϕ_soft(p) = -1/β · log Σ_i exp(-β·(‖p-c_i‖ - r_i))``
       (LogSumExp smooth min over primitive distances), with
       ``β = 1 / smooth_min_temperature``. As ``β`` grows it tightens onto the
       hard min ``min_i (‖p-c_i‖ - r_i)``.
    2. ``ϕ(p) ≈ σ(γ·ϕ_soft)·ϕ_soft + (1-σ(γ·ϕ_soft))·(-δ)``
       (sigmoid-blended inside-object penalty, paper III-D-1, Fig. 8).
       Deep penetration is clamped to ``-inside_penalty`` so the dynamics see a
       meaningful, bounded SDF inside the object instead of the LSE artefact
       described in Appendix B.
    3. Surface normal: the closed-form gradient of the LSE is
       ``∇ϕ_soft(p) = Σ_i w_i · (p - c_i)/‖p - c_i‖`` where ``w_i`` is the same
       softmax used in the LSE. The sigmoid blend only rescales magnitude, not
       direction, so this is also (up to a positive scalar) ``∇ϕ`` itself
       (paper III-D-1).
    """

    if softness <= 0.0:
        raise ValueError("softness must be positive.")
    if smooth_min_temperature <= 0.0:
        raise ValueError("smooth_min_temperature must be positive.")
    if inside_penalty <= 0.0:
        raise ValueError("inside_penalty must be positive.")
    if inside_sharpness <= 0.0:
        raise ValueError("inside_sharpness must be positive.")
    if query_points.ndim != 2 or query_points.shape[-1] != 3:
        raise ValueError("query_points must have shape (Q, 3).")
    if gaussian_centers.ndim != 2 or gaussian_centers.shape[-1] != 3:
        raise ValueError("gaussian_centers must have shape (G, 3).")
    if gaussian_radii.ndim != 1 or gaussian_radii.shape[0] != gaussian_centers.shape[0]:
        raise ValueError("gaussian_radii must have shape (G,).")

    centers = gaussian_centers.to(dtype=query_points.dtype, device=query_points.device)
    radii = gaussian_radii.to(dtype=query_points.dtype, device=query_points.device)
    plane_normal = collider_normal.to(dtype=query_points.dtype, device=query_points.device)
    plane_normal = plane_normal / torch.clamp(torch.linalg.norm(plane_normal), min=1e-12)

    offsets = query_points.unsqueeze(1) - centers.unsqueeze(0)
    center_distances = torch.linalg.norm(offsets, dim=-1)
    primitive_distances = center_distances - radii.unsqueeze(0)  # (Q, G)

    beta = 1.0 / smooth_min_temperature
    phi_soft = -smooth_min_temperature * torch.logsumexp(-beta * primitive_distances, dim=-1)
    primitive_weights = torch.softmax(-primitive_distances / smooth_min_temperature, dim=-1)

    sigma_blend = torch.sigmoid(inside_sharpness * phi_soft)
    signed_distances = sigma_blend * phi_soft + (1.0 - sigma_blend) * (-inside_penalty)

    direction_per_prim = offsets / torch.clamp(center_distances.unsqueeze(-1), min=1e-9)
    surface_normals = torch.sum(primitive_weights.unsqueeze(-1) * direction_per_prim, dim=1)
    surface_normals = surface_normals / torch.clamp(
        torch.linalg.norm(surface_normals, dim=-1, keepdim=True), min=1e-12
    )

    penetrations = F.softplus(-signed_distances / softness) * softness
    contact_weights = torch.sigmoid(-signed_distances / softness)
    weight_sum = torch.clamp(torch.sum(contact_weights, dim=-1, keepdim=True), min=1e-12)
    contact_point = torch.sum(query_points * contact_weights.unsqueeze(-1), dim=0) / weight_sum.squeeze(0)
    contact_normal = torch.sum(surface_normals * contact_weights.unsqueeze(-1), dim=0) / weight_sum.squeeze(0)
    contact_normal = contact_normal / torch.clamp(torch.linalg.norm(contact_normal), min=1e-12)
    contact_normal = torch.where(
        torch.sum(contact_normal * plane_normal) < 0.0,
        -contact_normal,
        contact_normal,
    )

    return GaussianUnionContacts(
        query_points=query_points,
        signed_distances=signed_distances,
        penetrations=penetrations,
        contact_weights=contact_weights,
        surface_normals=surface_normals,
        contact_point=contact_point,
        collider_normal=contact_normal,
    )


def load_gaussian_collision_primitives_from_ply(
    path: str | Path,
    *,
    radius_scale: float = 1.0,
    min_radius: float = 1e-4,
    recenter: bool = True,
    dtype=torch.float32,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load spherical collision primitives from a Stage 1 3DGS PLY.

    Stage 1 stores Gaussian centers as `x/y/z` and log-scales as `scale_*`.
    For collision smoke tests we use the mean exp-scale as each primitive's
    radius. This keeps the loader lightweight while matching the paper's
    spherical-Gaussian collision abstraction.
    """

    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plyfile is required to load Stage 1 Gaussian PLY files.") from exc

    ply = PlyData.read(str(path))
    vertices = ply["vertex"].data
    names = vertices.dtype.names or ()
    required = ("x", "y", "z")
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"{path} is missing required PLY fields: {missing}")

    centers_np = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1).astype(np.float32)
    scale_names = sorted(
        (name for name in names if name.startswith("scale_")),
        key=lambda name: int(name.split("_")[-1]),
    )
    if scale_names:
        log_scales = np.stack([vertices[name] for name in scale_names], axis=-1).astype(np.float32)
        radii_np = np.exp(log_scales).mean(axis=-1) * float(radius_scale)
    else:
        radii_np = np.full((centers_np.shape[0],), float(min_radius), dtype=np.float32)
    radii_np = np.maximum(radii_np, float(min_radius))

    if recenter:
        bbox_min = centers_np.min(axis=0)
        bbox_max = centers_np.max(axis=0)
        centers_np = centers_np - ((bbox_min + bbox_max) * 0.5)

    centers = torch.as_tensor(centers_np, dtype=dtype, device=device)
    radii = torch.as_tensor(radii_np, dtype=dtype, device=device)
    return centers, radii


def load_gaussian_points_from_ply(
    path: str | Path,
    *,
    max_points: int | None = None,
    recenter: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load Gaussian centers and approximate RGB colors from a Stage 1 PLY."""

    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plyfile is required to load Stage 1 Gaussian PLY files.") from exc

    ply = PlyData.read(str(path))
    vertices = ply["vertex"].data
    names = vertices.dtype.names or ()
    centers = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1).astype(np.float32)
    if recenter:
        centers = centers - ((centers.min(axis=0) + centers.max(axis=0)) * 0.5)

    if all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        sh_c0 = 0.28209479177387814
        colors = np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=-1).astype(np.float32)
        colors = np.clip(colors * sh_c0 + 0.5, 0.0, 1.0)
    else:
        height = centers[:, 2]
        denom = max(float(height.max() - height.min()), 1e-6)
        t = ((height - height.min()) / denom).reshape(-1, 1)
        colors = np.concatenate([0.2 + 0.6 * t, 0.45 + 0.3 * (1.0 - t), 0.95 - 0.3 * t], axis=1)

    if max_points is not None and centers.shape[0] > max_points:
        indices = np.linspace(0, centers.shape[0] - 1, max_points, dtype=np.int64)
        centers = centers[indices]
        colors = colors[indices]
    return centers.astype(np.float32), colors.astype(np.float32)


def detect_plane_contacts(
    position: torch.Tensor,
    local_query_points: torch.Tensor,
    collider: PlaneCollider,
    *,
    softness: float = 1e-3,
) -> ContactSamples:
    """Evaluate soft plane contacts for query points on a rigid body.

    This is complementarity-free: penetration and contact activity are smooth
    functions of signed distance rather than hard active-set decisions.
    """

    if softness <= 0.0:
        raise ValueError("softness must be positive.")
    if position.shape[-1] != 3:
        raise ValueError(f"position must end with 3 coordinates, got {tuple(position.shape)}.")
    if local_query_points.ndim != 2 or local_query_points.shape[-1] != 3:
        raise ValueError("local_query_points must have shape (N, 3).")

    local_query_points = local_query_points.to(dtype=position.dtype, device=position.device)
    world_points = position.unsqueeze(-2) + local_query_points
    collider = collider.on_like(world_points)
    signed_distances = collider.signed_distance(world_points)
    penetrations = F.softplus(-signed_distances / softness) * softness
    contact_weights = torch.sigmoid(-signed_distances / softness)
    normal = collider.normal / torch.clamp(torch.linalg.norm(collider.normal), min=1e-12)
    return ContactSamples(
        world_points=world_points,
        signed_distances=signed_distances,
        penetrations=penetrations,
        contact_weights=contact_weights,
        collider_normal=normal,
    )
