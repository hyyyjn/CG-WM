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
class CollisionEngineConfig:
    """Numerical knobs shared by the differentiable collision routines."""

    softness: float = 1e-3
    smooth_min_temperature: float = 2e-2
    inside_penalty: float = 0.02
    inside_sharpness: float = 50.0
    num_contact_patches: int = 1
    broad_phase_margin: float = 0.0
    broad_phase_mode: str = "sphere"
    patch_selection: str = "spatial"
    normal_mode: str = "phi_soft"


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
class GaussianUnionSDF:
    """Dense SDF evaluation of query points against spherical Gaussian geometry.

    The query points are assumed to already be in world space. This class is
    independent of floors, planes, or any particular query-point generator.
    """

    query_points: torch.Tensor
    primitive_signed_distances: torch.Tensor
    primitive_weights: torch.Tensor
    phi_soft: torch.Tensor
    signed_distances: torch.Tensor
    surface_normals: torch.Tensor

    @property
    def min_signed_distance(self) -> torch.Tensor:
        return torch.min(self.signed_distances)


@dataclass(frozen=True)
class GaussianUnionContacts:
    """Result of evaluating query points against a union of spherical Gaussians.

    `signed_distances` follows the paper's ϕ(p): a LogSumExp smooth min over
    primitives with a sigmoid-blended inside-object penalty so deep penetration
    is clamped to ≈ -inside_penalty instead of leaking back toward zero.

    `surface_normals` defaults to the analytic ∇ϕ_soft(p) / ‖∇ϕ_soft‖ per query
    point (a softmax-weighted blend of per-primitive outward directions). The
    SDF evaluator can alternatively use the sigmoid-blended signed-distance
    gradient via `normal_mode="signed_distance"` or the autograd version via
    `normal_mode="autograd"` for validation. The
    `collider_normal` field is the contact-weight-aggregated single normal used
    by downstream rigid-body dynamics. The name is kept for compatibility; use
    `contact_normal` in new code.
    """

    query_points: torch.Tensor
    primitive_signed_distances: torch.Tensor
    primitive_weights: torch.Tensor
    phi_soft: torch.Tensor
    signed_distances: torch.Tensor
    penetrations: torch.Tensor
    contact_weights: torch.Tensor
    surface_normals: torch.Tensor
    contact_point: torch.Tensor
    collider_normal: torch.Tensor
    patch_points: torch.Tensor
    patch_normals: torch.Tensor
    patch_weights: torch.Tensor
    patch_penetrations: torch.Tensor
    patch_signed_distances: torch.Tensor

    @property
    def min_signed_distance(self) -> torch.Tensor:
        return torch.min(self.signed_distances)

    @property
    def max_penetration(self) -> torch.Tensor:
        return torch.max(self.penetrations)

    @property
    def contact_normal(self) -> torch.Tensor:
        return self.collider_normal

    @property
    def num_contact_patches(self) -> int:
        return int(self.patch_points.shape[-2])


@dataclass(frozen=True)
class BodyPairContacts:
    """Bidirectional contact query result for two Gaussian collision bodies.

    `a_to_b` evaluates query points from body A against body B's SDF. Its
    normals point outward from B toward A. `b_to_a` is the opposite direction.
    `patch_*` merges the strongest patches from both directions.
    """

    a_to_b: GaussianUnionContacts
    b_to_a: GaussianUnionContacts
    patch_points: torch.Tensor
    patch_normals: torch.Tensor
    patch_weights: torch.Tensor
    patch_penetrations: torch.Tensor
    patch_signed_distances: torch.Tensor
    broad_phase_overlaps: torch.Tensor
    broad_phase_mode: str = "sphere"

    @property
    def has_contact(self) -> torch.Tensor:
        return torch.max(self.patch_weights, dim=-1).values > 0.5

    @property
    def min_signed_distance(self) -> torch.Tensor:
        return torch.minimum(self.a_to_b.min_signed_distance, self.b_to_a.min_signed_distance)


def _validate_query_points(query_points: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    if query_points.shape[-1:] != (3,):
        raise ValueError(f"query_points must end with 3 coordinates, got {tuple(query_points.shape)}.")
    leading_shape = query_points.shape[:-1]
    if query_points.numel() == 0:
        raise ValueError("query_points must contain at least one point.")
    return query_points.reshape(-1, 3), leading_shape


def _split_query_shape(query_points: torch.Tensor) -> tuple[torch.Size, int]:
    if query_points.ndim < 2 or query_points.shape[-1:] != (3,):
        raise ValueError(f"query_points must have shape (..., Q, 3), got {tuple(query_points.shape)}.")
    if query_points.shape[-2] == 0:
        raise ValueError("query_points must contain at least one point.")
    return query_points.shape[:-2], int(query_points.shape[-2])


def _validate_gaussian_primitives(
    query_points: torch.Tensor,
    gaussian_centers: torch.Tensor,
    gaussian_radii: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Size, int]:
    batch_shape, _ = _split_query_shape(query_points)
    if gaussian_centers.ndim < 2 or gaussian_centers.shape[-1] != 3:
        raise ValueError("gaussian_centers must have shape (G, 3) or (..., G, 3).")
    if gaussian_radii.ndim < 1:
        raise ValueError("gaussian_radii must have shape (G,) or (..., G).")
    if gaussian_centers.shape[-2] == 0:
        raise ValueError("At least one Gaussian primitive is required.")
    if gaussian_radii.shape[-1] != gaussian_centers.shape[-2]:
        raise ValueError("The last dimension of gaussian_radii must match the Gaussian count.")

    centers = gaussian_centers.to(dtype=query_points.dtype, device=query_points.device)
    radii = gaussian_radii.to(dtype=query_points.dtype, device=query_points.device)
    gaussian_batch_shape = centers.shape[:-2]
    radii_batch_shape = radii.shape[:-1]
    try:
        output_batch_shape = torch.broadcast_shapes(batch_shape, gaussian_batch_shape, radii_batch_shape)
    except RuntimeError as exc:
        raise ValueError(
            "query_points, gaussian_centers, and gaussian_radii batch shapes are not broadcastable: "
            f"{tuple(batch_shape)}, {tuple(gaussian_batch_shape)}, {tuple(radii_batch_shape)}."
        ) from exc

    centers = torch.broadcast_to(centers, (*output_batch_shape, centers.shape[-2], 3))
    radii = torch.broadcast_to(radii, (*output_batch_shape, radii.shape[-1]))
    if torch.any(radii <= 0.0):
        raise ValueError("gaussian_radii must be positive.")
    return centers, radii, output_batch_shape, int(centers.shape[-2])


@dataclass(frozen=True)
class GaussianCollisionBody:
    """Spherical-Gaussian collision proxy in a local rigid-body frame."""

    local_centers: torch.Tensor
    radii: torch.Tensor
    local_query_points: torch.Tensor | None = None

    def to(self, *, dtype=None, device=None) -> "GaussianCollisionBody":
        return GaussianCollisionBody(
            self.local_centers.to(dtype=dtype, device=device),
            self.radii.to(dtype=dtype, device=device),
            None if self.local_query_points is None else self.local_query_points.to(dtype=dtype, device=device),
        )

    def world_centers(
        self,
        position: torch.Tensor,
        *,
        quaternion_wxyz: torch.Tensor | None = None,
        rotation_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return transform_local_points(
            self.local_centers.to(dtype=position.dtype, device=position.device),
            position,
            quaternion_wxyz=quaternion_wxyz,
            rotation_matrix=rotation_matrix,
        )

    def query_points_world(
        self,
        position: torch.Tensor,
        *,
        quaternion_wxyz: torch.Tensor | None = None,
        rotation_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_queries = self.local_query_points
        if local_queries is None:
            local_queries = make_gaussian_proxy_query_points(
                self.local_centers.to(dtype=position.dtype, device=position.device),
                self.radii.to(dtype=position.dtype, device=position.device),
            )
        return transform_local_points(
            local_queries.to(dtype=position.dtype, device=position.device),
            position,
            quaternion_wxyz=quaternion_wxyz,
            rotation_matrix=rotation_matrix,
        )

    def local_bounding_sphere(self) -> tuple[torch.Tensor, torch.Tensor]:
        center = (self.local_centers.min(dim=0).values + self.local_centers.max(dim=0).values) * 0.5
        distances = torch.linalg.norm(self.local_centers - center.unsqueeze(0), dim=-1) + self.radii
        return center, torch.max(distances)

    def local_aabb(self) -> tuple[torch.Tensor, torch.Tensor]:
        radii = self.radii.unsqueeze(-1)
        return torch.min(self.local_centers - radii, dim=0).values, torch.max(self.local_centers + radii, dim=0).values

    def world_bounding_sphere(
        self,
        position: torch.Tensor,
        *,
        quaternion_wxyz: torch.Tensor | None = None,
        rotation_matrix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        center_local, radius = self.local_bounding_sphere()
        center_world = transform_local_points(
            center_local.reshape(1, 3).to(dtype=position.dtype, device=position.device),
            position,
            quaternion_wxyz=quaternion_wxyz,
            rotation_matrix=rotation_matrix,
        ).squeeze(-2)
        return center_world, radius.to(dtype=position.dtype, device=position.device)

    def world_aabb(
        self,
        position: torch.Tensor,
        *,
        quaternion_wxyz: torch.Tensor | None = None,
        rotation_matrix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centers = self.world_centers(
            position,
            quaternion_wxyz=quaternion_wxyz,
            rotation_matrix=rotation_matrix,
        )
        radii = self.radii.to(dtype=position.dtype, device=position.device)
        while radii.ndim < centers.ndim - 1:
            radii = radii.unsqueeze(0)
        radii = radii.unsqueeze(-1)
        return torch.min(centers - radii, dim=-2).values, torch.max(centers + radii, dim=-2).values


class DifferentiableCollisionEngine:
    """Small query-based collision engine for Gaussian rigid-body proxies.

    The engine does not own scene state. Callers provide world-space query
    points and world-space Gaussian centers, or use `contacts_for_body` with a
    local `GaussianCollisionBody` and a rigid pose.
    """

    def __init__(self, config: CollisionEngineConfig | None = None):
        self.config = config or CollisionEngineConfig()

    def evaluate_sdf(
        self,
        query_points: torch.Tensor,
        gaussian_centers: torch.Tensor,
        gaussian_radii: torch.Tensor,
    ) -> GaussianUnionSDF:
        cfg = self.config
        return evaluate_gaussian_union_sdf(
            query_points,
            gaussian_centers,
            gaussian_radii,
            smooth_min_temperature=cfg.smooth_min_temperature,
            inside_penalty=cfg.inside_penalty,
            inside_sharpness=cfg.inside_sharpness,
            normal_mode=cfg.normal_mode,
        )

    def contacts(
        self,
        query_points: torch.Tensor,
        gaussian_centers: torch.Tensor,
        gaussian_radii: torch.Tensor,
        *,
        normal_hint: torch.Tensor | None = None,
    ) -> GaussianUnionContacts:
        """Evaluate Gaussian contacts, optionally orienting normals to an environment hint.

        `normal_hint` is intended for contacts against a known external
        collider normal, such as a floor plane. Body-body contacts should use
        :meth:`body_pair_contacts`, which relies on SDF gradient normals and
        does not inject center-line normal hints.
        """

        sdf = self.evaluate_sdf(query_points, gaussian_centers, gaussian_radii)
        return aggregate_gaussian_union_contacts(
            sdf,
            softness=self.config.softness,
            normal_hint=normal_hint,
            num_contact_patches=self.config.num_contact_patches,
            patch_selection=self.config.patch_selection,
        )

    def contacts_for_body(
        self,
        query_points: torch.Tensor,
        body: GaussianCollisionBody,
        position: torch.Tensor,
        *,
        quaternion_wxyz: torch.Tensor | None = None,
        rotation_matrix: torch.Tensor | None = None,
        normal_hint: torch.Tensor | None = None,
    ) -> GaussianUnionContacts:
        centers = body.world_centers(
            position,
            quaternion_wxyz=quaternion_wxyz,
            rotation_matrix=rotation_matrix,
        )
        return self.contacts(query_points, centers, body.radii, normal_hint=normal_hint)

    def body_pair_contacts(
        self,
        body_a: GaussianCollisionBody,
        position_a: torch.Tensor,
        body_b: GaussianCollisionBody,
        position_b: torch.Tensor,
        *,
        quaternion_a_wxyz: torch.Tensor | None = None,
        quaternion_b_wxyz: torch.Tensor | None = None,
        rotation_a_matrix: torch.Tensor | None = None,
        rotation_b_matrix: torch.Tensor | None = None,
    ) -> BodyPairContacts:
        if self.config.broad_phase_mode not in ("sphere", "aabb"):
            raise ValueError("broad_phase_mode must be one of: sphere, aabb.")
        center_a, radius_a = body_a.world_bounding_sphere(
            position_a,
            quaternion_wxyz=quaternion_a_wxyz,
            rotation_matrix=rotation_a_matrix,
        )
        center_b, radius_b = body_b.world_bounding_sphere(
            position_b,
            quaternion_wxyz=quaternion_b_wxyz,
            rotation_matrix=rotation_b_matrix,
        )
        center_delta = center_a - center_b
        center_distance = torch.linalg.norm(center_delta, dim=-1)
        sphere_overlaps = center_distance <= (radius_a + radius_b + float(self.config.broad_phase_margin))
        if self.config.broad_phase_mode == "aabb":
            min_a, max_a = body_a.world_aabb(
                position_a,
                quaternion_wxyz=quaternion_a_wxyz,
                rotation_matrix=rotation_a_matrix,
            )
            min_b, max_b = body_b.world_aabb(
                position_b,
                quaternion_wxyz=quaternion_b_wxyz,
                rotation_matrix=rotation_b_matrix,
            )
            margin = float(self.config.broad_phase_margin)
            broad_phase_overlaps = torch.all((max_a + margin >= min_b) & (max_b + margin >= min_a), dim=-1)
        else:
            broad_phase_overlaps = sphere_overlaps

        query_a = body_a.query_points_world(
            position_a,
            quaternion_wxyz=quaternion_a_wxyz,
            rotation_matrix=rotation_a_matrix,
        )
        query_b = body_b.query_points_world(
            position_b,
            quaternion_wxyz=quaternion_b_wxyz,
            rotation_matrix=rotation_b_matrix,
        )
        centers_a = body_a.world_centers(
            position_a,
            quaternion_wxyz=quaternion_a_wxyz,
            rotation_matrix=rotation_a_matrix,
        )
        centers_b = body_b.world_centers(
            position_b,
            quaternion_wxyz=quaternion_b_wxyz,
            rotation_matrix=rotation_b_matrix,
        )

        # Object-object contact normals must come from the Gaussian SDF
        # gradient. A center-line hint can hide geometry errors and is only
        # appropriate for external colliders with a known normal, e.g. floors.
        a_to_b = self.contacts(query_a, centers_b, body_b.radii, normal_hint=None)
        b_to_a = self.contacts(query_b, centers_a, body_a.radii, normal_hint=None)
        merged = merge_contact_patches(
            a_to_b,
            b_to_a,
            num_contact_patches=self.config.num_contact_patches,
            patch_selection=self.config.patch_selection,
        )
        patch_weights = torch.where(
            broad_phase_overlaps.unsqueeze(-1),
            merged["patch_weights"],
            torch.zeros_like(merged["patch_weights"]),
        )
        patch_penetrations = torch.where(
            broad_phase_overlaps.unsqueeze(-1),
            merged["patch_penetrations"],
            torch.zeros_like(merged["patch_penetrations"]),
        )
        return BodyPairContacts(
            a_to_b=a_to_b,
            b_to_a=b_to_a,
            patch_points=merged["patch_points"],
            patch_normals=merged["patch_normals"],
            patch_weights=patch_weights,
            patch_penetrations=patch_penetrations,
            patch_signed_distances=merged["patch_signed_distances"],
            broad_phase_overlaps=broad_phase_overlaps,
            broad_phase_mode=self.config.broad_phase_mode,
        )


def _orient_contact_normal(contact_normal: torch.Tensor, normal_hint: torch.Tensor | None) -> torch.Tensor:
    normal_norm = torch.linalg.norm(contact_normal, dim=-1, keepdim=True)
    if normal_hint is None:
        return contact_normal / torch.clamp(normal_norm, min=1e-12)

    hint = normal_hint.to(dtype=contact_normal.dtype, device=contact_normal.device)
    if hint.shape[-1:] != (3,):
        raise ValueError(f"normal_hint must end with 3 coordinates, got {tuple(hint.shape)}.")
    hint = hint / torch.clamp(torch.linalg.norm(hint, dim=-1, keepdim=True), min=1e-12)
    hint = torch.broadcast_to(hint, contact_normal.shape)
    contact_normal = torch.where(normal_norm < 1e-8, hint, contact_normal / torch.clamp(normal_norm, min=1e-12))
    flip = torch.sum(contact_normal * hint, dim=-1, keepdim=True) < 0.0
    return torch.where(flip, -contact_normal, contact_normal)


def _normalize_vectors(vectors: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    return vectors / torch.clamp(torch.linalg.norm(vectors, dim=-1, keepdim=True), min=eps)


def quat_wxyz_to_matrix(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert normalized-or-not quaternions with shape (..., 4) to (..., 3, 3)."""

    if quaternion_wxyz.shape[-1:] != (4,):
        raise ValueError(f"quaternion_wxyz must end with 4 values, got {tuple(quaternion_wxyz.shape)}.")
    quat = quaternion_wxyz / torch.clamp(torch.linalg.norm(quaternion_wxyz, dim=-1, keepdim=True), min=1e-12)
    w, x, y, z = quat.unbind(dim=-1)
    one = torch.ones_like(w)
    two = 2.0
    return torch.stack(
        (
            torch.stack((one - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)), dim=-1),
            torch.stack((two * (x * y + z * w), one - two * (x * x + z * z), two * (y * z - x * w)), dim=-1),
            torch.stack((two * (x * z - y * w), two * (y * z + x * w), one - two * (x * x + y * y)), dim=-1),
        ),
        dim=-2,
    )


def transform_local_points(
    local_points: torch.Tensor,
    position: torch.Tensor,
    *,
    quaternion_wxyz: torch.Tensor | None = None,
    rotation_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transform local query/proxy points to world space with optional batching.

    `local_points` has shape `(N, 3)`. `position` has shape `(..., 3)`.
    The result has shape `(..., N, 3)`, or `(N, 3)` for an unbatched position.
    """

    if local_points.ndim != 2 or local_points.shape[-1] != 3:
        raise ValueError("local_points must have shape (N, 3).")
    if position.shape[-1:] != (3,):
        raise ValueError(f"position must end with 3 coordinates, got {tuple(position.shape)}.")
    if quaternion_wxyz is not None and rotation_matrix is not None:
        raise ValueError("Pass either quaternion_wxyz or rotation_matrix, not both.")

    local = local_points.to(dtype=position.dtype, device=position.device)
    if rotation_matrix is None:
        if quaternion_wxyz is None:
            rotation_matrix = torch.eye(3, dtype=position.dtype, device=position.device)
        else:
            rotation_matrix = quat_wxyz_to_matrix(quaternion_wxyz.to(dtype=position.dtype, device=position.device))
    else:
        rotation_matrix = rotation_matrix.to(dtype=position.dtype, device=position.device)
        if rotation_matrix.shape[-2:] != (3, 3):
            raise ValueError(f"rotation_matrix must end with shape (3, 3), got {tuple(rotation_matrix.shape)}.")

    rotated = torch.matmul(local, rotation_matrix.transpose(-1, -2))
    return rotated + position.unsqueeze(-2)


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


def make_box_surface_query_points(
    half_extents: Iterable[float],
    *,
    grid_resolution: int = 3,
    dtype=torch.float32,
    device=None,
) -> torch.Tensor:
    """Return local-space query points over all six faces of a box."""

    if grid_resolution < 2:
        raise ValueError("grid_resolution must be at least 2.")
    half_extents_tensor = _as_3_tensor(half_extents, dtype=dtype, device=device)
    hx, hy, hz = [float(v) for v in half_extents_tensor.detach().cpu().tolist()]
    xs = torch.linspace(-hx, hx, grid_resolution, dtype=dtype, device=device)
    ys = torch.linspace(-hy, hy, grid_resolution, dtype=dtype, device=device)
    zs = torch.linspace(-hz, hz, grid_resolution, dtype=dtype, device=device)

    faces = []
    for z in (-hz, hz):
        zz = torch.full((grid_resolution, grid_resolution), float(z), dtype=dtype, device=device)
        xx, yy = torch.meshgrid(xs, ys, indexing="ij")
        faces.append(torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3))
    for y in (-hy, hy):
        yy = torch.full((grid_resolution, grid_resolution), float(y), dtype=dtype, device=device)
        xx, zz = torch.meshgrid(xs, zs, indexing="ij")
        faces.append(torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3))
    for x in (-hx, hx):
        xx = torch.full((grid_resolution, grid_resolution), float(x), dtype=dtype, device=device)
        yy, zz = torch.meshgrid(ys, zs, indexing="ij")
        faces.append(torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3))
    return torch.unique(torch.cat(faces, dim=0), dim=0)


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


def make_gaussian_proxy_query_points(
    local_centers: torch.Tensor,
    radii: torch.Tensor,
    *,
    directions_per_gaussian: int = 6,
) -> torch.Tensor:
    """Sample local query points on each spherical Gaussian primitive."""

    if local_centers.ndim != 2 or local_centers.shape[-1] != 3:
        raise ValueError("local_centers must have shape (G, 3).")
    if radii.ndim != 1 or radii.shape[0] != local_centers.shape[0]:
        raise ValueError("radii must have shape (G,).")
    if directions_per_gaussian == 6:
        dirs = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=local_centers.dtype,
            device=local_centers.device,
        )
    else:
        dirs = make_sphere_query_points(
            1.0,
            num_points=max(int(directions_per_gaussian), 4),
            dtype=local_centers.dtype,
            device=local_centers.device,
        )
    return (local_centers.unsqueeze(1) + radii.reshape(-1, 1, 1) * dirs.unsqueeze(0)).reshape(-1, 3)


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
    collider_normal: torch.Tensor | None = None,
    *,
    softness: float = 1e-3,
    smooth_min_temperature: float = 2e-2,
    inside_penalty: float = 0.02,
    inside_sharpness: float = 50.0,
    num_contact_patches: int = 1,
    patch_selection: str = "spatial",
    normal_mode: str = "phi_soft",
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

    `collider_normal` is a legacy name for `normal_hint`. Pass it only for
    environment contacts with a physically known normal, such as floor queries.
    Object-object contacts should leave it as ``None`` so normals remain pure
    SDF-gradient normals.
    """

    sdf = evaluate_gaussian_union_sdf(
        query_points,
        gaussian_centers,
        gaussian_radii,
        smooth_min_temperature=smooth_min_temperature,
        inside_penalty=inside_penalty,
        inside_sharpness=inside_sharpness,
        normal_mode=normal_mode,
    )
    return aggregate_gaussian_union_contacts(
        sdf,
        softness=softness,
        normal_hint=collider_normal,
        num_contact_patches=num_contact_patches,
        patch_selection=patch_selection,
    )


def evaluate_gaussian_union_sdf(
    query_points: torch.Tensor,
    gaussian_centers: torch.Tensor,
    gaussian_radii: torch.Tensor,
    *,
    smooth_min_temperature: float = 2e-2,
    inside_penalty: float = 0.02,
    inside_sharpness: float = 50.0,
    normal_mode: str = "phi_soft",
) -> GaussianUnionSDF:
    """Evaluate the ContactGaussian-WM spherical-Gaussian SDF at query points.

    `query_points` are supplied by the caller and may have any leading shape
    ending in 3, for example `(Q, 3)` or `(B, Q, 3)`. No floor, plane, body
    transform, or query-point generation is assumed here.
    """

    if smooth_min_temperature <= 0.0:
        raise ValueError("smooth_min_temperature must be positive.")
    if inside_penalty <= 0.0:
        raise ValueError("inside_penalty must be positive.")
    if inside_sharpness <= 0.0:
        raise ValueError("inside_sharpness must be positive.")
    if normal_mode not in ("phi_soft", "signed_distance", "autograd"):
        raise ValueError("normal_mode must be one of: phi_soft, signed_distance, autograd.")

    _, query_count = _split_query_shape(query_points)
    centers, radii, batch_shape, gaussian_count = _validate_gaussian_primitives(
        query_points,
        gaussian_centers,
        gaussian_radii,
    )
    points = torch.broadcast_to(
        query_points,
        (*batch_shape, query_count, 3),
    )
    if normal_mode == "autograd" and not points.requires_grad:
        points = points.detach().clone().requires_grad_(True)

    offsets = points.unsqueeze(-2) - centers.unsqueeze(-3)
    center_distances = torch.linalg.norm(offsets, dim=-1)
    primitive_distances = center_distances - radii.unsqueeze(-2)

    primitive_weights = torch.softmax(-primitive_distances / smooth_min_temperature, dim=-1)
    phi_soft = -smooth_min_temperature * torch.logsumexp(
        -primitive_distances / smooth_min_temperature,
        dim=-1,
    )

    sigma_blend = torch.sigmoid(inside_sharpness * phi_soft)
    signed_distances = sigma_blend * phi_soft + (1.0 - sigma_blend) * (-inside_penalty)

    direction_per_prim = offsets / torch.clamp(center_distances.unsqueeze(-1), min=1e-9)
    phi_soft_gradient = torch.sum(primitive_weights.unsqueeze(-1) * direction_per_prim, dim=-2)
    if normal_mode == "phi_soft":
        surface_normals = _normalize_vectors(phi_soft_gradient)
    elif normal_mode == "signed_distance":
        blend_gradient_scale = sigma_blend + inside_sharpness * sigma_blend * (1.0 - sigma_blend) * (
            phi_soft + inside_penalty
        )
        surface_normals = _normalize_vectors(blend_gradient_scale.unsqueeze(-1) * phi_soft_gradient)
    else:
        gradients = torch.autograd.grad(
            signed_distances,
            points,
            grad_outputs=torch.ones_like(signed_distances),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        surface_normals = _normalize_vectors(gradients)

    return GaussianUnionSDF(
        query_points=points,
        primitive_signed_distances=primitive_distances.reshape(*batch_shape, query_count, gaussian_count),
        primitive_weights=primitive_weights.reshape(*batch_shape, query_count, gaussian_count),
        phi_soft=phi_soft.reshape(*batch_shape, query_count),
        signed_distances=signed_distances.reshape(*batch_shape, query_count),
        surface_normals=surface_normals.reshape(*batch_shape, query_count, 3),
    )


def _select_contact_patch_indices(
    query_points: torch.Tensor,
    contact_weights: torch.Tensor,
    patch_count: int,
    *,
    patch_selection: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if patch_selection not in ("topk", "spatial"):
        raise ValueError("_select_contact_patch_indices only supports discrete modes: topk, spatial.")
    if patch_selection == "topk" or patch_count <= 1:
        return torch.topk(contact_weights, k=patch_count, dim=-1)

    batch_count, query_count = contact_weights.shape
    selected_indices = []
    selected_weights = []
    large = torch.as_tensor(1e6, dtype=query_points.dtype, device=query_points.device)
    min_distances = torch.full((batch_count, query_count), float("inf"), dtype=query_points.dtype, device=query_points.device)
    score = contact_weights
    for patch_idx in range(patch_count):
        index = torch.argmax(score, dim=-1)
        selected_indices.append(index)
        selected_weights.append(torch.gather(contact_weights, dim=1, index=index.unsqueeze(-1)).squeeze(-1))
        selected_points = torch.gather(query_points, dim=1, index=index.reshape(batch_count, 1, 1).expand(batch_count, 1, 3))
        distances = torch.linalg.norm(query_points - selected_points, dim=-1)
        min_distances = torch.minimum(min_distances, distances)
        if patch_idx == 0:
            scale = torch.clamp(torch.quantile(min_distances.masked_fill(torch.isinf(min_distances), 0.0), 0.75, dim=-1, keepdim=True), min=1e-6)
        else:
            scale = torch.clamp(torch.max(min_distances, dim=-1, keepdim=True).values, min=1e-6)
        diversity = torch.clamp(min_distances / scale, max=1.0)
        score = contact_weights * (0.25 + 0.75 * diversity)
        score = score.scatter(1, torch.stack(selected_indices, dim=1), -large.expand(batch_count, len(selected_indices)))
    return torch.stack(selected_weights, dim=1), torch.stack(selected_indices, dim=1)


def _gather_patch_vectors(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, dim=1, index=indices.unsqueeze(-1).expand(indices.shape[0], indices.shape[1], values.shape[-1]))


def _soft_contact_patch_values(
    query_points: torch.Tensor,
    contact_weights: torch.Tensor,
    surface_normals: torch.Tensor,
    penetrations: torch.Tensor,
    signed_distances: torch.Tensor,
    patch_count: int,
) -> dict[str, torch.Tensor]:
    """Pool all query samples into differentiable contact patches.

    Unlike `topk` and `spatial`, this path does not pick discrete query
    indices. For K > 1 it creates smooth bins along a data-dependent but
    differentiable projection axis, then averages each bin with contact
    weights. Patch gates remain in [0, 1] as assignment-weighted mean contact
    probabilities.
    """

    if patch_count < 1:
        raise ValueError("patch_count must be at least 1.")
    batch_count, query_count, _ = query_points.shape
    if patch_count == 1:
        assignments = torch.ones((batch_count, query_count, 1), dtype=query_points.dtype, device=query_points.device)
    else:
        centered = query_points - torch.mean(query_points, dim=1, keepdim=True)
        axis_energy = torch.mean(centered * centered, dim=1)
        axis_weights = torch.softmax(axis_energy / torch.clamp(torch.mean(axis_energy, dim=-1, keepdim=True), min=1e-12), dim=-1)
        scalar = torch.sum(centered * axis_weights.unsqueeze(1), dim=-1)
        scalar_scale = torch.sqrt(torch.clamp(torch.mean(scalar * scalar, dim=-1, keepdim=True), min=1e-12))
        scalar = scalar / scalar_scale
        anchors = torch.linspace(-1.0, 1.0, patch_count, dtype=query_points.dtype, device=query_points.device)
        bin_softness = torch.clamp(torch.as_tensor(2.0 / max(patch_count - 1, 1), dtype=query_points.dtype, device=query_points.device), min=1e-3)
        logits = -((scalar.unsqueeze(-1) - anchors.reshape(1, 1, patch_count)) / bin_softness) ** 2
        assignments = torch.softmax(logits, dim=-1)

    mass = assignments * contact_weights.unsqueeze(-1)
    mass_sum = torch.clamp(torch.sum(mass, dim=1), min=1e-12)
    assignment_sum = torch.clamp(torch.sum(assignments, dim=1), min=1e-12)
    patch_points = torch.sum(mass.unsqueeze(-1) * query_points.unsqueeze(-2), dim=1) / mass_sum.unsqueeze(-1)
    patch_normals = torch.sum(mass.unsqueeze(-1) * surface_normals.unsqueeze(-2), dim=1) / mass_sum.unsqueeze(-1)
    patch_normals = patch_normals / torch.clamp(torch.linalg.norm(patch_normals, dim=-1, keepdim=True), min=1e-12)
    patch_penetrations = torch.sum(mass * penetrations.unsqueeze(-1), dim=1) / mass_sum
    patch_signed_distances = torch.sum(mass * signed_distances.unsqueeze(-1), dim=1) / mass_sum
    patch_weights = torch.sum(mass, dim=1) / assignment_sum
    return {
        "patch_points": patch_points,
        "patch_normals": patch_normals,
        "patch_weights": patch_weights,
        "patch_penetrations": patch_penetrations,
        "patch_signed_distances": patch_signed_distances,
    }


def merge_contact_patches(
    contacts_a: GaussianUnionContacts,
    contacts_b: GaussianUnionContacts,
    *,
    num_contact_patches: int,
    patch_selection: str = "spatial",
) -> dict[str, torch.Tensor]:
    """Merge bidirectional contact patches into one strongest patch set."""

    if num_contact_patches < 1:
        raise ValueError("num_contact_patches must be at least 1.")
    if patch_selection not in ("topk", "spatial", "soft"):
        raise ValueError("patch_selection must be one of: topk, spatial, soft.")

    points = torch.cat(
        (
            contacts_a.patch_points.reshape(-1, contacts_a.num_contact_patches, 3),
            contacts_b.patch_points.reshape(-1, contacts_b.num_contact_patches, 3),
        ),
        dim=1,
    )
    normals = torch.cat(
        (
            contacts_a.patch_normals.reshape(-1, contacts_a.num_contact_patches, 3),
            -contacts_b.patch_normals.reshape(-1, contacts_b.num_contact_patches, 3),
        ),
        dim=1,
    )
    weights = torch.cat(
        (
            contacts_a.patch_weights.reshape(-1, contacts_a.num_contact_patches),
            contacts_b.patch_weights.reshape(-1, contacts_b.num_contact_patches),
        ),
        dim=1,
    )
    penetrations = torch.cat(
        (
            contacts_a.patch_penetrations.reshape(-1, contacts_a.num_contact_patches),
            contacts_b.patch_penetrations.reshape(-1, contacts_b.num_contact_patches),
        ),
        dim=1,
    )
    signed_distances = torch.cat(
        (
            contacts_a.patch_signed_distances.reshape(-1, contacts_a.num_contact_patches),
            contacts_b.patch_signed_distances.reshape(-1, contacts_b.num_contact_patches),
        ),
        dim=1,
    )
    patch_count = min(int(num_contact_patches), int(weights.shape[-1]))
    if patch_selection == "soft":
        merged = _soft_contact_patch_values(
            points,
            weights,
            normals,
            penetrations,
            signed_distances,
            patch_count,
        )
        batch_shape = contacts_a.patch_points.shape[:-2]
        return {
            "patch_points": merged["patch_points"].reshape(*batch_shape, patch_count, 3),
            "patch_normals": merged["patch_normals"].reshape(*batch_shape, patch_count, 3),
            "patch_weights": merged["patch_weights"].reshape(*batch_shape, patch_count),
            "patch_penetrations": merged["patch_penetrations"].reshape(*batch_shape, patch_count),
            "patch_signed_distances": merged["patch_signed_distances"].reshape(*batch_shape, patch_count),
        }

    patch_weights, patch_indices = _select_contact_patch_indices(
        points,
        weights,
        patch_count,
        patch_selection=patch_selection,
    )
    batch_shape = contacts_a.patch_points.shape[:-2]
    return {
        "patch_points": _gather_patch_vectors(points, patch_indices).reshape(*batch_shape, patch_count, 3),
        "patch_normals": _gather_patch_vectors(normals, patch_indices).reshape(*batch_shape, patch_count, 3),
        "patch_weights": patch_weights.reshape(*batch_shape, patch_count),
        "patch_penetrations": torch.gather(penetrations, dim=1, index=patch_indices).reshape(*batch_shape, patch_count),
        "patch_signed_distances": torch.gather(signed_distances, dim=1, index=patch_indices).reshape(*batch_shape, patch_count),
    }


def aggregate_gaussian_union_contacts(
    sdf: GaussianUnionSDF,
    *,
    softness: float = 1e-3,
    normal_hint: torch.Tensor | None = None,
    num_contact_patches: int = 1,
    patch_selection: str = "spatial",
) -> GaussianUnionContacts:
    """Convert per-query SDF values into one smooth contact summary.

    `normal_hint` is optional. Pass an environment normal, such as a floor
    normal, only when downstream dynamics require a consistently oriented
    contact normal. Do not use it for object-object Gaussian contacts; those
    should be oriented by the SDF gradient itself. Inputs shaped `(B, Q, 3)` are
    aggregated independently per batch item. `num_contact_patches` keeps the
    top-k strongest query contacts for multi-contact dynamics while preserving
    the legacy single summary.
    """

    if softness <= 0.0:
        raise ValueError("softness must be positive.")
    if num_contact_patches < 1:
        raise ValueError("num_contact_patches must be at least 1.")
    if patch_selection not in ("topk", "spatial", "soft"):
        raise ValueError("patch_selection must be one of: topk, spatial, soft.")

    batch_shape, query_count = _split_query_shape(sdf.query_points)
    query_points = sdf.query_points.reshape(-1, query_count, 3)
    signed_distances = sdf.signed_distances.reshape(-1, query_count)
    surface_normals = sdf.surface_normals.reshape(-1, query_count, 3)
    batch_count = query_points.shape[0]

    penetrations = F.softplus(-signed_distances / softness) * softness
    contact_weights = torch.sigmoid(-signed_distances / softness)
    weight_sum = torch.clamp(torch.sum(contact_weights, dim=-1, keepdim=True), min=1e-12)
    contact_point = torch.sum(query_points * contact_weights.unsqueeze(-1), dim=-2) / weight_sum
    contact_normal = torch.sum(surface_normals * contact_weights.unsqueeze(-1), dim=-2) / weight_sum
    contact_normal = _orient_contact_normal(contact_normal, normal_hint)

    patch_count = min(int(num_contact_patches), int(query_count))
    if patch_selection == "soft":
        patch_values = _soft_contact_patch_values(
            query_points,
            contact_weights,
            surface_normals,
            penetrations,
            signed_distances,
            patch_count,
        )
        patch_points = patch_values["patch_points"]
        patch_normals = _orient_contact_normal(
            patch_values["patch_normals"],
            normal_hint.unsqueeze(-2) if normal_hint is not None and normal_hint.ndim > 1 else normal_hint,
        )
        patch_weights = patch_values["patch_weights"]
        patch_penetrations = patch_values["patch_penetrations"]
        patch_signed_distances = patch_values["patch_signed_distances"]
    else:
        patch_weights, patch_indices = _select_contact_patch_indices(
            query_points,
            contact_weights,
            patch_count,
            patch_selection=patch_selection,
        )
        patch_points = _gather_patch_vectors(query_points, patch_indices)
        patch_normals = _gather_patch_vectors(surface_normals, patch_indices)
        patch_normals = _orient_contact_normal(patch_normals, normal_hint.unsqueeze(-2) if normal_hint is not None and normal_hint.ndim > 1 else normal_hint)
        patch_penetrations = torch.gather(penetrations, dim=1, index=patch_indices)
        patch_signed_distances = torch.gather(signed_distances, dim=1, index=patch_indices)

    return GaussianUnionContacts(
        query_points=sdf.query_points,
        primitive_signed_distances=sdf.primitive_signed_distances,
        primitive_weights=sdf.primitive_weights,
        phi_soft=sdf.phi_soft,
        signed_distances=sdf.signed_distances,
        penetrations=penetrations.reshape(*batch_shape, query_count),
        contact_weights=contact_weights.reshape(*batch_shape, query_count),
        surface_normals=sdf.surface_normals,
        contact_point=contact_point.reshape(*batch_shape, 3),
        collider_normal=contact_normal.reshape(*batch_shape, 3),
        patch_points=patch_points.reshape(*batch_shape, patch_count, 3),
        patch_normals=patch_normals.reshape(*batch_shape, patch_count, 3),
        patch_weights=patch_weights.reshape(*batch_shape, patch_count),
        patch_penetrations=patch_penetrations.reshape(*batch_shape, patch_count),
        patch_signed_distances=patch_signed_distances.reshape(*batch_shape, patch_count),
    )


def compare_gaussian_union_normal_modes(
    query_points: torch.Tensor,
    gaussian_centers: torch.Tensor,
    gaussian_radii: torch.Tensor,
    *,
    smooth_min_temperature: float = 2e-2,
    inside_penalty: float = 0.02,
    inside_sharpness: float = 50.0,
) -> dict[str, torch.Tensor]:
    """Compare analytic normal modes against autograd normals.

    Returns normals and angular errors in degrees. This is intended as a
    diagnostic for validating whether `normal_mode="signed_distance"` materially
    differs from the default `normal_mode="phi_soft"` on a given proxy.
    """

    common_kwargs = {
        "smooth_min_temperature": smooth_min_temperature,
        "inside_penalty": inside_penalty,
        "inside_sharpness": inside_sharpness,
    }
    phi = evaluate_gaussian_union_sdf(
        query_points,
        gaussian_centers,
        gaussian_radii,
        normal_mode="phi_soft",
        **common_kwargs,
    )
    signed = evaluate_gaussian_union_sdf(
        query_points,
        gaussian_centers,
        gaussian_radii,
        normal_mode="signed_distance",
        **common_kwargs,
    )
    autograd = evaluate_gaussian_union_sdf(
        query_points,
        gaussian_centers,
        gaussian_radii,
        normal_mode="autograd",
        **common_kwargs,
    )

    def angle_deg(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        cosine = torch.sum(lhs * rhs, dim=-1)
        cosine = torch.clamp(cosine, min=-1.0, max=1.0)
        return torch.rad2deg(torch.acos(cosine))

    return {
        "phi_soft_normals": phi.surface_normals,
        "signed_distance_normals": signed.surface_normals,
        "autograd_normals": autograd.surface_normals,
        "phi_soft_vs_autograd_deg": angle_deg(phi.surface_normals, autograd.surface_normals),
        "signed_distance_vs_autograd_deg": angle_deg(signed.surface_normals, autograd.surface_normals),
        "phi_soft_vs_signed_distance_deg": angle_deg(phi.surface_normals, signed.surface_normals),
    }


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


def load_gaussian_collision_body_from_ply(
    path: str | Path,
    *,
    radius_scale: float = 1.0,
    min_radius: float = 1e-4,
    recenter: bool = False,
    object_id: int | None = None,
    foreground_threshold: float | None = None,
    opacity_threshold: float | None = None,
    max_primitives: int | None = None,
    use_centers_as_queries: bool = True,
    world_translation: "np.ndarray | torch.Tensor | tuple | list | None" = None,
    world_rotation: "np.ndarray | torch.Tensor | None" = None,
    dtype=torch.float32,
    device=None,
) -> GaussianCollisionBody:
    """Build a :class:`GaussianCollisionBody` directly from a Stage 1 PLY.

    This is the Stage 1 -> Stage 2 bridge: it keeps the spherical proxy logic
    from :func:`load_gaussian_collision_primitives_from_ply`, but also honors the
    object-aware fields Stage 1 writes (`object_id`, `foreground_logit`, and
    `opacity`) so downstream dynamics can consume one physical body at a time.

    Coordinate-frame handling
    -------------------------
    Stage 2 expects a rigid-body local frame. If the PLY stores world-frame
    means, pass ``world_translation`` and ``world_rotation`` from the Stage 2
    manifest contract to convert them into object-local coordinates:
    ``local = R^T (world - t)``. In row-vector form this is
    ``local = (world - t) @ R``. If the PLY is already object-local, leave both
    pose arguments unset. Bounding-box recentering is intentionally rejected for
    this bridge because it creates an implicit, data-dependent body frame.
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
    keep = np.ones((centers_np.shape[0],), dtype=bool)
    scores = np.ones((centers_np.shape[0],), dtype=np.float32)

    if object_id is not None:
        if "object_id" not in names:
            raise ValueError(f"{path} does not contain object_id fields needed for object filtering.")
        keep &= np.asarray(vertices["object_id"]).astype(np.int32) == int(object_id)

    if foreground_threshold is not None:
        if "foreground_logit" not in names:
            raise ValueError(f"{path} does not contain foreground_logit needed for foreground filtering.")
        foreground = 1.0 / (1.0 + np.exp(-np.asarray(vertices["foreground_logit"]).astype(np.float32)))
        keep &= foreground >= float(foreground_threshold)
        scores *= foreground

    if "opacity" in names:
        opacity = 1.0 / (1.0 + np.exp(-np.asarray(vertices["opacity"]).astype(np.float32)))
        scores *= opacity
        if opacity_threshold is not None:
            keep &= opacity >= float(opacity_threshold)
    elif opacity_threshold is not None:
        raise ValueError(f"{path} does not contain opacity needed for opacity filtering.")

    selected = np.nonzero(keep)[0]
    if selected.size == 0:
        raise ValueError(
            "No Gaussian primitives remain after filtering "
            f"(object_id={object_id}, foreground_threshold={foreground_threshold}, opacity_threshold={opacity_threshold})."
        )
    if max_primitives is not None and int(max_primitives) > 0 and selected.size > int(max_primitives):
        order = np.argsort(scores[selected])[::-1]
        selected = selected[order[: int(max_primitives)]]
        selected = np.sort(selected)

    centers_np = centers_np[selected]
    scale_names = sorted(
        (name for name in names if name.startswith("scale_")),
        key=lambda name: int(name.split("_")[-1]),
    )
    if scale_names:
        log_scales = np.stack([vertices[name] for name in scale_names], axis=-1).astype(np.float32)[selected]
        radii_np = np.exp(log_scales).mean(axis=-1) * float(radius_scale)
    else:
        radii_np = np.full((centers_np.shape[0],), float(min_radius), dtype=np.float32)
    radii_np = np.maximum(radii_np, float(min_radius))

    if world_rotation is not None and world_translation is None:
        raise ValueError("world_rotation requires world_translation so the object-local frame origin is explicit.")
    if recenter and world_translation is None:
        raise ValueError(
            "Implicit bbox recentering is not allowed for Stage1 -> Stage2 collision bodies. "
            "Declare stage1_gaussian_body.coordinate_frame='object_local' or provide stage1_gaussian_body.world_pose."
        )

    if world_translation is not None:
        t = np.asarray(world_translation, dtype=np.float32).reshape(3)
        centers_np = centers_np - t[None, :]
        if world_rotation is not None:
            R = np.asarray(world_rotation, dtype=np.float32).reshape(3, 3)
            centers_np = centers_np @ R

    centers = torch.as_tensor(centers_np, dtype=dtype, device=device)
    radii = torch.as_tensor(radii_np, dtype=dtype, device=device)
    query_points = centers if use_centers_as_queries else None
    return GaussianCollisionBody(centers, radii, query_points)


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
