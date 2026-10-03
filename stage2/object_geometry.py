"""Physical quantities derived from a Stage 1 spherical-Gaussian union.

* ``solid_mass_properties``: the region enclosed by the union of collision
  spheres (``r = 2s``), voxelized and hole-filled, gives the centre of mass and
  the inertia of a uniform-density solid.  No class or shape preset is used.
* ``plane_support_indices``: a plane can only touch the union on its convex
  hull, so the primitives that are extreme in some direction are the complete
  candidate set for Gaussian-plane contact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SolidMassProperties:
    center_of_mass: np.ndarray  # [3], Stage 1 world coordinates
    inertia_per_mass: np.ndarray  # [3, 3] about the COM, world axes, m^2
    volume: float
    voxel_size: float
    num_voxels: int

    def principal_inertia_per_mass(self) -> np.ndarray:
        return np.diag(self.inertia_per_mass).copy()

    def off_diagonal_ratio(self) -> float:
        diagonal = np.abs(np.diag(self.inertia_per_mass)).max()
        off = np.abs(self.inertia_per_mass - np.diag(np.diag(self.inertia_per_mass))).max()
        return float(off / max(diagonal, 1e-12))


def solidify_voxels(grid: np.ndarray, radius: int = 3) -> np.ndarray:
    """Fill the region enclosed by a possibly leaky voxel shell.

    The shell is dilated by ``radius`` voxels so gaps between surface
    primitives are sealed, the enclosed interior is filled, and the result is
    eroded by the same amount, which puts the boundary back on the shell.
    """
    kernel = 2 * int(radius) + 1
    pad = int(radius) + 1
    padded = np.pad(grid, pad)
    volume = torch.as_tensor(padded, dtype=torch.float32)[None, None]
    dilated = torch.nn.functional.max_pool3d(volume, kernel, stride=1, padding=int(radius))[0, 0] > 0.5
    filled = torch.as_tensor(fill_enclosed_voxels(dilated.numpy()), dtype=torch.float32)[None, None]
    eroded = 1.0 - torch.nn.functional.max_pool3d(1.0 - filled, kernel, stride=1, padding=int(radius))
    solid = (eroded[0, 0] > 0.5).numpy() | padded
    return solid[pad:-pad, pad:-pad, pad:-pad]


def fill_enclosed_voxels(grid: np.ndarray) -> np.ndarray:
    """Mark every empty voxel not 6-connected to the grid border as solid."""
    occupied = torch.as_tensor(grid)
    outside = torch.zeros_like(occupied)
    outside[0, :, :] = outside[-1, :, :] = True
    outside[:, 0, :] = outside[:, -1, :] = True
    outside[:, :, 0] = outside[:, :, -1] = True
    outside &= ~occupied
    kernel = torch.zeros((1, 1, 3, 3, 3))
    kernel[0, 0, 1, 1, :] = kernel[0, 0, 1, :, 1] = kernel[0, 0, :, 1, 1] = 1.0
    free = (~occupied).float()[None, None]
    while True:
        grown = torch.nn.functional.conv3d(outside.float()[None, None], kernel, padding=1)[0, 0] > 0
        grown = grown & ~occupied
        if torch.equal(grown | outside, outside):
            break
        outside = grown | outside
    del free
    return (~outside).numpy()


def solid_mass_properties(
    centers: np.ndarray, radii: np.ndarray, *, voxel_size: float | None = None, max_voxels_per_axis: int = 192,
) -> SolidMassProperties:
    """Uniform-density mass properties of the solid bounded by a sphere union."""
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    lower = (centers - radii[:, None]).min(axis=0)
    upper = (centers + radii[:, None]).max(axis=0)
    extent = upper - lower
    if voxel_size is None:
        voxel_size = float(extent.max() / max_voxels_per_axis)
    shape = np.maximum(np.ceil(extent / voxel_size).astype(int) + 3, 3)
    origin = lower - voxel_size
    grid = np.zeros(shape, dtype=bool)
    # Spheres no larger than a voxel mark the voxel holding their center;
    # larger ones mark every voxel whose center lies inside them.
    small = radii <= 0.5 * voxel_size
    cells = np.clip(np.floor((centers[small] - origin) / voxel_size).astype(int), 0, shape - 1)
    grid[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    for center, radius in zip(centers[~small], radii[~small]):
        lo = np.floor((center - radius - origin) / voxel_size).astype(int)
        hi = np.ceil((center + radius - origin) / voxel_size).astype(int) + 1
        lo = np.clip(lo, 0, shape - 1)
        hi = np.clip(hi, 1, shape)
        if np.any(hi <= lo):
            continue
        axes = [origin[a] + (np.arange(lo[a], hi[a]) + 0.5) * voxel_size for a in range(3)]
        dx, dy, dz = np.meshgrid(axes[0] - center[0], axes[1] - center[1], axes[2] - center[2], indexing="ij")
        inside = dx * dx + dy * dy + dz * dz <= max(radius, 0.5 * voxel_size) ** 2
        grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= inside
    # Seal sub-voxel gaps between surface primitives, then fill the interior.
    grid = solidify_voxels(grid)
    index = np.argwhere(grid)
    points = origin + (index + 0.5) * voxel_size
    com = points.mean(axis=0)
    offsets = points - com
    second = offsets.T @ offsets / float(points.shape[0])
    inertia = np.trace(second) * np.eye(3) - second
    inertia += np.eye(3) * (voxel_size ** 2) / 6.0  # each voxel's own inertia
    return SolidMassProperties(com, inertia, float(points.shape[0]) * voxel_size ** 3, float(voxel_size), int(points.shape[0]))


def fibonacci_directions(count: int) -> np.ndarray:
    index = np.arange(int(count), dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / float(count)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = math.pi * (3.0 - math.sqrt(5.0)) * index
    return np.stack((radius * np.cos(theta), radius * np.sin(theta), z), axis=1)


def _resolved_extreme_mask(centers: torch.Tensor, radii: torch.Tensor, directions: torch.Tensor, *,
                           tolerance: float, span: float, chunk: int = 64, pair_budget: int = 20_000_000) -> torch.Tensor:
    """Primitives that are extreme in a way the geometry's resolution can tell.

    A primitive counts in a direction when it reaches within ``tolerance`` of
    the support there and no primitive farther than ``span`` away reaches
    within ``tolerance`` of it.  On a face that is flat to within the
    tolerance over more than the span (or along the straight side of a
    cylinder) only its border survives; on an edge, a corner or a curved
    surface the extreme primitives themselves do.
    """
    keep = torch.zeros(centers.shape[0], dtype=torch.bool, device=centers.device)
    for start in range(0, directions.shape[0], chunk):
        block = directions[start:start + chunk]
        support = centers @ block.T + radii.unsqueeze(-1)
        top = support.max(dim=0).values
        for column in range(block.shape[0]):
            values = support[:, column]
            near = torch.nonzero(values >= top[column] - 2.0 * tolerance, as_tuple=False).squeeze(-1)
            near_values, near_centers = values[near], centers[near]
            candidates = torch.nonzero(near_values >= top[column] - tolerance, as_tuple=False).squeeze(-1)
            rows = max(1, pair_budget // max(1, int(near.numel())))
            for first in range(0, int(candidates.numel()), rows):
                part = candidates[first:first + rows]
                far = torch.cdist(near_centers[part], near_centers) > span
                reaches = near_values.unsqueeze(0) >= near_values[part].unsqueeze(1) - tolerance
                keep[near[part[~(far & reaches).any(dim=1)]]] = True
    return keep


def plane_support_indices(
    centers: torch.Tensor, radii: torch.Tensor, *, num_directions: int = 8192, chunk: int = 512,
    flat_tolerance: float = 0.0, flat_span: float | None = None,
) -> torch.Tensor:
    """Primitives that are the farthest along some direction (``c.u + r``).

    With a dense direction set these are the union's convex-hull primitives,
    i.e. every primitive a plane can touch first.

    ``flat_tolerance`` is the resolution of the geometry (how far the surface
    may be off).  A reconstructed flat face is never exactly flat: it bulges or
    ripples by a fraction of that resolution, and a plane would then touch it
    at an arbitrary interior point instead of along its border.  With a
    positive tolerance, primitives that are only extreme by less than the
    tolerance relative to primitives more than ``flat_span`` away are left out
    (see ``_resolved_extreme_mask``), so such a face is supported on its border.
    """
    directions = torch.as_tensor(fibonacci_directions(num_directions), dtype=centers.dtype, device=centers.device)
    with torch.no_grad():
        subset = torch.arange(centers.shape[0], device=centers.device)
        if flat_tolerance > 0.0:
            if flat_span is None:
                # a quarter of the body's smallest dimension: no face of the body is narrower than that
                flat_span = 0.25 * float((centers.max(dim=0).values - centers.min(dim=0).values).min())
            mask = _resolved_extreme_mask(centers, radii, directions, tolerance=float(flat_tolerance), span=float(flat_span))
            if bool(mask.any()):
                subset = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        chosen = []
        for start in range(0, directions.shape[0], chunk):
            block = directions[start:start + chunk]
            support = centers[subset] @ block.T + radii[subset].unsqueeze(-1)
            chosen.append(torch.argmax(support, dim=0))
        return subset[torch.unique(torch.cat(chosen))]
