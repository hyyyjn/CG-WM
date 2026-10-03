"""Silhouette- and support-consistent spherical Gaussian geometry for Stage 1.

Stage 2 treats every object Gaussian as a collision sphere of radius ``r = 2s``.
A multi-view reconstruction only constrains what the cameras see, so
Gaussians can stick out of the object's silhouettes (surface fuzz, floaters)
or hang below the surface the object rests on (its underside is never
observed).  Both would change the physical shape.

This module enforces two constraints that follow from the Stage 1 inputs alone:

* silhouette: in every training view where the center projects inside the
  image, the projected sphere must stay inside the object mask;
* support: when the dataset declares the plane the object rests on, no sphere
  may cross it.

``project`` prunes Gaussians whose center violates a constraint and clamps
the remaining scales so ``2s`` never exceeds the available clearance.  It is a
projection step applied between optimizer updates, so rendering and collision
keep sharing one isotropic scale ``s``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SilhouetteView:
    world_view: torch.Tensor  # [4, 4] row-vector world->camera (3DGS convention)
    fx: float
    fy: float
    cx: float  # principal point in pixel-index coordinates (pixel i covers [i - 0.5, i + 0.5])
    cy: float
    width: int
    height: int
    signed_distance: torch.Tensor  # [1, 1, H, W] pixels, positive inside the mask


MASK_UPSAMPLE = 4


def mask_signed_distance(mask: np.ndarray) -> np.ndarray:
    """Pixel distance to the mask boundary, positive inside and negative outside.

    The boundary is the 0.5 level of the mask interpolated between pixel
    centres, located on a ``MASK_UPSAMPLE`` times finer grid: a coverage
    (anti-aliased) mask gives the edge to a fraction of a pixel, a binary mask
    gives it half-way between the last inside and the first outside centre.
    """
    height, width = mask.shape
    fine = cv2.resize(mask.astype(np.float32), (width * MASK_UPSAMPLE, height * MASK_UPSAMPLE),
                      interpolation=cv2.INTER_LINEAR)
    inside = (fine > 0.5).astype(np.uint8)
    distance_inside = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    distance_outside = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
    # distances run between fine sample centres; the boundary is half a fine sample closer
    signed = np.where(inside > 0, distance_inside - 0.5, 0.5 - distance_outside) / MASK_UPSAMPLE
    return signed.reshape(height, MASK_UPSAMPLE, width, MASK_UPSAMPLE).mean(axis=(1, 3)).astype(np.float32)


def load_support_plane(source_path: str | Path) -> tuple[np.ndarray, float] | None:
    manifest_path = Path(source_path) / "dataset_manifest.json"
    if not manifest_path.exists():
        return None
    plane = json.loads(manifest_path.read_text(encoding="utf-8")).get("support_plane")
    if not plane:
        return None
    normal = np.asarray(plane["normal"], dtype=np.float32)
    return normal / np.linalg.norm(normal), float(plane.get("height", 0.0))


class SilhouetteGeometry:
    def __init__(self, cameras, *, support_plane=None, radius_factor: float = 2.0, device="cuda"):
        self.radius_factor = float(radius_factor)
        self.device = torch.device(device)
        self.views: list[SilhouetteView] = []
        for camera in cameras:
            mask = getattr(camera, "object_mask", None)
            if mask is None:
                continue
            mask_np = mask.detach().float().cpu().numpy()[0]
            height, width = mask_np.shape
            fx = width / (2.0 * math.tan(float(camera.FoVx) * 0.5))
            fy = height / (2.0 * math.tan(float(camera.FoVy) * 0.5))
            sdf = torch.from_numpy(mask_signed_distance(mask_np)).to(self.device)
            # The rasterizer places the optical axis at the image centre, which is
            # index (W - 1) / 2: the signed-distance image is sampled by pixel index.
            self.views.append(SilhouetteView(
                camera.world_view_transform.detach().to(self.device).float(),
                fx, fy, (width - 1) * 0.5, (height - 1) * 0.5, width, height, sdf[None, None],
            ))
        if not self.views:
            raise ValueError("silhouette geometry needs cameras with object masks")
        self.support_normal = None
        self.support_height = 0.0
        if support_plane is not None:
            normal, height = support_plane
            self.support_normal = torch.as_tensor(normal, dtype=torch.float32, device=self.device)
            self.support_height = float(height)

    @torch.no_grad()
    def clearance(self, xyz: torch.Tensor) -> torch.Tensor:
        """Metric room around each center before a sphere leaves a silhouette or crosses the support."""
        xyz = xyz.to(self.device).float()
        homogeneous = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=1)
        clearance = torch.full((xyz.shape[0],), float("inf"), device=self.device)
        for view in self.views:
            camera = homogeneous @ view.world_view
            depth = camera[:, 2]
            valid = depth > 1e-4
            u = view.fx * camera[:, 0] / depth.clamp_min(1e-4) + view.cx
            v = view.fy * camera[:, 1] / depth.clamp_min(1e-4) + view.cy
            in_frame = valid & (u >= 0) & (u <= view.width - 1) & (v >= 0) & (v <= view.height - 1)
            grid = torch.stack((2.0 * u / (view.width - 1) - 1.0, 2.0 * v / (view.height - 1) - 1.0), dim=-1)
            pixels = F.grid_sample(
                view.signed_distance, grid.view(1, -1, 1, 2), mode="bilinear", align_corners=True,
            ).view(-1)
            metric = pixels * depth / view.fy
            clearance = torch.where(in_frame, torch.minimum(clearance, metric), clearance)
        if self.support_normal is not None:
            support = xyz @ self.support_normal - self.support_height
            clearance = torch.minimum(clearance, support)
        return clearance

    @torch.no_grad()
    def hull_surface_samples(self, interior_points: torch.Tensor, *, max_points: int = 60000):
        """Points on the silhouette hull surface and their outward normals.

        Interior points within 5% of the object extent of the surface are moved
        outward along the clearance gradient by their clearance.
        """
        points = interior_points.to(self.device).float()
        clearance = self.clearance(points)
        self.extent = float((points.max(dim=0).values - points.min(dim=0).values).max())
        shell = 0.05 * self.extent
        near = (clearance > 0.0) & (clearance < shell)
        points, clearance = points[near], clearance[near]
        if points.shape[0] > int(max_points):
            keep = torch.randperm(points.shape[0], device=self.device)[: int(max_points)]
            points, clearance = points[keep], clearance[keep]
        step = 0.25 * shell
        gradient = torch.zeros_like(points)
        for axis in range(3):
            offset = torch.zeros(3, device=self.device)
            offset[axis] = step
            gradient[:, axis] = self.clearance(points + offset) - self.clearance(points - offset)
        outward = -gradient / gradient.norm(dim=1, keepdim=True).clamp_min(1e-9)
        surface = points + outward * clearance.unsqueeze(-1)
        # The clearance is a distance only to first order (the limiting view of
        # an interior point is rarely exactly tangent): iterate onto its zero level.
        for _ in range(3):
            surface = surface + outward * self.clearance(surface).unsqueeze(-1)
        residual = self.clearance(surface).abs()
        valid = residual < 0.01 * shell
        self.surface_residual = float(residual[valid].mean()) if bool(valid.any()) else float("nan")
        return surface[valid], outward[valid]

    @torch.no_grad()
    def _outermost_inside(self, origin: torch.Tensor, directions: torch.Tensor, lower: torch.Tensor,
                          upper: torch.Tensor, *, steps: int = 16) -> torch.Tensor:
        """Bisection for the hull crossing of each ray between an inside ``lower`` and an outside ``upper``."""
        lower, upper = lower.clone(), upper.clone()
        for _ in range(int(steps)):
            middle = 0.5 * (lower + upper)
            inside = self.clearance(origin + middle.unsqueeze(-1) * directions) > 0.0
            lower = torch.where(inside, middle, lower)
            upper = torch.where(inside, upper, middle)
        return lower

    @torch.no_grad()
    def hull_support_points(self, interior_points: torch.Tensor, *, num_directions: int = 8192,
                            rounds: int = 7, coarse_steps: int = 128):
        """Points of the silhouette hull that are extreme in some direction, with those directions.

        A plane touches a body first at such a point, so they are the exact
        candidates for contact with a flat support - including sharp edges and
        corners, which a union of spheres tangent to the hull from inside
        always rounds off.  Rays from the centre find the hull surface by
        bisection of the clearance; for each direction the ray whose crossing
        is farthest along it is refined on a shrinking patch of rays.
        """
        points = interior_points.to(self.device).float()
        origin = points.mean(dim=0)
        reach = 2.0 * float((points - origin).norm(dim=1).max())
        index = torch.arange(int(num_directions), device=self.device, dtype=torch.float32) + 0.5
        z = 1.0 - 2.0 * index / float(num_directions)
        ring = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        theta = math.pi * (3.0 - math.sqrt(5.0)) * index
        normals = torch.stack((ring * torch.cos(theta), ring * torch.sin(theta), z), dim=1)

        def crossing(directions, guess=None, margin=None):
            if guess is None:  # coarse march: the last inside sample along the whole ray
                ts = torch.linspace(0.0, reach, int(coarse_steps), device=self.device)
                inside = torch.stack([self.clearance(origin + t * directions) > 0.0 for t in ts], dim=1)
                last = (inside.float() * torch.arange(int(coarse_steps), device=self.device)).argmax(dim=1)
                lower, upper = ts[last], ts[torch.clamp(last + 1, max=int(coarse_steps) - 1)]
            else:  # near a known crossing
                lower, upper = torch.clamp(guess - margin, min=0.0), guess + margin
                # keep the bracket valid: fall back to the guess when it is not inside/outside
                lower = torch.where(self.clearance(origin + lower.unsqueeze(-1) * directions) > 0.0, lower,
                                    torch.zeros_like(lower))
            return self._outermost_inside(origin, directions, lower, upper)

        distance = crossing(normals)
        surface = origin + distance.unsqueeze(-1) * normals
        best = torch.argmax(normals @ surface.T, dim=1)  # for each direction, the farthest surface sample
        rays, reach_along = normals[best].clone(), distance[best].clone()
        spread = math.sqrt(4.0 * math.pi / float(num_directions))
        offsets = torch.tensor([(a, b) for a in (-1.0, -0.5, 0.0, 0.5, 1.0) for b in (-1.0, -0.5, 0.0, 0.5, 1.0)],
                               device=self.device)
        for _ in range(int(rounds)):
            helper = torch.where(rays[:, :1].abs() < 0.9, torch.tensor([1.0, 0.0, 0.0], device=self.device),
                                 torch.tensor([0.0, 1.0, 0.0], device=self.device))
            first = F.normalize(torch.cross(rays, helper, dim=1), dim=1)
            second = torch.cross(rays, first, dim=1)
            candidates = F.normalize(rays[:, None, :] + spread * (offsets[None, :, :1] * first[:, None, :]
                                                                    + offsets[None, :, 1:] * second[:, None, :]), dim=2)
            count = candidates.shape[1]
            flat = candidates.reshape(-1, 3)
            guess = reach_along.repeat_interleave(count)
            found = crossing(flat, guess, torch.full_like(guess, 2.0 * spread * reach))
            points_found = (origin + found.unsqueeze(-1) * flat).reshape(-1, count, 3)
            along = (points_found * normals[:, None, :]).sum(dim=2)
            pick = torch.argmax(along, dim=1)
            rows = torch.arange(rays.shape[0], device=self.device)
            rays, reach_along = candidates[rows, pick], found.reshape(-1, count)[rows, pick]
            spread *= 0.5
        support = origin + reach_along.unsqueeze(-1) * rays
        # one point per 0.01% of the extent
        cell = 1e-4 * reach
        _, first_index = np.unique(torch.round(support / cell).long().cpu().numpy(), axis=0, return_index=True)
        keep = torch.as_tensor(np.sort(first_index), device=self.device)
        return support[keep], normals[keep]

    @torch.no_grad()
    def add_support_points(self, gaussians, *, radius_fraction: float = 1e-3) -> dict:
        """Add a small Gaussian at every extreme point of the silhouette hull (see ``hull_support_points``).

        Its collision sphere (``radius_fraction`` of the object extent) touches
        the hull at the extreme point from inside; its appearance is copied
        from the nearest Gaussian and, at this size, does not change the renders.
        """
        support, normals = self.hull_support_points(gaussians.get_xyz.detach())
        radius = float(radius_fraction) * self.extent
        centers = support - normals * radius
        _, source = self.union_gap(gaussians, centers)
        log_scale = torch.full((centers.shape[0], 3), math.log(radius / self.radius_factor), device=self.device)
        had_tmp = getattr(gaussians, "tmp_radii", None) is not None
        if not had_tmp:
            gaussians.tmp_radii = torch.zeros(gaussians.get_xyz.shape[0], device=gaussians.get_xyz.device)
        gaussians.densification_postfix(
            centers, gaussians._features_dc[source], gaussians._features_rest[source],
            gaussians._features_geo[source], gaussians._foreground_logit[source],
            gaussians._object_ids[source], gaussians._opacity[source], log_scale,
            gaussians._rotation[source], torch.zeros(centers.shape[0], device=gaussians.get_xyz.device),
        )
        if not had_tmp:
            gaussians.tmp_radii = None
        count = gaussians.get_xyz.shape[0]
        gaussians.xyz_gradient_accum = torch.zeros((count, 1), device="cuda")
        gaussians.denom = torch.zeros((count, 1), device="cuda")
        gaussians.max_radii2D = torch.zeros((count), device="cuda")
        return {"support_points_added": int(centers.shape[0]), "support_point_radius_mm": 1e3 * radius,
                "num_gaussians": int(count)}

    @torch.no_grad()
    def union_gap(self, gaussians, points: torch.Tensor, *, chunk: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
        """Signed distance from ``points`` to the collision union, and the nearest primitive."""
        centers = gaussians.get_xyz
        radii = self.radius_factor * gaussians.get_scaling.mean(dim=1)
        gaps, nearest = [], []
        for start in range(0, points.shape[0], int(chunk)):
            distance = torch.cdist(points[start:start + int(chunk)], centers) - radii.unsqueeze(0)
            value, index = distance.min(dim=1)
            gaps.append(value)
            nearest.append(index)
        return torch.cat(gaps), torch.cat(nearest)

    @torch.no_grad()
    def add_surface_coverage(self, gaussians, surface_points: torch.Tensor, outward: torch.Tensor, *,
                             radius_fraction: float = 0.015, tolerance_fraction: float = 0.0015,
                             rounds: int = 3) -> dict:
        """Add Gaussians whose collision spheres touch the hull where the union does not reach it.

        The silhouette projection keeps the union inside the hull, but a sphere
        only touches it at one point, so the union surface is a set of peaks.
        Wherever a hull-surface sample is farther than the tolerance from the
        union, a Gaussian tangent to the hull from inside is added there, with
        the appearance of its nearest neighbour (later iterations refine it).
        """
        radius = float(radius_fraction) * self.extent
        tolerance = float(tolerance_fraction) * self.extent
        spacing = math.sqrt(2.0 * radius * tolerance)
        added = 0
        for _ in range(int(rounds)):
            gap, nearest = self.union_gap(gaussians, surface_points)
            uncovered = torch.nonzero(gap > tolerance, as_tuple=False).squeeze(-1)
            if uncovered.numel() == 0:
                break
            # one new sphere per ``spacing`` cell of uncovered surface
            cells = torch.floor(surface_points[uncovered] / spacing).long()
            _, first = np.unique(cells.cpu().numpy(), axis=0, return_index=True)
            chosen = uncovered[torch.as_tensor(first, device=self.device)]
            rho = torch.full((chosen.numel(),), radius, device=self.device)
            for _ in range(4):  # shrink until the sphere fits inside the hull
                centers = surface_points[chosen] - outward[chosen] * rho.unsqueeze(-1)
                rho = torch.minimum(rho, self.clearance(centers).clamp_min(1e-5))
            centers = surface_points[chosen] - outward[chosen] * rho.unsqueeze(-1)
            source = nearest[chosen]
            log_scale = torch.log(rho / self.radius_factor).unsqueeze(-1).expand(-1, 3)
            had_tmp = getattr(gaussians, "tmp_radii", None) is not None
            if not had_tmp:
                gaussians.tmp_radii = torch.zeros(gaussians.get_xyz.shape[0], device=gaussians.get_xyz.device)
            gaussians.densification_postfix(
                centers, gaussians._features_dc[source], gaussians._features_rest[source],
                gaussians._features_geo[source], gaussians._foreground_logit[source],
                gaussians._object_ids[source], gaussians._opacity[source], log_scale.contiguous(),
                gaussians._rotation[source], torch.zeros(chosen.numel(), device=gaussians.get_xyz.device),
            )
            if not had_tmp:
                gaussians.tmp_radii = None
            count = gaussians.get_xyz.shape[0]
            gaussians.xyz_gradient_accum = torch.zeros((count, 1), device="cuda")
            gaussians.denom = torch.zeros((count, 1), device="cuda")
            gaussians.max_radii2D = torch.zeros((count), device="cuda")
            added += int(chosen.numel())
        gap, _ = self.union_gap(gaussians, surface_points)
        return {"coverage_added": added, "coverage_gap_mean_mm": float(gap.clamp_min(0).mean() * 1e3),
                "coverage_gap_p95_mm": float(torch.quantile(gap.clamp_min(0)[:100000], 0.95) * 1e3),
                "num_gaussians": int(gaussians.get_xyz.shape[0])}

    @torch.no_grad()
    def project(self, gaussians, *, min_scale: float = 1e-5) -> dict:
        """Prune centers outside the silhouettes/support and clamp ``2s`` to the clearance."""
        clearance = self.clearance(gaussians.get_xyz)
        outside = ~(clearance > 0.0)
        pruned = int(outside.sum())
        if pruned:
            if getattr(gaussians, "tmp_radii", None) is None:
                gaussians.tmp_radii = torch.zeros(gaussians.get_xyz.shape[0], device=gaussians.get_xyz.device)
                gaussians.prune_points(outside)
                gaussians.tmp_radii = None
            else:
                gaussians.prune_points(outside)
            clearance = clearance[~outside]
        limit = torch.clamp(clearance / self.radius_factor, min=float(min_scale))
        log_limit = torch.log(limit).unsqueeze(-1)
        scaling = gaussians._scaling.data
        clamped = int((scaling > log_limit).any(dim=1).sum())
        scaling.copy_(torch.minimum(scaling, log_limit.expand_as(scaling)))
        return {
            "silhouette_pruned": pruned,
            "silhouette_scale_clamped": clamped,
            "num_gaussians": int(gaussians.get_xyz.shape[0]),
        }
