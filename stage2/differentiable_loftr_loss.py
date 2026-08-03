"""LoFTR correspondence-guided differentiable image loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rgb_to_gray(images: torch.Tensor) -> torch.Tensor:
    weights = images.new_tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1)
    return torch.sum(images * weights, dim=1, keepdim=True)


def _sample_patches(
    images: torch.Tensor,
    batch_indices: torch.Tensor,
    keypoints_xy: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Bilinearly sample RGB patches, returning ``(matches, C, patch_pixels)``."""
    _, _, height, width = images.shape
    offsets = torch.stack(torch.meshgrid(
        torch.arange(-radius, radius + 1, device=images.device, dtype=images.dtype),
        torch.arange(-radius, radius + 1, device=images.device, dtype=images.dtype),
        indexing="ij",
    ), dim=-1).reshape(-1, 2)
    # meshgrid gives y,x; keypoints use x,y.
    offsets = offsets[:, [1, 0]]
    points = keypoints_xy.unsqueeze(1) + offsets.unsqueeze(0)
    grid_x = 2.0 * points[..., 0] / max(width - 1, 1) - 1.0
    grid_y = 2.0 * points[..., 1] / max(height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(2)
    selected = images[batch_indices]
    sampled = F.grid_sample(
        selected, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return sampled.squeeze(-1)


class LoFTRCorrespondenceLoss(nn.Module):
    """Match detached images with LoFTR, then compare differentiable RGB patches.

    Matching is deliberately detached because LoFTR's discrete match selection
    is not differentiable. The selected rendered patches remain live tensors,
    so the loss differentiates to rendered pixels, poses, dynamics and geometry.
    """

    def __init__(
        self,
        *,
        pretrained: str = "outdoor",
        confidence_threshold: float = 0.2,
        max_matches: int = 1024,
        min_matches: int = 8,
        patch_radius: int = 2,
        matcher: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if matcher is None:
            try:
                from kornia.feature import LoFTR
            except ImportError as exc:
                raise RuntimeError(
                    "LoFTR loss requires Kornia. Install `kornia` in the "
                    "gaussian_splatting environment or select another image loss."
                ) from exc
            matcher = LoFTR(pretrained=pretrained)
        self.matcher = matcher.eval()
        for parameter in self.matcher.parameters():
            parameter.requires_grad_(False)
        self.confidence_threshold = float(confidence_threshold)
        self.max_matches = int(max_matches)
        self.min_matches = int(min_matches)
        self.patch_radius = int(patch_radius)
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.max_matches < 1 or self.min_matches < 0:
            raise ValueError("max_matches must be positive and min_matches non-negative")
        if self.patch_radius < 0:
            raise ValueError("patch_radius must be non-negative")

    def forward(
        self,
        rendered: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if rendered.shape != targets.shape or rendered.ndim != 4:
            raise ValueError("LoFTR loss expects equal BCHW rendered/target tensors.")
        with torch.no_grad():
            matches = self.matcher({
                "image0": _rgb_to_gray(rendered.detach()),
                "image1": _rgb_to_gray(targets.detach()),
            })
            keypoints0 = matches["keypoints0"]
            keypoints1 = matches["keypoints1"]
            confidence = matches.get(
                "confidence", torch.ones(keypoints0.shape[0], device=keypoints0.device)
            )
            batch_indices = matches.get(
                "batch_indexes",
                torch.zeros(keypoints0.shape[0], dtype=torch.long, device=keypoints0.device),
            ).long()
            raw_count = int(confidence.numel())
            keep = confidence >= self.confidence_threshold
            confidence_count = int(keep.sum())
            if masks is not None and keypoints0.numel() > 0:
                mask0 = _sample_patches(
                    masks, batch_indices, keypoints0, 0
                ).reshape(-1)
                mask1 = _sample_patches(
                    masks, batch_indices, keypoints1, 0
                ).reshape(-1)
                keep = keep & (mask0 >= 0.5) & (mask1 >= 0.5)
            keypoints0, keypoints1 = keypoints0[keep], keypoints1[keep]
            confidence, batch_indices = confidence[keep], batch_indices[keep]
            if confidence.numel() > self.max_matches:
                selected = torch.topk(confidence, k=self.max_matches, sorted=False).indices
                keypoints0, keypoints1 = keypoints0[selected], keypoints1[selected]
                confidence, batch_indices = confidence[selected], batch_indices[selected]
        count = int(confidence.numel())
        if count < self.min_matches:
            # Preserve a valid zero gradient to rendered images.
            loss = rendered.sum() * 0.0
            return loss, {
                "loftr_loss": 0.0,
                "loftr_matches": count,
                "loftr_raw_matches": raw_count,
                "loftr_confidence_matches": confidence_count,
                "loftr_sufficient_matches": False,
            }
        rendered_patches = _sample_patches(
            rendered, batch_indices, keypoints0, self.patch_radius
        )
        target_patches = _sample_patches(
            targets, batch_indices, keypoints1, self.patch_radius
        )
        # Normalize each patch to emphasize local features over exposure.
        rendered_features = F.normalize(
            rendered_patches - rendered_patches.mean(dim=-1, keepdim=True),
            dim=-1,
        )
        target_features = F.normalize(
            target_patches - target_patches.mean(dim=-1, keepdim=True),
            dim=-1,
        )
        per_match = torch.mean(torch.abs(rendered_features - target_features), dim=(1, 2))
        weights = confidence / torch.clamp(confidence.sum(), min=1e-12)
        loss = torch.sum(weights * per_match)
        return loss, {
            "loftr_loss": float(loss.detach().cpu().item()),
            "loftr_matches": count,
            "loftr_raw_matches": raw_count,
            "loftr_confidence_matches": confidence_count,
            "loftr_mean_confidence": float(confidence.mean().detach().cpu().item()),
            "loftr_sufficient_matches": True,
        }
