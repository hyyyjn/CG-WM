from __future__ import annotations

import argparse
import json
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

from gaussian_initiailization.stage2.differentiable_gaussian_render_loss import (  # noqa: E402
    GaussianRenderLossConfig,
    Stage2GaussianRenderLoss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test gradient flow from Gaussian RGB render loss back to Stage2 rigid pose tensors."
    )
    parser.add_argument("--stage1_ply", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--image_width", default=96, type=int)
    parser.add_argument("--image_height", default=72, type=int)
    parser.add_argument("--scale_multiplier", default=1.0, type=float)
    parser.add_argument("--position_delta", default="0.018,-0.012,0.006")
    parser.add_argument("--quaternion_delta", default="0.996,0.0,0.0,0.087")
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--allow_cpu_skip", action="store_true")
    parser.add_argument("--min_position_grad_norm", default=0.0, type=float)
    parser.add_argument("--min_quaternion_grad_norm", default=0.0, type=float)
    return parser.parse_args()


def parse_vec(text: str, *, length: int, label: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != length:
        raise ValueError(f"{label} must contain {length} comma-separated values, got {text!r}.")
    return values


def save_rgb_tensor(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (image.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(array).save(path)


def assert_grad(name: str, grad: torch.Tensor | None, *, min_norm: float) -> dict:
    if grad is None:
        raise RuntimeError(f"{name} did not receive a gradient.")
    finite = bool(torch.isfinite(grad).all().detach().cpu().item())
    norm = float(torch.linalg.norm(grad.detach()).cpu().item())
    if not finite:
        raise RuntimeError(f"{name} gradient contains NaN or Inf.")
    if norm < float(min_norm):
        raise RuntimeError(f"{name} gradient norm {norm:.6g} is below required minimum {min_norm:.6g}.")
    return {f"{name}_grad_norm": norm, f"{name}_grad_finite": finite}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        message = "Gaussian render-loss backward smoke requires CUDA because gaussian_renderer.render uses CUDA tensors."
        if args.allow_cpu_skip:
            print(f"[SKIP] {message}")
            return
        raise RuntimeError(message)

    device = torch.device("cuda")
    config = GaussianRenderLossConfig(
        image_width=max(16, int(args.image_width)),
        image_height=max(16, int(args.image_height)),
        white_background=bool(args.white_background),
        scale_multiplier=float(args.scale_multiplier),
        loss="l1",
    )
    output_dir = args.output_dir.resolve()
    gt_rgb_dir = output_dir / "gt_rgb"
    target_position = torch.tensor([[[0.0, 0.0, 0.08]]], dtype=torch.float32, device=device)
    target_quaternion = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32, device=device)

    renderer = Stage2GaussianRenderLoss(
        stage1_ply=args.stage1_ply.resolve(),
        gt_rgb_dir=gt_rgb_dir,
        frame_indices=[],
        config=config,
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        target_rgb = renderer.render_frame(target_position[0], target_quaternion[0])
    save_rgb_tensor(gt_rgb_dir / "000000.png", target_rgb)

    loss_module = Stage2GaussianRenderLoss(
        stage1_ply=args.stage1_ply.resolve(),
        gt_rgb_dir=gt_rgb_dir,
        frame_indices=[0],
        config=config,
        dtype=torch.float32,
        device=device,
    )
    position_delta = torch.tensor(parse_vec(args.position_delta, length=3, label="--position_delta"), dtype=torch.float32, device=device)
    quaternion_delta = torch.tensor(
        parse_vec(args.quaternion_delta, length=4, label="--quaternion_delta"),
        dtype=torch.float32,
        device=device,
    )
    positions = (target_position + position_delta.reshape(1, 1, 3)).detach().clone().requires_grad_(True)
    quaternions = quaternion_delta.reshape(1, 1, 4).detach().clone().requires_grad_(True)
    loss, diagnostics = loss_module(positions, quaternions)
    loss.backward()

    report = {
        "stage1_ply": str(args.stage1_ply.resolve()),
        "gt_rgb_dir": str(gt_rgb_dir),
        "loss": float(loss.detach().cpu().item()),
        **diagnostics,
        **assert_grad("position", positions.grad, min_norm=float(args.min_position_grad_norm)),
        **assert_grad("quaternion", quaternions.grad, min_norm=float(args.min_quaternion_grad_norm)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "gaussian_render_loss_backward_smoke.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    report["report"] = str(report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
