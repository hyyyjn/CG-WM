"""Side-by-side GIFs for a video physics fit (``run_video_physics_pipeline.py`` report).

Panels per frame:
  1. MuJoCo ground-truth frame (the observation)
  2. raw Gaussian render of the predicted rollout: Stage 1 object Gaussians at
     the predicted poses plus the Stage 1 floor Gaussians, one rasterizer call
  3. auxiliary overlay: ground-truth frame with the ground-truth object outline
     (red) and the predicted object outline (green)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1.gaussian_renderer import GaussianModel  # noqa: E402
from stage1.gaussian_renderer import render as gs_render  # noqa: E402
from stage2.renderable_gaussian_asset import (  # noqa: E402
    concatenate_gaussian_assets,
    copy_asset_to_gaussian_model,
    load_renderable_gaussian_asset,
    rigid_transform_asset,
)
from stage2.video_physics_fit import ObjectRenderer, _Pipe, build_object_model, load_episode  # noqa: E402


def outline(mask: np.ndarray) -> np.ndarray:
    inside = mask > 0.5
    edge = np.zeros_like(inside)
    edge[1:-1, 1:-1] = inside[1:-1, 1:-1] & ~(
        inside[:-2, 1:-1] & inside[2:, 1:-1] & inside[1:-1, :-2] & inside[1:-1, 2:])
    return edge


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--floor_ply", default=None, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--background", default="0.85,0.88,0.92")
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--source", default="open_loop", choices=("open_loop", "rolling_forecast"),
                        help="open_loop: one rollout from the first window; rolling_forecast: every frame "
                             "predicted a few frames after the last observed one (held-out episodes only)")
    parser.add_argument("--floor_band", default=0.01, type=float,
                        help="keep floor Gaussians within this distance of the support plane (visual only)")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    dataset_root = Path(report["dataset_root"])
    obj = build_object_model(report["stage1_ply"], report.get("stage1_dataset", dataset_root / "stage1" / "can"),
                             device="cuda")
    floor = None
    if args.floor_ply is not None:
        floor = load_renderable_gaussian_asset(args.floor_ply, device="cuda")
        normal = torch.as_tensor(obj.support_normal, dtype=floor.xyz.dtype, device="cuda")
        # The floor reconstruction leaves floaters where the floor was never
        # visible (sky, under the object); only primitives on the support plane
        # are part of the floor surface.
        on_plane = (floor.xyz @ normal - obj.support_height).abs() <= float(args.floor_band)
        floor = floor.index_select(torch.nonzero(on_plane & (floor.activated_opacity[:, 0] > 0.05)).squeeze(-1))
    background = torch.tensor([float(v) for v in args.background.split(",")], device="cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots = {Path(p).name + "@" + Path(p).parent.parent.name: Path(p)
             for p in report["train_episodes"] + report["eval_episodes"]}
    for group in ("train", "holdout"):
        for row in report["results"][group]:
            split, name = row["episode"].split("/")
            title = "Gaussian render of open-loop rollout"
            if args.source == "rolling_forecast":
                if "rolling_forecast" not in row:
                    continue
                row = row["rolling_forecast"]
                title = "Gaussian render, predicted {}-{} frames ahead".format(*row["lead_frames"])
            root = roots[f"{name}@{split}"]
            episode = load_episode(root, device="cuda", load_ground_truth=True)
            renderer = ObjectRenderer(obj.asset, episode.camera)
            positions = torch.tensor(row["predicted_com_positions"], device="cuda")
            quaternions = torch.tensor(row["predicted_com_quaternions_wxyz"], device="cuda")
            frames = []
            for index in range(0, len(episode.frames), max(1, args.stride)):
                with torch.no_grad():
                    posed = rigid_transform_asset(obj.asset, positions[index], quaternions[index])
                    scene = posed if floor is None else concatenate_gaussian_assets(posed, floor)
                    model = copy_asset_to_gaussian_model(scene, GaussianModel(scene.sh_degree))
                    image = gs_render(episode.camera, model, _Pipe(), background, separate_sh=False)["render"]
                    _, alpha = renderer(positions[index], quaternions[index])
                gt = (episode.rgb[index].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pred = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                overlay = gt.copy()
                overlay[outline(episode.mask[index, 0].cpu().numpy())] = (255, 40, 40)
                overlay[outline(alpha[0].cpu().numpy())] = (40, 230, 40)
                panel = Image.fromarray(np.concatenate((gt, pred, overlay), axis=1))
                draw = ImageDraw.Draw(panel)
                width = gt.shape[1]
                draw.text((8, 8), f"MuJoCo GT  frame {episode.frames[index]}", fill=(0, 0, 0))
                draw.text((width + 8, 8), title, fill=(0, 0, 0))
                draw.text((2 * width + 8, 8),
                          f"outline GT(red) pred(green)  err {row['per_frame_translation_m'][index] * 100:.1f} cm"
                          f" / {row['per_frame_rotation_deg'][index]:.1f} deg", fill=(0, 0, 0))
                frames.append(panel.resize((panel.width // 2, panel.height // 2), Image.BILINEAR))
            path = args.output_dir / f"{args.source}_{group}_{split}_{name}.gif"
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=66, loop=0)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
