import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from scene.dataset_readers import storePly
from utils.graphics_utils import fov2focal, focal2fov


@dataclass
class HullView:
    image_name: str
    image_path: str
    mask_path: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray
    T: np.ndarray
    split_name: str = ""

    @property
    def camera_center(self):
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = self.R.transpose()
        w2c[:3, 3] = self.T
        c2w = np.linalg.inv(w2c)
        return c2w[:3, 3]


def parse_args():
    parser = argparse.ArgumentParser(description="Build a dense visual hull seed point cloud from multi-view masks.")
    parser.add_argument("--source_path", required=True, type=str)
    parser.add_argument("--masks_dir", required=True, type=str)
    parser.add_argument("--images", default="images", type=str)
    parser.add_argument("--output_ply", default="", type=str)
    parser.add_argument("--grid_resolution", default=128, type=int)
    parser.add_argument("--bounds_scale", default=1.2, type=float)
    parser.add_argument("--mask_threshold", default=0.5, type=float)
    parser.add_argument("--max_points", default=200000, type=int)
    parser.add_argument("--chunk_size", default=262144, type=int)
    parser.add_argument("--skip_color", action="store_true")
    return parser.parse_args()


def resolve_mask_path(masks_root: Path, split_name: str, image_name: str):
    image_stem = Path(image_name).stem
    candidates = []
    if split_name:
        candidates.extend(
            [
                masks_root / split_name / f"{image_name}.png",
                masks_root / split_name / f"{image_name}.npy",
                masks_root / split_name / f"{image_stem}.png",
                masks_root / split_name / f"{image_stem}.npy",
            ]
        )
    candidates.extend(
        [
            masks_root / f"{image_name}.png",
            masks_root / f"{image_name}.npy",
            masks_root / f"{image_stem}.png",
            masks_root / f"{image_stem}.npy",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_mask(mask_path: Path, width: int, height: int, threshold: float):
    if mask_path.suffix == ".npy":
        mask = np.load(mask_path)
    else:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Unable to load mask: {mask_path}")

    mask = np.asarray(mask)
    if mask.ndim == 3:
        if mask.shape[-1] == 4:
            mask = mask[..., 3]
        else:
            mask = mask[..., :3].mean(axis=-1)
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask >= float(threshold)


def load_colmap_views(source_path: Path, images_dir: str, masks_root: Path):
    sparse_dir = source_path / "sparse" / "0"
    try:
        cam_extrinsics = read_extrinsics_binary(str(sparse_dir / "images.bin"))
        cam_intrinsics = read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
    except Exception:
        cam_extrinsics = read_extrinsics_text(str(sparse_dir / "images.txt"))
        cam_intrinsics = read_intrinsics_text(str(sparse_dir / "cameras.txt"))

    views = []
    for key in sorted(cam_extrinsics.keys()):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        image_path = source_path / images_dir / extr.name
        mask_path = resolve_mask_path(masks_root, "", extr.name)
        if mask_path is None:
            continue

        if intr.model == "SIMPLE_PINHOLE":
            fx = fy = float(intr.params[0])
            cx = float(intr.params[1])
            cy = float(intr.params[2])
        elif intr.model == "PINHOLE":
            fx = float(intr.params[0])
            fy = float(intr.params[1])
            cx = float(intr.params[2])
            cy = float(intr.params[3])
        else:
            raise ValueError(f"Unsupported COLMAP camera model for visual hull: {intr.model}")

        views.append(
            HullView(
                image_name=extr.name,
                image_path=str(image_path),
                mask_path=str(mask_path),
                width=int(intr.width),
                height=int(intr.height),
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                R=np.transpose(qvec2rotmat(extr.qvec)).astype(np.float32),
                T=np.asarray(extr.tvec, dtype=np.float32),
            )
        )
    return views


def load_synthetic_views(source_path: Path, masks_root: Path):
    views = []
    for split_name in ("train", "test", "val"):
        transforms_path = source_path / f"transforms_{split_name}.json"
        if not transforms_path.exists():
            continue
        import json

        with open(transforms_path, "r") as f:
            contents = json.load(f)
        fovx = float(contents["camera_angle_x"])
        for idx, frame in enumerate(contents["frames"]):
            image_path = source_path / f"{frame['file_path']}.png"
            image_name = image_path.name
            if not image_path.exists():
                continue
            image = Image.open(image_path)
            width, height = image.size
            focal = float(fov2focal(fovx, width))
            fx = focal
            fy = focal

            c2w = np.array(frame["transform_matrix"], dtype=np.float32)
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            mask_path = resolve_mask_path(masks_root, split_name, image_name)
            if mask_path is None:
                continue

            views.append(
                HullView(
                    image_name=image_name,
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                    width=int(width),
                    height=int(height),
                    fx=fx,
                    fy=fy,
                    cx=(width - 1) * 0.5,
                    cy=(height - 1) * 0.5,
                    R=np.transpose(w2c[:3, :3]).astype(np.float32),
                    T=w2c[:3, 3].astype(np.float32),
                    split_name=split_name,
                )
            )
    return views


def load_views(source_path: Path, images_dir: str, masks_root: Path):
    if (source_path / "sparse" / "0").exists():
        return load_colmap_views(source_path, images_dir, masks_root)
    if (source_path / "transforms_train.json").exists():
        return load_synthetic_views(source_path, masks_root)
    raise ValueError(f"Could not infer camera source for {source_path}")


def build_bounds(views, bounds_scale: float):
    centers = np.stack([view.camera_center for view in views], axis=0)
    center = centers.mean(axis=0)
    radius = np.linalg.norm(centers - center[None, :], axis=1).max()
    radius = max(float(radius * bounds_scale), 1e-2)
    bbox_min = center - radius
    bbox_max = center + radius
    return bbox_min.astype(np.float32), bbox_max.astype(np.float32)


def generate_grid_points(bbox_min, bbox_max, grid_resolution: int):
    axes = [
        np.linspace(float(bbox_min[i]), float(bbox_max[i]), int(grid_resolution), dtype=np.float32)
        for i in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    return grid.reshape(-1, 3), axes


def project_points(points, view: HullView):
    rot = view.R.transpose()
    cam = points @ rot.T + view.T[None, :]
    z = cam[:, 2]
    valid = z > 1e-6
    u = np.full((points.shape[0],), -1, dtype=np.int32)
    v = np.full((points.shape[0],), -1, dtype=np.int32)
    if np.any(valid):
        x = cam[valid, 0] / z[valid]
        y = cam[valid, 1] / z[valid]
        u_valid = np.round(view.fx * x + view.cx).astype(np.int32)
        v_valid = np.round(view.fy * y + view.cy).astype(np.int32)
        inside = (
            (u_valid >= 0) & (u_valid < view.width) &
            (v_valid >= 0) & (v_valid < view.height)
        )
        valid_indices = np.nonzero(valid)[0]
        selected = valid_indices[inside]
        u[selected] = u_valid[inside]
        v[selected] = v_valid[inside]
        valid = np.zeros_like(valid)
        valid[selected] = True
    return valid, u, v


def carve_visual_hull(points, views, masks_by_path, chunk_size):
    keep = np.ones((points.shape[0],), dtype=bool)
    for start in range(0, points.shape[0], chunk_size):
        end = min(start + chunk_size, points.shape[0])
        chunk = points[start:end]
        chunk_keep = np.ones((chunk.shape[0],), dtype=bool)
        for view in views:
            mask = masks_by_path[view.mask_path]
            valid, u, v = project_points(chunk, view)
            current = np.zeros((chunk.shape[0],), dtype=bool)
            if np.any(valid):
                current[valid] = mask[v[valid], u[valid]]
            chunk_keep &= current
            if not np.any(chunk_keep):
                break
        keep[start:end] = chunk_keep
    return keep

def colorize_points(points, views, chunk_size):
    colors = np.zeros((points.shape[0], 3), dtype=np.float32)
    counts = np.zeros((points.shape[0], 1), dtype=np.float32)
    image_cache = {}
    for view in views:
        if not os.path.exists(view.image_path):
            continue
        image_cache[view.image_path] = np.asarray(Image.open(view.image_path).convert("RGB"), dtype=np.float32) / 255.0

    for start in range(0, points.shape[0], chunk_size):
        end = min(start + chunk_size, points.shape[0])
        chunk = points[start:end]
        chunk_color = np.zeros((chunk.shape[0], 3), dtype=np.float32)
        chunk_count = np.zeros((chunk.shape[0], 1), dtype=np.float32)
        for view in views:
            image = image_cache.get(view.image_path)
            if image is None:
                continue
            valid, u, v = project_points(chunk, view)
            if not np.any(valid):
                continue
            chunk_color[valid] += image[v[valid], u[valid]]
            chunk_count[valid] += 1.0
        chunk_count = np.maximum(chunk_count, 1.0)
        colors[start:end] = chunk_color / chunk_count
        counts[start:end] = chunk_count

    missing = counts.squeeze(-1) <= 1e-6
    if np.any(missing):
        colors[missing] = 0.7
    return np.clip(colors, 0.0, 1.0)


def maybe_downsample(points, colors, max_points):
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    indices = np.random.choice(points.shape[0], size=max_points, replace=False)
    return points[indices], colors[indices]


def main():
    args = parse_args()
    source_path = Path(args.source_path).expanduser().resolve()
    masks_root = Path(args.masks_dir).expanduser().resolve()
    output_ply = Path(args.output_ply).expanduser() if args.output_ply else source_path / "visual_hull" / "visual_hull.ply"
    output_ply = output_ply.resolve()
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    views = load_views(source_path, args.images, masks_root)
    if not views:
        raise RuntimeError("No usable views were found. Check masks_dir/image names and camera inputs.")

    masks_by_path = {
        view.mask_path: load_mask(Path(view.mask_path), view.width, view.height, args.mask_threshold)
        for view in views
    }

    bbox_min, bbox_max = build_bounds(views, args.bounds_scale)
    points, _ = generate_grid_points(bbox_min, bbox_max, args.grid_resolution)
    keep = carve_visual_hull(points, views, masks_by_path, args.chunk_size)
    hull_points = points[keep]
    if hull_points.shape[0] == 0:
        raise RuntimeError("Visual hull carving removed every voxel. Try a larger bounds_scale or lower mask_threshold.")

    if args.skip_color:
        hull_colors = np.full((hull_points.shape[0], 3), 0.7, dtype=np.float32)
    else:
        hull_colors = colorize_points(hull_points, views, args.chunk_size)

    hull_points, hull_colors = maybe_downsample(hull_points, hull_colors, args.max_points)
    storePly(str(output_ply), hull_points.astype(np.float32), np.clip(hull_colors * 255.0, 0, 255).astype(np.uint8))

    occupied_ratio = float(keep.sum()) / float(max(points.shape[0], 1))
    print(f"Saved visual hull seed to {output_ply}")
    print(f"Views used: {len(views)}")
    print(f"Grid resolution: {args.grid_resolution}")
    print(f"Output points: {hull_points.shape[0]}")
    print(f"Occupied ratio: {occupied_ratio:.6f}")


if __name__ == "__main__":
    main()
