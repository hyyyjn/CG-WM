import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Extract SAM2 feature maps for SG-GS Stage 1 images.")
    parser.add_argument("--source_path", required=True, type=str, help="Dataset root containing images or transforms files.")
    parser.add_argument(
        "--output_dir",
        default="sam_features_sam2",
        type=str,
        help="Output directory name or absolute path for saved .npy feature maps.",
    )
    parser.add_argument(
        "--checkpoint",
        default="/home/cgr-ugrad-2026/work/sam2_repo/checkpoints/sam2.1_hiera_tiny.pt",
        type=str,
        help="Path to the SAM2 checkpoint.",
    )
    parser.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        type=str,
        help="SAM2 config name understood by build_sam2.",
    )
    parser.add_argument(
        "--feature_source",
        default="high_res0",
        choices=["high_res0", "high_res1", "image_embed"],
        help="Which SAM2 feature map to export before reducing channels.",
    )
    parser.add_argument(
        "--output_channels",
        default=3,
        type=int,
        help="Number of channels to save after reducing the raw SAM2 feature map.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "val"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--images_root",
        default="images",
        type=str,
        help="Preferred image root for split-aware datasets, e.g. source_path/images/train.",
    )
    return parser.parse_args()


def collect_images_from_dir(split_dir: Path):
    return sorted(
        path for path in split_dir.glob("*.png")
        if path.is_file() and "_depth_" not in path.name
    )


def resolve_transform_image_path(source_path: Path, file_path: str) -> Path:
    raw = Path(file_path)
    candidate = raw if raw.is_absolute() else source_path / raw
    if candidate.suffix:
        return candidate
    for suffix in (".png", ".jpg", ".jpeg"):
        with_suffix = candidate.with_suffix(suffix)
        if with_suffix.exists():
            return with_suffix
    return candidate.with_suffix(".png")


def collect_images(source_path: Path, split: str, images_root: str):
    # edit this: prefer the Stage 1 layout, then keep legacy NeRF synthetic fallbacks.
    split_dirs = [
        source_path / images_root / split,
        source_path / split,
    ]
    for split_dir in split_dirs:
        if split_dir.exists():
            image_paths = collect_images_from_dir(split_dir)
            if image_paths:
                return image_paths

    transforms_path = source_path / f"transforms_{split}.json"
    if transforms_path.exists():
        with open(transforms_path, "r") as f:
            frames = json.load(f).get("frames", [])
        image_paths = []
        for frame in frames:
            image_path = resolve_transform_image_path(source_path, frame["file_path"])
            if image_path.exists() and "_depth_" not in image_path.name:
                image_paths.append(image_path)
        return sorted(image_paths)

    return []


def reduce_feature_channels(feature_map: torch.Tensor, output_channels: int) -> np.ndarray:
    if feature_map.ndim != 3:
        raise ValueError(f"Expected CHW feature map, got shape {tuple(feature_map.shape)}")
    if output_channels <= 0:
        raise ValueError(f"output_channels must be positive, got {output_channels}")

    channels = feature_map.shape[0]
    if channels == output_channels:
        reduced = feature_map
    elif channels < output_channels:
        repeats = int(np.ceil(output_channels / float(channels)))
        reduced = feature_map.repeat(repeats, 1, 1)[:output_channels]
    else:
        chunk_size = int(np.ceil(channels / float(output_channels)))
        chunks = []
        for chunk_idx in range(output_channels):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, channels)
            if start >= channels:
                chunks.append(chunks[-1])
            else:
                chunks.append(feature_map[start:end].mean(dim=0, keepdim=True))
        reduced = torch.cat(chunks, dim=0)

    return reduced.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32)


def select_feature_map(predictor: SAM2ImagePredictor, feature_source: str) -> torch.Tensor:
    if feature_source == "image_embed":
        return predictor._features["image_embed"][0]
    if feature_source == "high_res0":
        return predictor._features["high_res_feats"][0][0]
    if feature_source == "high_res1":
        return predictor._features["high_res_feats"][1][0]
    raise ValueError(f"Unsupported feature source: {feature_source}")


def resolve_output_dir(source_path: Path, output_dir_arg: str) -> Path:
    output_dir = Path(output_dir_arg)
    return output_dir if output_dir.is_absolute() else source_path / output_dir


def main():
    args = parse_args()
    source_path = Path(args.source_path).expanduser().resolve()
    output_root = resolve_output_dir(source_path, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_model = build_sam2(args.config, args.checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam_model)

    total_saved = 0
    for split in args.splits:
        image_paths = collect_images(source_path, split, args.images_root)
        if not image_paths:
            print(f"Skipping split with no images: {split}")
            continue

        split_output_dir = output_root / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        for image_path in tqdm(image_paths, desc=f"Extracting {split}", unit="image"):
            image = np.array(Image.open(image_path).convert("RGB"))
            predictor.set_image(image)
            feature_map = select_feature_map(predictor, args.feature_source)
            reduced = reduce_feature_channels(feature_map, args.output_channels)
            np.save(split_output_dir / f"{image_path.stem}.npy", reduced)
            total_saved += 1

    print(f"Saved {total_saved} SAM2 feature maps to {output_root}")


if __name__ == "__main__":
    main()
