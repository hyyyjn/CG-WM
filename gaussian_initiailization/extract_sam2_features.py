import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Extract SAM2 feature maps for NeRF synthetic images.")
    parser.add_argument("--source_path", required=True, type=str, help="Dataset root containing train/test/val image folders.")
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
        "--splits",
        nargs="+",
        default=["train", "test", "val"],
        help="Dataset splits to process.",
    )
    return parser.parse_args()


def collect_images(split_dir: Path):
    return sorted(
        path for path in split_dir.glob("*.png")
        if path.is_file() and "_depth_" not in path.name
    )


def reduce_to_three_channels(feature_map: torch.Tensor) -> np.ndarray:
    if feature_map.ndim != 3:
        raise ValueError(f"Expected CHW feature map, got shape {tuple(feature_map.shape)}")

    channels = feature_map.shape[0]
    if channels == 1:
        reduced = feature_map.repeat(3, 1, 1)
    elif channels == 2:
        reduced = torch.cat([feature_map, feature_map[:1]], dim=0)
    elif channels == 3:
        reduced = feature_map
    else:
        chunk_size = int(np.ceil(channels / 3.0))
        chunks = []
        for chunk_idx in range(3):
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
        split_dir = source_path / split
        if not split_dir.exists():
            print(f"Skipping missing split directory: {split_dir}")
            continue

        image_paths = collect_images(split_dir)
        split_output_dir = output_root / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        for image_path in tqdm(image_paths, desc=f"Extracting {split}", unit="image"):
            image = np.array(Image.open(image_path).convert("RGB"))
            predictor.set_image(image)
            feature_map = select_feature_map(predictor, args.feature_source)
            reduced = reduce_to_three_channels(feature_map)
            np.save(split_output_dir / f"{image_path.stem}.npy", reduced)
            total_saved += 1

    print(f"Saved {total_saved} SAM2 feature maps to {output_root}")


if __name__ == "__main__":
    main()
