"""Build deterministic geometry feature targets from synthetic instance masks.

This is a dependency-free substitute for SAM2 features when exact simulator
instance masks are available.  Real captures should still use
``extract_sam2_features.py``.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--masks_dir", default="instance_masks")
    parser.add_argument("--output_dir", default="mask_geometry_features")
    parser.add_argument("--channels", type=int, default=3)
    return parser.parse_args()


def label_features(labels: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 0:
        raise ValueError("--channels must be positive")
    labels = labels.astype(np.int64)
    features = np.zeros((*labels.shape, channels), dtype=np.float32)
    # A stable sinusoidal code works for any integer instance label and keeps
    # nearby/background labels distinct without hard-coding object names.
    for channel in range(channels):
        frequency = channel // 2 + 1
        if channel % 2 == 0:
            features[..., channel] = np.cos(labels * frequency * np.pi / 3.0)
        else:
            features[..., channel] = np.sin(labels * frequency * np.pi / 3.0)
    return features


def main():
    args = parse_args()
    source = Path(args.source_path).expanduser().resolve()
    masks_root = source / args.masks_dir
    output_root = source / args.output_dir
    saved = 0
    for split in ("train", "test", "val"):
        split_root = masks_root / split
        if not split_root.exists():
            continue
        output_split = output_root / split
        output_split.mkdir(parents=True, exist_ok=True)
        for mask_path in sorted(split_root.glob("*.png")):
            labels = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if labels is None:
                raise FileNotFoundError(mask_path)
            if labels.ndim == 3:
                labels = labels[..., 0]
            np.save(output_split / f"{mask_path.stem}.npy", label_features(labels, args.channels))
            saved += 1
    if saved == 0:
        raise RuntimeError(f"No PNG masks found below {masks_root}")
    print(f"Saved {saved} geometry feature maps to {output_root}")


if __name__ == "__main__":
    main()
