import json
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


def collect_frames(source_path, splits):
    frames = []
    for split in splits:
        split_dir = source_path / split
        if not split_dir.exists():
            continue
        for image_path in sorted(split_dir.glob("*.png")):
            if "_depth_" in image_path.name:
                continue
            frames.append((split, image_path))
    return frames


def estimate_background_color(rgb):
    corners = np.concatenate(
        [
            rgb[:16, :16].reshape(-1, 3),
            rgb[:16, -16:].reshape(-1, 3),
            rgb[-16:, :16].reshape(-1, 3),
            rgb[-16:, -16:].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(corners.astype(np.float32), axis=0)


def normalize_confidence(confidence):
    confidence = np.asarray(confidence, dtype=np.float32)
    if confidence.size == 0:
        return confidence
    cmin = float(confidence.min())
    cmax = float(confidence.max())
    if cmax > cmin:
        confidence = (confidence - cmin) / (cmax - cmin)
    else:
        confidence = np.zeros_like(confidence, dtype=np.float32)
    return np.clip(confidence, 0.0, 1.0)


def extract_alpha_mask(image_rgba, alpha_threshold):
    alpha = image_rgba[..., 3].astype(np.float32) / 255.0
    labels = (alpha >= alpha_threshold).astype(np.int32)
    return labels, alpha.astype(np.float32), "alpha"


def extract_bg_subtract_mask(
    image_rgb,
    diff_threshold,
    blur_ksize,
    morph_kernel,
    min_area,
    force_background=None,
):
    rgb = image_rgb.astype(np.float32)
    if force_background == "white":
        bg_color = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    elif force_background == "black":
        bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        bg_color = estimate_background_color(rgb)

    diff = np.linalg.norm(rgb - bg_color[None, None, :], axis=-1)
    diff = normalize_confidence(diff)

    if blur_ksize > 1:
        ksize = int(blur_ksize)
        if ksize % 2 == 0:
            ksize += 1
        diff = cv2.GaussianBlur(diff, (ksize, ksize), 0)

    mask = (diff >= diff_threshold).astype(np.uint8)

    if morph_kernel > 0:
        kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)
        for label_idx in range(1, num_labels):
            if stats[label_idx, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == label_idx] = 1
        mask = filtered

    confidence = diff * mask.astype(np.float32)
    return mask.astype(np.int32), confidence.astype(np.float32), "bg_subtract"


def extract_mask(image_bgra, args):
    if image_bgra.ndim == 2:
        image_bgra = cv2.cvtColor(image_bgra, cv2.COLOR_GRAY2BGRA)
    if image_bgra.shape[-1] == 3:
        image_bgra = cv2.cvtColor(image_bgra, cv2.COLOR_BGR2BGRA)

    rgba = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2RGBA)
    rgb = rgba[..., :3]

    if args.method in {"auto", "alpha"} and rgba.shape[-1] == 4 and np.any(rgba[..., 3] < 255):
        return extract_alpha_mask(rgba, args.alpha_threshold)

    if args.method == "alpha":
        raise ValueError("Requested alpha-based mask extraction, but images do not contain usable alpha.")

    return extract_bg_subtract_mask(
        rgb,
        diff_threshold=args.diff_threshold,
        blur_ksize=args.blur_ksize,
        morph_kernel=args.morph_kernel,
        min_area=args.min_area,
        force_background=args.background_hint,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Automatically extract foreground object masks from dataset images.")
    parser.add_argument("--source_path", required=True, type=str, help="Dataset root containing train/test/val image folders.")
    parser.add_argument("--output_masks_dir", required=True, type=str, help="Output directory for split-aware mask label maps.")
    parser.add_argument("--output_confidence_dir", default="", type=str, help="Optional output directory for confidence maps.")
    parser.add_argument("--splits", nargs="+", default=["train", "test", "val"])
    parser.add_argument("--method", choices=["auto", "alpha", "bg_subtract"], default="auto")
    parser.add_argument("--alpha_threshold", default=0.01, type=float)
    parser.add_argument("--diff_threshold", default=0.12, type=float)
    parser.add_argument("--blur_ksize", default=5, type=int)
    parser.add_argument("--morph_kernel", default=5, type=int)
    parser.add_argument("--min_area", default=256, type=int)
    parser.add_argument("--background_hint", choices=["auto", "white", "black"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.source_path).expanduser().resolve()
    output_masks_dir = Path(args.output_masks_dir).expanduser().resolve()
    output_confidence_dir = Path(args.output_confidence_dir).expanduser().resolve() if args.output_confidence_dir else None
    background_hint = None if args.background_hint == "auto" else args.background_hint
    args.background_hint = background_hint

    frames = collect_frames(source_path, args.splits)
    if not frames:
        raise ValueError(f"No dataset frames found under {source_path}.")

    prepared_masks = 0
    prepared_confidence = 0
    extraction_methods = {}

    for split_name, image_path in frames:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {image_path}")

        label_map, confidence, resolved_method = extract_mask(image, args)

        split_mask_dir = output_masks_dir / split_name
        split_mask_dir.mkdir(parents=True, exist_ok=True)
        output_mask_path = split_mask_dir / f"{image_path.stem}.npy"
        if output_mask_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing mask: {output_mask_path}")
        np.save(output_mask_path, label_map.astype(np.int32))
        prepared_masks += 1
        extraction_methods.setdefault(resolved_method, 0)
        extraction_methods[resolved_method] += 1

        if output_confidence_dir is not None:
            split_conf_dir = output_confidence_dir / split_name
            split_conf_dir.mkdir(parents=True, exist_ok=True)
            output_conf_path = split_conf_dir / f"{image_path.stem}.npy"
            if output_conf_path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite existing confidence map: {output_conf_path}")
            np.save(output_conf_path, normalize_confidence(confidence).astype(np.float32))
            prepared_confidence += 1

    manifest = {
        "source_path": str(source_path),
        "output_masks_dir": str(output_masks_dir),
        "output_confidence_dir": str(output_confidence_dir) if output_confidence_dir is not None else "",
        "splits": args.splits,
        "requested_method": args.method,
        "resolved_method_counts": extraction_methods,
        "alpha_threshold": float(args.alpha_threshold),
        "diff_threshold": float(args.diff_threshold),
        "blur_ksize": int(args.blur_ksize),
        "morph_kernel": int(args.morph_kernel),
        "min_area": int(args.min_area),
        "background_hint": args.background_hint if args.background_hint is not None else "auto",
        "prepared_masks": int(prepared_masks),
        "prepared_confidence": int(prepared_confidence),
    }

    output_masks_dir.mkdir(parents=True, exist_ok=True)
    with open(output_masks_dir / "extract_object_masks_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if output_confidence_dir is not None:
        output_confidence_dir.mkdir(parents=True, exist_ok=True)
        with open(output_confidence_dir / "extract_object_masks_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"Prepared {prepared_masks} automatic masks into {output_masks_dir}.")
    if output_confidence_dir is not None:
        print(f"Prepared {prepared_confidence} confidence maps into {output_confidence_dir}.")
