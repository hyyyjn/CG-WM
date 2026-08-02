import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


COMMON_MASK_SUFFIXES = (".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
SCORE_KEYS = ("score", "predicted_iou", "stability_score", "confidence")
TRACK_ID_KEYS = ("track_id", "object_id", "instance_id", "id")


def encode_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim == 2:
        return mask.astype(np.int32)

    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            return mask[..., 0].astype(np.int32)
        if mask.shape[-1] == 4:
            mask = mask[..., :3]
        if mask.shape[-1] == 3:
            if np.issubdtype(mask.dtype, np.floating):
                mask = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
            return (
                (mask[..., 0].astype(np.int32) << 16)
                | (mask[..., 1].astype(np.int32) << 8)
                | mask[..., 2].astype(np.int32)
            )

    raise ValueError(f"Unsupported mask shape: {mask.shape}")


def load_array(path):
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True)

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            for key in ("mask", "masks", "segmentation", "labels", "panoptic_seg", "sem_seg", "logits", "scores"):
                if key in data:
                    return data[key]
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"Empty npz file: {path}")
            return data[keys[0]]

    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise FileNotFoundError(f"Unable to load file: {path}")
    if array.ndim == 3 and array.shape[-1] >= 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    return array


def normalize_confidence_map(array):
    array = np.asarray(array)
    if array.ndim == 3:
        if array.shape[-1] == 1:
            array = array[..., 0]
        else:
            array = array[..., :3].mean(axis=-1)

    array = array.astype(np.float32)
    if array.size == 0:
        raise ValueError("Confidence map is empty.")

    arr_min = float(array.min())
    arr_max = float(array.max())
    if arr_max > 1.0 or arr_min < 0.0:
        if arr_max > arr_min:
            array = (array - arr_min) / (arr_max - arr_min)
        else:
            array = np.ones_like(array, dtype=np.float32)
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def collect_dataset_frames(source_path, splits):
    frames = []
    for split in splits:
        split_dir = source_path / split
        if not split_dir.exists():
            continue
        for image_path in sorted(split_dir.glob("*.png")):
            if "_depth_" in image_path.name:
                continue
            frames.append((split, image_path.stem, image_path.name))
    return frames


def build_search_roots(root_dir, split_name, input_format):
    roots = [root_dir / split_name, root_dir]
    if input_format in {"sam2", "deva"}:
        roots.extend(
            [
                root_dir / "Annotations" / split_name,
                root_dir / "Annotations",
                root_dir / "masks" / split_name,
                root_dir / "masks",
                root_dir / "masklets" / split_name,
                root_dir / "masklets",
            ]
        )
    if input_format == "mask2former":
        roots.extend(
            [
                root_dir / "predictions" / split_name,
                root_dir / "predictions",
                root_dir / "semantic" / split_name,
                root_dir / "semantic",
                root_dir / "panoptic" / split_name,
                root_dir / "panoptic",
            ]
        )

    ordered = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            ordered.append(root)
            seen.add(key)
    return ordered


def is_valid_mask_candidate(path):
    if not path.exists():
        return False
    if path.is_file():
        return True
    if path.is_dir():
        return any(child.is_file() and child.suffix.lower() in COMMON_MASK_SUFFIXES for child in path.iterdir())
    return False


def find_matching_path(root_dir, split_name, image_stem, image_name, suffixes, input_format):
    roots = build_search_roots(root_dir, split_name, input_format)
    candidates = []
    suffixes = tuple(suffixes)

    for base in roots:
        candidates.extend(
            [
                base / image_name,
                base / image_stem,
            ]
        )
        for suffix in suffixes:
            candidates.extend(
                [
                    base / f"{image_name}{suffix}",
                    base / f"{image_stem}{suffix}",
                ]
            )

    for candidate in candidates:
        if is_valid_mask_candidate(candidate):
            return candidate

    recursive_names = [image_name, image_stem]
    for base in roots:
        if not base.exists():
            continue
        for name in recursive_names:
            matches = sorted(base.rglob(name))
            valid_matches = [match for match in matches if is_valid_mask_candidate(match)]
            if valid_matches:
                return valid_matches[0]
            for suffix in suffixes:
                matches = sorted(base.rglob(f"{name}{suffix}"))
                valid_matches = [match for match in matches if is_valid_mask_candidate(match)]
                if valid_matches:
                    return valid_matches[0]
    return None


def to_label_map(mask_array):
    encoded = encode_mask(mask_array)
    unique_values = np.unique(encoded)
    if unique_values.size <= 1:
        return encoded.astype(np.int32)

    if np.issubdtype(encoded.dtype, np.integer):
        min_value = int(unique_values.min())
        max_value = int(unique_values.max())
        if min_value >= 0 and max_value <= 65535:
            return encoded.astype(np.int32)

    remapped = np.zeros(encoded.shape, dtype=np.int32)
    next_label = 1
    for value in unique_values.tolist():
        if int(value) == 0:
            continue
        remapped[encoded == value] = next_label
        next_label += 1
    return remapped


def infer_channel_axis(array):
    if array.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {array.shape}")
    if array.shape[0] <= 256 and array.shape[0] < array.shape[1] and array.shape[0] < array.shape[2]:
        return 0
    if array.shape[-1] <= 256 and array.shape[-1] < array.shape[0] and array.shape[-1] < array.shape[1]:
        return 2
    raise ValueError(f"Unable to infer channel axis for tensor shape {array.shape}")


def logits_to_labels_and_confidence(array):
    array = np.asarray(array, dtype=np.float32)
    channel_axis = infer_channel_axis(array)
    if channel_axis != array.ndim - 1:
        array = np.moveaxis(array, channel_axis, -1)

    if array.shape[-1] == 1:
        probs = normalize_confidence_map(array[..., 0])
        labels = (probs >= 0.5).astype(np.int32)
        return labels, probs

    shifted = array - array.max(axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    denom = np.maximum(exp_scores.sum(axis=-1, keepdims=True), 1e-8)
    probs = exp_scores / denom
    labels = probs.argmax(axis=-1).astype(np.int32)
    confidence = probs.max(axis=-1).astype(np.float32)
    return labels, confidence


def load_json_score(path):
    if not path.exists():
        return 1.0
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return 1.0

    if isinstance(data, dict):
        for key in SCORE_KEYS:
            if key in data:
                return float(data[key])
    return 1.0


def load_json_dict(path):
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def extract_int_from_name(path):
    digits = []
    for char in path.stem:
        if char.isdigit():
            digits.append(char)
    if not digits:
        return None
    return int("".join(digits))


def infer_track_id(mask_file, metadata, preserve_track_ids):
    if not preserve_track_ids:
        return None

    for key in TRACK_ID_KEYS:
        if key in metadata:
            try:
                return int(metadata[key])
            except (TypeError, ValueError):
                pass

    return extract_int_from_name(mask_file)


def combine_binary_masks(mask_dir, preserve_track_ids=False):
    mask_files = sorted(
        [
            path
            for path in mask_dir.iterdir()
            if path.is_file() and path.suffix.lower() in COMMON_MASK_SUFFIXES
        ]
    )
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in directory: {mask_dir}")

    label_map = None
    confidence = None
    next_label = 1
    used_track_ids = set()

    for mask_file in mask_files:
        array = np.asarray(load_array(mask_file))
        if array.ndim == 3 and array.shape[-1] > 1:
            if array.shape[-1] <= 4:
                array = array[..., 0]
            else:
                raise ValueError(f"Binary mask stack file should be single-channel: {mask_file}")
        if array.ndim != 2:
            raise ValueError(f"Unsupported binary mask shape {array.shape} in {mask_file}")

        metadata = load_json_dict(mask_file.with_suffix(".json"))
        score = float(metadata.get("score", metadata.get("predicted_iou", metadata.get("stability_score", metadata.get("confidence", 1.0)))))
        track_id = infer_track_id(mask_file, metadata, preserve_track_ids)
        if np.issubdtype(array.dtype, np.bool_):
            mask_binary = array
            mask_conf = np.where(mask_binary, score, 0.0).astype(np.float32)
        elif np.issubdtype(array.dtype, np.floating):
            mask_conf = np.clip(array.astype(np.float32), 0.0, 1.0) * float(score)
            mask_binary = mask_conf > 0.5
        else:
            if int(array.max()) > 1:
                mask_conf = normalize_confidence_map(array) * float(score)
                mask_binary = array > 0
            else:
                mask_binary = array > 0
                mask_conf = np.where(mask_binary, float(score), 0.0).astype(np.float32)

        if label_map is None:
            label_map = np.zeros(mask_binary.shape, dtype=np.int32)
            confidence = np.zeros(mask_binary.shape, dtype=np.float32)

        assigned_label = next_label
        if track_id is not None and track_id > 0 and track_id not in used_track_ids:
            assigned_label = int(track_id)
        else:
            while next_label in used_track_ids or next_label <= 0:
                next_label += 1
            assigned_label = next_label

        overwrite = mask_binary & (mask_conf >= confidence)
        label_map[overwrite] = assigned_label
        confidence[overwrite] = mask_conf[overwrite]
        used_track_ids.add(int(assigned_label))
        if assigned_label >= next_label:
            next_label = assigned_label + 1

    return label_map.astype(np.int32), confidence.astype(np.float32)


def load_mask_and_confidence(mask_path, input_format):
    if mask_path.is_dir():
        return combine_binary_masks(mask_path, preserve_track_ids=(input_format == "deva"))

    array = load_array(mask_path)
    array = np.asarray(array)

    if input_format == "mask2former" and array.ndim == 3:
        return logits_to_labels_and_confidence(array)

    if input_format in {"sam2", "deva"} and array.ndim == 3 and array.shape[-1] > 4:
        return logits_to_labels_and_confidence(array)

    return to_label_map(array), None


def maybe_resize(array, width, height, interpolation):
    if array.shape[:2] == (height, width):
        return array
    return cv2.resize(array, (width, height), interpolation=interpolation)


if __name__ == "__main__":
    parser = ArgumentParser(description="Normalize external instance segmentation outputs for auto_assign_object_ids.py.")
    parser.add_argument("--source_path", required=True, type=str, help="Dataset root containing train/test/val images.")
    parser.add_argument("--input_masks_dir", required=True, type=str, help="External mask output root.")
    parser.add_argument("--output_masks_dir", required=True, type=str, help="Normalized output mask directory.")
    parser.add_argument("--input_confidence_dir", default="", type=str, help="Optional external confidence output root.")
    parser.add_argument("--output_confidence_dir", default="", type=str, help="Optional normalized confidence output directory.")
    parser.add_argument("--splits", nargs="+", default=["train", "test", "val"], help="Dataset splits to prepare.")
    parser.add_argument("--mask_suffixes", nargs="+", default=[".npy", ".npz", ".png"], help="Suffix search order for masks.")
    parser.add_argument("--confidence_suffixes", nargs="+", default=[".npy", ".npz", ".png"], help="Suffix search order for confidence maps.")
    parser.add_argument(
        "--input_format",
        default="generic",
        choices=["generic", "sam2", "mask2former", "deva"],
        help="Known external segmentation output format.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.source_path).expanduser().resolve()
    input_masks_dir = Path(args.input_masks_dir).expanduser().resolve()
    output_masks_dir = Path(args.output_masks_dir).expanduser().resolve()
    input_confidence_dir = Path(args.input_confidence_dir).expanduser().resolve() if args.input_confidence_dir else None
    output_confidence_dir = Path(args.output_confidence_dir).expanduser().resolve() if args.output_confidence_dir else None

    frames = collect_dataset_frames(source_path, args.splits)
    if not frames:
        raise ValueError(f"No dataset frames found under {source_path} for splits {args.splits}.")

    prepared_masks = 0
    prepared_confidence = 0
    missing_masks = []
    missing_confidence = []
    format_resolved_count = 0
    directory_stack_count = 0
    inline_confidence_count = 0
    explicit_confidence_count = 0

    for split_name, image_stem, image_name in frames:
        dataset_image_path = source_path / split_name / image_name
        dataset_image = cv2.imread(str(dataset_image_path), cv2.IMREAD_UNCHANGED)
        if dataset_image is None:
            raise FileNotFoundError(f"Unable to load dataset image: {dataset_image_path}")
        height, width = dataset_image.shape[:2]

        mask_path = find_matching_path(
            input_masks_dir,
            split_name,
            image_stem,
            image_name,
            args.mask_suffixes,
            args.input_format,
        )
        if mask_path is None:
            missing_masks.append(f"{split_name}:{image_name}")
            continue

        label_map, inline_confidence = load_mask_and_confidence(mask_path, args.input_format)
        label_map = maybe_resize(label_map.astype(np.int32), width, height, cv2.INTER_NEAREST)

        output_split_dir = output_masks_dir / split_name
        output_split_dir.mkdir(parents=True, exist_ok=True)
        output_mask_path = output_split_dir / f"{image_stem}.npy"
        if output_mask_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing mask: {output_mask_path}")
        np.save(output_mask_path, label_map.astype(np.int32))
        prepared_masks += 1
        format_resolved_count += 1
        if mask_path.is_dir():
            directory_stack_count += 1

        confidence = None
        if inline_confidence is not None:
            confidence = maybe_resize(
                normalize_confidence_map(inline_confidence),
                width,
                height,
                cv2.INTER_LINEAR,
            )
            inline_confidence_count += 1

        if input_confidence_dir is not None and output_confidence_dir is not None:
            confidence_path = find_matching_path(
                input_confidence_dir,
                split_name,
                image_stem,
                image_name,
                args.confidence_suffixes,
                args.input_format,
            )
            if confidence_path is None:
                if confidence is None:
                    missing_confidence.append(f"{split_name}:{image_name}")
            else:
                confidence = maybe_resize(
                    normalize_confidence_map(load_array(confidence_path)),
                    width,
                    height,
                    cv2.INTER_LINEAR,
                )
                explicit_confidence_count += 1

        if confidence is not None and output_confidence_dir is not None:
            conf_split_dir = output_confidence_dir / split_name
            conf_split_dir.mkdir(parents=True, exist_ok=True)
            output_conf_path = conf_split_dir / f"{image_stem}.npy"
            if output_conf_path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite existing confidence map: {output_conf_path}")
            np.save(output_conf_path, confidence.astype(np.float32))
            prepared_confidence += 1

    manifest = {
        "source_path": str(source_path),
        "input_masks_dir": str(input_masks_dir),
        "output_masks_dir": str(output_masks_dir),
        "input_confidence_dir": str(input_confidence_dir) if input_confidence_dir is not None else "",
        "output_confidence_dir": str(output_confidence_dir) if output_confidence_dir is not None else "",
        "input_format": args.input_format,
        "preserve_track_ids": args.input_format == "deva",
        "splits": args.splits,
        "prepared_masks": prepared_masks,
        "prepared_confidence": prepared_confidence,
        "format_resolved_count": format_resolved_count,
        "directory_stack_count": directory_stack_count,
        "inline_confidence_count": inline_confidence_count,
        "explicit_confidence_count": explicit_confidence_count,
        "missing_masks": missing_masks,
        "missing_confidence": missing_confidence,
    }

    output_masks_dir.mkdir(parents=True, exist_ok=True)
    with open(output_masks_dir / "prepare_instance_masks_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if output_confidence_dir is not None:
        output_confidence_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_masks_dir / "prepare_instance_masks_manifest.json", output_confidence_dir / "prepare_instance_masks_manifest.json")

    print(f"Prepared {prepared_masks} masks into {output_masks_dir}.")
    if output_confidence_dir is not None:
        print(f"Prepared {prepared_confidence} confidence maps into {output_confidence_dir}.")
    if missing_masks:
        print(f"Missing masks for {len(missing_masks)} frames.")
    if missing_confidence:
        print(f"Missing confidence maps for {len(missing_confidence)} frames.")
