import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate COLMAP poses from foreground-masked multi-view images."
    )
    parser.add_argument("--source_path", required=True, type=str)
    parser.add_argument("--masks_dir", required=True, type=str)
    parser.add_argument("--images", default="images", type=str)
    parser.add_argument("--workspace_dir", default="", type=str)
    parser.add_argument("--camera", default="OPENCV", type=str)
    parser.add_argument("--colmap_executable", default="colmap", type=str)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--skip_matching", action="store_true")
    parser.add_argument("--mask_threshold", default=0.5, type=float)
    parser.add_argument("--mask_dilate", default=0, type=int)
    parser.add_argument("--background_mode", default="white", choices=["white", "black", "keep"])
    parser.add_argument("--copy_to_images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_mask_path(masks_root: Path, image_name: str):
    image_stem = Path(image_name).stem
    candidates = [
        masks_root / f"{image_name}.png",
        masks_root / f"{image_name}.npy",
        masks_root / f"{image_stem}.png",
        masks_root / f"{image_stem}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_mask(mask_path: Path, width: int, height: int, threshold: float, mask_dilate: int):
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
        mask /= 255.0
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = mask >= float(threshold)

    if mask_dilate > 0:
        kernel = np.ones((mask_dilate, mask_dilate), dtype=np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray, background_mode: str):
    if background_mode == "keep":
        return image

    background_value = 255 if background_mode == "white" else 0
    output = np.full_like(image, background_value)
    output[mask] = image[mask]
    return output


def run_command(command):
    logging.info("Running: %s", " ".join(command))
    subprocess.run(command, check=True)


def prepare_workspace(workspace_dir: Path, overwrite: bool):
    if workspace_dir.exists() and overwrite:
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "input").mkdir(exist_ok=True)
    (workspace_dir / "distorted" / "sparse").mkdir(parents=True, exist_ok=True)


def write_masked_inputs(source_images_dir: Path, masks_root: Path, workspace_input_dir: Path, threshold: float, mask_dilate: int, background_mode: str):
    image_paths = sorted(
        path for path in source_images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        raise RuntimeError(f"No images were found in {source_images_dir}")

    used = 0
    skipped = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped.append(image_path.name)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        mask_path = resolve_mask_path(masks_root, image_path.name)
        if mask_path is None:
            skipped.append(image_path.name)
            continue

        mask = load_mask(mask_path, width, height, threshold, mask_dilate)
        masked = apply_mask(image, mask, background_mode)
        output_path = workspace_input_dir / image_path.name
        cv2.imwrite(str(output_path), cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
        used += 1

    if used == 0:
        raise RuntimeError("No masked input images were generated. Check image names and masks_dir.")
    return used, skipped


def run_colmap_pipeline(workspace_dir: Path, colmap_executable: str, camera_model: str, use_gpu: bool, skip_matching: bool):
    database_path = workspace_dir / "distorted" / "database.db"
    input_dir = workspace_dir / "input"
    sparse_dir = workspace_dir / "distorted" / "sparse"
    use_gpu_int = "0" if not use_gpu else "1"

    if not skip_matching:
        run_command(
            [
                colmap_executable,
                "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(input_dir),
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", camera_model,
                "--SiftExtraction.use_gpu", use_gpu_int,
            ]
        )

        run_command(
            [
                colmap_executable,
                "exhaustive_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", use_gpu_int,
            ]
        )

        run_command(
            [
                colmap_executable,
                "mapper",
                "--database_path", str(database_path),
                "--image_path", str(input_dir),
                "--output_path", str(sparse_dir),
            ]
        )

    run_command(
        [
            colmap_executable,
            "image_undistorter",
            "--image_path", str(input_dir),
            "--input_path", str(sparse_dir / "0"),
            "--output_path", str(workspace_dir),
            "--output_type", "COLMAP",
        ]
    )


def finalize_workspace(workspace_dir: Path, source_path: Path, copy_to_images: bool):
    sparse_root = workspace_dir / "sparse"
    sparse_zero = sparse_root / "0"
    sparse_zero.mkdir(parents=True, exist_ok=True)
    for child in list(sparse_root.iterdir()):
        if child.name == "0":
            continue
        shutil.move(str(child), str(sparse_zero / child.name))

    if copy_to_images:
        target_images_dir = source_path / "images"
        target_images_dir.mkdir(exist_ok=True)
        for image_path in (workspace_dir / "images").iterdir():
            if image_path.is_file():
                shutil.copy2(image_path, target_images_dir / image_path.name)

    target_sparse_dir = source_path / "sparse"
    if target_sparse_dir.exists():
        shutil.rmtree(target_sparse_dir)
    shutil.copytree(sparse_root, target_sparse_dir)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    source_path = Path(args.source_path).expanduser().resolve()
    source_images_dir = source_path / args.images
    masks_root = Path(args.masks_dir).expanduser().resolve()
    workspace_dir = (
        Path(args.workspace_dir).expanduser().resolve()
        if args.workspace_dir
        else source_path / "masked_colmap"
    )

    prepare_workspace(workspace_dir, args.overwrite)
    used, skipped = write_masked_inputs(
        source_images_dir=source_images_dir,
        masks_root=masks_root,
        workspace_input_dir=workspace_dir / "input",
        threshold=args.mask_threshold,
        mask_dilate=args.mask_dilate,
        background_mode=args.background_mode,
    )
    logging.info("Prepared %d masked input images", used)
    if skipped:
        logging.info("Skipped %d images without usable masks", len(skipped))

    run_colmap_pipeline(
        workspace_dir=workspace_dir,
        colmap_executable=args.colmap_executable,
        camera_model=args.camera,
        use_gpu=not args.no_gpu,
        skip_matching=args.skip_matching,
    )
    finalize_workspace(workspace_dir, source_path, args.copy_to_images)

    logging.info("Masked COLMAP workspace: %s", workspace_dir)
    logging.info("Updated sparse reconstruction at: %s", source_path / "sparse" / "0")


if __name__ == "__main__":
    main()
