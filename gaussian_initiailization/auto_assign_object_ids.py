import json
import os
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

import cv2
import numpy as np
import torch

from arguments import ModelParams
from gaussian_renderer import GaussianModel
from scene import Scene
from utils.general_utils import safe_state


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


def load_mask(mask_path, width, height):
    if mask_path.suffix == ".npy":
        mask = np.load(mask_path)
    else:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Unable to load mask: {mask_path}")
        if mask.ndim == 3 and mask.shape[-1] >= 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    encoded = encode_mask(mask)
    if encoded.shape[:2] != (height, width):
        encoded = cv2.resize(encoded, (width, height), interpolation=cv2.INTER_NEAREST)
    return encoded.astype(np.int32)


def load_confidence_map(confidence_path, width, height):
    if confidence_path.suffix == ".npy":
        confidence = np.load(confidence_path)
    else:
        confidence = cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED)
        if confidence is None:
            raise FileNotFoundError(f"Unable to load confidence map: {confidence_path}")

    confidence = np.asarray(confidence)
    if confidence.ndim == 3:
        if confidence.shape[-1] == 1:
            confidence = confidence[..., 0]
        else:
            confidence = confidence[..., :3].mean(axis=-1)

    confidence = confidence.astype(np.float32)
    if confidence.size == 0:
        raise ValueError(f"Empty confidence map: {confidence_path}")

    conf_min = float(confidence.min())
    conf_max = float(confidence.max())
    if conf_max > 1.0 or conf_min < 0.0:
        if conf_max > conf_min:
            confidence = (confidence - conf_min) / (conf_max - conf_min)
        else:
            confidence = np.ones_like(confidence, dtype=np.float32)

    if confidence.shape[:2] != (height, width):
        confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def compute_boundary_weights(mask, ignored_labels, boundary_band_width, boundary_min_weight):
    weights = np.ones(mask.shape, dtype=np.float32)
    if boundary_band_width <= 0:
        return weights

    unique_labels = np.unique(mask)
    for label in unique_labels.tolist():
        if int(label) in ignored_labels:
            continue
        region = (mask == label).astype(np.uint8)
        if region.sum() == 0:
            continue

        distance = cv2.distanceTransform(region, cv2.DIST_L2, 3)
        local_weights = np.ones_like(distance, dtype=np.float32)
        band_mask = distance < float(boundary_band_width)
        if np.any(band_mask):
            normalized = np.clip(distance[band_mask] / float(boundary_band_width), 0.0, 1.0)
            local_weights[band_mask] = boundary_min_weight + (1.0 - boundary_min_weight) * normalized
        weights[region.astype(bool)] = np.minimum(weights[region.astype(bool)], local_weights[region.astype(bool)])

    return np.clip(weights, boundary_min_weight, 1.0).astype(np.float32)


def resolve_mask_path(masks_dir, split_name, image_name):
    image_stem = Path(image_name).stem
    candidates = [
        masks_dir / split_name / f"{image_name}.png",
        masks_dir / split_name / f"{image_name}.npy",
        masks_dir / split_name / f"{image_stem}.png",
        masks_dir / split_name / f"{image_stem}.npy",
        masks_dir / f"{image_name}.png",
        masks_dir / f"{image_name}.npy",
        masks_dir / f"{image_stem}.png",
        masks_dir / f"{image_stem}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_aux_path(root_dir, split_name, image_name):
    if root_dir is None:
        return None
    image_stem = Path(image_name).stem
    candidates = [
        root_dir / split_name / f"{image_name}.npy",
        root_dir / split_name / f"{image_name}.png",
        root_dir / split_name / f"{image_stem}.npy",
        root_dir / split_name / f"{image_stem}.png",
        root_dir / f"{image_name}.npy",
        root_dir / f"{image_name}.png",
        root_dir / f"{image_stem}.npy",
        root_dir / f"{image_stem}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_ignore_ids(ignore_ids_arg):
    if not ignore_ids_arg:
        return set()
    return {int(item.strip()) for item in ignore_ids_arg.split(",") if item.strip()}


def project_gaussians(camera, xyz):
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    xyz_h = torch.cat((xyz, ones), dim=1)
    view = xyz_h @ camera.world_view_transform
    clip = xyz_h @ camera.full_proj_transform
    w = clip[:, 3]
    valid = w > 1e-8

    ndc = torch.zeros_like(clip[:, :3])
    ndc[valid] = clip[valid, :3] / w[valid, None]

    x = ((ndc[:, 0] + 1.0) * 0.5) * (camera.image_width - 1)
    y = ((1.0 - ndc[:, 1]) * 0.5) * (camera.image_height - 1)

    visible = (
        valid
        & (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
        & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
        & (ndc[:, 2] >= 0.0)
    )
    depth = view[:, 2]
    visible = visible & (depth > 0.0)
    return visible, x.round().long(), y.round().long(), depth


def compute_frontmost_mask(visible_idx, px, py, depth, width, height, depth_tolerance):
    if visible_idx.size == 0:
        return np.zeros((0,), dtype=bool), 0

    pixel_ids = py.astype(np.int64) * int(width) + px.astype(np.int64)
    order = np.lexsort((depth, pixel_ids))
    sorted_pixels = pixel_ids[order]
    sorted_depth = depth[order]

    front_depth_by_pixel = {}
    unique_front_pixels = 0
    for pix, dep in zip(sorted_pixels, sorted_depth):
        if pix not in front_depth_by_pixel:
            front_depth_by_pixel[pix] = dep
            unique_front_pixels += 1

    keep = np.zeros((visible_idx.shape[0],), dtype=bool)
    for local_idx, (pix, dep) in enumerate(zip(pixel_ids, depth)):
        if dep <= front_depth_by_pixel[pix] + depth_tolerance:
            keep[local_idx] = True
    return keep, unique_front_pixels


def save_grouped_model(scene, dataset, output_model_path, save_iteration, object_ids, source_model_path):
    scene.model_path = output_model_path
    dataset.model_path = output_model_path
    os.makedirs(output_model_path, exist_ok=True)

    point_cloud_dir = os.path.join(output_model_path, f"point_cloud/iteration_{save_iteration}")
    os.makedirs(point_cloud_dir, exist_ok=True)
    scene.gaussians.set_object_ids(object_ids)
    scene.gaussians.save_ply(os.path.join(point_cloud_dir, "point_cloud.ply"))

    source_exposure_path = os.path.join(source_model_path, "exposure.json")
    if os.path.exists(source_exposure_path):
        shutil.copy2(source_exposure_path, os.path.join(output_model_path, "exposure.json"))

    with open(os.path.join(output_model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(dataset))))


def accumulate_view_votes(
    views_by_split,
    masks_dir,
    xyz,
    ignored_labels,
    confidence_maps_dir,
    confidence_threshold,
    boundary_band_width,
    boundary_min_weight,
    disable_occlusion_filter,
    depth_tolerance,
):
    vote_counts = {}
    used_views = []
    skipped_views = []
    total_visible_projections = 0
    total_frontmost_projections = 0
    total_confident_frontmost_projections = 0

    per_view_observations = []

    for split_name, views in views_by_split:
        for view in views:
            mask_path = resolve_mask_path(masks_dir, split_name, view.image_name)
            if mask_path is None:
                skipped_views.append(f"{split_name}:{view.image_name}")
                continue

            mask = load_mask(mask_path, view.image_width, view.image_height)
            boundary_weights = compute_boundary_weights(
                mask,
                ignored_labels,
                boundary_band_width,
                boundary_min_weight,
            )
            confidence_map = None
            confidence_path = resolve_aux_path(confidence_maps_dir, split_name, view.image_name)
            if confidence_path is not None:
                confidence_map = load_confidence_map(confidence_path, view.image_width, view.image_height)

            visible, px, py, depth = project_gaussians(view, xyz)
            visible_idx = torch.nonzero(visible, as_tuple=False).squeeze(1).detach().cpu().numpy()
            if visible_idx.size == 0:
                skipped_views.append(f"{split_name}:{view.image_name}")
                continue

            vis_px = px[visible].detach().cpu().numpy()
            vis_py = py[visible].detach().cpu().numpy()
            vis_depth = depth[visible].detach().cpu().numpy()
            total_visible_projections += int(visible_idx.size)

            if disable_occlusion_filter:
                frontmost_mask = np.ones((visible_idx.shape[0],), dtype=bool)
            else:
                frontmost_mask, _ = compute_frontmost_mask(
                    visible_idx,
                    vis_px,
                    vis_py,
                    vis_depth,
                    view.image_width,
                    view.image_height,
                    depth_tolerance,
                )

            frontmost_idx = visible_idx[frontmost_mask]
            total_frontmost_projections += int(frontmost_idx.size)
            if frontmost_idx.size == 0:
                skipped_views.append(f"{split_name}:{view.image_name}")
                continue

            sampled_labels = mask[vis_py[frontmost_mask], vis_px[frontmost_mask]]
            sampled_boundary_weights = boundary_weights[vis_py[frontmost_mask], vis_px[frontmost_mask]]
            if confidence_map is not None:
                sampled_confidence = confidence_map[vis_py[frontmost_mask], vis_px[frontmost_mask]]
                confident_mask = sampled_confidence >= confidence_threshold
                frontmost_idx = frontmost_idx[confident_mask]
                sampled_labels = sampled_labels[confident_mask]
                sampled_boundary_weights = sampled_boundary_weights[confident_mask]
                sampled_confidence = sampled_confidence[confident_mask]
            else:
                sampled_confidence = np.ones((frontmost_idx.shape[0],), dtype=np.float32)

            if frontmost_idx.size == 0:
                skipped_views.append(f"{split_name}:{view.image_name}")
                continue

            valid_label_mask = ~np.isin(sampled_labels, list(ignored_labels))
            frontmost_idx = frontmost_idx[valid_label_mask]
            sampled_labels = sampled_labels[valid_label_mask]
            sampled_boundary_weights = sampled_boundary_weights[valid_label_mask]
            sampled_confidence = sampled_confidence[valid_label_mask]
            total_confident_frontmost_projections += int(frontmost_idx.size)
            if frontmost_idx.size == 0:
                skipped_views.append(f"{split_name}:{view.image_name}")
                continue

            sampled_weights = sampled_confidence * sampled_boundary_weights

            unique_labels = np.unique(sampled_labels)
            for label in unique_labels.tolist():
                label_mask = sampled_labels == label
                if label not in vote_counts:
                    vote_counts[label] = np.zeros((xyz.shape[0],), dtype=np.float32)
                np.add.at(vote_counts[label], frontmost_idx[label_mask], sampled_weights[label_mask])

            per_view_observations.append(
                {
                    "view_name": f"{split_name}:{view.image_name}",
                    "gaussian_indices": frontmost_idx,
                    "labels": sampled_labels.astype(np.int32),
                    "weights": sampled_weights.astype(np.float32),
                }
            )
            used_views.append(f"{split_name}:{view.image_name}")

    return {
        "vote_counts": vote_counts,
        "used_views": used_views,
        "skipped_views": skipped_views,
        "total_visible_projections": total_visible_projections,
        "total_frontmost_projections": total_frontmost_projections,
        "total_confident_frontmost_projections": total_confident_frontmost_projections,
        "per_view_observations": per_view_observations,
    }


def assign_labels_from_votes(vote_counts, num_gaussians, background_id, min_votes):
    if vote_counts:
        sorted_labels = sorted(vote_counts.keys())
        stacked_votes = np.stack([vote_counts[label] for label in sorted_labels], axis=1)
        best_label_idx = stacked_votes.argmax(axis=1)
        best_votes = stacked_votes[np.arange(num_gaussians), best_label_idx]
        object_ids = np.full((num_gaussians,), background_id, dtype=np.int32)
        has_votes = best_votes >= float(min_votes)
        object_ids[has_votes] = np.asarray(sorted_labels, dtype=np.int32)[best_label_idx[has_votes]]
    else:
        object_ids = np.full((num_gaussians,), background_id, dtype=np.int32)
    return object_ids


def refine_object_ids(object_ids, vote_counts, per_view_observations, background_id, consistency_threshold, max_iterations):
    if max_iterations <= 0:
        return object_ids, []
    if not vote_counts:
        return object_ids, [{"iteration": 0, "num_reassigned": 0, "mean_consistency": 1.0, "status": "skipped_no_votes"}]
    if not per_view_observations:
        return object_ids, [{"iteration": 0, "num_reassigned": 0, "mean_consistency": 1.0, "status": "skipped_no_observations"}]

    refined = object_ids.copy()
    refinement_stats = []
    labels_sorted = sorted(vote_counts.keys())
    vote_matrix = np.stack([vote_counts[label] for label in labels_sorted], axis=1)
    label_to_col = {label: idx for idx, label in enumerate(labels_sorted)}

    for refine_iter in range(max_iterations):
        total_weight = np.zeros((refined.shape[0],), dtype=np.float32)
        matched_weight = np.zeros((refined.shape[0],), dtype=np.float32)

        for obs in per_view_observations:
            idx = obs["gaussian_indices"]
            labels = obs["labels"]
            weights = obs["weights"]
            np.add.at(total_weight, idx, weights)
            matches = refined[idx] == labels
            if np.any(matches):
                np.add.at(matched_weight, idx[matches], weights[matches])

        consistency = np.ones((refined.shape[0],), dtype=np.float32)
        valid = total_weight > 0
        consistency[valid] = matched_weight[valid] / np.maximum(total_weight[valid], 1e-8)

        inconsistent = valid & (consistency < float(consistency_threshold))
        if not np.any(inconsistent):
            refinement_stats.append(
                {
                    "iteration": refine_iter + 1,
                    "num_reassigned": 0,
                    "mean_consistency": float(consistency[valid].mean()) if np.any(valid) else 1.0,
                    "status": "converged",
                }
            )
            break

        candidate_votes = vote_matrix[inconsistent]
        best_alt_idx = candidate_votes.argmax(axis=1)
        best_alt_label = np.asarray(labels_sorted, dtype=np.int32)[best_alt_idx]
        best_alt_vote = candidate_votes[np.arange(candidate_votes.shape[0]), best_alt_idx]

        inconsistent_indices = np.nonzero(inconsistent)[0]
        current_labels = refined[inconsistent_indices]
        current_vote = np.zeros((inconsistent_indices.shape[0],), dtype=np.float32)
        for row_idx, label in enumerate(current_labels.tolist()):
            col = label_to_col.get(int(label))
            if col is not None:
                current_vote[row_idx] = candidate_votes[row_idx, col]

        should_reassign = (best_alt_label != current_labels) & (best_alt_vote > current_vote)
        reassigned_indices = inconsistent_indices[should_reassign]
        refined[reassigned_indices] = best_alt_label[should_reassign]

        refinement_stats.append(
            {
                "iteration": refine_iter + 1,
                "num_reassigned": int(reassigned_indices.shape[0]),
                "mean_consistency": float(consistency[valid].mean()) if np.any(valid) else 1.0,
                "status": "updated" if reassigned_indices.shape[0] > 0 else "stable",
            }
        )

        if reassigned_indices.shape[0] == 0:
            break

    return refined, refinement_stats


if __name__ == "__main__":
    parser = ArgumentParser(description="Automatically assign Gaussian object ids from 2D instance masks.")
    model = ModelParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--masks_dir", required=True, type=str)
    parser.add_argument("--output_model_path", default="", type=str)
    parser.add_argument("--save_iteration", default=-1, type=int)
    parser.add_argument("--background_id", default=0, type=int)
    parser.add_argument("--min_votes", default=1, type=int)
    parser.add_argument("--ignore_ids", default="", type=str)
    parser.add_argument("--confidence_maps_dir", default="", type=str)
    parser.add_argument("--confidence_threshold", default=0.0, type=float)
    parser.add_argument("--boundary_band_width", default=0.0, type=float)
    parser.add_argument("--boundary_min_weight", default=0.25, type=float)
    parser.add_argument("--disable_occlusion_filter", action="store_true")
    parser.add_argument("--depth_tolerance", default=1e-3, type=float)
    parser.add_argument("--refine_iterations", default=0, type=int)
    parser.add_argument("--refine_consistency_threshold", default=0.6, type=float)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    safe_state(args.quiet)

    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, geometry_feature_dim=getattr(dataset, "geometry_feature_dim", 3))
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    views_by_split = []
    if not args.skip_train:
        views_by_split.append(("train", scene.getTrainCameras()))
    if not args.skip_test:
        views_by_split.append(("test", scene.getTestCameras()))

    if not views_by_split:
        raise ValueError("Nothing to process. Both --skip_train and --skip_test are set.")

    xyz = gaussians.get_xyz.detach()
    num_gaussians = int(xyz.shape[0])
    masks_dir = Path(args.masks_dir)
    confidence_maps_dir = Path(args.confidence_maps_dir) if args.confidence_maps_dir else None
    ignored_labels = parse_ignore_ids(args.ignore_ids)
    accumulation = accumulate_view_votes(
        views_by_split=views_by_split,
        masks_dir=masks_dir,
        xyz=xyz,
        ignored_labels=ignored_labels,
        confidence_maps_dir=confidence_maps_dir,
        confidence_threshold=args.confidence_threshold,
        boundary_band_width=args.boundary_band_width,
        boundary_min_weight=args.boundary_min_weight,
        disable_occlusion_filter=args.disable_occlusion_filter,
        depth_tolerance=args.depth_tolerance,
    )
    vote_counts = accumulation["vote_counts"]
    used_views = accumulation["used_views"]
    skipped_views = accumulation["skipped_views"]
    total_visible_projections = accumulation["total_visible_projections"]
    total_frontmost_projections = accumulation["total_frontmost_projections"]
    total_confident_frontmost_projections = accumulation["total_confident_frontmost_projections"]
    per_view_observations = accumulation["per_view_observations"]

    object_ids = assign_labels_from_votes(vote_counts, num_gaussians, args.background_id, args.min_votes)
    object_ids, refinement_stats = refine_object_ids(
        object_ids=object_ids,
        vote_counts=vote_counts,
        per_view_observations=per_view_observations,
        background_id=args.background_id,
        consistency_threshold=args.refine_consistency_threshold,
        max_iterations=args.refine_iterations,
    )

    output_model_path = args.output_model_path if args.output_model_path else args.model_path
    save_iteration = scene.loaded_iter if args.save_iteration == -1 else args.save_iteration
    save_grouped_model(scene, dataset, output_model_path, save_iteration, object_ids, args.model_path)

    metadata = {
        "source_model_path": os.path.abspath(args.model_path),
        "output_model_path": os.path.abspath(output_model_path),
        "loaded_iteration": int(scene.loaded_iter),
        "saved_iteration": int(save_iteration),
        "masks_dir": str(masks_dir.resolve()),
        "num_gaussians": num_gaussians,
        "num_used_views": len(used_views),
        "num_skipped_views": len(skipped_views),
        "background_id": int(args.background_id),
        "min_votes": int(args.min_votes),
        "ignored_labels": sorted(int(x) for x in ignored_labels),
        "confidence_maps_dir": str(confidence_maps_dir.resolve()) if confidence_maps_dir is not None else "",
        "confidence_threshold": float(args.confidence_threshold),
        "confidence_weighted_votes": confidence_maps_dir is not None,
        "boundary_band_width": float(args.boundary_band_width),
        "boundary_min_weight": float(args.boundary_min_weight),
        "boundary_downweighting_enabled": args.boundary_band_width > 0,
        "occlusion_filter_enabled": not args.disable_occlusion_filter,
        "depth_tolerance": float(args.depth_tolerance),
        "refine_iterations": int(args.refine_iterations),
        "refine_consistency_threshold": float(args.refine_consistency_threshold),
        "refinement_stats": refinement_stats,
        "total_visible_projections": int(total_visible_projections),
        "total_frontmost_projections": int(total_frontmost_projections),
        "total_confident_frontmost_projections": int(total_confident_frontmost_projections),
        "unique_object_ids": np.unique(object_ids).astype(np.int32).tolist(),
    }

    metadata_path = os.path.join(output_model_path, f"auto_object_ids_iter_{save_iteration}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    np.save(os.path.join(output_model_path, f"auto_object_ids_iter_{save_iteration}.npy"), object_ids)

    print(f"Assigned object ids to {num_gaussians} gaussians.")
    print(f"Used {len(used_views)} views and skipped {len(skipped_views)} views.")
    print(f"Saved grouped model to {output_model_path} at iteration {save_iteration}.")
