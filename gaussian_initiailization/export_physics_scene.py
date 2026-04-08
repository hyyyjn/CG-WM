import json
import os
from argparse import ArgumentParser

import numpy as np
import torch

from arguments import ModelParams
from scene import Scene
from gaussian_renderer import GaussianModel
from utils.general_utils import safe_state


def infer_source_track_id(object_id, background_id):
    if int(object_id) == int(background_id):
        return None
    return int(object_id)


def compute_rigid_metadata(center, bbox_min, bbox_max, radius, density, friction, restitution, dynamic):
    extents = bbox_max - bbox_min
    half_extents = extents * 0.5
    sphere_radius = max(float(radius), 1e-6)
    approx_volume = (4.0 / 3.0) * np.pi * (sphere_radius ** 3)
    mass = float(max(density * approx_volume, 1e-8)) if dynamic else 0.0
    inertia_scalar = 0.4 * mass * (sphere_radius ** 2) if dynamic else 0.0
    return {
        "body_frame": {
            "translation": center.tolist(),
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "collision_proxy": {
            "type": "sphere_and_aabb",
            "sphere_center": center.tolist(),
            "sphere_radius": float(sphere_radius),
            "aabb_min": bbox_min.tolist(),
            "aabb_max": bbox_max.tolist(),
            "half_extents": half_extents.tolist(),
        },
        "physics": {
            "dynamic": bool(dynamic),
            "mass": float(mass),
            "density": float(density),
            "friction": float(friction),
            "restitution": float(restitution),
            "inertia_diag": [float(inertia_scalar), float(inertia_scalar), float(inertia_scalar)],
            "initial_linear_velocity": [0.0, 0.0, 0.0],
            "initial_angular_velocity": [0.0, 0.0, 0.0],
        },
    }


def build_object_export(gaussians, object_id, background_id, density, friction, restitution):
    mask = gaussians.get_object_ids == object_id
    xyz = gaussians.get_xyz[mask].detach().cpu().numpy()
    scaling = gaussians.get_scaling[mask].detach().cpu().numpy()
    opacity = gaussians.get_opacity[mask].detach().cpu().numpy()
    features_dc = gaussians.get_features_dc[mask].detach().cpu().numpy()
    features_rest = gaussians.get_features_rest[mask].detach().cpu().numpy()
    features_geo = gaussians.get_geometry_features[mask].detach().cpu().numpy()
    source_track_id = infer_source_track_id(object_id, background_id)

    center = xyz.mean(axis=0)
    distances = np.linalg.norm(xyz - center[None, :], axis=1)
    gaussian_radius = np.max(np.linalg.norm(scaling, axis=1))
    radius = float(distances.max() + gaussian_radius)
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    dynamic = int(object_id) != int(background_id)
    rigid_metadata = compute_rigid_metadata(
        center=center,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        radius=radius,
        density=density,
        friction=friction,
        restitution=restitution,
        dynamic=dynamic,
    )

    return {
        "object_id": int(object_id),
        "source_track_id": source_track_id,
        "num_gaussians": int(xyz.shape[0]),
        "center": center.tolist(),
        "radius": radius,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "mean_scale": scaling.mean(axis=0).tolist(),
        "mean_opacity": float(opacity.mean()),
        **rigid_metadata,
    }, {
        "xyz": xyz,
        "scaling": scaling,
        "opacity": opacity,
        "features_dc": features_dc,
        "features_rest": features_rest,
        "features_geo": features_geo,
        "object_ids": np.full((xyz.shape[0],), int(object_id), dtype=np.int32),
        "source_track_ids": np.full(
            (xyz.shape[0],),
            -1 if source_track_id is None else int(source_track_id),
            dtype=np.int32,
        ),
    }


if __name__ == "__main__":
    parser = ArgumentParser(description="Export a trained Gaussian scene to a physics-friendly intermediate format.")
    model = ModelParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--background_id", default=0, type=int)
    parser.add_argument("--density", default=1000.0, type=float)
    parser.add_argument("--friction", default=0.5, type=float)
    parser.add_argument("--restitution", default=0.1, type=float)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    safe_state(args.quiet)

    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    output_dir = args.output_dir if args.output_dir else os.path.join(dataset.model_path, "physics_export", f"iteration_{scene.loaded_iter}")
    os.makedirs(output_dir, exist_ok=True)

    unique_object_ids = torch.unique(gaussians.get_object_ids).detach().cpu().numpy().astype(np.int32)
    object_summaries = []
    npz_payload = {}

    for object_id in unique_object_ids.tolist():
        summary, arrays = build_object_export(
            gaussians,
            object_id,
            args.background_id,
            density=args.density,
            friction=args.friction,
            restitution=args.restitution,
        )
        summary["is_background"] = int(object_id) == args.background_id
        object_summaries.append(summary)
        prefix = f"object_{int(object_id)}"
        for name, array in arrays.items():
            npz_payload[f"{prefix}_{name}"] = array

    manifest = {
        "model_path": os.path.abspath(dataset.model_path),
        "iteration": int(scene.loaded_iter),
        "num_gaussians": int(gaussians.get_xyz.shape[0]),
        "num_objects": int(len(object_summaries)),
        "background_id": int(args.background_id),
        "default_physics": {
            "density": float(args.density),
            "friction": float(args.friction),
            "restitution": float(args.restitution),
        },
        "objects": object_summaries,
    }

    manifest_path = os.path.join(output_dir, "physics_scene.json")
    arrays_path = os.path.join(output_dir, "physics_scene_arrays.npz")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    np.savez_compressed(arrays_path, **npz_payload)

    print(f"Exported {manifest['num_objects']} object groups from iteration {scene.loaded_iter}.")
    print(f"JSON manifest: {manifest_path}")
    print(f"Array bundle: {arrays_path}")
