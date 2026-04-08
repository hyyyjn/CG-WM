import json
import os
import shutil
from argparse import ArgumentParser
from argparse import Namespace

import numpy as np
import torch

from arguments import ModelParams
from scene import Scene
from gaussian_renderer import GaussianModel
from utils.general_utils import safe_state


def load_object_ids(path):
    if path.endswith(".npy"):
        object_ids = np.load(path)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            payload = json.load(f)
        object_ids = payload["object_ids"] if isinstance(payload, dict) and "object_ids" in payload else payload
    else:
        raise ValueError("Unsupported object id file format. Use .npy or .json.")
    return np.asarray(object_ids, dtype=np.int32)


if __name__ == "__main__":
    parser = ArgumentParser(description="Assign object ids to a trained Gaussian model.")
    model = ModelParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--object_ids_path", required=True, type=str)
    parser.add_argument("--output_model_path", default="", type=str)
    parser.add_argument("--save_iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    safe_state(args.quiet)

    output_model_path = args.output_model_path if args.output_model_path else args.model_path
    dataset = model.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    scene.model_path = output_model_path
    dataset.model_path = output_model_path

    object_ids = load_object_ids(args.object_ids_path)
    gaussians.set_object_ids(object_ids)

    save_iteration = scene.loaded_iter if args.save_iteration == -1 else args.save_iteration
    point_cloud_dir = os.path.join(output_model_path, f"point_cloud/iteration_{save_iteration}")
    os.makedirs(point_cloud_dir, exist_ok=True)
    gaussians.save_ply(os.path.join(point_cloud_dir, "point_cloud.ply"))

    source_exposure_path = os.path.join(args.model_path, "exposure.json")
    if os.path.exists(source_exposure_path):
        shutil.copy2(source_exposure_path, os.path.join(output_model_path, "exposure.json"))

    metadata = {
        "source_model_path": args.model_path,
        "output_model_path": output_model_path,
        "loaded_iteration": scene.loaded_iter,
        "saved_iteration": save_iteration,
        "num_gaussians": int(gaussians.get_xyz.shape[0]),
        "num_objects": int(np.unique(object_ids).shape[0]),
        "object_ids_path": os.path.abspath(args.object_ids_path),
    }

    os.makedirs(output_model_path, exist_ok=True)
    metadata_path = os.path.join(output_model_path, f"object_ids_assignment_iter_{save_iteration}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(output_model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(dataset))))

    print(f"Assigned object ids for {metadata['num_gaussians']} gaussians.")
    print(f"Saved updated model to {output_model_path} at iteration {save_iteration}.")
    print(f"Metadata written to {metadata_path}.")
