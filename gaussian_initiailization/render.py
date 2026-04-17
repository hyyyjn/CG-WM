#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def grayscale_to_rgb(image):
    if image.shape[0] == 1:
        return image.repeat(3, 1, 1)
    return image

def build_foreground_debug_render(view, gaussians, pipeline, separate_sh, gaussian_mask=None):
    foreground_scores = gaussians.get_foreground_scores
    if gaussian_mask is not None:
        foreground_scores = foreground_scores[gaussian_mask]

    foreground_color = foreground_scores.repeat(1, 3)
    foreground_render = render(
        view,
        gaussians,
        pipeline,
        torch.zeros((3), dtype=torch.float32, device="cuda"),
        use_trained_exp=False,
        separate_sh=separate_sh,
        override_color=foreground_color,
        gaussian_mask=gaussian_mask,
    )["render"][:1, ...]
    return foreground_render.clamp(0.0, 1.0)

def build_gaussian_mask(gaussians, object_id=None, foreground_threshold=None):
    gaussian_mask = torch.ones(
        (gaussians.get_xyz.shape[0],),
        dtype=torch.bool,
        device=gaussians.get_xyz.device,
    )

    if object_id is not None:
        gaussian_mask = gaussian_mask & (gaussians.get_object_ids == int(object_id))

    if foreground_threshold is not None:
        gaussian_mask = gaussian_mask & (gaussians.get_foreground_scores[:, 0] >= float(foreground_threshold))

    return gaussian_mask

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, object_id=None, foreground_threshold=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    foreground_scores_path = os.path.join(model_path, name, "ours_{}".format(iteration), "foreground_scores")
    object_mask_prior_path = os.path.join(model_path, name, "ours_{}".format(iteration), "object_mask_prior")
    foreground_overlay_path = os.path.join(model_path, name, "ours_{}".format(iteration), "foreground_overlay")
    suffix_parts = []
    if object_id is not None:
        suffix_parts.append(f"object_{object_id}")
    if foreground_threshold is not None:
        threshold_tag = str(foreground_threshold).replace(".", "p")
        suffix_parts.append(f"fgthr_{threshold_tag}")
    output_name = name if not suffix_parts else f"{name}_{'_'.join(suffix_parts)}"
    render_path = os.path.join(model_path, output_name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, output_name, "ours_{}".format(iteration), "gt")
    foreground_scores_path = os.path.join(model_path, output_name, "ours_{}".format(iteration), "foreground_scores")
    object_mask_prior_path = os.path.join(model_path, output_name, "ours_{}".format(iteration), "object_mask_prior")
    foreground_overlay_path = os.path.join(model_path, output_name, "ours_{}".format(iteration), "foreground_overlay")

    gaussian_mask = build_gaussian_mask(gaussians, object_id=object_id, foreground_threshold=foreground_threshold)
    if not torch.any(gaussian_mask):
        raise ValueError(
            f"No Gaussians survived the render filter "
            f"(object_id={object_id}, foreground_threshold={foreground_threshold})."
        )

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(foreground_scores_path, exist_ok=True)
    makedirs(object_mask_prior_path, exist_ok=True)
    makedirs(foreground_overlay_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(
            view,
            gaussians,
            pipeline,
            background,
            use_trained_exp=train_test_exp,
            separate_sh=separate_sh,
            gaussian_mask=gaussian_mask,
        )["render"]
        foreground_render = build_foreground_debug_render(
            view,
            gaussians,
            pipeline,
            separate_sh,
            gaussian_mask=gaussian_mask,
        )
        gt = view.original_image[0:3, :, :]
        object_mask_prior = None
        if getattr(view, "has_object_mask_prior", False) and view.object_mask is not None:
            object_mask_prior = view.object_mask[:1, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            foreground_render = foreground_render[..., foreground_render.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]
            if object_mask_prior is not None:
                object_mask_prior = object_mask_prior[..., object_mask_prior.shape[-1] // 2:]

        foreground_rgb = grayscale_to_rgb(foreground_render)
        overlay = 0.65 * rendering + 0.35 * foreground_rgb

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(foreground_rgb, os.path.join(foreground_scores_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(overlay.clamp(0.0, 1.0), os.path.join(foreground_overlay_path, '{0:05d}'.format(idx) + ".png"))
        if object_mask_prior is not None:
            torchvision.utils.save_image(
                grayscale_to_rgb(object_mask_prior.clamp(0.0, 1.0)),
                os.path.join(object_mask_prior_path, '{0:05d}'.format(idx) + ".png"),
            )

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, object_id=None, foreground_threshold=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, geometry_feature_dim=getattr(dataset, "geometry_feature_dim", 3))
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, object_id=object_id, foreground_threshold=foreground_threshold)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, object_id=object_id, foreground_threshold=foreground_threshold)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--object_id", default=None, type=int)
    parser.add_argument("--foreground_threshold", default=None, type=float)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    if not hasattr(args, "object_id"):
        args.object_id = None
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        SPARSE_ADAM_AVAILABLE,
        object_id=args.object_id,
        foreground_threshold=args.foreground_threshold,
    )
