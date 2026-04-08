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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

class _DisabledNetworkGUI:
    def __init__(self):
        self.conn = None

    def init(self, *args, **kwargs):
        self.conn = None

    def try_connect(self):
        self.conn = None

    def receive(self):
        return None, None, None, None, None, None

    def send(self, *args, **kwargs):
        return None

NETWORK_GUI_IMPORT_ERROR = None
try:
    import gaussian_renderer.network_gui as network_gui
except Exception as e:
    network_gui = _DisabledNetworkGUI()
    NETWORK_GUI_IMPORT_ERROR = e

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

GEOMETRY_SCALE_REG_WEIGHT = 1e-4
GEOMETRY_OPACITY_REG_WEIGHT = 1e-4

def zero_active_optimizers(gaussians, use_decoupled_optimization):
    if use_decoupled_optimization:
        gaussians.geometry_optimizer.zero_grad(set_to_none=True)
        gaussians.appearance_optimizer.zero_grad(set_to_none=True)
        gaussians.exposure_optimizer.zero_grad(set_to_none=True)
    else:
        gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians.exposure_optimizer.zero_grad(set_to_none=True)

def compute_losses(render_pkg, viewpoint_cam, opt, depth_l1_weight_value):
    image = render_pkg["render"]
    if viewpoint_cam.alpha_mask is not None:
        alpha_mask = viewpoint_cam.alpha_mask.cuda()
        image = image * alpha_mask

    gt_image = viewpoint_cam.original_image.cuda()
    ll1 = l1_loss(image, gt_image)
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    else:
        ssim_value = ssim(image, gt_image)

    appearance_loss = (1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * (1.0 - ssim_value)

    depth_loss_value = 0.0
    depth_loss_tensor = None
    if depth_l1_weight_value > 0 and viewpoint_cam.depth_reliable:
        invDepth = render_pkg["depth"]
        mono_invdepth = viewpoint_cam.invdepthmap.cuda()
        depth_mask = viewpoint_cam.depth_mask.cuda()

        depth_loss_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
        depth_loss_tensor = depth_l1_weight_value * depth_loss_pure
        depth_loss_value = depth_loss_tensor.item()

    return image, ll1, appearance_loss, depth_loss_tensor, depth_loss_value

def compute_sam_feature_loss(viewpoint_cam, gaussians, pipe, bg, separate_sh, weight, shared_screenspace_points=None):
    if weight <= 0 or viewpoint_cam.sam_feature_map is None:
        return None, 0.0

    geometry_feature_render = render(
        viewpoint_cam,
        gaussians,
        pipe,
        bg,
        override_color=gaussians.get_geometry_features,
        use_trained_exp=False,
        separate_sh=separate_sh,
        screenspace_points=shared_screenspace_points,
    )["render"]

    target_feature_map = viewpoint_cam.sam_feature_map
    feature_mask = viewpoint_cam.alpha_mask
    geometry_feature_loss = weight * l1_loss(geometry_feature_render * feature_mask, target_feature_map * feature_mask)
    return geometry_feature_loss, geometry_feature_loss.item()

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    use_decoupled_optimization = opt.alternating_optimization or opt.joint_optimization

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_geometry_loss_for_log = 0.0
    ema_appearance_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if opt.alternating_optimization:
            cycle_length = opt.geometry_iters + opt.appearance_iters
            phase_position = (iteration - 1) % cycle_length
            optimize_geometry = phase_position < opt.geometry_iters
            optimize_appearance = not optimize_geometry
        elif opt.joint_optimization:
            optimize_geometry = True
            optimize_appearance = True
        else:
            optimize_geometry = True
            optimize_appearance = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        zero_active_optimizers(gaussians, use_decoupled_optimization)

        Ll1 = torch.tensor(0.0, device="cuda")
        loss = torch.tensor(0.0, device="cuda")
        Ll1depth = 0.0
        geometry_loss_value = 0.0
        appearance_loss_value = 0.0
        geometry_feature_loss_value = 0.0
        densify_render_pkg = None
        densify_visibility_filter = None
        densify_radii = None
        densify_viewspace_points = None

        if use_decoupled_optimization:
            depth_weight_value = depth_l1_weight(iteration)

            if optimize_geometry:
                gaussians.set_parameter_requires_grad(True, False, False)
                geometry_render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                geometry_image, geometry_l1, geometry_appearance_loss, geometry_depth_loss_tensor, geometry_depth_loss_value = compute_losses(
                    geometry_render_pkg, viewpoint_cam, opt, depth_weight_value
                )
                geometry_feature_loss_tensor, geometry_feature_loss_value = compute_sam_feature_loss(
                    viewpoint_cam,
                    gaussians,
                    pipe,
                    bg,
                    SPARSE_ADAM_AVAILABLE,
                    opt.sam_feature_weight,
                    shared_screenspace_points=geometry_render_pkg["viewspace_points"],
                )
                geometry_reg = GEOMETRY_SCALE_REG_WEIGHT * gaussians.get_scaling.mean() + GEOMETRY_OPACITY_REG_WEIGHT * gaussians.get_opacity.mean()
                geometry_loss = geometry_appearance_loss + geometry_reg
                if geometry_depth_loss_tensor is not None:
                    geometry_loss = geometry_loss + geometry_depth_loss_tensor
                if geometry_feature_loss_tensor is not None:
                    geometry_loss = geometry_loss + geometry_feature_loss_tensor
                geometry_loss.backward()

                densify_render_pkg = geometry_render_pkg
                densify_visibility_filter = geometry_render_pkg["visibility_filter"]
                densify_radii = geometry_render_pkg["radii"]
                densify_viewspace_points = geometry_render_pkg["viewspace_points"]
                geometry_loss_value = geometry_loss.item()
                Ll1depth = geometry_depth_loss_value
                Ll1 = geometry_l1.detach()

            if optimize_appearance:
                gaussians.set_parameter_requires_grad(False, True, True)
                appearance_render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                _, appearance_l1, appearance_loss_tensor, _, _ = compute_losses(
                    appearance_render_pkg, viewpoint_cam, opt, 0.0
                )
                appearance_loss_tensor.backward()

                if densify_render_pkg is None:
                    densify_render_pkg = appearance_render_pkg
                    densify_visibility_filter = appearance_render_pkg["visibility_filter"]
                    densify_radii = appearance_render_pkg["radii"]
                    densify_viewspace_points = appearance_render_pkg["viewspace_points"]
                    Ll1 = appearance_l1.detach()
                appearance_loss_value = appearance_loss_tensor.item()

            gaussians.set_parameter_requires_grad(True, True, True)
            combined_loss_value = geometry_loss_value + appearance_loss_value
            loss = torch.tensor(combined_loss_value, device="cuda")
        else:
            gaussians.set_parameter_requires_grad(True, True, True)
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            image, Ll1, appearance_loss_tensor, depth_loss_tensor, depth_loss_value = compute_losses(
                render_pkg, viewpoint_cam, opt, depth_l1_weight(iteration)
            )
            loss = appearance_loss_tensor
            if depth_loss_tensor is not None:
                loss = loss + depth_loss_tensor
            loss.backward()

            densify_render_pkg = render_pkg
            densify_visibility_filter = render_pkg["visibility_filter"]
            densify_radii = render_pkg["radii"]
            densify_viewspace_points = render_pkg["viewspace_points"]
            Ll1depth = depth_loss_value
            geometry_loss_value = loss.item()
            appearance_loss_value = appearance_loss_tensor.item()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            ema_geometry_loss_for_log = 0.4 * geometry_loss_value + 0.6 * ema_geometry_loss_for_log
            ema_appearance_loss_for_log = 0.4 * appearance_loss_value + 0.6 * ema_appearance_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Geom": f"{ema_geometry_loss_for_log:.{7}f}",
                    "App": f"{ema_appearance_loss_for_log:.{7}f}",
                    "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if tb_writer and use_decoupled_optimization:
                tb_writer.add_scalar('train_loss_patches/geometry_loss', geometry_loss_value, iteration)
                tb_writer.add_scalar('train_loss_patches/appearance_loss', appearance_loss_value, iteration)
                tb_writer.add_scalar('train_loss_patches/geometry_feature_loss', geometry_feature_loss_value, iteration)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            should_update_geometry_stats = (not use_decoupled_optimization) or optimize_geometry
            if iteration < opt.densify_until_iter and should_update_geometry_stats:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[densify_visibility_filter] = torch.max(gaussians.max_radii2D[densify_visibility_filter], densify_radii[densify_visibility_filter])
                gaussians.add_densification_stats(densify_viewspace_points, densify_visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, densify_radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                visible = densify_radii > 0
                if use_decoupled_optimization:
                    if optimize_geometry:
                        if use_sparse_adam:
                            gaussians.geometry_optimizer.step(visible, densify_radii.shape[0])
                        else:
                            gaussians.geometry_optimizer.step()
                    if optimize_appearance:
                        if use_sparse_adam:
                            gaussians.appearance_optimizer.step(visible, densify_radii.shape[0])
                        else:
                            gaussians.appearance_optimizer.step()
                        gaussians.exposure_optimizer.step()
                    gaussians.geometry_optimizer.zero_grad(set_to_none = True)
                    gaussians.appearance_optimizer.zero_grad(set_to_none = True)
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                    if use_sparse_adam:
                        gaussians.optimizer.step(visible, radii.shape[0])
                    else:
                        gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    if args.alternating_optimization and args.joint_optimization:
        raise ValueError("Choose either --alternating_optimization or --joint_optimization, not both.")
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer and NETWORK_GUI_IMPORT_ERROR is not None:
        print(f"Viewer disabled because network GUI could not be initialized: {NETWORK_GUI_IMPORT_ERROR}")
    elif not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
