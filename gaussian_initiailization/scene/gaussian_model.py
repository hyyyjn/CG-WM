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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:
    # add function
    @staticmethod
    def _expand_isotropic_scaling(scaling):
        isotropic = scaling.mean(dim=1, keepdim=True)
        return isotropic.repeat(1, 3)

    # add function
    @staticmethod
    def _identity_rotation(num_points, device):
        rots = torch.zeros((num_points, 4), device=device)
        rots[:, 0] = 1
        return rots

    # add function
    def _get_isotropic_scaling(self):
        return self._expand_isotropic_scaling(self.scaling_activation(self._scaling))

    def setup_functions(self):
        # edit this
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            isotropic_scaling = self._expand_isotropic_scaling(scaling_modifier * scaling)
            identity_rotation = self._identity_rotation(isotropic_scaling.shape[0], isotropic_scaling.device)
            L = build_scaling_rotation(isotropic_scaling, identity_rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._features_geo = torch.empty(0)
        self._object_ids = torch.empty(0, dtype=torch.int32)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.geometry_optimizer = None
        self.appearance_optimizer = None
        self.exposure_optimizer = None
        self.use_decoupled_optimization = False
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return {
            "active_sh_degree": self.active_sh_degree,
            "xyz": self._xyz,
            "features_dc": self._features_dc,
            "features_rest": self._features_rest,
            "features_geo": self._features_geo,
            "object_ids": self._object_ids,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "exposure": self._exposure,
            "max_radii2D": self.max_radii2D,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "geometry_optimizer_state": self.geometry_optimizer.state_dict() if self.geometry_optimizer is not None else None,
            "appearance_optimizer_state": self.appearance_optimizer.state_dict() if self.appearance_optimizer is not None else None,
            "exposure_optimizer_state": self.exposure_optimizer.state_dict() if self.exposure_optimizer is not None else None,
            "spatial_lr_scale": self.spatial_lr_scale,
            "use_decoupled_optimization": self.use_decoupled_optimization,
        }
    
    def restore(self, model_args, training_args):
        if isinstance(model_args, dict):
            self.active_sh_degree = model_args["active_sh_degree"]
            self._xyz = model_args["xyz"]
            self._features_dc = model_args["features_dc"]
            self._features_rest = model_args["features_rest"]
            self._features_geo = model_args.get("features_geo", self._features_geo)
            self._object_ids = model_args.get("object_ids", self._object_ids)
            self._scaling = model_args["scaling"]
            self._rotation = model_args["rotation"]
            self._opacity = model_args["opacity"]
            self._exposure = model_args.get("exposure", self._exposure)
            self.max_radii2D = model_args["max_radii2D"]
            xyz_gradient_accum = model_args["xyz_gradient_accum"]
            denom = model_args["denom"]
            opt_dict = model_args.get("optimizer_state")
            geometry_opt_dict = model_args.get("geometry_optimizer_state")
            appearance_opt_dict = model_args.get("appearance_optimizer_state")
            exposure_opt_dict = model_args.get("exposure_optimizer_state")
            self.spatial_lr_scale = model_args["spatial_lr_scale"]
            saved_decoupled = model_args.get("use_decoupled_optimization", False)
        else:
            if len(model_args) == 13:
                (self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._features_geo,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale) = model_args
            else:
                (self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale) = model_args
                self._features_geo = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), device=self._xyz.device).requires_grad_(True))
            geometry_opt_dict = None
            appearance_opt_dict = None
            exposure_opt_dict = None
            saved_decoupled = False

        if self._features_geo.numel() == 0:
            self._features_geo = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), device=self._xyz.device).requires_grad_(True))
        if self._object_ids.numel() == 0:
            self._object_ids = torch.zeros((self._xyz.shape[0],), dtype=torch.int32, device=self._xyz.device)

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        requested_decoupled = training_args.alternating_optimization or training_args.joint_optimization
        if saved_decoupled != requested_decoupled:
            raise RuntimeError("Checkpoint optimization mode does not match current training configuration.")

        if self.optimizer is not None and opt_dict is not None:
            self._load_optimizer_state(self.optimizer, opt_dict)
        if self.geometry_optimizer is not None and geometry_opt_dict is not None:
            self._load_optimizer_state(self.geometry_optimizer, geometry_opt_dict)
        if self.appearance_optimizer is not None and appearance_opt_dict is not None:
            self._load_optimizer_state(self.appearance_optimizer, appearance_opt_dict)
        if exposure_opt_dict is not None:
            self._load_optimizer_state(self.exposure_optimizer, exposure_opt_dict)

    # edit this
    @property
    def get_scaling(self):
        return self._get_isotropic_scaling()
    
    # edit this
    @property
    def get_rotation(self):
        return self._identity_rotation(self._rotation.shape[0], self._rotation.device)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_geometry_features(self):
        return torch.sigmoid(self._features_geo)

    @property
    def get_object_ids(self):
        return self._object_ids
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    # edit this
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_geo = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 3), device="cuda").requires_grad_(True))
        self._object_ids = torch.zeros((fused_point_cloud.shape[0],), dtype=torch.int32, device="cuda")
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.use_decoupled_optimization = training_args.alternating_optimization or training_args.joint_optimization
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._features_geo], 'lr': training_args.feature_lr, "name": "f_geo"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        def build_optimizer(param_groups):
            if self.optimizer_type == "default":
                return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
            if self.optimizer_type == "sparse_adam":
                try:
                    return SparseGaussianAdam(param_groups, lr=0.0, eps=1e-15)
                except:
                    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
            return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

        self.optimizer = None
        self.geometry_optimizer = None
        self.appearance_optimizer = None

        if self.use_decoupled_optimization:
            geometry_groups = [group for group in l if group["name"] in {"xyz", "scaling", "rotation", "f_geo"}]
            appearance_groups = [group for group in l if group["name"] in {"f_dc", "f_rest", "opacity"}]
            self.geometry_optimizer = build_optimizer(geometry_groups)
            self.appearance_optimizer = build_optimizer(appearance_groups)
        else:
            self.optimizer = build_optimizer(l)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        lr = self.xyz_scheduler_args(iteration)
        for optimizer in self._get_gaussian_optimizers():
            for param_group in optimizer.param_groups:
                if param_group["name"] == "xyz":
                    param_group['lr'] = lr
        return lr

    def _get_gaussian_optimizers(self):
        optimizers = []
        if self.optimizer is not None:
            optimizers.append(self.optimizer)
        if self.geometry_optimizer is not None:
            optimizers.append(self.geometry_optimizer)
        if self.appearance_optimizer is not None:
            optimizers.append(self.appearance_optimizer)
        return optimizers

    @staticmethod
    def _state_shapes_match(saved_state, current_param):
        for value in saved_state.values():
            if torch.is_tensor(value) and value.shape != current_param.shape:
                return False
        return True

    def _load_optimizer_state(self, optimizer, state_dict):
        try:
            optimizer.load_state_dict(state_dict)
            return
        except ValueError:
            pass

        saved_groups_by_name = {
            group.get("name"): group for group in state_dict["param_groups"]
            if group.get("name") is not None
        }
        if not saved_groups_by_name:
            raise

        optimizer.state.clear()
        saved_state = state_dict["state"]
        for current_group in optimizer.param_groups:
            group_name = current_group.get("name")
            saved_group = saved_groups_by_name.get(group_name)
            if saved_group is None:
                continue

            for key, value in saved_group.items():
                if key not in {"params", "name"}:
                    current_group[key] = value

            saved_param_ids = saved_group.get("params", [])
            if len(saved_param_ids) != 1 or len(current_group["params"]) != 1:
                continue

            current_param = current_group["params"][0]
            param_state = saved_state.get(saved_param_ids[0])
            if param_state is None or not self._state_shapes_match(param_state, current_param):
                continue

            optimizer.state[current_param] = {
                key: value.to(current_param.device) if torch.is_tensor(value) else value
                for key, value in param_state.items()
            }

    def geometry_parameters(self):
        return [self._xyz, self._scaling, self._rotation, self._features_geo]

    def appearance_parameters(self):
        return [self._features_dc, self._features_rest, self._opacity]

    def exposure_parameters(self):
        return [self._exposure]

    def set_parameter_requires_grad(self, geometry_enabled, appearance_enabled, exposure_enabled):
        for param in self.geometry_parameters():
            param.requires_grad_(geometry_enabled)
        for param in self.appearance_parameters():
            param.requires_grad_(appearance_enabled)
        for param in self.exposure_parameters():
            param.requires_grad_(exposure_enabled)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._features_geo.shape[1]):
            l.append('f_geo_{}'.format(i))
        l.append('object_id')
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_geo = self._features_geo.detach().cpu().numpy()
        object_ids = self._object_ids.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = []
        for attribute in self.construct_list_of_attributes():
            dtype_full.append((attribute, 'i4' if attribute == 'object_id' else 'f4'))

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, f_geo, object_ids[:, None], opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        geo_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_geo_")]
        geo_f_names = sorted(geo_f_names, key = lambda x: int(x.split('_')[-1]))
        if len(geo_f_names) > 0:
            features_geo = np.zeros((xyz.shape[0], len(geo_f_names)))
            for idx, attr_name in enumerate(geo_f_names):
                features_geo[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            features_geo = np.zeros((xyz.shape[0], 3), dtype=np.float32)

        if "object_id" in plydata.elements[0].data.dtype.names:
            object_ids = np.asarray(plydata.elements[0]["object_id"]).astype(np.int32)
        else:
            object_ids = np.zeros((xyz.shape[0],), dtype=np.int32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_geo = nn.Parameter(torch.tensor(features_geo, dtype=torch.float, device="cuda").requires_grad_(True))
        self._object_ids = torch.tensor(object_ids, dtype=torch.int32, device="cuda")
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for optimizer in self._get_gaussian_optimizers():
            for group in optimizer.param_groups:
                if group["name"] == name:
                    stored_state = optimizer.state.get(group['params'][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = torch.zeros_like(tensor)
                        stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                        del optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    if stored_state is not None:
                        optimizer.state[group["params"][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for optimizer in self._get_gaussian_optimizers():
            for group in optimizer.param_groups:
                stored_state = optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._features_geo = optimizable_tensors["f_geo"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._object_ids = self._object_ids[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for optimizer in self._get_gaussian_optimizers():
            for group in optimizer.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_features_geo, new_object_ids, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "f_geo": new_features_geo,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._features_geo = optimizable_tensors["f_geo"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._object_ids = torch.cat((self._object_ids, new_object_ids.to(self._object_ids.device, dtype=self._object_ids.dtype)))

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # edit this
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.get_rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._identity_rotation(selected_pts_mask.sum() * N, self._rotation.device)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_features_geo = self._features_geo[selected_pts_mask].repeat(N,1)
        new_object_ids = self._object_ids[selected_pts_mask].repeat(N)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_features_geo, new_object_ids, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_features_geo = self._features_geo[selected_pts_mask]
        new_object_ids = self._object_ids[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_features_geo, new_object_ids, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def set_object_ids(self, object_ids):
        if torch.is_tensor(object_ids):
            object_ids_tensor = object_ids.to(device=self.get_xyz.device, dtype=torch.int32)
        else:
            object_ids_tensor = torch.tensor(object_ids, dtype=torch.int32, device=self.get_xyz.device)

        if object_ids_tensor.ndim != 1:
            raise ValueError(f"Expected 1D object id tensor, got shape {tuple(object_ids_tensor.shape)}")
        if object_ids_tensor.shape[0] != self.get_xyz.shape[0]:
            raise ValueError(
                f"Object id count mismatch: got {object_ids_tensor.shape[0]} ids for {self.get_xyz.shape[0]} gaussians."
            )
        self._object_ids = object_ids_tensor.contiguous()
