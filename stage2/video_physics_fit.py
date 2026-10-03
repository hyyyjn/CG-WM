"""Learn a rigid object's contact physics from video alone.

Pipeline (no per-dataset constants; everything comes from the data):

1. ``build_object_model``: the Stage 1 spherical Gaussians define both the
   rendered appearance and the collision geometry (``r = 2s``).  The body frame
   is the centre of mass of the solid they enclose and the inertia is that
   solid's uniform-density inertia; mass is a gauge (the soft-constraint
   contact is mass invariant) and is fixed to 1.
2. ``estimate_initial_state``: frame-0 pose by a coarse orientation search and
   gradient refinement, then the initial velocity/angular velocity by fitting
   the simulator's free flight to the frames before the first contact.
3. ``fit_physics``: soft-constraint contact parameters (time constant, damping
   ratio, friction) and generalized damping, jointly over training episodes,
   by back-propagating an image loss through the rendered rollout.
4. Held-out episodes reuse the learned physics: only their initial state is
   estimated from their own frames, then the rollout is open loop.

Ground-truth trajectories are read only by ``evaluate_trajectory``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from stage1.gaussian_renderer import GaussianModel
from stage1.gaussian_renderer import render as gs_render
from stage1.scene.cameras import MiniCam
from stage1.utils.graphics_utils import getProjectionMatrix, getWorld2View2
from stage2.differentiable_collision_detection import GaussianCollisionBody, PlaneCollider
from stage2.differentiable_complementarity_free_contact_dynamics import (
    GaussianPlaneContactPair,
    MultiBodyGaussianImpedanceDynamics,
    MultiBodyImpedanceDynamicsConfig,
    RigidBodyState,
    angular_velocity_body_to_world,
)
from stage2.object_geometry import plane_support_indices, solid_mass_properties
from stage2.renderable_gaussian_asset import (
    RenderableGaussianAsset,
    copy_asset_to_gaussian_model,
    load_renderable_gaussian_asset,
    quat_mul_wxyz,
    quat_to_matrix_wxyz,
    rigid_transform_asset,
)


class _Pipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


# --------------------------------------------------------------------------- camera


@dataclass(frozen=True)
class CameraSpec:
    c2w: np.ndarray  # 4x4, MuJoCo/OpenGL camera frame (x right, y up, looks along -z)
    fovy_deg: float
    width: int
    height: int


def camera_from_episode_manifest(manifest: dict) -> CameraSpec:
    camera = manifest["camera"]
    width, height = camera.get("image_size", manifest.get("image_size"))
    if "c2w" in camera:
        c2w = np.asarray(camera["c2w"], dtype=np.float64)
    else:  # legacy look-at record from (0, -distance, height) towards target
        eye = np.array([0.0, -float(camera["distance"]), float(camera["height"])])
        target = np.asarray(camera.get("target", [0.0, 0.0, 0.0]), dtype=np.float64)
        forward = (target - eye) / np.linalg.norm(target - eye)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        c2w = np.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, -forward, eye
    return CameraSpec(c2w, float(camera["fovy_deg"]), int(width), int(height))


def make_minicam(spec: CameraSpec, *, width: int, height: int, device) -> MiniCam:
    c2w = np.array(spec.c2w, dtype=np.float64, copy=True)
    c2w[:3, 1:3] *= -1.0  # OpenGL -> 3DGS/COLMAP camera axes
    w2c = np.linalg.inv(c2w)
    R, T = w2c[:3, :3].T.astype(np.float32), w2c[:3, 3].astype(np.float32)
    fovy = math.radians(spec.fovy_deg)
    fovx = 2.0 * math.atan(math.tan(0.5 * fovy) * float(width) / float(height))
    world_view = torch.tensor(getWorld2View2(R, T), dtype=torch.float32).T.to(device)
    projection = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).T.to(device)
    full = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return MiniCam(int(width), int(height), fovy, fovx, 0.01, 100.0, world_view, full)


# --------------------------------------------------------------------------- episodes


@dataclass
class Episode:
    name: str
    root: Path
    frames: list[int]
    rgb: torch.Tensor  # [F, 3, H, W] on the render device
    mask: torch.Tensor  # [F, 1, H, W]
    camera_spec: CameraSpec
    camera: MiniCam
    frame_dt: float
    physics_dt: float
    substeps: int
    gt_positions: np.ndarray | None = None  # MuJoCo body frame, evaluation only
    gt_quaternions: np.ndarray | None = None
    boxes: dict = field(default_factory=dict)  # per-frame object windows (see frame_box)


def load_episode(root: str | Path, *, resolution_scale: float = 1.0, max_frames: int | None = None,
                 device="cuda", load_ground_truth: bool = True) -> Episode:
    root = Path(root)
    manifest = _read_json(root / "episode_manifest.json")
    spec = camera_from_episode_manifest(manifest)
    width = int(round(spec.width * resolution_scale))
    height = int(round(spec.height * resolution_scale))
    frames = sorted(int(path.stem) for path in (root / "rgb").glob("*.png") if path.stem.isdigit())
    if max_frames is not None:
        frames = frames[: int(max_frames)]
    rgb, mask = [], []
    for frame in frames:
        image = Image.open(root / "rgb" / f"{frame:06d}.png").convert("RGB").resize((width, height), Image.BILINEAR)
        rgb.append(torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1))
        m = Image.open(root / "masks" / f"{frame:06d}.png").convert("L").resize((width, height), Image.BILINEAR)
        mask.append(torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0)[None])
    timing = manifest.get("timing") or {}
    physics_dt = float(timing.get("physics_timestep", manifest.get("mujoco_body_properties", {}).get("internal_timestep", 0.002)))
    fps = float(manifest.get("fps", 30))
    substeps = int(timing.get("steps_per_frame", max(1, round(1.0 / (fps * physics_dt)))))
    episode = Episode(
        name=f"{manifest.get('split', root.parent.parent.name)}/{root.name}", root=root, frames=frames,
        rgb=torch.stack(rgb).to(device), mask=torch.stack(mask).to(device), camera_spec=spec,
        camera=make_minicam(spec, width=width, height=height, device=device),
        frame_dt=physics_dt * substeps, physics_dt=physics_dt, substeps=substeps,
    )
    if load_ground_truth and (root / "state" / "trajectory.json").exists():
        states = {int(s["frame_index"]): s for s in _read_json(root / "state" / "trajectory.json")["states"]}
        episode.gt_positions = np.asarray([states[f]["position"] for f in frames], dtype=np.float64)
        episode.gt_quaternions = np.asarray([states[f]["quaternion_wxyz"] for f in frames], dtype=np.float64)
    return episode


# --------------------------------------------------------------------------- object


@dataclass
class ObjectModel:
    asset: RenderableGaussianAsset  # centers in the COM body frame, render device
    collision: GaussianCollisionBody  # plane-support primitives, COM frame, CPU
    inertia_per_mass: np.ndarray  # principal inertia / mass in the body frame
    center_of_mass: np.ndarray  # Stage 1 world
    support_normal: np.ndarray
    support_height: float
    reference_pose: dict | None  # Stage 1 object body pose (evaluation frame only)
    report: dict = field(default_factory=dict)


def build_object_model(stage1_ply: str | Path, stage1_dataset: str | Path, *, device="cuda") -> ObjectModel:
    stage1_dataset = Path(stage1_dataset)
    manifest = _read_json(stage1_dataset / "dataset_manifest.json")
    plane = manifest.get("floor_plane") or manifest["support_plane"]
    normal = np.asarray(plane["normal"], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    asset = load_renderable_gaussian_asset(stage1_ply, device=device)
    centers = asset.xyz.detach().cpu().numpy().astype(np.float64)
    radii = 2.0 * np.exp(asset.scaling.detach().cpu().numpy().astype(np.float64)).mean(axis=1)
    solid = solid_mass_properties(centers, radii)
    com = solid.center_of_mass
    local_asset = asset.subtract_local_offset(torch.as_tensor(com, dtype=asset.xyz.dtype))
    local_centers = torch.as_tensor(centers - com, dtype=torch.float32)
    local_radii = torch.as_tensor(radii, dtype=torch.float32)
    # Resolution of the Stage 1 silhouettes in metres (half a pixel at the scan distance):
    # the collision surface is not known better than this.
    resolution = 0.0
    if manifest.get("camera_radius") and manifest.get("fovy_deg") and manifest.get("height"):
        focal = 0.5 * float(manifest["height"]) / math.tan(0.5 * math.radians(float(manifest["fovy_deg"])))
        resolution = 0.5 * float(manifest["camera_radius"]) / focal
    support = plane_support_indices(local_centers, local_radii, flat_tolerance=resolution)
    collision = GaussianCollisionBody(local_centers[support], local_radii[support], local_centers[support], support)
    report = {
        "stage1_ply": str(stage1_ply),
        "num_gaussians": int(centers.shape[0]),
        "num_plane_support_primitives": int(support.numel()),
        "center_of_mass_stage1": com.tolist(),
        "inertia_per_mass": solid.principal_inertia_per_mass().tolist(),
        "inertia_off_diagonal_ratio": solid.off_diagonal_ratio(),
        "solid_volume_m3": solid.volume,
        "voxel_size_m": solid.voxel_size,
        "union_extent_below_com": float((local_centers[:, 2] - local_radii).min()),
        "union_extent_above_com": float((local_centers[:, 2] + local_radii).max()),
        "union_radial_extent": float((local_centers[:, :2].norm(dim=1) + local_radii).max()),
        "support_plane": {"normal": normal.tolist(), "height": float(plane.get("height", 0.0))},
        "stage1_silhouette_resolution_m": resolution,
    }
    # Mean object size in the Stage 1 training images: the pixel scale at which
    # the Gaussians (and the rasterizer's fixed screen-space blur) were fitted.
    mask_paths = sorted((stage1_dataset / "object_masks" / "train").glob("*.png"))[:24]
    if mask_paths:
        report["stage1_object_pixels"] = float(np.mean([
            (np.asarray(Image.open(path).convert("L")) > 127).sum() for path in mask_paths]))
    reference = manifest.get("object_pose_world")
    if reference is not None:
        report["com_minus_reference_body_origin"] = (com - np.asarray(reference["position"])).tolist()
    return ObjectModel(local_asset, collision, solid.principal_inertia_per_mass(), com, normal,
                       float(plane.get("height", 0.0)), reference, report)


# --------------------------------------------------------------------------- physics


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value > 20.0, value, torch.log(torch.expm1(torch.clamp(value, max=20.0))))


class ContactPhysics(torch.nn.Module):
    """Learnable soft-constraint contact and generalized damping, all positive."""

    def __init__(self, timeconst=0.05, dampratio=0.7, friction=0.5, damping=1e-3):
        super().__init__()
        self.log_timeconst = torch.nn.Parameter(torch.tensor(math.log(timeconst)))
        self.log_dampratio = torch.nn.Parameter(torch.tensor(math.log(dampratio)))
        self.log_friction = torch.nn.Parameter(torch.tensor(math.log(friction)))
        self.log_damping = torch.nn.Parameter(torch.tensor(math.log(damping)))

    def values(self) -> dict[str, torch.Tensor]:
        return {
            "timeconst": torch.exp(self.log_timeconst),
            "dampratio": torch.exp(self.log_dampratio),
            "friction": torch.exp(self.log_friction),
            "damping": torch.exp(self.log_damping),
        }

    def as_dict(self) -> dict[str, float]:
        return {key: float(value.detach()) for key, value in self.values().items()}


def make_dynamics(obj: ObjectModel, physics: ContactPhysics, *, dt: float, damping_override=None):
    values = physics.values()
    floor = GaussianCollisionBody(torch.zeros(1, 3), torch.full((1,), 1e-6), torch.zeros(1, 3))
    config = MultiBodyImpedanceDynamicsConfig(
        dt=float(dt), masses=(1.0, 1.0),
        inertia_diags=(tuple(float(v) for v in obj.inertia_per_mass), (1.0, 1.0, 1.0)),
        generalized_damping=(0.0, 0.0), dynamic_flags=(True, False), kinematic_flags=(False, False),
        candidate_pair_mode="all", contact_model="soft_constraint",
    )
    plane = PlaneCollider(torch.as_tensor(obj.support_normal, dtype=torch.float32), float(obj.support_height))
    dynamics = MultiBodyGaussianImpedanceDynamics(
        [obj.collision, floor],
        stiffness=_inverse_softplus(values["timeconst"]).reshape(1),
        damping=_inverse_softplus(values["dampratio"]).reshape(1),
        friction_coefficient=_inverse_softplus(values["friction"]).reshape(1),
        candidate_pairs=[], plane_contact_pairs=[GaussianPlaneContactPair(0, 1, plane, 0)], config=config,
    )
    damping = values["damping"] if damping_override is None else damping_override
    dynamics.generalized_damping_parameters = torch.stack((
        _inverse_softplus(damping), torch.tensor(-30.0),
    ))
    return dynamics


def _static_state() -> RigidBodyState:
    return RigidBodyState(torch.zeros(3), torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.zeros(3), torch.zeros(3))


def rollout(dynamics, state0: RigidBodyState, num_frames: int, substeps: int):
    """Poses at frames 0..num_frames-1; frame 0 is ``state0`` itself."""
    states = (state0, _static_state())
    positions, quaternions = [state0.position], [F.normalize(state0.quaternion_wxyz, dim=-1)]
    for _ in range(int(num_frames) - 1):
        for _ in range(int(substeps)):
            states, _ = dynamics.step(states)
        positions.append(states[0].position)
        quaternions.append(states[0].quaternion_wxyz)
    return torch.stack(positions), torch.stack(quaternions)


def first_contact_frame(obj: ObjectModel, state0: RigidBodyState, *, num_frames: int, substeps: int,
                        physics: ContactPhysics, dt: float) -> int:
    """First frame whose pose (from ``state0``) has any primitive below the support plane."""
    dynamics = make_dynamics(obj, physics, dt=dt)
    normal = torch.as_tensor(obj.support_normal, dtype=torch.float32)
    with torch.no_grad():
        positions, quaternions = rollout(dynamics, state0, num_frames, substeps)
        for frame in range(num_frames):
            rotation = quat_to_matrix_wxyz(quaternions[frame])
            world = obj.collision.local_centers @ rotation.T + positions[frame]
            if bool(((world @ normal) - obj.support_height - obj.collision.radii < 0.0).any()):
                return frame
    return num_frames


# --------------------------------------------------------------------------- rendering


class ObjectRenderer:
    """Object-only Gaussian render: colour over black and accumulated opacity.

    The 3DGS rasterizer adds a fixed 0.3 px^2 screen-space blur to every
    Gaussian, which inflates the silhouette of a small, distant object by about
    half a pixel per side (a ~3% depth bias for a 40 px object).  Rendering at
    ``supersample`` times the resolution and box-filtering back integrates each
    pixel like a camera and removes most of that bias.

    ``box = (x0, y0, x1, y1)`` renders only that pixel window of the image (an
    off-centre sub-frustum of the same camera), so the supersampling cost
    follows the object's size instead of the whole frame.
    """

    def __init__(self, asset: RenderableGaussianAsset, camera: MiniCam, *, camera_spec: "CameraSpec | None" = None,
                 supersample: int = 1):
        self.asset = asset
        self.supersample = max(1, int(supersample))
        self.base_camera = camera
        self.width, self.height = int(camera.image_width), int(camera.image_height)
        self._cameras: dict = {}
        device = asset.xyz.device
        self.black = torch.zeros(3, device=device)
        self.ones = torch.ones((asset.num_gaussians, 3), device=device)

    def _camera(self, box):
        if box is None:
            box = (0, 0, self.width, self.height)
        camera = self._cameras.get(box)
        if camera is None:
            x0, y0, x1, y1 = box
            base = self.base_camera
            scale_x, scale_y = (x1 - x0) / self.width, (y1 - y0) / self.height
            centre_x, centre_y = (x0 + x1) / self.width - 1.0, (y0 + y1) / self.height - 1.0
            window = torch.eye(4, device=base.full_proj_transform.device)
            window[0, 0], window[1, 1] = 1.0 / scale_x, 1.0 / scale_y
            window[3, 0], window[3, 1] = -centre_x / scale_x, -centre_y / scale_y  # row-vector convention
            fovx = 2.0 * math.atan(scale_x * math.tan(0.5 * base.FoVx))
            fovy = 2.0 * math.atan(scale_y * math.tan(0.5 * base.FoVy))
            camera = MiniCam((x1 - x0) * self.supersample, (y1 - y0) * self.supersample, fovy, fovx,
                             base.znear, base.zfar, base.world_view_transform, base.full_proj_transform @ window)
            if len(self._cameras) > 512:
                self._cameras.clear()
            self._cameras[box] = camera
        return camera

    def __call__(self, position: torch.Tensor, quaternion: torch.Tensor, box=None, *, colour: bool = True,
                 opacity: bool = True):
        """``(rgb, alpha)``; a pass that is switched off returns ``None`` (each pass is one rasterization)."""
        camera = self._camera(box)
        posed = rigid_transform_asset(self.asset, position, F.normalize(quaternion, dim=-1))
        model = copy_asset_to_gaussian_model(posed, GaussianModel(posed.sh_degree))
        rgb = alpha = None
        if colour:
            rgb = gs_render(camera, model, _Pipe(), self.black, separate_sh=False)["render"]
        if opacity:
            alpha = gs_render(camera, model, _Pipe(), self.black, separate_sh=False,
                              override_color=self.ones)["render"][:1]
        if self.supersample > 1:
            rgb = None if rgb is None else F.avg_pool2d(rgb[None], self.supersample)[0]
            alpha = None if alpha is None else F.avg_pool2d(alpha[None], self.supersample)[0]
        return rgb, alpha


def frame_box(episode: Episode, frame: int, *, align: int = 16):
    """Pixel window around the observed object (three times its size), or None without a mask."""
    cached = episode.boxes.get(frame, False)
    if cached is not False:
        return cached
    mask = episode.mask[frame, 0]
    height, width = mask.shape
    ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
    box = None
    if ys.numel() > 0:
        x_lo, x_hi, y_lo, y_hi = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        pad = max(24, x_hi - x_lo, y_hi - y_lo)
        x0 = max(0, (x_lo - pad) // align * align)
        y0 = max(0, (y_lo - pad) // align * align)
        x1 = min(width // align * align, -(-(x_hi + pad) // align) * align)
        y1 = min(height // align * align, -(-(y_hi + pad) // align) * align)
        if x1 > x0 and y1 > y0:
            box = (x0, y0, x1, y1)
    episode.boxes[frame] = box
    return box


def observed_frame_loss(renderer: "ObjectRenderer", episode: Episode, frame: int, position, quaternion):
    """Image loss of a pose against one observed frame, evaluated in the object's window."""
    box = frame_box(episode, frame)
    rgb, alpha = renderer(position, quaternion, box)
    if box is None:
        return frame_image_loss(rgb, alpha, episode.rgb[frame], episode.mask[frame])
    x0, y0, x1, y1 = box
    return frame_image_loss(rgb, alpha, episode.rgb[frame][:, y0:y1, x0:x1], episode.mask[frame][:, y0:y1, x0:x1])


SILHOUETTE_SCALES = (1, 2, 4, 8, 16)


def chromaticity(rgb: torch.Tensor, floor: float = 0.05) -> torch.Tensor:
    """Colour with the brightness divided out (``rgb / (r + g + b)``).

    The Gaussians carry the shading of the Stage 1 scan baked in; a body that
    tumbles under the same lights is shaded differently in every frame, so raw
    intensities favour whichever orientation happens to put a bright face where
    the image is bright.  Chromaticity is unchanged by white-light shading.
    """
    return rgb / (rgb.sum(dim=0, keepdim=True) + float(floor))


def frame_image_loss(rgb, alpha, gt_rgb, gt_mask):
    """Silhouette (multi-scale, area normalized) + in-mask chromaticity L1."""
    area = gt_mask.sum().clamp_min(1.0)
    silhouette = 0.0
    for scale in SILHOUETTE_SCALES:
        a = F.avg_pool2d(alpha[None], scale)[0] if scale > 1 else alpha
        m = F.avg_pool2d(gt_mask[None], scale)[0] if scale > 1 else gt_mask
        silhouette = silhouette + (a - m).abs().sum() * float(scale * scale) / area
    silhouette = silhouette / len(SILHOUETTE_SCALES)
    colour = ((chromaticity(rgb) - chromaticity(gt_rgb)) * gt_mask).abs().sum() / area
    return silhouette + colour, float(silhouette), float(colour)


def rendered_loss_and_backward(renderer: ObjectRenderer, positions: torch.Tensor, quaternions: torch.Tensor,
                               episode: Episode, frame_ids: list[int], *, weight: float = 1.0,
                               backward: bool = True, frame_offset: int = 0) -> dict:
    """Image loss over ``frame_ids`` with a per-frame GPU backward.

    ``positions[i]`` is the pose of episode frame ``frame_offset + i``.

    Rendering gradients are accumulated frame by frame on detached pose leaves
    and then pushed through the (CPU) dynamics graph in one call, so GPU memory
    stays constant in the number of frames.
    """
    device = episode.rgb.device
    pos_leaf = positions.detach().to(device).requires_grad_(backward)
    quat_leaf = quaternions.detach().to(device).requires_grad_(backward)
    total, silhouettes, colours = 0.0, [], []
    for frame in frame_ids:
        loss, silhouette, colour = observed_frame_loss(
            renderer, episode, frame, pos_leaf[frame - frame_offset], quat_leaf[frame - frame_offset])
        loss = loss * (float(weight) / len(frame_ids))
        if backward:
            loss.backward()
        total += float(loss.detach())
        silhouettes.append(silhouette)
        colours.append(colour)
    if backward and (positions.requires_grad or quaternions.requires_grad):
        torch.autograd.backward(
            [positions, quaternions],
            [pos_leaf.grad.to(positions.device), quat_leaf.grad.to(quaternions.device)],
        )
    return {"loss": total, "silhouette_mean": float(np.mean(silhouettes)), "colour_mean": float(np.mean(colours))}


def auto_supersample(obj: ObjectModel, episode: Episode, *, maximum: int = 8) -> int:
    """Render scale that shows the object at about its Stage 1 pixel size.

    The rasterizer's 0.3 px^2 blur is a fixed number of pixels, so its
    silhouette inflation (a depth bias) grows as the object gets smaller in the
    image.  Rendering the episode at this factor and box-filtering back keeps
    the object near the resolution the Gaussians were trained at.
    """
    reference = obj.report.get("stage1_object_pixels")
    areas = episode.mask.sum(dim=(1, 2, 3))
    areas = areas[areas > 0]
    if reference is None or areas.numel() == 0:
        return 2
    factor = math.sqrt(float(reference) / float(areas.median()))
    return int(min(max(1, math.ceil(factor - 0.25)), int(maximum)))


# --------------------------------------------------------------------------- initial state


def _uniform_quaternions(count: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn((count, 4), generator=generator)
    q = F.normalize(q, dim=-1)
    q = torch.where(q[:, :1] < 0, -q, q)
    return torch.cat((torch.tensor([[1.0, 0.0, 0.0, 0.0]]), q), dim=0)


def _pixel_ray(camera_spec: CameraSpec, u: float, v: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    fy = 0.5 * height / math.tan(0.5 * math.radians(camera_spec.fovy_deg))
    x = (u + 0.5 - 0.5 * width) / fy
    y = -(v + 0.5 - 0.5 * height) / fy
    direction_cam = np.array([x, y, -1.0])
    rotation = camera_spec.c2w[:3, :3]
    direction = rotation @ direction_cam
    return camera_spec.c2w[:3, 3].copy(), direction / np.linalg.norm(direction)


def _camera_axes(episode: Episode):
    """Camera centre, right/up/forward axes (world) and the focal length in pixels."""
    c2w = torch.as_tensor(episode.camera_spec.c2w, dtype=torch.float32, device=episode.rgb.device)
    focal = 0.5 * episode.rgb.shape[-2] / math.tan(0.5 * math.radians(episode.camera_spec.fovy_deg))
    return c2w[:3, 3], c2w[:3, 0], c2w[:3, 1], -c2w[:3, 2], focal


def _rotation_quaternion(vector: torch.Tensor) -> torch.Tensor:
    """Unit quaternion of a rotation vector (smooth at zero)."""
    half = 0.5 * torch.sqrt((vector * vector).sum() + 1e-16)
    return torch.cat((torch.cos(half).reshape(1), torch.sinc(half / torch.pi) * 0.5 * vector))


def _rotation_ball(count: int, max_angle: float, seed: int = 0) -> torch.Tensor:
    """``count`` rotations spread over all axes and angles up to ``max_angle`` (deterministic)."""
    generator = torch.Generator().manual_seed(seed)
    axes = F.normalize(torch.randn((count, 3), generator=generator), dim=-1)
    angles = (torch.arange(count, dtype=torch.float32).reshape(-1, 1) + 1.0) / count * float(max_angle)
    return torch.cat((torch.cos(0.5 * angles), torch.sin(0.5 * angles) * axes), dim=-1)


def align_to_silhouette(renderer: ObjectRenderer, episode: Episode, frame: int, position: torch.Tensor,
                        quaternion: torch.Tensor, *, rounds: int = 3) -> torch.Tensor:
    """Closed-form position update: match the rendered silhouette's centroid and area to the mask's.

    Across the viewing ray one pixel is ``depth / focal`` metres and along it
    the silhouette area scales with ``1 / range^2``, so a render without
    gradients puts the object within a fraction of a pixel of the observation,
    wherever the previous frame left it.
    """
    box = frame_box(episode, frame)
    if box is None:
        return position
    x0, y0, x1, y1 = box
    device = episode.rgb.device
    mask = episode.mask[frame, 0, y0:y1, x0:x1]
    area = mask.sum()
    us = torch.arange(x0, x1, device=device, dtype=torch.float32) + 0.5
    vs = torch.arange(y0, y1, device=device, dtype=torch.float32) + 0.5
    mask_u, mask_v = (mask.sum(0) * us).sum() / area, (mask.sum(1) * vs).sum() / area
    centre, right, up, forward, focal = _camera_axes(episode)
    height, width = episode.rgb.shape[-2:]
    position = position.detach()
    with torch.no_grad():
        for _ in range(int(rounds)):
            alpha = renderer(position, quaternion, box, colour=False)[1][0]
            rendered = alpha.sum()
            if float(rendered) < 0.05 * float(area):
                # the guess is (almost) outside the window: restart on the mask centroid's ray at the same range
                direction = F.normalize(forward * focal + right * (mask_u - 0.5 * width) - up * (mask_v - 0.5 * height), dim=0)
                position = centre + (position - centre).norm() * direction
                continue
            u, v = (alpha.sum(0) * us).sum() / rendered, (alpha.sum(1) * vs).sum() / rendered
            depth = (position - centre) @ forward
            position = position + ((mask_u - u) * right - (mask_v - v) * up) * depth / focal
            position = centre + (position - centre) * torch.sqrt(rendered / area)
    return position


def colour_score(renderer: ObjectRenderer, episode: Episode, frame: int, position, quaternion) -> torch.Tensor:
    """In-mask chromaticity error of a pose (one rasterization; used to rank orientation candidates)."""
    box = frame_box(episode, frame)
    rgb = renderer(position, quaternion, box, opacity=False)[0]
    gt_rgb, gt_mask = episode.rgb[frame], episode.mask[frame]
    if box is not None:
        x0, y0, x1, y1 = box
        gt_rgb, gt_mask = gt_rgb[:, y0:y1, x0:x1], gt_mask[:, y0:y1, x0:x1]
    return ((chromaticity(rgb) - chromaticity(gt_rgb)) * gt_mask).abs().sum() / gt_mask.sum().clamp_min(1.0)


def refine_pose(renderer: ObjectRenderer, episode: Episode, frame: int, position: torch.Tensor,
                quaternion: torch.Tensor, *, iters: int = 30, step_pixels: float = 0.5):
    """Gradient refinement of one frame's pose with steps measured in pixels of image motion.

    The parameters are offsets across the viewing ray, along it and a rotation
    vector, each scaled so that a unit change moves the image by about one
    pixel (``range / focal`` metres across, ``range / R_px`` along, ``1 / R_px``
    radians; ``R_px`` is the silhouette radius).  Depth therefore converges as
    fast as the image-plane position although the image depends on it weakly.
    Returns ``(loss, position, quaternion)`` of the best iterate.
    """
    centre, _, up, _, focal = _camera_axes(episode)
    radius = torch.sqrt(episode.mask[frame].sum().clamp_min(1.0) / math.pi)
    position, quaternion = position.detach(), F.normalize(quaternion.detach(), dim=-1)
    offset = position - centre
    distance = offset.norm()
    ray = offset / distance
    across_1 = F.normalize(torch.cross(ray, up, dim=0), dim=0)
    across_2 = torch.cross(ray, across_1, dim=0)
    basis = torch.stack((across_1 * distance / focal, across_2 * distance / focal, ray * distance / radius), dim=1)
    shift = torch.nn.Parameter(torch.zeros(3, device=position.device))
    turn = torch.nn.Parameter(torch.zeros(3, device=position.device))
    optimizer = torch.optim.Adam([shift, turn], lr=float(step_pixels))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(iters), eta_min=0.1 * float(step_pixels))
    best = None
    for _ in range(int(iters)):
        optimizer.zero_grad()
        p = position + basis @ shift
        q = quat_mul_wxyz(_rotation_quaternion(turn / radius), quaternion)
        loss = observed_frame_loss(renderer, episode, frame, p, q)[0]
        loss.backward()
        value = float(loss.detach())
        if best is None or value < best[0]:
            best = (value, p.detach().clone(), F.normalize(q.detach(), dim=-1))
        optimizer.step()
        scheduler.step()
    return best


HALF_TURNS = ((0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))  # about the body's principal axes


def estimate_initial_pose(renderer: ObjectRenderer, episode: Episode, *, frame: int = 0,
                          num_orientations: int = 256, coarse_candidates: int = 16, coarse_iters: int = 15,
                          refine_candidates: int = 3, refine_iters: int = 60, lookahead: int = 10,
                          log=print) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Pose of ``frame`` from the images: orientation search, refinement, short look-ahead.

    1. ``num_orientations`` orientations, each placed by its silhouette, ranked by colour;
    2. the best ``coarse_candidates`` get a short refinement, and so do the half
       turns of the best of them about the body's principal axes - the poses a
       nearly symmetric object is most easily confused with;
    3. the best distinct ``refine_candidates`` are refined fully and each is
       tracked ``lookahead`` frames: a single view of a small object can look
       alike in several orientations, the following frames tell them apart.
    """
    device = episode.rgb.device
    if float(episode.mask[frame].sum()) < 1.0:
        raise ValueError(f"{episode.name}: empty mask at frame {frame}")
    centre, _, _, forward, _ = _camera_axes(episode)
    start = centre + float(np.linalg.norm(episode.camera_spec.c2w[:3, 3])) * forward
    quaternions = _uniform_quaternions(num_orientations).to(device)
    with torch.no_grad():
        placed = [align_to_silhouette(renderer, episode, frame, start, q) for q in quaternions]
        scores = torch.stack([colour_score(renderer, episode, frame, p, q) for p, q in zip(placed, quaternions)])
    coarse = [refine_pose(renderer, episode, frame, placed[index], quaternions[index], iters=coarse_iters)
              for index in torch.argsort(scores)[: int(coarse_candidates)].tolist()]
    coarse.sort(key=lambda item: item[0])
    for _, p, q in list(coarse[:4]):
        for half_turn in HALF_TURNS:
            turned = quat_mul_wxyz(q, torch.tensor(half_turn, device=device))
            coarse.append(refine_pose(renderer, episode, frame, align_to_silhouette(renderer, episode, frame, p, turned),
                                      turned, iters=coarse_iters))
    coarse.sort(key=lambda item: item[0])
    distinct = []
    for item in coarse:
        if all(float(_rotation_vector(other[2], item[2]).norm()) > math.radians(20.0) for other in distinct):
            distinct.append(item)
    distinct = [refine_pose(renderer, episode, frame, p, q, iters=refine_iters)
                for _, p, q in distinct[: int(refine_candidates)]]
    totals = [item[0] for item in distinct]
    last = min(len(episode.frames), frame + 1 + int(lookahead))
    if len(distinct) > 1 and last > frame + 1:
        totals = []
        for _, p, q in distinct:
            losses = track_poses(renderer, episode, p, q, first=frame, last=last, search=0, log=None)[2]
            totals.append(float(np.nanmean(losses)))
    loss, p, q = distinct[int(np.argmin(totals))]
    log(f"  [{episode.name}] frame-{frame} pose: loss {loss:.4f} (candidates {[round(b[0], 4) for b in distinct]}, "
        f"look-ahead {[round(t, 4) for t in totals]})")
    return p.cpu(), q.cpu(), {"frame": frame, "loss": loss, "candidate_losses": [b[0] for b in distinct],
                              "lookahead_losses": totals}


def fit_free_flight(obj: ObjectModel, physics: ContactPhysics, renderer: ObjectRenderer, episode: Episode,
                    position: torch.Tensor, quaternion: torch.Tensor, *, iters: int = 250,
                    learn_damping: bool = True, rounds: int = 2, log=print):
    """Initial pose/velocity (and damping) from the frames before first contact."""
    p = torch.nn.Parameter(position.clone())
    q = torch.nn.Parameter(quaternion.clone())
    v = torch.nn.Parameter(torch.zeros(3))
    w = torch.nn.Parameter(torch.zeros(3))
    log_damping = torch.nn.Parameter(physics.log_damping.detach().clone())
    history = []
    for round_index in range(int(rounds)):
        state = RigidBodyState(p.detach(), F.normalize(q.detach(), dim=-1), v.detach(), w.detach())
        contact = first_contact_frame(obj, state, num_frames=len(episode.frames), substeps=episode.substeps,
                                      physics=physics, dt=episode.physics_dt)
        count = max(3, min(contact, len(episode.frames)))
        frame_ids = list(range(count))
        groups = [{"params": [p], "lr": 1e-3}, {"params": [q], "lr": 3e-3},
                  {"params": [v], "lr": 1e-2}, {"params": [w], "lr": 3e-2}]
        if learn_damping:
            groups.append({"params": [log_damping], "lr": 3e-2})
        optimizer = torch.optim.Adam(groups)
        for iteration in range(int(iters)):
            optimizer.zero_grad()
            dynamics = make_dynamics(obj, physics, dt=episode.physics_dt, damping_override=torch.exp(log_damping))
            state = RigidBodyState(p, F.normalize(q, dim=-1), v, w)
            positions, quaternions = rollout(dynamics, state, count, episode.substeps)
            stats = rendered_loss_and_backward(renderer, positions, quaternions, episode, frame_ids)
            optimizer.step()
            if iteration % 50 == 0 or iteration == int(iters) - 1:
                history.append({"round": round_index, "iteration": iteration, "frames": count, **stats,
                                "damping": float(torch.exp(log_damping))})
        log(f"  [{episode.name}] free flight round {round_index}: frames 0..{count - 1}, "
            f"loss {history[-1]['loss']:.4f}, damping {float(torch.exp(log_damping)):.4f}")
    state = RigidBodyState(p.detach(), F.normalize(q.detach(), dim=-1), v.detach(), w.detach())
    return state, float(torch.exp(log_damping).detach()), history


def estimate_initial_state(obj, physics, renderer, episode, *, learn_damping=True, free_flight_iters=250, log=print):
    position, quaternion, pose_report = estimate_initial_pose(renderer, episode, log=log)
    state, damping, history = fit_free_flight(obj, physics, renderer, episode, position, quaternion,
                                              iters=free_flight_iters, learn_damping=learn_damping, log=log)
    return state, damping, {"pose": pose_report, "free_flight": history}


# --------------------------------------------------------------------------- physics learning


def fit_physics(obj: ObjectModel, episodes: list[Episode], renderers: list[ObjectRenderer],
                initial_states: list[RigidBodyState], physics: ContactPhysics, *, iters: int = 120,
                train_frames: int | None = None, physics_lr: float = 0.03, state_lr_scale: float = 0.1,
                log=print):
    """Jointly learn (tau, zeta, mu, damping) and refine initial states over episodes."""
    leaves = []
    for state in initial_states:
        leaves.append([torch.nn.Parameter(t.detach().clone()) for t in (
            state.position, state.quaternion_wxyz, state.linear_velocity, state.angular_velocity)])
    groups = [{"params": list(physics.parameters()), "lr": physics_lr}]
    for p, q, v, w in leaves:
        groups += [{"params": [p], "lr": 1e-3 * state_lr_scale}, {"params": [q], "lr": 3e-3 * state_lr_scale},
                   {"params": [v], "lr": 1e-2 * state_lr_scale}, {"params": [w], "lr": 3e-2 * state_lr_scale}]
    optimizer = torch.optim.Adam(groups)
    history, best = [], None
    for iteration in range(int(iters)):
        optimizer.zero_grad()
        total = 0.0
        for episode, renderer, (p, q, v, w) in zip(episodes, renderers, leaves):
            count = len(episode.frames) if train_frames is None else min(int(train_frames), len(episode.frames))
            dynamics = make_dynamics(obj, physics, dt=episode.physics_dt)
            positions, quaternions = rollout(dynamics, RigidBodyState(p, F.normalize(q, dim=-1), v, w),
                                             count, episode.substeps)
            stats = rendered_loss_and_backward(renderer, positions, quaternions, episode, list(range(count)),
                                               weight=1.0 / len(episodes))
            total += stats["loss"]
        record = {"iteration": iteration, "loss": total, **physics.as_dict()}
        history.append(record)
        if best is None or total < best[0]:
            best = (total, {k: v.detach().clone() for k, v in physics.state_dict().items()},
                    [[t.detach().clone() for t in leaf] for leaf in leaves])
        optimizer.step()
        if iteration % 10 == 0 or iteration == int(iters) - 1:
            log(f"  physics it {iteration:3d}: loss {total:.4f} tau {record['timeconst']:.4f} "
                f"zeta {record['dampratio']:.3f} mu {record['friction']:.3f} damping {record['damping']:.4f}")
    physics.load_state_dict(best[1])
    states = [RigidBodyState(p, F.normalize(q, dim=-1), v, w) for p, q, v, w in best[2]]
    return physics, states, history


# --------------------------------------------------------------------------- multiple shooting


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def _rotation_vector(q_from: torch.Tensor, q_to: torch.Tensor) -> torch.Tensor:
    """World-frame rotation vector taking ``q_from`` to ``q_to`` (shortest arc)."""
    delta = quat_mul_wxyz(q_to, _quat_conjugate(q_from))
    delta = torch.where(delta[..., :1] < 0, -delta, delta)
    vector = delta[..., 1:]
    norm = vector.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, delta[..., :1])
    return torch.where(norm > 1e-9, vector / norm.clamp_min(1e-12) * angle, 2.0 * vector)


def track_poses(renderer: ObjectRenderer, episode: Episode, position: torch.Tensor, quaternion: torch.Tensor,
                *, iters: int = 30, search: int = 48, search_angle: float = 0.5 * math.pi, first: int = 0,
                last: int | None = None, log=print) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Physics-free per-frame pose tracking from the images (frames ``first .. last-1``).

    Position: the silhouette's centroid and area give it in closed form every
    frame (``align_to_silhouette``), so it never depends on a motion model.
    Orientation: starts from a constant-angular-velocity extrapolation; because
    an impact can change the spin arbitrarily, ``search`` rotations around the
    extrapolation are ranked by colour as well and the best one is refined too
    whenever it ranks above the extrapolation.
    """
    device = episode.rgb.device
    last = len(episode.frames) if last is None else int(last)
    positions, quaternions = [position.to(device)], [F.normalize(quaternion.to(device), dim=-1)]
    losses = [float(observed_frame_loss(renderer, episode, first, positions[0], quaternions[0])[0].detach())]
    offsets = _rotation_ball(int(search), search_angle).to(device) if search > 0 else None
    for frame in range(first + 1, last):
        if len(positions) >= 2:
            guess_p = 2.0 * positions[-1] - positions[-2]
            guess_q = F.normalize(quat_mul_wxyz(
                _rotation_quaternion(_rotation_vector(quaternions[-2], quaternions[-1])), quaternions[-1]), dim=-1)
        else:
            guess_p, guess_q = positions[-1], quaternions[-1]
        if float(episode.mask[frame].sum()) < 1.0:  # object left the image: keep the extrapolation
            positions.append(guess_p.detach())
            quaternions.append(guess_q.detach())
            losses.append(float("nan"))
            continue
        placed = align_to_silhouette(renderer, episode, frame, positions[-1], guess_q)
        best = refine_pose(renderer, episode, frame, placed, guess_q, iters=iters)
        if offsets is not None:
            candidates = F.normalize(quat_mul_wxyz(offsets, guess_q.expand_as(offsets)), dim=-1)
            with torch.no_grad():
                scores = torch.stack([colour_score(renderer, episode, frame, placed, q) for q in candidates])
                reference = colour_score(renderer, episode, frame, placed, guess_q)
            index = int(torch.argmin(scores))
            if float(scores[index]) < float(reference):
                start = align_to_silhouette(renderer, episode, frame, placed, candidates[index])
                attempt = refine_pose(renderer, episode, frame, start, candidates[index], iters=iters)
                if attempt[0] < best[0]:
                    best = attempt
        positions.append(best[1])
        quaternions.append(best[2])
        losses.append(best[0])
    finite = [value for value in losses if value == value]
    if log is not None:
        log(f"  [{episode.name}] tracked {len(positions)} frames, frame loss mean {np.mean(finite):.4f} max {np.max(finite):.4f}")
    return torch.stack(positions).cpu(), torch.stack(quaternions).cpu(), losses


def states_from_track(positions: torch.Tensor, quaternions: torch.Tensor, frame_dt: float) -> list[RigidBodyState]:
    """Per-frame states with central finite-difference velocities."""
    count = positions.shape[0]
    states = []
    for frame in range(count):
        lo, hi = max(0, frame - 1), min(count - 1, frame + 1)
        span = float(hi - lo) * float(frame_dt)
        velocity = (positions[hi] - positions[lo]) / span
        omega = _rotation_vector(quaternions[lo], quaternions[hi]) / span
        states.append(RigidBodyState(positions[frame].clone(), quaternions[frame].clone(), velocity, omega))
    return states


def window_starts(num_frames: int, window: int) -> list[int]:
    return list(range(0, max(1, num_frames - 1), int(window)))


def fit_physics_multiple_shooting(
    obj: ObjectModel, episodes: list[Episode], renderers: list[ObjectRenderer],
    tracked_states: list[list[RigidBodyState]], physics: ContactPhysics, *, window: int = 10,
    iters: int = 100, physics_lr: float = 0.05, log=print,
):
    """Learn shared physics with one free start state per ``window`` frames.

    A long bouncing/tumbling rollout is chaotic, so its gradient with respect to
    the physics is useless.  Each window is rolled out only ``window`` frames
    from its own start state (initialised from the image track), which keeps
    every gradient path short while all windows share the physics parameters.
    """
    leaves, starts = [], []
    for episode, states in zip(episodes, tracked_states):
        episode_starts = window_starts(len(episode.frames), window)
        starts.append(episode_starts)
        leaves.append([[torch.nn.Parameter(t.detach().clone()) for t in (
            states[t0].position, states[t0].quaternion_wxyz, states[t0].linear_velocity, states[t0].angular_velocity)]
            for t0 in episode_starts])
    groups = [{"params": list(physics.parameters()), "lr": physics_lr}]
    for episode_leaves in leaves:
        for p, q, v, w in episode_leaves:
            groups += [{"params": [p], "lr": 1e-3}, {"params": [q], "lr": 2e-3},
                       {"params": [v], "lr": 1e-2}, {"params": [w], "lr": 3e-2}]
    optimizer = torch.optim.Adam(groups)
    total_windows = sum(len(item) for item in starts)
    history, best = [], None
    for iteration in range(int(iters)):
        optimizer.zero_grad()
        total = 0.0
        for episode, renderer, episode_starts, episode_leaves in zip(episodes, renderers, starts, leaves):
            for t0, (p, q, v, w) in zip(episode_starts, episode_leaves):
                end = min(len(episode.frames), t0 + int(window) + 1)
                dynamics = make_dynamics(obj, physics, dt=episode.physics_dt)
                positions, quaternions = rollout(dynamics, RigidBodyState(p, F.normalize(q, dim=-1), v, w),
                                                 end - t0, episode.substeps)
                stats = rendered_loss_and_backward(renderer, positions, quaternions, episode, list(range(t0, end)),
                                                   weight=1.0 / total_windows, frame_offset=t0)
                total += stats["loss"]
        record = {"iteration": iteration, "loss": total, **physics.as_dict()}
        history.append(record)
        if best is None or total < best[0]:
            best = (total, {k: v.detach().clone() for k, v in physics.state_dict().items()},
                    [[[t.detach().clone() for t in leaf] for leaf in episode_leaves] for episode_leaves in leaves])
        optimizer.step()
        if iteration % 10 == 0 or iteration == int(iters) - 1:
            log(f"  physics it {iteration:3d}: loss {total:.4f} tau {record['timeconst']:.4f} "
                f"zeta {record['dampratio']:.3f} mu {record['friction']:.3f} damping {record['damping']:.4f}")
    physics.load_state_dict(best[1])
    fitted = [[RigidBodyState(p, F.normalize(q, dim=-1), v, w) for p, q, v, w in episode_leaves]
              for episode_leaves in best[2]]
    return physics, starts, fitted, history


def fit_window_state(obj: ObjectModel, physics: ContactPhysics, renderer: ObjectRenderer, episode: Episode,
                     state: RigidBodyState, t0: int, t1: int, *, iters: int = 40) -> RigidBodyState:
    """State at frame ``t0`` that best explains frames ``t0..t1`` under fixed physics."""
    p, q, v, w = [torch.nn.Parameter(t.detach().clone()) for t in (
        state.position, state.quaternion_wxyz, state.linear_velocity, state.angular_velocity)]
    optimizer = torch.optim.Adam([{"params": [p], "lr": 1e-3}, {"params": [q], "lr": 2e-3},
                                  {"params": [v], "lr": 1e-2}, {"params": [w], "lr": 3e-2}])
    best = None
    for _ in range(int(iters)):
        optimizer.zero_grad()
        dynamics = make_dynamics(obj, physics, dt=episode.physics_dt)
        positions, quaternions = rollout(dynamics, RigidBodyState(p, F.normalize(q, dim=-1), v, w),
                                         t1 - t0 + 1, episode.substeps)
        stats = rendered_loss_and_backward(renderer, positions, quaternions, episode, list(range(t0, t1 + 1)),
                                           frame_offset=t0)
        if best is None or stats["loss"] < best[0]:
            best = (stats["loss"], [t.detach().clone() for t in (p, q, v, w)])
        optimizer.step()
    p, q, v, w = best[1]
    return RigidBodyState(p, F.normalize(q, dim=-1), v, w)


def forecast_from_observations(obj: ObjectModel, physics: ContactPhysics, renderer: ObjectRenderer,
                               episode: Episode, tracked: list[RigidBodyState], *, window: int = 10,
                               horizon: int = 15, iters: int = 40, stride: int | None = None) -> list[dict]:
    """Short-horizon prediction from the last ``window`` observed frames.

    For each window ``[t - window, t]`` the start state is fitted to those
    frames only (learned physics fixed); the same rollout then continues
    ``horizon`` unobserved frames.  Returns poses for ``t .. t + horizon``.
    """
    results = []
    count = len(episode.frames)
    for t in range(int(window), count - int(horizon), int(window if stride is None else stride)):
        t0 = t - int(window)
        state = fit_window_state(obj, physics, renderer, episode, tracked[t0], t0, t, iters=iters)
        with torch.no_grad():
            positions, quaternions = rollout(make_dynamics(obj, physics, dt=episode.physics_dt), state,
                                             t - t0 + int(horizon) + 1, episode.substeps)
        results.append({"observed_until": t, "frames": list(range(t, t + int(horizon) + 1)),
                        "positions": positions[t - t0:], "quaternions": quaternions[t - t0:]})
    return results


def pose_errors(obj: ObjectModel, positions: torch.Tensor, quaternions: torch.Tensor,
                gt_positions: np.ndarray, gt_quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translation [m] and rotation [deg] error of COM-frame poses against body-frame ground truth."""
    body_p, body_q = to_reference_frame(obj, positions.detach().cpu(), quaternions.detach().cpu())
    translation = np.linalg.norm(body_p.numpy().astype(np.float64) - gt_positions, axis=1)
    body_q = F.normalize(body_q, dim=-1).numpy().astype(np.float64)
    dots = np.clip(np.abs(np.sum(body_q * gt_quaternions, axis=1)), 0.0, 1.0)
    return translation, np.degrees(2.0 * np.arccos(dots))


# --------------------------------------------------------------------------- evaluation


def ray_error_components(position_error: np.ndarray, gt_positions: np.ndarray, camera_position) -> tuple[np.ndarray, np.ndarray]:
    """Split position errors into the part across the camera's viewing ray and the part along it.

    One camera sees the first directly (it moves the object in the image) and
    the second only through the object's apparent size.
    """
    ray = gt_positions - np.asarray(camera_position, dtype=np.float64)
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    along = np.sum(position_error * ray, axis=1)
    return np.linalg.norm(position_error - along[:, None] * ray, axis=1), np.abs(along)


def to_reference_frame(obj: ObjectModel, positions: torch.Tensor, quaternions: torch.Tensor):
    """COM-frame poses -> poses of the Stage 1 reference body frame (evaluation only)."""
    if obj.reference_pose is None:
        return positions, quaternions
    origin = torch.as_tensor(obj.reference_pose["position"], dtype=positions.dtype) - torch.as_tensor(
        obj.center_of_mass, dtype=positions.dtype)
    reference_q = torch.as_tensor(obj.reference_pose["quaternion_wxyz"], dtype=quaternions.dtype)
    rotation = quat_to_matrix_wxyz(quaternions)
    body_positions = positions + (rotation @ origin.unsqueeze(-1)).squeeze(-1)
    body_quaternions = quat_mul_wxyz(F.normalize(quaternions, dim=-1), reference_q.expand_as(quaternions))
    return body_positions, body_quaternions


def reference_state_from_ground_truth(obj: ObjectModel, trajectory_path: Path, frame: int = 0) -> RigidBodyState:
    """GT body-frame state -> COM-frame state (for oracle diagnostics only)."""
    payload = _read_json(trajectory_path)
    state = next(s for s in payload["states"] if int(s["frame_index"]) == frame)
    q_body = torch.tensor(state["quaternion_wxyz"], dtype=torch.float32)
    omega = torch.tensor(state["angular_velocity"], dtype=torch.float32)
    if payload.get("angular_velocity_frame") != "world":
        omega = angular_velocity_body_to_world(q_body, omega)
    reference_q = torch.tensor(obj.reference_pose["quaternion_wxyz"], dtype=torch.float32)
    q = quat_mul_wxyz(q_body, torch.cat((reference_q[:1], -reference_q[1:])))
    origin = torch.as_tensor(obj.reference_pose["position"], dtype=torch.float32) - torch.as_tensor(
        obj.center_of_mass, dtype=torch.float32)
    lever = quat_to_matrix_wxyz(q) @ origin
    position = torch.tensor(state["position"], dtype=torch.float32) - lever
    velocity = torch.tensor(state["linear_velocity"], dtype=torch.float32) - torch.cross(omega, lever, dim=-1)
    return RigidBodyState(position, q, velocity, omega)


def evaluate_trajectory(obj: ObjectModel, episode: Episode, positions: torch.Tensor, quaternions: torch.Tensor) -> dict:
    if episode.gt_positions is None:
        return {}
    body_p, body_q = to_reference_frame(obj, positions.detach().cpu(), quaternions.detach().cpu())
    body_p = body_p.numpy().astype(np.float64)
    body_q = F.normalize(body_q, dim=-1).numpy().astype(np.float64)
    translation = np.linalg.norm(body_p - episode.gt_positions, axis=1)
    across, along = ray_error_components(body_p - episode.gt_positions, episode.gt_positions,
                                         episode.camera_spec.c2w[:3, 3])
    dots = np.clip(np.abs(np.sum(body_q * episode.gt_quaternions, axis=1)), 0.0, 1.0)
    rotation = np.degrees(2.0 * np.arccos(dots))
    return {
        "translation_across_ray_mean_m": float(across.mean()),
        "translation_along_ray_mean_m": float(along.mean()),
        "per_frame_translation_across_ray_m": across.tolist(),
        "frames": len(episode.frames),
        "translation_mean_m": float(translation.mean()),
        "translation_rmse_m": float(np.sqrt(np.mean(translation ** 2))),
        "translation_max_m": float(translation.max()),
        "translation_final_m": float(translation[-1]),
        "rotation_mean_deg": float(rotation.mean()),
        "rotation_max_deg": float(rotation.max()),
        "rotation_final_deg": float(rotation[-1]),
        "per_frame_translation_m": translation.tolist(),
        "per_frame_rotation_deg": rotation.tolist(),
        "predicted_body_positions": body_p.tolist(),
        "predicted_body_quaternions_wxyz": body_q.tolist(),
        "predicted_com_positions": positions.detach().cpu().tolist(),
        "predicted_com_quaternions_wxyz": F.normalize(quaternions.detach().cpu(), dim=-1).tolist(),
        "ground_truth_body_positions": episode.gt_positions.tolist(),
    }
