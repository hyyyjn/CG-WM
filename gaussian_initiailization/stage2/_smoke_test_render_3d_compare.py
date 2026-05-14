"""Render a 3D side-by-side comparison of GT vs predicted sphere trajectory.

Uses MuJoCo's renderer for BOTH halves so the camera, lighting and materials
are identical — the only thing that differs is the sphere's pose. The left
panel uses the GT trajectory saved during dataset generation; the right
panel rolls out the learned ImpedanceFloorContactDynamics from the GT first
frame and re-renders MuJoCo with that pose stream.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco

from gaussian_initiailization.stage2.differentiable_collision_detection import (
    PlaneCollider,
    detect_gaussian_union_contacts,
    load_gaussian_collision_primitives_from_ply,
    make_floor_disk_query_points,
)
from gaussian_initiailization.stage2.differentiable_complementarity_free_contact_dynamics import (
    _smooth_min,
)
import torch.nn.functional as F


E2E_ROOT = REPO_ROOT / "gaussian_initiailization" / "stage2" / "_outputs" / "e2e"
DATASET_ROOT = E2E_ROOT / "dataset"
PLY_PATH = E2E_ROOT / "stage1_sphere" / "point_cloud" / "iteration_0" / "point_cloud.ply"
FIT_SUMMARY = E2E_ROOT / "fit" / "fit_summary.json"
TEST_EPISODE = DATASET_ROOT / "stage2" / "fall_and_rebound" / "test" / "sphere" / "episode_000"
OUTPUT_GIF = E2E_ROOT / "test_episode_000_3d_compare.gif"

IMAGE_W, IMAGE_H = 640, 480
GRAVITY = -9.81
TIMESTEP = 0.002
GROUND_SIZE = 2.0
CAMERA_DISTANCE = 1.4
CAMERA_HEIGHT = 0.8
SPHERE_RADIUS = 0.10


def build_mjcf() -> str:
    """Single-sphere scene matching the dataset generator (used for the original
    side-by-side render)."""
    return f"""
<mujoco model="sphere_compare">
  <option timestep="{TIMESTEP}" gravity="0 0 {GRAVITY}" integrator="Euler"/>
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <rgba haze="1 1 1 1"/>
  </visual>
  <asset>
    <texture name="texplane" type="2d" builtin="checker" rgb1="0.96 0.96 0.96" rgb2="0.88 0.88 0.88" width="256" height="256"/>
    <material name="matplane" texture="texplane" texrepeat="2 2" reflectance="0.05"/>
    <texture name="texobject" type="cube" builtin="checker" rgb1="0.92 0.28 0.22" rgb2="0.20 0.55 0.92" width="256" height="256"/>
    <material name="matobject" texture="texobject" texuniform="true" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="{GROUND_SIZE} {GROUND_SIZE} 0.1" material="matplane" rgba="0.92 0.92 0.92 1"/>
    <camera name="cam0" pos="0 -{CAMERA_DISTANCE} {CAMERA_HEIGHT}" xyaxes="1 0 0 0 0.5 0.8660254"/>
    <body name="sphere" pos="0 0 1.0">
      <joint name="root_free" type="free" damping="0.05"/>
      <geom name="sphere_geom" type="sphere" size="{SPHERE_RADIUS}" rgba="0.85 0.25 0.20 1"
            density="1000" friction="0.02 0.001 0.0001" solref="-2000 -50"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


def build_overlay_mjcf() -> str:
    """Scene with both balls visible at once: GT in red, prediction in blue,
    rendered semi-transparent so overlap is obvious. Both spheres are static
    (we manipulate qpos directly via mocap bodies — collisions disabled)."""
    return f"""
<mujoco model="sphere_overlay">
  <option timestep="{TIMESTEP}" gravity="0 0 0" integrator="Euler"/>
  <visual>
    <headlight diffuse="0.85 0.85 0.85" ambient="0.32 0.32 0.32" specular="0.15 0.15 0.15"/>
    <rgba haze="1 1 1 1"/>
  </visual>
  <asset>
    <texture name="texplane" type="2d" builtin="checker" rgb1="0.96 0.96 0.96" rgb2="0.86 0.86 0.86" width="256" height="256"/>
    <material name="matplane" texture="texplane" texrepeat="2 2" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true"/>
    <light pos="1.0 -1.0 2.0" dir="-0.5 0.5 -1" directional="false" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" type="plane" size="{GROUND_SIZE} {GROUND_SIZE} 0.1" material="matplane" rgba="0.95 0.95 0.95 1"/>
    <camera name="cam0" pos="0 -{CAMERA_DISTANCE} {CAMERA_HEIGHT}" xyaxes="1 0 0 0 0.5 0.8660254"/>
    <body name="gt_sphere" mocap="true" pos="0 0 1.0">
      <geom name="gt_geom" type="sphere" size="{SPHERE_RADIUS}" rgba="0.90 0.25 0.20 0.85"
            contype="0" conaffinity="0"/>
    </body>
    <body name="pred_sphere" mocap="true" pos="0 0 1.0">
      <geom name="pred_geom" type="sphere" size="{SPHERE_RADIUS}" rgba="0.20 0.50 0.92 0.60"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


def render_pose(renderer, data, model, position, quaternion):
    data.qpos[:3] = position
    data.qpos[3:7] = quaternion
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="cam0")
    rgb = np.asarray(renderer.render(), dtype=np.uint8)
    rgb = np.ascontiguousarray(rgb[..., :3])
    if rgb.ndim == 3 and rgb.shape[2] == 3:
        rgb[(rgb.sum(axis=-1) == 0)] = [255, 255, 255]
    return rgb


def rollout_impedance(
    initial_position,
    initial_velocity,
    gravity_z,
    K,
    D,
    local_centers,
    radii,
    floor_offsets,
    steps,
    dt,
):
    collider = PlaneCollider.floor(dtype=initial_position.dtype, device=initial_position.device)
    position = initial_position.clone()
    velocity = initial_velocity.clone()
    gravity = torch.tensor([0.0, 0.0, gravity_z], dtype=position.dtype, device=position.device)
    K_t = torch.tensor(K, dtype=position.dtype, device=position.device)
    D_t = torch.tensor(D, dtype=position.dtype, device=position.device)
    positions = [position.clone()]
    for _ in range(steps - 1):
        b = velocity + dt * gravity
        floor_points = torch.cat(
            (
                position[:2].unsqueeze(0) + floor_offsets,
                torch.zeros((floor_offsets.shape[0], 1), dtype=position.dtype, device=position.device),
            ),
            dim=-1,
        )
        gaussian_centers = local_centers + position.unsqueeze(0)
        contacts = detect_gaussian_union_contacts(
            floor_points,
            gaussian_centers,
            radii,
            collider.normal.to(dtype=position.dtype, device=position.device),
            softness=8e-4,
            smooth_min_temperature=8e-3,
            inside_penalty=0.025,
            inside_sharpness=100.0,
        )
        phi_agg = _smooth_min(contacts.signed_distances, 8e-3)
        normal = contacts.collider_normal.to(dtype=position.dtype, device=position.device)
        Jb = torch.sum(b * normal)
        lam = F.softplus(-K_t * (dt * Jb + phi_agg) - D_t * Jb)
        velocity = b + (dt / 1.0) * lam * normal
        position = position + dt * velocity
        positions.append(position.clone())
    return torch.stack(positions, dim=0).numpy()


def stitch_side_by_side(left: np.ndarray, right: np.ndarray, t_text: str) -> Image.Image:
    h, w = left.shape[:2]
    pair = np.concatenate(
        (left, np.full((h, 4, 3), 28, dtype=np.uint8), right),
        axis=1,
    )
    img = Image.fromarray(pair)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 32), fill=(28, 28, 30))
    draw.text((12, 8), "MuJoCo ground truth", fill=(240, 240, 240))
    draw.text((w + 16, 8), "Stage 2 prediction (learned K/D/g)", fill=(240, 240, 240))
    draw.text((img.width - 110, 8), t_text, fill=(200, 200, 200))
    return img


def main():
    with open(FIT_SUMMARY, "r", encoding="utf-8") as h:
        fit = json.load(h)
    with open(TEST_EPISODE / "state" / "trajectory.json", "r", encoding="utf-8") as h:
        gt = json.load(h)
    states = gt["states"]
    n_frames = len(states)
    gt_positions = np.array([s["position"] for s in states], dtype=np.float64)
    gt_quats = np.array([s["quaternion_wxyz"] for s in states], dtype=np.float64)
    times = np.array([s["time"] for s in states], dtype=np.float64)
    dt = float(np.median(np.diff(times)))

    device = torch.device("cpu")
    local_centers, radii = load_gaussian_collision_primitives_from_ply(
        PLY_PATH, radius_scale=1.0, recenter=True, dtype=torch.float32, device=device
    )
    xy_extent = torch.linalg.norm(local_centers[:, :2], dim=-1) + radii
    query_radius = float(torch.quantile(xy_extent, 0.98).item()) * 1.20
    floor_offsets = make_floor_disk_query_points(
        query_radius, num_rings=4, num_angles=24, dtype=torch.float32, device=device
    )
    initial_position = torch.tensor(states[0]["position"], dtype=torch.float32)
    initial_velocity_gt = torch.tensor(states[0]["linear_velocity"], dtype=torch.float32)

    pred_positions = rollout_impedance(
        initial_position,
        initial_velocity_gt,
        fit["learned_gravity_z"],
        fit["learned_stiffness"],
        fit["learned_damping"],
        local_centers,
        radii,
        floor_offsets,
        steps=n_frames,
        dt=dt,
    )

    # ---------- Overlay (both balls in one 3D scene) ----------
    overlay_model = mujoco.MjModel.from_xml_string(build_overlay_mjcf())
    overlay_data = mujoco.MjData(overlay_model)
    overlay_renderer = mujoco.Renderer(overlay_model, height=IMAGE_H, width=IMAGE_W)
    gt_body_id = mujoco.mj_name2id(overlay_model, mujoco.mjtObj.mjOBJ_BODY, "gt_sphere")
    pred_body_id = mujoco.mj_name2id(overlay_model, mujoco.mjtObj.mjOBJ_BODY, "pred_sphere")
    gt_mocap_id = overlay_model.body_mocapid[gt_body_id]
    pred_mocap_id = overlay_model.body_mocapid[pred_body_id]

    overlay_frames = []
    # Skip the very first frames where the ball drops in from above the camera
    # frustum (drop_height_test ≈ 0.7 m, camera pitched down → top of arc is
    # off-screen). Start at the frame where both balls are visible.
    start_frame = 14
    stride = 2
    for i in range(start_frame, n_frames, stride):
        gt_pos = gt_positions[i]
        pred_pos = pred_positions[i]
        overlay_data.mocap_pos[gt_mocap_id] = gt_pos
        overlay_data.mocap_pos[pred_mocap_id] = pred_pos
        overlay_data.mocap_quat[gt_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
        overlay_data.mocap_quat[pred_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(overlay_model, overlay_data)
        overlay_renderer.update_scene(overlay_data, camera="cam0")
        rgb = np.asarray(overlay_renderer.render(), dtype=np.uint8)
        rgb = np.ascontiguousarray(rgb[..., :3])
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            rgb[(rgb.sum(axis=-1) == 0)] = [255, 255, 255]
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, img.width, 38), fill=(28, 28, 30))
        draw.text((12, 6), "Sphere fall: GT (red) vs Stage 2 prediction (blue)", fill=(240, 240, 240))
        draw.text((12, 22), f"t = {times[i]:.2f} s   frame {i:03d}/{n_frames-1}", fill=(200, 200, 200))
        err_xyz = pred_pos - gt_pos
        draw.text(
            (img.width - 260, 6),
            f"err Δx={err_xyz[0]*1000:+5.1f}mm  Δy={err_xyz[1]*1000:+5.1f}mm  Δz={err_xyz[2]*1000:+5.1f}mm",
            fill=(220, 220, 200),
        )
        overlay_frames.append(img)

    duration_ms = max(1, int(round(1000.0 * dt * stride)))
    overlay_gif = E2E_ROOT / "test_episode_000_3d_overlay.gif"
    overlay_frames[0].save(
        overlay_gif,
        save_all=True,
        append_images=overlay_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"wrote {overlay_gif}  ({len(overlay_frames)} frames, stride={stride})")

    # Montage of overlay frames at well-spaced times
    picks = np.linspace(0, len(overlay_frames) - 1, 4, dtype=int)
    rows = [np.asarray(overlay_frames[p]) for p in picks]
    montage_arr = np.concatenate(rows, axis=0)
    Image.fromarray(montage_arr).save(E2E_ROOT / "test_episode_000_3d_overlay_montage.png")
    print(f"wrote {E2E_ROOT / 'test_episode_000_3d_overlay_montage.png'}")

    # ---------- Side-by-side (original, kept for completeness) ----------
    model = mujoco.MjModel.from_xml_string(build_mjcf())
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMAGE_H, width=IMAGE_W)
    frames_out = []
    for i in range(start_frame, n_frames, stride):
        gt_pos = gt_positions[i]
        gt_quat = gt_quats[i]
        pred_pos = pred_positions[i]
        gt_rgb = render_pose(renderer, data, model, gt_pos, gt_quat)
        pred_rgb = render_pose(renderer, data, model, pred_pos, gt_quat)
        pair = stitch_side_by_side(
            gt_rgb,
            pred_rgb,
            f"t={times[i]:.2f}s f={i:03d}/{n_frames-1}",
        )
        frames_out.append(pair)
    frames_out[0].save(
        OUTPUT_GIF,
        save_all=True,
        append_images=frames_out[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"wrote {OUTPUT_GIF}  ({len(frames_out)} frames, stride={stride})")
    picks = np.linspace(0, len(frames_out) - 1, 4, dtype=int)
    rows = [np.asarray(frames_out[p]) for p in picks]
    montage_arr = np.concatenate(rows, axis=0)
    Image.fromarray(montage_arr).save(E2E_ROOT / "test_episode_000_3d_compare_montage.png")
    print(f"wrote {E2E_ROOT / 'test_episode_000_3d_compare_montage.png'}")


if __name__ == "__main__":
    main()
