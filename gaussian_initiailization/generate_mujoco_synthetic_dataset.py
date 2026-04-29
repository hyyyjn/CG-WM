import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a simple MuJoCo synthetic dataset in the Blender/NeRF format used by scene initialization."
    )
    parser.add_argument("--output_root", required=True, type=str, help="Directory where the dataset will be created.")
    parser.add_argument(
        "--scene_name",
        default="simple_box",
        type=str,
        help="Dataset subdirectory name written under output_root.",
    )
    parser.add_argument(
        "--object_type",
        default="box",
        choices=["box", "sphere", "cylinder"],
        help="Simple object primitive to render.",
    )
    parser.add_argument("--train_views", default=24, type=int, help="Number of train views on the orbit.")
    parser.add_argument("--test_views", default=8, type=int, help="Number of test views on the orbit.")
    parser.add_argument("--width", default=512, type=int, help="Rendered image width.")
    parser.add_argument("--height", default=512, type=int, help="Rendered image height.")
    parser.add_argument("--fovy_deg", default=45.0, type=float, help="Vertical field of view in degrees.")
    parser.add_argument("--camera_radius", default=1.45, type=float, help="Distance from camera to object center.")
    parser.add_argument("--elevation_deg", default=25.0, type=float, help="Orbit elevation angle in degrees.")
    parser.add_argument(
        "--object_height",
        default=None,
        type=float,
        help="Approximate object center height above the plane. Defaults to a primitive-specific value.",
    )
    parser.add_argument(
        "--settle_steps",
        default=200,
        type=int,
        help="Number of simulation steps before rendering each scene.",
    )
    parser.add_argument(
        "--mujoco_gl",
        default="auto",
        choices=["auto", "egl", "osmesa", "glfw"],
        help="OpenGL backend for offscreen rendering. 'auto' prefers MUJOCO_GL if set, otherwise egl.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed for deterministic camera ordering and optional pose jitter.",
    )
    parser.add_argument(
        "--add_pose_jitter",
        action="store_true",
        help="Apply a small random yaw rotation to the object before settling.",
    )
    return parser.parse_args()


def resolve_mujoco_gl(mode: str) -> str:
    if mode == "auto":
        return os.environ.get("MUJOCO_GL", "egl")
    return mode


def object_geom_xml(object_type: str) -> tuple[str, float]:
    if object_type == "box":
        return (
            """
        <body name="target_body" pos="0 0 0.08">
          <freejoint/>
          <geom name="target_geom" type="box" size="0.08 0.08 0.08" rgba="0.85 0.25 0.2 1"/>
        </body>
            """.strip(),
            0.08,
        )
    if object_type == "sphere":
        return (
            """
        <body name="target_body" pos="0 0 0.09">
          <freejoint/>
          <geom name="target_geom" type="sphere" size="0.09" rgba="0.2 0.45 0.85 1"/>
        </body>
            """.strip(),
            0.09,
        )
    if object_type == "cylinder":
        return (
            """
        <body name="target_body" pos="0 0 0.10">
          <freejoint/>
          <geom name="target_geom" type="cylinder" size="0.07 0.10" rgba="0.2 0.75 0.4 1"/>
        </body>
            """.strip(),
            0.10,
        )
    raise ValueError(f"Unsupported object_type: {object_type}")


def build_model_xml(object_type: str, fovy_deg: float, width: int, height: int) -> str:
    object_xml, _ = object_geom_xml(object_type)
    return f"""
<mujoco model="cgwm_simple_scene">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="{int(width)}" offheight="{int(height)}"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.65 0.65 0.65" specular="0.15 0.15 0.15"/>
    <rgba haze="0.95 0.95 0.95 1"/>
  </visual>
  <asset>
    <texture name="checker" type="2d" builtin="checker" rgb1="0.92 0.92 0.92" rgb2="0.84 0.84 0.84" width="256" height="256"/>
    <material name="floor_mat" texture="checker" texrepeat="3 3" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light name="key" pos="1.5 -1.5 2.8" dir="-0.4 0.3 -1"/>
    <light name="fill" pos="-1.2 1.0 2.1" dir="0 -0.1 -1" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="2 2 0.1" material="floor_mat" rgba="0.95 0.95 0.95 1"/>
    <camera name="render_cam" mode="fixed" pos="1 0 0.5" xyaxes="0 1 0 -0.3 0 1" fovy="{float(fovy_deg)}"/>
    {object_xml}
  </worldbody>
</mujoco>
""".strip()


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / norm


def camera_to_world_matrix(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, up_hint))
    up = normalize(np.cross(right, forward))
    back = -forward

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = back
    c2w[:3, 3] = eye
    return c2w


def orbit_camera_positions(count: int, radius: float, elevation_deg: float, lookat: np.ndarray) -> list[np.ndarray]:
    positions = []
    elevation = math.radians(float(elevation_deg))
    z = radius * math.sin(elevation)
    xy_radius = radius * math.cos(elevation)
    for index in range(count):
        theta = (2.0 * math.pi * index) / float(count)
        eye = np.array(
            [
                lookat[0] + xy_radius * math.cos(theta),
                lookat[1] + xy_radius * math.sin(theta),
                lookat[2] + z,
            ],
            dtype=np.float32,
        )
        positions.append(eye)
    return positions


def set_camera_pose(mujoco, model, camera_name: str, eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera not found: {camera_name}")

    forward = normalize(target - eye)
    right = normalize(np.cross(forward, up_hint))
    up = normalize(np.cross(right, forward))
    rot = np.stack([right, up, -forward], axis=1).astype(np.float64)
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, rot.reshape(-1))

    model.cam_pos[cam_id] = eye
    model.cam_quat[cam_id] = quat
    if hasattr(model, "cam_mat0"):
        model.cam_mat0[cam_id] = rot.reshape(-1)


def save_png(imageio, path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, array)


def segmentation_to_mask(segmentation: np.ndarray, target_geom_id: int, floor_geom_id: int | None = None) -> np.ndarray:
    if segmentation.ndim == 2:
        mask = segmentation == target_geom_id
    elif segmentation.ndim == 3:
        mask = np.zeros(segmentation.shape[:2], dtype=bool)
        for channel in range(segmentation.shape[-1]):
            mask |= segmentation[..., channel] == target_geom_id
        if not mask.any() and floor_geom_id is not None:
            non_floor = np.ones(segmentation.shape[:2], dtype=bool)
            for channel in range(segmentation.shape[-1]):
                non_floor &= segmentation[..., channel] != floor_geom_id
            mask = non_floor
    else:
        raise ValueError(f"Unsupported segmentation shape: {segmentation.shape}")
    return mask.astype(np.uint8)


def build_rgba(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = (mask * 255).astype(np.uint8)[..., None]
    return np.concatenate([rgb.astype(np.uint8), alpha], axis=-1)


def write_transforms(path: Path, camera_angle_x: float, frames: list[dict]):
    payload = {
        "camera_angle_x": float(camera_angle_x),
        "frames": frames,
    }
    path.write_text(json.dumps(payload, indent=2))


def render_split(
    mujoco,
    imageio,
    model,
    data,
    renderer,
    split_name: str,
    dataset_dir: Path,
    positions: list[np.ndarray],
    lookat: np.ndarray,
    target_geom_id: int,
    floor_geom_id: int | None,
    width: int,
    height: int,
):
    image_dir = dataset_dir / "images" / split_name
    mask_dir = dataset_dir / "masks" / split_name
    frames = []
    for index, eye in enumerate(positions):
        set_camera_pose(mujoco, model, "render_cam", eye, lookat, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        mujoco.mj_forward(model, data)

        renderer.disable_segmentation_rendering()
        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera="render_cam")
        rgb = renderer.render()

        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="render_cam")
        segmentation = renderer.render()

        mask = segmentation_to_mask(segmentation, target_geom_id=target_geom_id, floor_geom_id=floor_geom_id)
        rgba = build_rgba(rgb, mask)

        stem = f"{split_name}_{index:03d}"
        save_png(imageio, image_dir / f"{stem}.png", rgba)
        save_png(imageio, mask_dir / f"{stem}.png", mask * 255)

        c2w = camera_to_world_matrix(eye, lookat, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        frames.append(
            {
                "file_path": f"images/{split_name}/{stem}",
                "transform_matrix": c2w.tolist(),
            }
        )

    camera_angle_x = 2.0 * math.atan(math.tan(math.radians(model.cam_fovy[0]) * 0.5) * (float(width) / float(height)))
    write_transforms(dataset_dir / f"transforms_{split_name}.json", camera_angle_x=camera_angle_x, frames=frames)


def maybe_jitter_object_pose(mujoco, model, data, rng: np.random.Generator, enabled: bool):
    if not enabled or model.nq < 7:
        return
    yaw = rng.uniform(-math.pi / 8.0, math.pi / 8.0)
    half = yaw * 0.5
    quat = np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)
    data.qpos[:7] = np.array([0.0, 0.0, data.qpos[2], *quat], dtype=np.float64)


def main():
    args = parse_args()
    mujoco_gl = resolve_mujoco_gl(args.mujoco_gl)
    os.environ.setdefault("MUJOCO_GL", mujoco_gl)

    import imageio.v2 as imageio
    import mujoco

    rng = np.random.default_rng(args.seed)
    dataset_dir = Path(args.output_root).expanduser().resolve() / args.scene_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    xml = build_model_xml(args.object_type, args.fovy_deg, args.width, args.height)
    (dataset_dir / "scene.xml").write_text(xml)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    maybe_jitter_object_pose(mujoco, model, data, rng, args.add_pose_jitter)
    for _ in range(int(args.settle_steps)):
        mujoco.mj_step(model, data)

    try:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    except Exception as exc:
        raise RuntimeError(
            "Failed to create a MuJoCo renderer. In a headless environment, try "
            "`MUJOCO_GL=egl` first, then `MUJOCO_GL=osmesa` if EGL is unavailable."
        ) from exc

    target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    _, default_object_height = object_geom_xml(args.object_type)
    object_height = float(default_object_height if args.object_height is None else args.object_height)
    lookat = np.array([0.0, 0.0, object_height], dtype=np.float32)
    train_positions = orbit_camera_positions(args.train_views, args.camera_radius, args.elevation_deg, lookat)
    test_positions = orbit_camera_positions(args.test_views, args.camera_radius, args.elevation_deg, lookat)
    if args.test_views > 0:
        test_positions = test_positions[1:] + test_positions[:1]

    render_split(
        mujoco=mujoco,
        imageio=imageio,
        model=model,
        data=data,
        renderer=renderer,
        split_name="train",
        dataset_dir=dataset_dir,
        positions=train_positions,
        lookat=lookat,
        target_geom_id=target_geom_id,
        floor_geom_id=floor_geom_id,
        width=args.width,
        height=args.height,
    )
    render_split(
        mujoco=mujoco,
        imageio=imageio,
        model=model,
        data=data,
        renderer=renderer,
        split_name="test",
        dataset_dir=dataset_dir,
        positions=test_positions,
        lookat=lookat,
        target_geom_id=target_geom_id,
        floor_geom_id=floor_geom_id,
        width=args.width,
        height=args.height,
    )

    manifest = {
        "scene_name": args.scene_name,
        "object_type": args.object_type,
        "train_views": args.train_views,
        "test_views": args.test_views,
        "width": args.width,
        "height": args.height,
        "fovy_deg": args.fovy_deg,
        "camera_radius": args.camera_radius,
        "elevation_deg": args.elevation_deg,
        "mujoco_gl": os.environ.get("MUJOCO_GL", mujoco_gl),
    }
    (dataset_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[DONE] MuJoCo synthetic dataset created at: {dataset_dir}")
    print("[INFO] Expected layout: images/train|test, masks/train|test, transforms_train.json, transforms_test.json")


if __name__ == "__main__":
    main()
