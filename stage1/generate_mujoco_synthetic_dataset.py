from __future__ import annotations

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
        choices=["box", "sphere", "cylinder", "dice", "cola_can"],
        help="Simple object primitive to render.",
    )
    parser.add_argument("--train_views", default=24, type=int, help="Number of train views on the orbit.")
    parser.add_argument("--test_views", default=8, type=int, help="Number of test views on the orbit.")
    parser.add_argument("--width", default=512, type=int, help="Rendered image width.")
    parser.add_argument("--height", default=512, type=int, help="Rendered image height.")
    parser.add_argument("--fovy_deg", default=45.0, type=float, help="Vertical field of view in degrees.")
    parser.add_argument("--camera_radius", default=1.45, type=float, help="Distance from camera to object center.")
    parser.add_argument("--elevation_deg", default=25.0, type=float, help="Orbit elevation angle in degrees.")
    parser.add_argument("--camera_sampling", default="orbit", choices=("orbit", "hemisphere"))
    parser.add_argument("--min_elevation_deg", default=12.0, type=float)
    parser.add_argument("--max_elevation_deg", default=70.0, type=float)
    parser.add_argument(
        "--appearance_preset", default="legacy", choices=("legacy", "realistic_contact"),
        help="Use the same floor, sky, and lighting family as the realistic Stage2 contact scene.",
    )
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
    parser.add_argument(
        "--alpha_subject",
        default="object",
        choices=["object", "floor", "scene"],
        help=(
            "Content represented by the PNG alpha channel. 'object' keeps the legacy "
            "can-only behavior; 'floor' isolates the support surface; 'scene' includes both for a joint "
            "can/floor Gaussian reconstruction. Per-instance label maps are always written."
        ),
    )
    parser.add_argument(
        "--box_face_colors",
        default=(
            "0.96 0.28 0.22;"
            "0.22 0.72 0.32;"
            "0.22 0.52 0.96;"
            "0.98 0.82 0.20;"
            "0.68 0.34 0.92;"
            "0.20 0.84 0.88"
        ),
        type=str,
        help="Six RGB triplets for box face colors, separated by ';'.",
    )
    return parser.parse_args()


def resolve_mujoco_gl(mode: str) -> str:
    if mode == "auto":
        return os.environ.get("MUJOCO_GL", "egl")
    return mode


def parse_face_colors(face_colors_arg: str) -> list[list[float]]:
    colors = []
    for raw_color in str(face_colors_arg).split(";"):
        raw_color = raw_color.strip()
        if not raw_color:
            continue
        values = [float(value) for value in raw_color.split()]
        if len(values) != 3:
            raise ValueError(
                f"--box_face_colors expects RGB triplets separated by ';', got: {face_colors_arg}"
            )
        colors.append(values)
    if len(colors) < 6:
        raise ValueError("--box_face_colors must provide at least 6 RGB triplets.")
    return colors[:6]


def colored_box_geom_xml(half_extent: float, face_colors_arg: str) -> str:
    h = float(half_extent)
    face_colors = parse_face_colors(face_colors_arg)
    face_pad = max(h * 0.02, 0.0015)
    face_geoms = [
        ("target_face_px", f"{h + face_pad * 0.5} 0 0", f"{face_pad * 0.5} {h} {h}", face_colors[0]),
        ("target_face_nx", f"{-h - face_pad * 0.5} 0 0", f"{face_pad * 0.5} {h} {h}", face_colors[1]),
        ("target_face_py", f"0 {h + face_pad * 0.5} 0", f"{h} {face_pad * 0.5} {h}", face_colors[2]),
        ("target_face_ny", f"0 {-h - face_pad * 0.5} 0", f"{h} {face_pad * 0.5} {h}", face_colors[3]),
        ("target_face_pz", f"0 0 {h + face_pad * 0.5}", f"{h} {h} {face_pad * 0.5}", face_colors[4]),
        ("target_face_nz", f"0 0 {-h - face_pad * 0.5}", f"{h} {h} {face_pad * 0.5}", face_colors[5]),
    ]
    face_geom_xml = "\n          ".join(
        (
            f'<geom name="{name}" type="box" pos="{pos}" size="{size}" '
            f'rgba="{color[0]} {color[1]} {color[2]} 1" '
            'contype="0" conaffinity="0" density="0"/>'
        )
        for name, pos, size, color in face_geoms
    )
    return f"""
          <geom name="target_geom" type="box" size="{h} {h} {h}" rgba="0.14 0.14 0.16 1"/>
          {face_geom_xml}
    """.strip()


def dice_pip_positions(number_name: str) -> list[tuple[float, float]]:
    layouts = {
        "one": [(0.0, 0.0)],
        "two": [(-0.022, -0.022), (0.022, 0.022)],
        "three": [(-0.024, -0.024), (0.0, 0.0), (0.024, 0.024)],
        "four": [(-0.023, -0.023), (0.023, -0.023), (-0.023, 0.023), (0.023, 0.023)],
        "five": [(-0.024, -0.024), (0.024, -0.024), (0.0, 0.0), (-0.024, 0.024), (0.024, 0.024)],
        "six": [(-0.024, -0.028), (0.024, -0.028), (-0.024, 0.0), (0.024, 0.0), (-0.024, 0.028), (0.024, 0.028)],
    }
    return layouts[number_name]


def dice_geom_xml(half_extent: float) -> str:
    h = float(half_extent)
    face_specs = [
        ("px", "three", (h + 0.001, 0, 0), "y", "z"),
        ("nx", "four", (-h - 0.001, 0, 0), "y", "z"),
        ("py", "two", (0, h + 0.001, 0), "x", "z"),
        ("ny", "five", (0, -h - 0.001, 0), "x", "z"),
        ("pz", "six", (0, 0, h + 0.001), "x", "y"),
        ("nz", "one", (0, 0, -h - 0.001), "x", "y"),
    ]
    axis_index = {"x": 0, "y": 1, "z": 2}
    pips = []
    for face_name, number_name, base, axis_u, axis_v in face_specs:
        for pip_idx, (u, v) in enumerate(dice_pip_positions(number_name)):
            pos = [float(base[0]), float(base[1]), float(base[2])]
            pos[axis_index[axis_u]] += u
            pos[axis_index[axis_v]] += v
            pips.append(
                f'<geom name="target_pip_{face_name}_{pip_idx:02d}" '
                f'type="sphere" pos="{pos[0]} {pos[1]} {pos[2]}" size="0.0085" '
                'rgba="0.02 0.02 0.018 1" contype="0" conaffinity="0" density="0"/>'
            )
    pip_xml = "\n          ".join(pips)
    return f"""
          <geom name="target_geom" type="box" size="{h} {h} {h}" rgba="0.93 0.92 0.86 1"/>
          {pip_xml}
    """.strip()


def cola_can_geom_xml(radius: float = 0.07, half_height: float = 0.10) -> str:
    r = float(radius)
    h = float(half_height)
    stripe = 0.006
    stripe_offset = r + stripe * 0.5
    top_z = h + 0.001
    bottom_z = -h - 0.001
    return f"""
          <geom name="target_geom" type="cylinder" size="{r} {h}" rgba="0.82 0.02 0.02 1"/>
          <geom name="target_top_cap" type="cylinder" pos="0 0 {top_z}" size="{r * 0.96} 0.002" rgba="0.86 0.86 0.82 1" contype="0" conaffinity="0" density="0"/>
          <geom name="target_bottom_cap" type="cylinder" pos="0 0 {bottom_z}" size="{r * 0.96} 0.002" rgba="0.74 0.74 0.70 1" contype="0" conaffinity="0" density="0"/>
          <geom name="target_label_front" type="box" pos="0 {-stripe_offset} 0.000" size="{r * 0.44} {stripe * 0.5} {h * 0.54}" rgba="1 1 1 1" contype="0" conaffinity="0" density="0"/>
          <geom name="target_label_back" type="box" pos="0 {stripe_offset} 0.000" size="{r * 0.28} {stripe * 0.5} {h * 0.42}" rgba="1 1 1 1" contype="0" conaffinity="0" density="0"/>
          <geom name="target_label_left" type="box" pos="{-stripe_offset} 0 0.030" size="{stripe * 0.5} {r * 0.24} {h * 0.30}" rgba="1 1 1 1" contype="0" conaffinity="0" density="0"/>
          <geom name="target_label_right" type="box" pos="{stripe_offset} 0 -0.035" size="{stripe * 0.5} {r * 0.20} {h * 0.26}" rgba="1 1 1 1" contype="0" conaffinity="0" density="0"/>
          <geom name="target_pull_tab" type="box" pos="0 {-r * 0.22} {h + 0.004}" size="{r * 0.22} {r * 0.065} 0.002" rgba="0.55 0.55 0.52 1" contype="0" conaffinity="0" density="0"/>
    """.strip()


def object_geom_xml(object_type: str, box_face_colors: str) -> tuple[str, float]:
    if object_type == "box":
        geom_xml = colored_box_geom_xml(0.08, box_face_colors)
        return (
            f"""
        <body name="target_body" pos="0 0 0.08">
          <freejoint/>
          {geom_xml}
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
    if object_type == "dice":
        geom_xml = dice_geom_xml(0.08)
        return (
            f"""
        <body name="target_body" pos="0 0 0.08">
          <freejoint/>
          {geom_xml}
        </body>
            """.strip(),
            0.08,
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
    if object_type == "cola_can":
        geom_xml = cola_can_geom_xml(0.07, 0.10)
        return (
            f"""
        <body name="target_body" pos="0 0 0.10">
          <freejoint/>
          {geom_xml}
        </body>
            """.strip(),
            0.10,
        )
    raise ValueError(f"Unsupported object_type: {object_type}")


def build_model_xml(object_type: str, fovy_deg: float, width: int, height: int,
                    box_face_colors: str, appearance_preset: str = "legacy") -> str:
    object_xml, _ = object_geom_xml(object_type, box_face_colors)
    if appearance_preset == "realistic_contact":
        visual_xml = """
    <headlight ambient="0.22 0.22 0.22" diffuse="0.65 0.65 0.65" specular="0.18 0.18 0.18"/>
    <rgba haze="0.82 0.86 0.92 1"/>"""
        asset_xml = """
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.68 0.77 0.88" rgb2="0.95 0.96 0.98" width="512" height="3072"/>
    <texture name="checker" type="2d" builtin="checker" rgb1="0.60 0.62 0.64" rgb2="0.56 0.58 0.60" width="512" height="512"/>
    <material name="floor_mat" texture="checker" texrepeat="16 16" reflectance="0.03" shininess="0.08"/>"""
        light_xml = """
    <light name="key" pos="-2 -3 5" dir="0.25 0.35 -1" directional="true" diffuse="0.85 0.82 0.78" castshadow="true"/>
    <light name="fill" pos="3 -1 2.5" dir="-1 0.2 -0.5" directional="true" diffuse="0.25 0.28 0.35" specular="0.1 0.1 0.1"/>"""
        floor_rgba = "0.92 0.92 0.92 1"
    else:
        visual_xml = """
    <headlight ambient="0.35 0.35 0.35" diffuse="0.65 0.65 0.65" specular="0.15 0.15 0.15"/>
    <rgba haze="0.95 0.95 0.95 1"/>"""
        asset_xml = """
    <texture name="checker" type="2d" builtin="checker" rgb1="0.92 0.92 0.92" rgb2="0.84 0.84 0.84" width="256" height="256"/>
    <material name="floor_mat" texture="checker" texrepeat="3 3" reflectance="0.05"/>"""
        light_xml = """
    <light name="key" pos="1.5 -1.5 2.8" dir="-0.4 0.3 -1"/>
    <light name="fill" pos="-1.2 1.0 2.1" dir="0 -0.1 -1" diffuse="0.5 0.5 0.5"/>"""
        floor_rgba = "0.95 0.95 0.95 1"
    return f"""
<mujoco model="cgwm_simple_scene">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="{int(width)}" offheight="{int(height)}"/>
    {visual_xml}
  </visual>
  <asset>
    {asset_xml}
  </asset>
  <worldbody>
    {light_xml}
    <geom name="floor" type="plane" size="5 5 0.1" material="floor_mat" rgba="{floor_rgba}"/>
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


def hemisphere_camera_positions(count: int, radius: float, min_elevation_deg: float,
                                max_elevation_deg: float, lookat: np.ndarray) -> list[np.ndarray]:
    """Deterministic golden-angle sampling over an upper spherical band."""
    if count < 1:
        return []
    min_sin = math.sin(math.radians(float(min_elevation_deg)))
    max_sin = math.sin(math.radians(float(max_elevation_deg)))
    if not 0.0 <= min_sin < max_sin <= 1.0:
        raise ValueError("hemisphere elevations must satisfy 0 <= min < max <= 90 degrees")
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    positions = []
    for index in range(count):
        fraction = (index + 0.5) / float(count)
        sin_elevation = min_sin + fraction * (max_sin - min_sin)
        elevation = math.asin(sin_elevation)
        theta = index * golden_angle
        xy_radius = radius * math.cos(elevation)
        positions.append(lookat + np.array(
            [xy_radius * math.cos(theta), xy_radius * math.sin(theta), radius * math.sin(elevation)],
            dtype=np.float32,
        ))
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


def segmentation_to_mask(segmentation: np.ndarray, target_geom_ids: list[int], floor_geom_id: int | None = None) -> np.ndarray:
    target_geom_ids = [int(geom_id) for geom_id in target_geom_ids if int(geom_id) >= 0]
    if segmentation.ndim == 2:
        mask = np.isin(segmentation, target_geom_ids)
    elif segmentation.ndim == 3:
        mask = np.isin(segmentation[..., 0], target_geom_ids)
        if not mask.any() and floor_geom_id is not None:
            mask = segmentation[..., 0] != floor_geom_id
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
    target_geom_ids: list[int],
    floor_geom_id: int | None,
    width: int,
    height: int,
    alpha_subject: str,
):
    image_dir = dataset_dir / "images" / split_name
    mask_dir = dataset_dir / "masks" / split_name
    instance_mask_dir = dataset_dir / "instance_masks" / split_name
    object_mask_dir = dataset_dir / "object_masks" / split_name
    floor_mask_dir = dataset_dir / "floor_masks" / split_name
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

        mask = segmentation_to_mask(segmentation, target_geom_ids=target_geom_ids, floor_geom_id=floor_geom_id)
        geom_labels = segmentation if segmentation.ndim == 2 else segmentation[..., 0]
        floor_mask = geom_labels == int(floor_geom_id) if floor_geom_id is not None else np.zeros_like(mask, dtype=bool)
        # Stable IDs consumed by auto_assign_object_ids.py: 0=background, 1=can, 2=floor.
        instance_mask = np.zeros(mask.shape, dtype=np.uint8)
        instance_mask[floor_mask] = 2
        instance_mask[mask.astype(bool)] = 1
        if alpha_subject == "scene":
            alpha_mask = (instance_mask > 0).astype(np.uint8)
        elif alpha_subject == "floor":
            alpha_mask = floor_mask.astype(np.uint8)
        else:
            alpha_mask = mask
        rgba = build_rgba(rgb, alpha_mask)

        stem = f"{split_name}_{index:03d}"
        save_png(imageio, image_dir / f"{stem}.png", rgba)
        # ``masks`` is the binary prior for the selected Stage-1 subject.
        save_png(imageio, mask_dir / f"{stem}.png", alpha_mask * 255)
        save_png(imageio, instance_mask_dir / f"{stem}.png", instance_mask)
        save_png(imageio, object_mask_dir / f"{stem}.png", mask * 255)
        save_png(imageio, floor_mask_dir / f"{stem}.png", floor_mask.astype(np.uint8) * 255)

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

    xml = build_model_xml(
        args.object_type, args.fovy_deg, args.width, args.height,
        args.box_face_colors, args.appearance_preset,
    )
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

    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    target_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) != "floor"
    ]

    _, default_object_height = object_geom_xml(args.object_type, args.box_face_colors)
    object_height = float(default_object_height if args.object_height is None else args.object_height)
    lookat = np.array([0.0, 0.0, object_height], dtype=np.float32)
    if args.camera_sampling == "hemisphere":
        position_builder = lambda count: hemisphere_camera_positions(
            count, args.camera_radius, args.min_elevation_deg, args.max_elevation_deg, lookat
        )
    else:
        position_builder = lambda count: orbit_camera_positions(
            count, args.camera_radius, args.elevation_deg, lookat
        )
    train_positions = position_builder(args.train_views)
    test_positions = position_builder(args.test_views)
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
        target_geom_ids=target_geom_ids,
        floor_geom_id=floor_geom_id,
        width=args.width,
        height=args.height,
        alpha_subject=args.alpha_subject,
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
        target_geom_ids=target_geom_ids,
        floor_geom_id=floor_geom_id,
        width=args.width,
        height=args.height,
        alpha_subject=args.alpha_subject,
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
        "camera_sampling": args.camera_sampling,
        "min_elevation_deg": args.min_elevation_deg,
        "max_elevation_deg": args.max_elevation_deg,
        "appearance_preset": args.appearance_preset,
        "mujoco_gl": os.environ.get("MUJOCO_GL", mujoco_gl),
        "alpha_subject": args.alpha_subject,
        "instance_labels": {"0": "background", "1": "object", "2": "floor"},
    }
    (dataset_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[DONE] MuJoCo synthetic dataset created at: {dataset_dir}")
    print("[INFO] Expected layout: images/, masks/ (selected alpha subject), object_masks/, floor_masks/, instance_masks/, transforms_*.json")


if __name__ == "__main__":
    main()
