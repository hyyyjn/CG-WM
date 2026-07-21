import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "MuJoCo Python package is required. Install it in your runtime environment first."
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Populate a ContactGaussian-WM-style fall_and_rebound dataset with MuJoCo free-fall rollouts."
    )
    parser.add_argument("--dataset_root", required=True, type=str)
    parser.add_argument("--object_name", required=True, type=str)
    parser.add_argument("--split", default="all", choices=("all", "train", "test"))
    parser.add_argument("--camera_count", default=1, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--timestep", default=0.002, type=float)
    parser.add_argument("--gravity", default=-9.81, type=float)
    parser.add_argument("--ground_size", default=4.0, type=float)
    parser.add_argument("--camera_distance", default=6.0, type=float)
    parser.add_argument("--camera_height", default=3.0, type=float)
    parser.add_argument(
        "--camera_target_z",
        default=0.0,
        type=float,
        help="World z the camera aims at. Raise it (e.g. 0.7) so a close camera still frames the fall.",
    )
    parser.add_argument(
        "--camera_target_x",
        default=0.0,
        type=float,
        help="World x the camera aims at. Shift toward the settle point to recompose the frame.",
    )
    parser.add_argument(
        "--camera_target_y",
        default=0.0,
        type=float,
        help="World y the camera aims at (negative = toward the camera).",
    )
    parser.add_argument("--drop_height_train", default=2.0, type=float)
    parser.add_argument("--drop_height_test", default=2.6, type=float)
    parser.add_argument("--xy_range_train", default=0.10, type=float)
    parser.add_argument("--xy_range_test", default=0.25, type=float)
    parser.add_argument("--max_tilt_deg_train", default=12.0, type=float)
    parser.add_argument("--max_tilt_deg_test", default=35.0, type=float)
    parser.add_argument("--planar_speed_train", default=0.5, type=float)
    parser.add_argument("--planar_speed_test", default=0.8, type=float)
    parser.add_argument("--spin_speed_train", default=5.0, type=float)
    parser.add_argument("--spin_speed_test", default=8.0, type=float)
    parser.add_argument("--object_rgba", default="0.85 0.25 0.20 1")
    parser.add_argument("--object_rgb1", default="0.92 0.28 0.22", type=str)
    parser.add_argument("--object_rgb2", default="0.20 0.55 0.92", type=str)
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
    )
    parser.add_argument("--floor_rgba", default="0.92 0.92 0.92 1")
    parser.add_argument("--skybox_rgb", default="255,255,255")
    parser.add_argument("--sphere_solref", default="0.02 0.2", type=str)
    parser.add_argument("--box_solref", default="", type=str)
    parser.add_argument("--cylinder_solref", default="", type=str)
    parser.add_argument("--floor_solref", default="", type=str)
    parser.add_argument("--sphere_friction", default="0.02 0.001 0.0001", type=str)
    parser.add_argument("--box_friction", default="0.35 0.01 0.001", type=str)
    parser.add_argument("--cylinder_friction", default="0.55 0.02 0.001", type=str)
    parser.add_argument("--freejoint_damping", default=0.05, type=float)
    parser.add_argument(
        "--vertical_speed_train",
        default=0.0,
        type=float,
        help="Optional initial downward z velocity magnitude for train episodes.",
    )
    parser.add_argument(
        "--vertical_speed_test",
        default=0.0,
        type=float,
        help="Optional initial downward z velocity magnitude for test episodes.",
    )
    parser.add_argument(
        "--max_floor_penetration",
        default=0.03,
        type=float,
        help="Abort sphere rollouts if the sphere bottom goes this far below z=0.",
    )
    parser.add_argument(
        "--skip_render",
        action="store_true",
        help="Run MuJoCo physics and write simple placeholder RGB/masks without creating an OpenGL renderer.",
    )
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def derive_half_extents(object_manifest):
    normalization = object_manifest.get("normalization", {})
    bbox_min = normalization.get("bbox_min")
    bbox_max = normalization.get("bbox_max")
    scale = float(normalization.get("scale", 1.0))
    if bbox_min is None or bbox_max is None:
        return np.array([0.5, 0.5, 0.5], dtype=np.float32)

    bbox_min = np.asarray(bbox_min, dtype=np.float32)
    bbox_max = np.asarray(bbox_max, dtype=np.float32)
    extents = (bbox_max - bbox_min) * scale
    extents = np.maximum(extents, 1e-3)
    return extents * 0.5


def parse_face_colors(face_colors_arg):
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


def cola_can_visual_xml(radius: float, half_height: float) -> str:
    r = float(radius)
    h = float(half_height)
    stripe = max(r * 0.085, 0.004)
    stripe_offset = r + stripe * 0.5
    return f"""
      <geom name="can_top_cap" type="cylinder" pos="0 0 {h + 0.001}" size="{r * 0.96} 0.002" rgba="0.86 0.86 0.82 1" contype="0" conaffinity="0" density="0"/>
      <geom name="can_bottom_cap" type="cylinder" pos="0 0 {-h - 0.001}" size="{r * 0.96} 0.002" rgba="0.74 0.74 0.70 1" contype="0" conaffinity="0" density="0"/>
      <geom name="can_label_front" type="box" pos="0 {-stripe_offset} 0.000" size="{r * 0.44} {stripe * 0.5} {h * 0.54}" rgba="1 1 1 1" contype="0" conaffinity="0" density="0"/>
      <geom name="can_label_back" type="box" pos="0 {stripe_offset} 0.000" size="{r * 0.28} {stripe * 0.5} {h * 0.42}" rgba="1 1 1 1" contype="0" conaffinity="0" density="0"/>
      <geom name="can_label_left" type="box" pos="{-stripe_offset} 0 0.030" size="{stripe * 0.5} {r * 0.24} {h * 0.30}" rgba="1 1 1 1" contype="0" conaffinity="0" density="0"/>
      <geom name="can_label_right" type="box" pos="{stripe_offset} 0 -0.035" size="{stripe * 0.5} {r * 0.20} {h * 0.26}" rgba="1 1 1 1" contype="0" conaffinity="0" density="0"/>
      <geom name="can_pull_tab" type="box" pos="0 {-r * 0.22} {h + 0.004}" size="{r * 0.22} {r * 0.065} 0.002" rgba="0.55 0.55 0.52 1" contype="0" conaffinity="0" density="0"/>
    """.strip()


def euler_xyz_to_quat(rx, ry, rz):
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def build_mjcf(
    half_extents,
    physics_shape,
    object_rgba,
    floor_rgba,
    gravity,
    timestep,
    ground_size,
    camera_distance,
    camera_height,
    camera_target_z,
    camera_target_x,
    camera_target_y,
    sphere_solref,
    box_solref,
    cylinder_solref,
    floor_solref,
    sphere_friction,
    box_friction,
    cylinder_friction,
    freejoint_damping,
    object_rgb1,
    object_rgb2,
    box_face_colors,
    visual_model,
):
    hx, hy, hz = [float(v) for v in half_extents]
    # General look-at from camera pos (0, -camera_distance, camera_height) to the
    # target point. MuJoCo cameras look along -z_cam with xyaxes = right, up.
    _px, _py, _pz = 0.0, -float(camera_distance), float(camera_height)
    _fx = float(camera_target_x) - _px
    _fy = float(camera_target_y) - _py
    _fz = float(camera_target_z) - _pz
    _fn = max((_fx * _fx + _fy * _fy + _fz * _fz) ** 0.5, 1e-9)
    _fx, _fy, _fz = _fx / _fn, _fy / _fn, _fz / _fn
    # right = forward x world_up; degenerate only if looking straight down.
    _rx, _ry = _fy, -_fx
    _rn = max((_rx * _rx + _ry * _ry) ** 0.5, 1e-9)
    _rx, _ry = _rx / _rn, _ry / _rn
    # up = right x forward
    _ux = _ry * _fz
    _uy = -_rx * _fz
    _uz = _rx * _fy - _ry * _fx
    camera_xyaxes = f"{_rx:.6f} {_ry:.6f} 0 {_ux:.6f} {_uy:.6f} {_uz:.6f}"
    box_solref_attr = f' solref="{box_solref}"' if str(box_solref).strip() else ""
    cylinder_solref_attr = f' solref="{cylinder_solref}"' if str(cylinder_solref).strip() else ""
    floor_solref_attr = f' solref="{floor_solref}"' if str(floor_solref).strip() else ""
    if physics_shape == "sphere":
        radius = max(hx, hy, hz)
        geom_name = "sphere_geom"
        body_name = "sphere"
        geom_xml = (
            f'<geom name="{geom_name}" type="sphere" size="{radius}" rgba="{object_rgba}" '
            f'density="1000" friction="{sphere_friction}" solref="{sphere_solref}"/>'
        )
    elif physics_shape == "cylinder":
        radius = max(hx, hy)
        half_height = hz
        geom_name = "cylinder_geom"
        body_name = "cylinder"
        body_rgba = "0.82 0.02 0.02 1" if visual_model == "cola_can" else object_rgba
        extras = cola_can_visual_xml(radius, half_height) if visual_model == "cola_can" else ""
        geom_xml = (
            f'<geom name="{geom_name}" type="cylinder" size="{radius} {half_height}" rgba="{body_rgba}" '
            f'density="1000" friction="{cylinder_friction}"{cylinder_solref_attr}/>\n'
            f"      {extras}"
        ).rstrip()
    else:
        geom_name = "box_geom"
        body_name = "box"
        face_colors = parse_face_colors(box_face_colors)
        face_pad = max(min(hx, hy, hz) * 0.02, 0.0015)
        face_geoms = [
            ("face_px", f"{hx + face_pad * 0.5} 0 0", f"{face_pad * 0.5} {hy} {hz}", face_colors[0]),
            ("face_nx", f"{-hx - face_pad * 0.5} 0 0", f"{face_pad * 0.5} {hy} {hz}", face_colors[1]),
            ("face_py", f"0 {hy + face_pad * 0.5} 0", f"{hx} {face_pad * 0.5} {hz}", face_colors[2]),
            ("face_ny", f"0 {-hy - face_pad * 0.5} 0", f"{hx} {face_pad * 0.5} {hz}", face_colors[3]),
            ("face_pz", f"0 0 {hz + face_pad * 0.5}", f"{hx} {hy} {face_pad * 0.5}", face_colors[4]),
            ("face_nz", f"0 0 {-hz - face_pad * 0.5}", f"{hx} {hy} {face_pad * 0.5}", face_colors[5]),
        ]
        face_geom_xml = "\n      ".join(
            (
                f'<geom name="{name}" type="box" pos="{pos}" size="{size}" '
                f'rgba="{color[0]} {color[1]} {color[2]} 1" '
                'contype="0" conaffinity="0" density="0"/>'
            )
            for name, pos, size, color in face_geoms
        )
        geom_xml = (
            f'<geom name="{geom_name}" type="box" size="{hx} {hy} {hz}" rgba="0.14 0.14 0.16 1" '
            f'density="1000" friction="{box_friction}"{box_solref_attr}/>\n'
            f"      {face_geom_xml}"
        )
    return f"""
<mujoco model="contactwm_cube_fall">
  <option timestep="{timestep}" gravity="0 0 {gravity}" integrator="Euler"/>
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <rgba haze="1 1 1 1"/>
  </visual>
  <asset>
    <texture name="texplane" type="2d" builtin="checker" rgb1="0.96 0.96 0.96" rgb2="0.88 0.88 0.88" width="256" height="256"/>
    <material name="matplane" texture="texplane" texrepeat="2 2" reflectance="0.05"/>
    <texture name="texobject" type="cube" builtin="checker" rgb1="{object_rgb1}" rgb2="{object_rgb2}" width="256" height="256"/>
    <material name="matobject" texture="texobject" texuniform="true" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="{ground_size} {ground_size} 0.1" material="matplane" rgba="{floor_rgba}"{floor_solref_attr}/>
    <camera name="cam0" pos="0 -{camera_distance} {camera_height}" xyaxes="{camera_xyaxes}"/>
    <body name="{body_name}" pos="0 0 2">
      <joint name="root_free" type="free" damping="{freejoint_damping}"/>
      {geom_xml}
    </body>
  </worldbody>
</mujoco>
""".strip()


def save_rgb(path: Path, rgb):
    Image.fromarray(rgb).save(path)


def save_mask(path: Path, segmentation, object_geom_ids):
    mask = np.isin(segmentation[..., 0], object_geom_ids).astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(path)


def save_placeholder_frame(path: Path, mask_path: Path, position, image_size):
    width, height = [int(v) for v in image_size]
    rgb = Image.new("RGB", (width, height), (246, 247, 248))
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(rgb)
    mask_draw = ImageDraw.Draw(mask)
    floor_y = int(height * 0.82)
    draw.rectangle((0, floor_y, width, height), fill=(224, 226, 226))
    draw.line((0, floor_y, width, floor_y), fill=(80, 82, 82), width=max(2, width // 256))

    z = float(position[2])
    x = float(position[0])
    cube_size = max(12, int(min(width, height) * 0.12))
    x_px = int(width * 0.5 + x * width * 0.7)
    y_px = int(floor_y - max(z, 0.0) * height * 0.55)
    half = cube_size // 2
    depth = max(8, cube_size // 3)
    front = [
        (x_px - half, y_px - half),
        (x_px + half, y_px - half),
        (x_px + half, y_px + half),
        (x_px - half, y_px + half),
    ]
    offset = (depth, -depth)
    back = [(px + offset[0], py + offset[1]) for px, py in front]
    top = [front[0], front[1], back[1], back[0]]
    side = [front[1], front[2], back[2], back[1]]
    outline = (35, 35, 35)
    draw.polygon(top, fill=(242, 132, 78), outline=outline)
    draw.polygon(side, fill=(172, 52, 44), outline=outline)
    draw.polygon(front, fill=(220, 68, 48), outline=outline)
    draw.line(front + [front[0]], fill=outline, width=max(2, width // 256))
    draw.line(back + [back[0]], fill=outline, width=max(1, width // 384))
    draw.line([front[0], back[0], back[1], front[1]], fill=outline, width=max(1, width // 384))
    mask_draw.polygon(front + back, fill=255)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(path)
    mask.save(mask_path)


def get_episode_roots(dataset_root: Path, object_name: str, split: str):
    scenario_root = dataset_root / "stage2" / "fall_and_rebound"
    splits = ["train", "test"] if split == "all" else [split]
    episode_roots = []
    for split_name in splits:
        object_root = scenario_root / split_name / object_name
        if not object_root.exists():
            continue
        episode_roots.extend(sorted(path for path in object_root.iterdir() if path.is_dir()))
    return episode_roots


def rollout_episode(
    model,
    data,
    renderer,
    camera_name,
    fps,
    skybox_rgb,
    episode_root: Path,
    initial_pos,
    initial_quat,
    initial_linvel,
    initial_angvel,
    object_geom_ids,
):
    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    manifest = read_json(episode_root / "episode_manifest.json")
    frames_per_episode = int(manifest["frames_per_episode"])

    data.qpos[:3] = initial_pos
    data.qpos[3:7] = initial_quat
    data.qvel[:] = 0.0
    data.qvel[:3] = initial_linvel
    data.qvel[3:6] = initial_angvel
    mujoco.mj_forward(model, data)

    rgb_dir = episode_root / "rgb"
    mask_dir = episode_root / "masks"
    state_dir = episode_root / "state"
    action_dir = episode_root / "actions"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    action_dir.mkdir(parents=True, exist_ok=True)

    states = []
    actions = []
    bg_rgb = np.asarray([int(v) for v in skybox_rgb.split(",")], dtype=np.uint8)

    for frame_idx in range(frames_per_episode):
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

        stem = f"{frame_idx:06d}"
        if renderer is None:
            save_placeholder_frame(
                rgb_dir / f"{stem}.png",
                mask_dir / f"{stem}.png",
                data.qpos[:3],
                manifest.get("image_size", [640, 480]),
            )
        else:
            renderer.update_scene(data, camera=camera_name)
            rgb = renderer.render()
            rgb = np.asarray(rgb, dtype=np.uint8)
            rgb = np.ascontiguousarray(rgb[..., :3])

            # fill alpha-less background consistently for threshold masks
            if rgb.ndim == 3 and rgb.shape[2] == 3:
                rgb[(rgb.sum(axis=-1) == 0)] = bg_rgb

            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera=camera_name)
            segmentation = np.asarray(renderer.render(), dtype=np.int32)
            renderer.disable_segmentation_rendering()

            save_rgb(rgb_dir / f"{stem}.png", rgb)
            save_mask(mask_dir / f"{stem}.png", segmentation, object_geom_ids)

        states.append(
            {
                "frame_index": frame_idx,
                "time": float(data.time),
                "qpos": data.qpos.tolist(),
                "qvel": data.qvel.tolist(),
                "position": data.qpos[:3].tolist(),
                "quaternion_wxyz": data.qpos[3:7].tolist(),
                "linear_velocity": data.qvel[:3].tolist(),
                "angular_velocity": data.qvel[3:6].tolist(),
            }
        )
        actions.append(
            {
                "frame_index": frame_idx,
                "action_type": "none",
                "control": [],
            }
        )

    write_json(state_dir / "trajectory.json", {"states": states})
    write_json(action_dir / "trajectory.json", {"actions": actions})

    manifest["generator"] = {
        "name": "generate_mujoco_fall_dataset.py",
        "camera_name": camera_name,
    }
    manifest["initial_state"] = {
        "position": initial_pos.tolist(),
        "quaternion_wxyz": initial_quat.tolist(),
        "linear_velocity": initial_linvel.tolist(),
        "angular_velocity": initial_angvel.tolist(),
    }
    write_json(episode_root / "episode_manifest.json", manifest)
    return states


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    object_manifest_path = dataset_root / "objects" / args.object_name / "object_manifest.json"
    if not object_manifest_path.exists():
        raise FileNotFoundError(f"Missing object manifest: {object_manifest_path}")

    object_manifest = read_json(object_manifest_path)
    half_extents = derive_half_extents(object_manifest)
    physics_shape = str(object_manifest.get("physics_shape", "box"))
    visual_model = str(object_manifest.get("visual_model", object_manifest.get("object_type", physics_shape)))

    mjcf = build_mjcf(
        half_extents=half_extents,
        physics_shape=physics_shape,
        object_rgba=args.object_rgba,
        floor_rgba=args.floor_rgba,
        gravity=args.gravity,
        timestep=args.timestep,
        ground_size=args.ground_size,
        camera_distance=args.camera_distance,
        camera_height=args.camera_height,
        camera_target_z=args.camera_target_z,
        camera_target_x=args.camera_target_x,
        camera_target_y=args.camera_target_y,
        sphere_solref=args.sphere_solref,
        box_solref=args.box_solref,
        cylinder_solref=args.cylinder_solref,
        floor_solref=args.floor_solref,
        sphere_friction=args.sphere_friction,
        box_friction=args.box_friction,
        cylinder_friction=args.cylinder_friction,
        freejoint_damping=args.freejoint_damping,
        object_rgb1=args.object_rgb1,
        object_rgb2=args.object_rgb2,
        box_face_colors=args.box_face_colors,
        visual_model=visual_model,
    )
    model = mujoco.MjModel.from_xml_string(mjcf)
    object_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) != "floor"
    ]

    scenario_manifest = read_json(dataset_root / "stage2" / "fall_and_rebound" / "scenario_manifest.json")
    width, height = scenario_manifest["image_size"]
    renderer = None if bool(args.skip_render) else mujoco.Renderer(model, height=int(height), width=int(width))
    data = mujoco.MjData(model)

    rng = np.random.default_rng(args.seed)
    episode_roots = get_episode_roots(dataset_root, args.object_name, args.split)
    if not episode_roots:
        raise FileNotFoundError("No episode directories found for the requested object/split.")

    for episode_root in episode_roots:
        split_name = episode_root.parent.parent.name
        if split_name == "train":
            xy_range = float(args.xy_range_train)
            drop_height = float(args.drop_height_train)
            max_tilt_deg = float(args.max_tilt_deg_train)
            planar_speed = float(args.planar_speed_train)
            spin_speed = float(args.spin_speed_train)
            vertical_speed = float(args.vertical_speed_train)
        else:
            xy_range = float(args.xy_range_test)
            drop_height = float(args.drop_height_test)
            max_tilt_deg = float(args.max_tilt_deg_test)
            planar_speed = float(args.planar_speed_test)
            spin_speed = float(args.spin_speed_test)
            vertical_speed = float(args.vertical_speed_test)

        init_xy = rng.uniform(-xy_range, xy_range, size=2)
        init_pos = np.array([init_xy[0], init_xy[1], drop_height], dtype=np.float64)
        tilt = np.deg2rad(rng.uniform(-max_tilt_deg, max_tilt_deg, size=3))
        init_quat = euler_xyz_to_quat(float(tilt[0]), float(tilt[1]), float(tilt[2]))
        heading = float(rng.uniform(0.0, 2.0 * math.pi))
        planar_mag = float(rng.uniform(0.4 * planar_speed, planar_speed))
        init_linvel = np.array(
            [
                planar_mag * math.cos(heading),
                planar_mag * math.sin(heading),
                -float(rng.uniform(0.65 * vertical_speed, vertical_speed)) if vertical_speed > 0.0 else 0.0,
            ],
            dtype=np.float64,
        )
        init_angvel = rng.uniform(-spin_speed, spin_speed, size=3).astype(np.float64)

        states = rollout_episode(
            model,
            data,
            renderer,
            camera_name="cam0",
            fps=args.fps,
            skybox_rgb=args.skybox_rgb,
            episode_root=episode_root,
            initial_pos=init_pos,
            initial_quat=init_quat,
            initial_linvel=init_linvel,
            initial_angvel=init_angvel,
            object_geom_ids=object_geom_ids,
        )
        if physics_shape == "sphere":
            radius = float(max(half_extents))
            min_bottom_z = min(float(state["position"][2]) - radius for state in states)
            if min_bottom_z < -float(args.max_floor_penetration):
                raise RuntimeError(
                    f"Sphere rollout penetrated the floor by {-min_bottom_z:.4f} m "
                    f"in {episode_root}. Check --sphere_solref/--timestep."
                )

    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "object_name": args.object_name,
                "episodes_written": len(episode_roots),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
