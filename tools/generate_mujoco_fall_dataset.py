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
    parser.add_argument(
        "--scene_xml", default="", type=str,
        help="Stage1 dataset scene.xml to reuse (lights, floor, sky and object visuals), so "
             "Stage1 multiview and Stage2 rollouts are rendered from one scene.",
    )
    parser.add_argument("--split", default="all", choices=("all", "train", "test"))
    parser.add_argument("--camera_count", default=1, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--timestep", default=0.002, type=float)
    parser.add_argument("--gravity", default=-9.81, type=float)
    parser.add_argument("--ground_size", default=4.0, type=float)
    parser.add_argument("--camera_distance", default=6.0, type=float)
    parser.add_argument("--camera_height", default=3.0, type=float)
    parser.add_argument("--camera_fovy", default=45.0, type=float)
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
    parser.add_argument("--floor_rgb1", default="0.62 0.64 0.66", type=str)
    parser.add_argument("--floor_rgb2", default="0.55 0.57 0.59", type=str)
    parser.add_argument("--floor_texrepeat", default=12.0, type=float)
    parser.add_argument("--sky_rgb1", default="0.72 0.80 0.90", type=str)
    parser.add_argument("--sky_rgb2", default="0.96 0.97 0.98", type=str)
    parser.add_argument("--skybox_rgb", default="255,255,255")
    parser.add_argument("--sphere_solref", default="0.02 0.2", type=str)
    parser.add_argument("--box_solref", default="", type=str)
    parser.add_argument("--cylinder_solref", default="", type=str)
    parser.add_argument("--floor_solref", default="", type=str)
    parser.add_argument("--floor_friction", default="", type=str,
                        help="floor geom friction; MuJoCo uses the element-wise max of the two geoms")
    parser.add_argument("--sphere_friction", default="0.02 0.001 0.0001", type=str)
    parser.add_argument("--box_friction", default="0.35 0.01 0.001", type=str)
    parser.add_argument("--cylinder_friction", default="0.55 0.02 0.001", type=str)
    parser.add_argument("--freejoint_damping", default=0.05, type=float)
    parser.add_argument("--initial_position", default="", type=str, help="Optional fixed x,y,z for every episode.")
    parser.add_argument("--initial_euler_deg", default="", type=str, help="Optional fixed XYZ Euler angles in degrees.")
    parser.add_argument(
        "--initial_quaternion_wxyz", default="", type=str,
        help="Optional fixed w,x,y,z quaternion; takes precedence over --initial_euler_deg.",
    )
    parser.add_argument("--initial_linear_velocity", default="", type=str, help="Optional fixed vx,vy,vz.")
    parser.add_argument("--initial_angular_velocity", default="", type=str, help="Optional fixed wx,wy,wz.")
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


def parse_optional_vec3(value, name):
    if not str(value).strip():
        return None
    values = [float(item) for item in str(value).replace(",", " ").split()]
    if len(values) != 3:
        raise ValueError(f"{name} expects exactly three comma- or space-separated values")
    return np.asarray(values, dtype=np.float64)


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
    object_mass,
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
    camera_fovy,
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
    floor_rgb1,
    floor_rgb2,
    floor_texrepeat,
    sky_rgb1,
    sky_rgb2,
):
    hx, hy, hz = [float(v) for v in half_extents]
    object_mass = float(object_mass)
    if object_mass <= 0.0:
        raise ValueError("object_mass must be positive")
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
            f'mass="{object_mass}" friction="{sphere_friction}" solref="{sphere_solref}"/>'
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
            f'mass="{object_mass}" friction="{cylinder_friction}"{cylinder_solref_attr}/>\n'
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
            f'mass="{object_mass}" friction="{box_friction}"{box_solref_attr}/>\n'
            f"      {face_geom_xml}"
        )
    return f"""
<mujoco model="contactwm_cube_fall">
  <option timestep="{timestep}" gravity="0 0 {gravity}" integrator="Euler"/>
  <visual>
    <headlight diffuse="0.65 0.65 0.65" ambient="0.22 0.22 0.22" specular="0.18 0.18 0.18"/>
    <rgba haze="0.82 0.86 0.92 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="{sky_rgb1}" rgb2="{sky_rgb2}" width="512" height="3072"/>
    <texture name="texplane" type="2d" builtin="checker" rgb1="{floor_rgb1}" rgb2="{floor_rgb2}" width="512" height="512"/>
    <material name="matplane" texture="texplane" texrepeat="{floor_texrepeat} {floor_texrepeat}" reflectance="0.03" shininess="0.08"/>
    <texture name="texobject" type="cube" builtin="checker" rgb1="{object_rgb1}" rgb2="{object_rgb2}" width="256" height="256"/>
    <material name="matobject" texture="texobject" texuniform="true" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="-2 -3 5" dir="0.25 0.35 -1" directional="true" diffuse="0.85 0.82 0.78" castshadow="true"/>
    <light pos="3 -1 2.5" dir="-1 0.2 -0.5" directional="true" diffuse="0.25 0.28 0.35" specular="0.1 0.1 0.1"/>
    <geom name="floor" type="plane" size="{ground_size} {ground_size} 0.1" material="matplane" rgba="{floor_rgba}"{floor_solref_attr}/>
    <camera name="cam0" pos="0 -{camera_distance} {camera_height}" xyaxes="{camera_xyaxes}" fovy="{camera_fovy}"/>
    <body name="{body_name}" pos="0 0 2">
      <joint name="root_free" type="free" damping="{freejoint_damping}"/>
      {geom_xml}
    </body>
  </worldbody>
</mujoco>
""".strip()


def look_at_xyaxes(camera_distance, camera_height, target):
    """MuJoCo ``xyaxes`` of a camera at (0, -distance, height) looking at ``target``."""
    eye = np.array([0.0, -float(camera_distance), float(camera_height)])
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    right = np.array([forward[1], -forward[0], 0.0])
    right /= max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, forward)
    return eye, f"{right[0]:.6f} {right[1]:.6f} {right[2]:.6f} {up[0]:.6f} {up[1]:.6f} {up[2]:.6f}"


def build_mjcf_from_scene_xml(
    scene_xml: str,
    *,
    object_mass: float,
    object_friction: str,
    object_solref: str,
    floor_friction: str,
    floor_solref: str,
    freejoint_damping: float,
    timestep: float,
    gravity: float,
    camera_distance: float,
    camera_height: float,
    camera_target,
    camera_fovy: float,
    width: int,
    height: int,
) -> str:
    """Turn a Stage1 dataset ``scene.xml`` into the Stage2 rollout model.

    Appearance (lights, sky, floor material, object visuals) is kept verbatim;
    only physics attributes and the episode camera are added.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(scene_xml)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", f"{float(timestep)}")
    option.set("gravity", f"0 0 {float(gravity)}")
    option.set("integrator", "Euler")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", str(max(int(width), int(global_visual.get("offwidth", 0)))))
    global_visual.set("offheight", str(max(int(height), int(global_visual.get("offheight", 0)))))
    # Round primitives are drawn as prisms; use enough sides that the rendered
    # shape is the analytic one that collides (default 28 sides: 0.44 mm off for the can).
    quality = visual.find("quality")
    if quality is None:
        quality = ET.SubElement(visual, "quality")
    quality.set("numslices", "160")
    quality.set("numstacks", "160")
    worldbody = root.find("worldbody")
    body = worldbody.find(".//body[@name='target_body']")
    if body is None:
        raise ValueError("scene_xml must contain a body named 'target_body'")
    for free in list(body.findall("freejoint")) + list(body.findall("joint")):
        body.remove(free)
    joint = ET.Element("joint", {"name": "root_free", "type": "free", "damping": f"{float(freejoint_damping)}"})
    body.insert(0, joint)
    target = body.find("geom[@name='target_geom']")
    if target is None:
        raise ValueError("scene_xml target_body must contain a geom named 'target_geom'")
    target.set("mass", f"{float(object_mass)}")
    target.attrib.pop("density", None)
    if str(object_friction).strip():
        target.set("friction", str(object_friction))
    if str(object_solref).strip():
        target.set("solref", str(object_solref))
    floor = worldbody.find("geom[@name='floor']")
    if floor is not None:
        if str(floor_friction).strip():
            floor.set("friction", str(floor_friction))
        if str(floor_solref).strip():
            floor.set("solref", str(floor_solref))
    eye, xyaxes = look_at_xyaxes(camera_distance, camera_height, camera_target)
    for existing in worldbody.findall("camera[@name='cam0']"):
        worldbody.remove(existing)
    ET.SubElement(worldbody, "camera", {
        "name": "cam0", "pos": f"{eye[0]} {eye[1]} {eye[2]}",
        "xyaxes": xyaxes, "fovy": f"{float(camera_fovy)}",
    })
    return ET.tostring(root, encoding="unicode")


def camera_record(model, data, camera_name: str, width: int, height: int) -> dict:
    """Exact camera used for rendering: MuJoCo frame (x right, y up, looking along -z)."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    c2w = np.eye(4)
    c2w[:3, :3] = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    c2w[:3, 3] = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
    return {
        "name": camera_name,
        "convention": "mujoco_camera_x_right_y_up_looks_along_minus_z",
        "c2w": c2w.tolist(),
        "fovy_deg": float(model.cam_fovy[cam_id]),
        "image_size": [int(width), int(height)],
    }


def save_rgb(path: Path, rgb):
    Image.fromarray(rgb).save(path)


def save_rollout_gif(rgb_dir: Path, output_path: Path, fps: int):
    paths = sorted(rgb_dir.glob("*.png"))
    if not paths:
        return
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB").copy())
    frames[0].save(
        output_path, save_all=True, append_images=frames[1:],
        duration=max(1, round(1000 / fps)), loop=0,
    )


MASK_SUPERSAMPLE = 4  # object masks are pixel coverages from a segmentation render at this multiple of the image size


def save_mask(path: Path, segmentation, object_geom_ids, factor: int = 1):
    """8-bit object mask; with ``factor > 1`` the segmentation is box-filtered to a pixel coverage."""
    fine = np.isin(segmentation[..., 0], object_geom_ids).astype(np.float32)
    height, width = fine.shape[0] // factor, fine.shape[1] // factor
    coverage = fine.reshape(height, factor, width, factor).mean(axis=(1, 3))
    Image.fromarray(np.round(coverage * 255.0).astype(np.uint8), mode="L").save(path)


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
        # mj_step leaves the body poses of the step's start in ``data``; refresh
        # them so the rendered frame and the recorded state are the same instant.
        mujoco.mj_forward(model, data)

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

            mask_renderer = renderer.mask_renderer  # same camera, MASK_SUPERSAMPLE x the image size
            mask_renderer.enable_segmentation_rendering()
            mask_renderer.update_scene(data, camera=camera_name)
            segmentation = np.asarray(mask_renderer.render(), dtype=np.int32)
            mask_renderer.disable_segmentation_rendering()

            save_rgb(rgb_dir / f"{stem}.png", rgb)
            save_mask(mask_dir / f"{stem}.png", segmentation, object_geom_ids, MASK_SUPERSAMPLE)

        # MuJoCo free joints store the angular velocity in the body frame.
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
        angular_velocity_world = rotation.reshape(3, 3) @ data.qvel[3:6]
        states.append(
            {
                "frame_index": frame_idx,
                "time": float(data.time),
                "qpos": data.qpos.tolist(),
                "qvel": data.qvel.tolist(),
                "position": data.qpos[:3].tolist(),
                "quaternion_wxyz": data.qpos[3:7].tolist(),
                "linear_velocity": data.qvel[:3].tolist(),
                "angular_velocity": angular_velocity_world.tolist(),
                "angular_velocity_body": data.qvel[3:6].tolist(),
            }
        )
        actions.append(
            {
                "frame_index": frame_idx,
                "action_type": "none",
                "control": [],
            }
        )

    timing = {
        "physics_timestep": float(model.opt.timestep),
        "steps_per_frame": int(steps_per_frame),
        "frame_dt": float(steps_per_frame * model.opt.timestep),
        # The simulator is stepped before each frame is rendered, so frame 0
        # is one frame interval after the released initial state.
        "first_frame_time": float(steps_per_frame * model.opt.timestep),
    }
    write_json(state_dir / "trajectory.json", {
        "angular_velocity_frame": "world", "timing": timing, "states": states,
    })
    write_json(action_dir / "trajectory.json", {"actions": actions})

    manifest["generator"] = {
        "name": "generate_mujoco_fall_dataset.py",
        "camera_name": camera_name,
    }
    manifest["timing"] = timing
    manifest["initial_state"] = {
        "position": initial_pos.tolist(),
        "quaternion_wxyz": initial_quat.tolist(),
        "linear_velocity": initial_linvel.tolist(),
        "angular_velocity_body": initial_angvel.tolist(),
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
    physics_prior = dict(object_manifest.get("physics_prior") or {})
    object_mass = float(physics_prior.get("mass_kg", 1.0))
    if object_mass <= 0.0:
        raise ValueError("object_manifest physics_prior.mass_kg must be positive")

    scenario_manifest = read_json(dataset_root / "stage2" / "fall_and_rebound" / "scenario_manifest.json")
    width, height = scenario_manifest["image_size"]
    if str(args.scene_xml).strip():
        scene_xml_path = Path(args.scene_xml).expanduser().resolve()
        shape_friction = {
            "sphere": args.sphere_friction, "cylinder": args.cylinder_friction,
        }.get(physics_shape, args.box_friction)
        shape_solref = {
            "sphere": args.sphere_solref, "cylinder": args.cylinder_solref,
        }.get(physics_shape, args.box_solref)
        mjcf = build_mjcf_from_scene_xml(
            scene_xml_path.read_text(encoding="utf-8"),
            object_mass=object_mass, object_friction=shape_friction, object_solref=shape_solref,
            floor_friction=args.floor_friction, floor_solref=args.floor_solref,
            freejoint_damping=args.freejoint_damping, timestep=args.timestep, gravity=args.gravity,
            camera_distance=args.camera_distance, camera_height=args.camera_height,
            camera_target=(args.camera_target_x, args.camera_target_y, args.camera_target_z),
            camera_fovy=args.camera_fovy, width=int(width), height=int(height),
        )
    else:
        scene_xml_path = None
        mjcf = legacy_mjcf(args, half_extents, physics_shape, object_mass, visual_model)
    (dataset_root / "stage2" / "fall_and_rebound" / "rollout_model.xml").write_text(mjcf, encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(mjcf)
    moving_body_id = int(model.jnt_bodyid[0])
    actual_mass = float(model.body_mass[moving_body_id])
    actual_inertia = [float(value) for value in model.body_inertia[moving_body_id]]
    if not math.isclose(actual_mass, object_mass, rel_tol=1e-6, abs_tol=1e-8):
        raise RuntimeError(
            f"MuJoCo body mass {actual_mass} does not match requested mass {object_mass}"
        )
    object_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) != "floor"
    ]

    renderer = None
    if not bool(args.skip_render):
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), MASK_SUPERSAMPLE * int(width))
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), MASK_SUPERSAMPLE * int(height))
        renderer = mujoco.Renderer(model, height=int(height), width=int(width))
        renderer.mask_renderer = mujoco.Renderer(model, height=MASK_SUPERSAMPLE * int(height),
                                                 width=MASK_SUPERSAMPLE * int(width))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    exact_camera = camera_record(model, data, "cam0", int(width), int(height))
    run_episodes(args, dataset_root, model, data, renderer, object_geom_ids, physics_shape,
                 half_extents, object_mass, actual_mass, actual_inertia, moving_body_id,
                 exact_camera, scene_xml_path)


def legacy_mjcf(args, half_extents, physics_shape, object_mass, visual_model):
    return build_mjcf(
        half_extents=half_extents,
        physics_shape=physics_shape,
        object_mass=object_mass,
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
        camera_fovy=args.camera_fovy,
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
        floor_rgb1=args.floor_rgb1,
        floor_rgb2=args.floor_rgb2,
        floor_texrepeat=args.floor_texrepeat,
        sky_rgb1=args.sky_rgb1,
        sky_rgb2=args.sky_rgb2,
    )


def run_episodes(args, dataset_root, model, data, renderer, object_geom_ids, physics_shape,
                 half_extents, object_mass, actual_mass, actual_inertia, moving_body_id,
                 exact_camera, scene_xml_path):
    rng = np.random.default_rng(args.seed)
    fixed_position = parse_optional_vec3(args.initial_position, "--initial_position")
    fixed_euler_deg = parse_optional_vec3(args.initial_euler_deg, "--initial_euler_deg")
    fixed_quaternion = None
    if str(args.initial_quaternion_wxyz).strip():
        values = [float(value) for value in str(args.initial_quaternion_wxyz).replace(",", " ").split()]
        if len(values) != 4:
            raise ValueError("--initial_quaternion_wxyz expects four values")
        fixed_quaternion = np.asarray(values, dtype=np.float64)
        norm = float(np.linalg.norm(fixed_quaternion))
        if norm <= 1e-12:
            raise ValueError("--initial_quaternion_wxyz must be non-zero")
        fixed_quaternion = fixed_quaternion / norm
    fixed_linear_velocity = parse_optional_vec3(args.initial_linear_velocity, "--initial_linear_velocity")
    fixed_angular_velocity = parse_optional_vec3(args.initial_angular_velocity, "--initial_angular_velocity")
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
        if fixed_position is not None:
            init_pos = fixed_position.copy()
        if fixed_euler_deg is not None:
            fixed_euler = np.deg2rad(fixed_euler_deg)
            init_quat = euler_xyz_to_quat(*[float(value) for value in fixed_euler])
        if fixed_quaternion is not None:
            init_quat = fixed_quaternion.copy()
        if fixed_linear_velocity is not None:
            init_linvel = fixed_linear_velocity.copy()
        if fixed_angular_velocity is not None:
            init_angvel = fixed_angular_velocity.copy()

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

        half_height = float(half_extents[2])
        speeds = [float(np.linalg.norm(state["linear_velocity"])) for state in states]
        angular_speeds = [float(np.linalg.norm(state["angular_velocity"])) for state in states]
        impact_frames = [
            int(state["frame_index"]) for state in states
            if float(state["position"][2]) <= half_height + 0.015
        ]
        episode_manifest = read_json(episode_root / "episode_manifest.json")
        episode_manifest["rollout_diagnostics"] = {
            "first_near_floor_frame": None if not impact_frames else impact_frames[0],
            "final_linear_speed": speeds[-1],
            "final_angular_speed": angular_speeds[-1],
            "final_position": states[-1]["position"],
            "duration_seconds": len(states) / float(args.fps),
        }
        episode_manifest["camera"] = {
            **exact_camera,
            "distance": args.camera_distance, "height": args.camera_height,
            "target": [args.camera_target_x, args.camera_target_y, args.camera_target_z],
        }
        if scene_xml_path is not None:
            episode_manifest["scene_xml"] = str(scene_xml_path)
        episode_manifest["mujoco_body_properties"] = {
            "body_id": int(moving_body_id),
            "requested_mass_kg": object_mass,
            "actual_mass_kg": actual_mass,
            "inertia_diagonal_kg_m2": actual_inertia,
            "freejoint_damping": float(args.freejoint_damping),
            "internal_timestep": float(model.opt.timestep),
        }
        # Ground-truth contact physics, for evaluation only (MuJoCo mixes a
        # geom pair by taking the element-wise max friction).
        target_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        if target_geom < 0:
            target_geom = object_geom_ids[0]
        floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        episode_manifest["mujoco_contact_properties"] = {
            "object_friction": model.geom_friction[target_geom].tolist(),
            "floor_friction": None if floor_geom < 0 else model.geom_friction[floor_geom].tolist(),
            "effective_sliding_friction": float(max(
                model.geom_friction[target_geom][0],
                0.0 if floor_geom < 0 else model.geom_friction[floor_geom][0],
            )),
            "object_solref": model.geom_solref[target_geom].tolist(),
            "floor_solref": None if floor_geom < 0 else model.geom_solref[floor_geom].tolist(),
            "solimp": model.geom_solimp[target_geom].tolist(),
            "cone": "pyramidal" if int(model.opt.cone) == 0 else "elliptic",
        }
        write_json(episode_root / "episode_manifest.json", episode_manifest)
        save_rollout_gif(episode_root / "rgb", episode_root / "rollout.gif", args.fps)

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
