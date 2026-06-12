from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an actual MuJoCo multi-dice rollout.")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--dice_count", default=5, type=int)
    parser.add_argument("--frames", default=180, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--width", default=960, type=int)
    parser.add_argument("--height", default=540, type=int)
    parser.add_argument("--timestep", default=0.002, type=float)
    parser.add_argument("--mujoco_gl", default="glfw", choices=("glfw", "egl", "osmesa"))
    parser.add_argument("--seed", default=11, type=int)
    parser.add_argument(
        "--skip_render",
        action="store_true",
        help="Run MuJoCo physics and write trajectory/top-down GIF without requiring an OpenGL renderer.",
    )
    return parser.parse_args()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def euler_xyz_to_quat(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


def pip_positions(face: str) -> list[tuple[float, float]]:
    layouts = {
        "one": [(0.0, 0.0)],
        "two": [(-0.022, -0.022), (0.022, 0.022)],
        "three": [(-0.024, -0.024), (0.0, 0.0), (0.024, 0.024)],
        "four": [(-0.023, -0.023), (0.023, -0.023), (-0.023, 0.023), (0.023, 0.023)],
        "five": [(-0.024, -0.024), (0.024, -0.024), (0.0, 0.0), (-0.024, 0.024), (0.024, 0.024)],
        "six": [(-0.024, -0.028), (0.024, -0.028), (-0.024, 0.0), (0.024, 0.0), (-0.024, 0.028), (0.024, 0.028)],
    }
    return layouts[face]


def pip_xml(die: int, half: float) -> str:
    # Small visual-only raised dots.  The central cube is the only collider.
    specs = [
        ("px", "three", (half + 0.001, 0, 0), "y", "z"),
        ("nx", "four", (-half - 0.001, 0, 0), "y", "z"),
        ("py", "two", (0, half + 0.001, 0), "x", "z"),
        ("ny", "five", (0, -half - 0.001, 0), "x", "z"),
        ("pz", "six", (0, 0, half + 0.001), "x", "y"),
        ("nz", "one", (0, 0, -half - 0.001), "x", "y"),
    ]
    axis_index = {"x": 0, "y": 1, "z": 2}
    rows = []
    for face_name, number_name, base, axis_u, axis_v in specs:
        for pip_idx, (u, v) in enumerate(pip_positions(number_name)):
            pos = [float(base[0]), float(base[1]), float(base[2])]
            pos[axis_index[axis_u]] += u
            pos[axis_index[axis_v]] += v
            rows.append(
                f'<geom name="die_{die:02d}_pip_{face_name}_{pip_idx:02d}" '
                f'type="sphere" pos="{pos[0]} {pos[1]} {pos[2]}" size="0.0075" '
                'rgba="0.02 0.02 0.018 1" contype="0" conaffinity="0" density="0"/>'
            )
    return "\n      ".join(rows)


def build_mjcf(dice_count: int, half: float, timestep: float, width: int, height: int) -> str:
    colors = [
        "0.92 0.10 0.12 1",
        "0.10 0.34 0.92 1",
        "0.12 0.62 0.28 1",
        "0.95 0.70 0.12 1",
        "0.48 0.25 0.82 1",
        "0.05 0.66 0.72 1",
    ]
    bodies = []
    for idx in range(dice_count):
        color = colors[idx % len(colors)]
        bodies.append(
            f"""
    <body name="die_{idx:02d}" pos="0 0 1">
      <joint name="die_{idx:02d}_free" type="free" damping="0.0"/>
      <geom name="die_{idx:02d}_collider" type="box" size="{half} {half} {half}" rgba="{color}" density="1200"
            friction="0.48 0.025 0.004" solref="0.010 0.62" solimp="0.88 0.96 0.001"/>
      {pip_xml(idx, half)}
    </body>
""".rstrip()
        )
    body_xml = "\n".join(bodies)
    return f"""
<mujoco model="multi_dice_rollout">
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="Euler"/>
  <visual>
    <global offwidth="{int(width)}" offheight="{int(height)}"/>
    <quality shadowsize="2048"/>
    <headlight diffuse="0.75 0.75 0.75" ambient="0.26 0.26 0.26" specular="0.18 0.18 0.18"/>
    <rgba haze="1 1 1 1"/>
  </visual>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.95 0.95 0.92" rgb2="0.84 0.84 0.80" width="256" height="256"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="5 5" reflectance="0.08"/>
  </asset>
  <worldbody>
    <light name="key" pos="-1.5 -2.0 4.0" dir="0.4 0.6 -1" directional="true"/>
    <geom name="floor" type="plane" size="2.2 2.2 0.1" material="floor_mat" rgba="0.92 0.92 0.88 1"
          friction="0.52 0.025 0.004" solref="0.010 0.72"/>
    <camera name="cam0" pos="0 -1.12 0.66" xyaxes="1 0 0 0 0.48 0.8772685" fovy="40"/>
{body_xml}
  </worldbody>
</mujoco>
""".strip()


def set_initial_state(model, data, rng: np.random.Generator, dice_count: int) -> list[dict]:
    import mujoco

    starts = np.linspace(-0.50, 0.50, dice_count)
    initial_states = []
    for idx in range(dice_count):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"die_{idx:02d}_free")
        qpos_adr = model.jnt_qposadr[joint_id]
        qvel_adr = model.jnt_dofadr[joint_id]
        x = starts[idx] + rng.uniform(-0.025, 0.025)
        y = -0.48 + rng.uniform(-0.035, 0.035)
        target_x = rng.uniform(-0.10, 0.10)
        target_y = rng.uniform(0.38, 0.58)
        travel = np.array([target_x - x, target_y - y], dtype=np.float64)
        travel_norm = max(float(np.linalg.norm(travel)), 1e-6)
        travel_dir = travel / travel_norm
        speed = rng.uniform(1.55, 2.15)
        pos = np.array(
            [
                x,
                y,
                0.62 + 0.075 * idx + rng.uniform(-0.018, 0.018),
            ],
            dtype=np.float64,
        )
        quat = euler_xyz_to_quat(
            rng.uniform(-math.pi, math.pi),
            rng.uniform(-math.pi, math.pi),
            rng.uniform(-math.pi, math.pi),
        )
        linvel = np.array(
            [
                speed * travel_dir[0],
                speed * travel_dir[1],
                rng.uniform(0.05, 0.28),
            ],
            dtype=np.float64,
        )
        angvel = rng.uniform(-34.0, 34.0, size=3).astype(np.float64)
        data.qpos[qpos_adr : qpos_adr + 3] = pos
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
        data.qvel[qvel_adr : qvel_adr + 3] = linvel
        data.qvel[qvel_adr + 3 : qvel_adr + 6] = angvel
        initial_states.append(
            {
                "die": idx,
                "position": pos.tolist(),
                "quaternion_wxyz": quat.tolist(),
                "linear_velocity": linvel.tolist(),
                "angular_velocity": angvel.tolist(),
            }
        )
    mujoco.mj_forward(model, data)
    return initial_states


def die_geom_ids(model, dice_count: int) -> dict[int, list[int]]:
    import mujoco

    mapping: dict[int, list[int]] = {}
    for idx in range(dice_count):
        prefix = f"die_{idx:02d}_"
        ids = []
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith(prefix):
                ids.append(geom_id)
        mapping[idx] = ids
    return mapping


def capture_state(model, data, dice_count: int) -> list[dict]:
    import mujoco

    dice = []
    for idx in range(dice_count):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"die_{idx:02d}_free")
        qpos_adr = model.jnt_qposadr[joint_id]
        qvel_adr = model.jnt_dofadr[joint_id]
        dice.append(
            {
                "die": idx,
                "position": data.qpos[qpos_adr : qpos_adr + 3].tolist(),
                "quaternion_wxyz": data.qpos[qpos_adr + 3 : qpos_adr + 7].tolist(),
                "linear_velocity": data.qvel[qvel_adr : qvel_adr + 3].tolist(),
                "angular_velocity": data.qvel[qvel_adr + 3 : qvel_adr + 6].tolist(),
            }
        )
    return dice


def capture_contacts(model, data) -> list[dict]:
    import mujoco

    contacts = []
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        geom_names = []
        dice = []
        for geom_id in (int(contact.geom1), int(contact.geom2)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            geom_names.append(name)
            if name.startswith("die_"):
                dice.append(int(name.split("_")[1]))
        if len(set(dice)) == 2:
            pair = sorted(set(dice))
            contacts.append(
                {
                    "body_i": pair[0],
                    "body_j": pair[1],
                    "name_i": f"die_{pair[0]:02d}",
                    "name_j": f"die_{pair[1]:02d}",
                    "geom1": geom_names[0],
                    "geom2": geom_names[1],
                    "distance": float(contact.dist),
                }
            )
    unique = {}
    for contact in contacts:
        key = (contact["body_i"], contact["body_j"])
        if key not in unique or contact["distance"] < unique[key]["distance"]:
            unique[key] = contact
    return [unique[key] for key in sorted(unique)]


def save_masks(segmentation: np.ndarray, geom_ids_by_die: dict[int, list[int]], masks_root: Path, stem: str) -> None:
    geom_channel = segmentation[..., 0]
    union = np.zeros(geom_channel.shape, dtype=np.uint8)
    for die_idx, geom_ids in geom_ids_by_die.items():
        mask = np.isin(geom_channel, geom_ids).astype(np.uint8) * 255
        union = np.maximum(union, mask)
        Image.fromarray(mask, mode="L").save(masks_root / f"die_{die_idx:02d}" / f"{stem}.png")
    Image.fromarray(union, mode="L").save(masks_root / "all" / f"{stem}.png")


def quat_wxyz_to_rotmat_xy(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z)],
        ],
        dtype=np.float64,
    )


def draw_topdown_frame(frame: dict, *, half: float, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    image = Image.new("RGB", (width, height), (245, 245, 242))
    pixels = image.load()
    tile = 28
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                for yy in range(y, min(y + tile, height)):
                    for xx in range(x, min(x + tile, width)):
                        pixels[xx, yy] = (235, 235, 230)
    colors = [
        (235, 42, 48),
        (48, 95, 235),
        (46, 170, 86),
        (236, 178, 40),
        (130, 78, 210),
        (35, 176, 186),
    ]
    draw = ImageDraw.Draw(image, "RGBA")
    bounds = (-0.78, 0.78, -0.72, 0.72)

    def project_xy(point: np.ndarray) -> tuple[float, float]:
        xmin, xmax, ymin, ymax = bounds
        px = 28 + (float(point[0]) - xmin) / (xmax - xmin) * (width - 56)
        py = height - 28 - (float(point[1]) - ymin) / (ymax - ymin) * (height - 56)
        return px, py

    draw.text((12, 10), f"frame {int(frame.get('frame_index', 0)):03d}  t={float(frame.get('time', 0.0)):.2f}s", fill=(30, 30, 30, 255))
    for die in frame["dice"]:
        idx = int(die["die"])
        pos = np.asarray(die["position"], dtype=np.float64)
        quat = np.asarray(die["quaternion_wxyz"], dtype=np.float64)
        rot_xy = quat_wxyz_to_rotmat_xy(quat)
        corners = []
        for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            local = np.array([sx * half, sy * half], dtype=np.float64)
            corner_xy = pos[:2] + rot_xy @ local
            corners.append(project_xy(corner_xy))
        color = colors[idx % len(colors)]
        draw.polygon(corners, fill=(*color, 220), outline=(25, 25, 25, 210))
        cx, cy = project_xy(pos[:2])
        draw.text((cx - 4, cy - 7), str(idx), fill=(255, 255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def main() -> None:
    args = parse_args()
    os.environ["MUJOCO_GL"] = args.mujoco_gl
    import mujoco

    dice_count = max(1, min(int(args.dice_count), 8))
    half = 0.055
    output_dir = args.output_dir.resolve()
    rgb_dir = output_dir / "rgb"
    masks_root = output_dir / "masks"
    for path in [rgb_dir, masks_root / "all"]:
        path.mkdir(parents=True, exist_ok=True)
    for die_idx in range(dice_count):
        (masks_root / f"die_{die_idx:02d}").mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_string(
        build_mjcf(dice_count, half, float(args.timestep), int(args.width), int(args.height))
    )
    data = mujoco.MjData(model)
    rng = np.random.default_rng(int(args.seed))
    initial_states = set_initial_state(model, data, rng, dice_count)
    geom_ids_by_die = die_geom_ids(model, dice_count)
    renderer = None if args.skip_render else mujoco.Renderer(model, height=int(args.height), width=int(args.width))
    steps_per_frame = max(1, int(round(1.0 / (float(args.fps) * float(args.timestep)))))

    states = []
    gif_frames = []
    for frame_idx in range(int(args.frames)):
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

        stem = f"{frame_idx:06d}"
        frame_payload = {
            "frame_index": frame_idx,
            "time": float(data.time),
            "dice": capture_state(model, data, dice_count),
            "mujoco_contacts": capture_contacts(model, data),
        }
        states.append(frame_payload)

        if renderer is None:
            gif_frames.append(draw_topdown_frame(frame_payload, half=half, size=(int(args.width), int(args.height))))
            continue

        renderer.update_scene(data, camera="cam0")
        rgb = np.asarray(renderer.render(), dtype=np.uint8)[..., :3]
        rgb[(rgb.sum(axis=-1) == 0)] = np.array([248, 248, 244], dtype=np.uint8)
        Image.fromarray(rgb).save(rgb_dir / f"{stem}.png")
        gif_frames.append(rgb)

        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="cam0")
        segmentation = np.asarray(renderer.render(), dtype=np.int32)
        renderer.disable_segmentation_rendering()
        save_masks(segmentation, geom_ids_by_die, masks_root, stem)

    gif_path = output_dir / "multi_dice_mujoco_gt.gif"
    imageio.mimsave(gif_path, gif_frames, fps=int(args.fps))
    montage_path = output_dir / "multi_dice_mujoco_gt_montage.png"
    montage_count = min(12, len(gif_frames))
    montage_indices = np.linspace(0, len(gif_frames) - 1, montage_count).round().astype(int).tolist()
    thumbs = [
        Image.fromarray(gif_frames[idx]).resize((320, 180), Image.Resampling.LANCZOS)
        for idx in montage_indices
    ]
    montage = Image.new("RGB", (1280, int(math.ceil(montage_count / 4)) * 180), (245, 245, 242))
    for tile_idx, thumb in enumerate(thumbs):
        montage.paste(thumb, ((tile_idx % 4) * 320, (tile_idx // 4) * 180))
    montage.save(montage_path)
    write_json(
        output_dir / "trajectory.json",
        {
            "generator": "generate_mujoco_multi_dice_rollout.py",
            "dice_count": dice_count,
            "fps": int(args.fps),
            "timestep": float(args.timestep),
            "half_extent": half,
            "initial_states": initial_states,
            "states": states,
        },
    )
    write_json(
        output_dir / "manifest.json",
        {
            "kind": "mujoco_multi_dice_rollout",
            "render_mode": "topdown_fallback" if args.skip_render else "mujoco_renderer",
            "rgb_dir": str(rgb_dir),
            "masks_dir": str(masks_root),
            "gif": str(gif_path),
            "montage": str(montage_path),
            "trajectory": str(output_dir / "trajectory.json"),
            "image_size": [int(args.width), int(args.height)],
            "frames": int(args.frames),
            "fps": int(args.fps),
            "dice_count": dice_count,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "gif": str(gif_path),
                "montage": str(montage_path),
                "trajectory": str(output_dir / "trajectory.json"),
                "frames": int(args.frames),
                "dice_count": dice_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
