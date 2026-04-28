import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:  # pragma: no cover
    raise ImportError("MuJoCo and mujoco.viewer are required for preview.") from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Preview a generated MuJoCo fall_and_rebound episode.")
    parser.add_argument("--dataset_root", required=True, type=str)
    parser.add_argument("--object_name", required=True, type=str)
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--episode_index", default=0, type=int)
    parser.add_argument("--gravity", default=-9.81, type=float)
    parser.add_argument("--timestep", default=0.002, type=float)
    parser.add_argument("--ground_size", default=4.0, type=float)
    parser.add_argument("--camera_distance", default=6.0, type=float)
    parser.add_argument("--camera_height", default=3.0, type=float)
    parser.add_argument("--freejoint_damping", default=0.12, type=float)
    parser.add_argument("--object_rgba", default="0.85 0.25 0.20 1")
    parser.add_argument("--object_rgb1", default="0.92 0.28 0.22")
    parser.add_argument("--object_rgb2", default="0.20 0.55 0.92")
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
    )
    parser.add_argument("--floor_rgba", default="0.92 0.92 0.92 1")
    parser.add_argument("--sphere_solref", default="-1000 0")
    parser.add_argument("--sphere_friction", default="0.02 0.001 0.0001")
    parser.add_argument("--box_friction", default="0.35 0.01 0.001")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--loop", action="store_true", help="Replay the episode continuously until the viewer is closed.")
    parser.add_argument(
        "--hold_after",
        action="store_true",
        help="Keep the final frame visible after playback until the viewer is closed.",
    )
    return parser.parse_args()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


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
    sphere_solref,
    sphere_friction,
    box_friction,
    freejoint_damping,
    object_rgb1,
    object_rgb2,
    box_face_colors,
):
    hx, hy, hz = [float(v) for v in half_extents]
    if physics_shape == "sphere":
        radius = max(hx, hy, hz)
        body_name = "sphere"
        geom_xml = (
            f'<geom name="sphere_geom" type="sphere" size="{radius}" rgba="{object_rgba}" '
            f'density="1000" friction="{sphere_friction}" solref="{sphere_solref}"/>'
        )
    else:
        body_name = "box"
        face_colors = parse_face_colors(box_face_colors)
        face_pad = max(min(hx, hy, hz) * 0.04, 0.01)
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
            f'<geom name="box_geom" type="box" size="{hx} {hy} {hz}" rgba="0.14 0.14 0.16 1" '
            f'density="1000" friction="{box_friction}"/>\n'
            f"      {face_geom_xml}"
        )
    return f"""
<mujoco model="contactwm_cube_fall_preview">
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
    <geom name="floor" type="plane" size="{ground_size} {ground_size} 0.1" material="matplane" rgba="{floor_rgba}"/>
    <camera name="cam0" pos="0 -{camera_distance} {camera_height}" xyaxes="1 0 0 0 0.5 0.8660254"/>
    <body name="{body_name}" pos="0 0 2">
      <joint name="root_free" type="free" damping="{freejoint_damping}"/>
      {geom_xml}
    </body>
  </worldbody>
</mujoco>
""".strip()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    object_manifest_path = dataset_root / "objects" / args.object_name / "object_manifest.json"
    episode_root = (
        dataset_root
        / "stage2"
        / "fall_and_rebound"
        / args.split
        / args.object_name
        / f"episode_{args.episode_index:03d}"
    )
    episode_manifest_path = episode_root / "episode_manifest.json"
    state_path = episode_root / "state" / "trajectory.json"

    if not object_manifest_path.exists():
        raise FileNotFoundError(object_manifest_path)
    if not episode_manifest_path.exists():
        raise FileNotFoundError(episode_manifest_path)
    if not state_path.exists():
        raise FileNotFoundError(state_path)

    object_manifest = read_json(object_manifest_path)
    episode_manifest = read_json(episode_manifest_path)
    state_payload = read_json(state_path)
    states = state_payload["states"]
    if not states:
        raise ValueError("No states found in trajectory.")

    half_extents = derive_half_extents(object_manifest)
    physics_shape = str(object_manifest.get("physics_shape", "box"))
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
        sphere_solref=args.sphere_solref,
        sphere_friction=args.sphere_friction,
        box_friction=args.box_friction,
        freejoint_damping=args.freejoint_damping,
        object_rgb1=args.object_rgb1,
        object_rgb2=args.object_rgb2,
        box_face_colors=args.box_face_colors,
    )
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)

    first = states[0]
    data.qpos[:7] = np.asarray(first["qpos"], dtype=np.float64)
    if "qvel" in first:
        qvel = np.asarray(first["qvel"], dtype=np.float64)
        data.qvel[: qvel.shape[0]] = qvel
    mujoco.mj_forward(model, data)

    frame_dt = 1.0 / float(episode_manifest.get("fps", args.fps))
    title = f"MuJoCo fall preview: {args.object_name} / {args.split} / episode_{args.episode_index:03d}"

    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam0")
        viewer.sync()

        while viewer.is_running():
            for frame in states:
                if not viewer.is_running():
                    break
                data.qpos[:7] = np.asarray(frame["qpos"], dtype=np.float64)
                qvel = np.asarray(frame.get("qvel", []), dtype=np.float64)
                if qvel.size > 0:
                    data.qvel[: qvel.shape[0]] = qvel
                data.time = float(frame["time"])
                mujoco.mj_forward(model, data)
                viewer.sync()
                print(
                    f"{title} | frame={frame['frame_index']:04d} "
                    f"time={frame['time']:.3f} z={frame['position'][2]:.4f}"
                )
                time.sleep(frame_dt)

            if args.loop:
                continue

            if args.hold_after:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(frame_dt)
            break


if __name__ == "__main__":
    main()
