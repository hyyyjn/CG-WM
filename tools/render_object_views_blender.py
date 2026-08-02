import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import bpy
import mathutils


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Render an object-centric SG-GS Stage 1 dataset from a mesh in Blender."
    )
    parser.add_argument("--mesh_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--object_name", default="", type=str)
    parser.add_argument(
        "--physics_shape",
        default="box",
        choices=("box", "sphere"),
        help="Physics proxy shape to record for downstream MuJoCo dataset generation.",
    )
    parser.add_argument("--num_views", default=72, type=int)
    parser.add_argument("--test_hold", default=8, type=int, help="Every Nth view is assigned to test.")
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--radius_scale", default=2.4, type=float)
    parser.add_argument("--lens", default=45.0, type=float, help="Perspective camera focal length in mm.")
    parser.add_argument("--elevation_min", default=-30.0, type=float)
    parser.add_argument("--elevation_max", default=60.0, type=float)
    parser.add_argument("--point_count", default=100000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--density", default=1000.0, type=float)
    parser.add_argument("--friction", default=0.5, type=float)
    parser.add_argument("--restitution", default=0.1, type=float)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument(
        "--material_mode",
        default="gray",
        choices=("gray", "position_bands", "face_palette"),
        help="Use gray for geometry checks, position_bands for generic color checks, or face_palette for box-like objects.",
    )
    return parser.parse_args(argv)


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_mesh(mesh_path):
    suffix = mesh_path.suffix.lower()
    if suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(mesh_path))
        else:
            bpy.ops.import_scene.obj(filepath=str(mesh_path))
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(mesh_path))
    elif suffix == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=str(mesh_path))
        else:
            bpy.ops.import_mesh.ply(filepath=str(mesh_path))
    else:
        raise ValueError(f"Unsupported mesh extension: {suffix}")

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise ValueError(f"No mesh objects were imported from {mesh_path}")
    return mesh_objects


def normalize_objects(mesh_objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]

    bbox_min = mathutils.Vector((float("inf"), float("inf"), float("inf")))
    bbox_max = mathutils.Vector((float("-inf"), float("-inf"), float("-inf")))
    for obj in mesh_objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ mathutils.Vector(corner)
            bbox_min.x = min(bbox_min.x, world_corner.x)
            bbox_min.y = min(bbox_min.y, world_corner.y)
            bbox_min.z = min(bbox_min.z, world_corner.z)
            bbox_max.x = max(bbox_max.x, world_corner.x)
            bbox_max.y = max(bbox_max.y, world_corner.y)
            bbox_max.z = max(bbox_max.z, world_corner.z)

    center = (bbox_min + bbox_max) * 0.5
    extent = max((bbox_max - bbox_min).x, (bbox_max - bbox_min).y, (bbox_max - bbox_min).z)
    scale = 2.0 / max(extent, 1e-8)

    for obj in mesh_objects:
        obj.location = (obj.location - center) * scale
        obj.scale = obj.scale * scale
        bpy.context.view_layer.update()

    return {
        "bbox_min": [float(bbox_min.x), float(bbox_min.y), float(bbox_min.z)],
        "bbox_max": [float(bbox_max.x), float(bbox_max.y), float(bbox_max.z)],
        "center": [float(center.x), float(center.y), float(center.z)],
        "scale": float(scale),
        "target_extent": 2.0,
    }


def make_material(name, color):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.68
    return material


def band_index(point):
    if point.z > 0.42:
        return 3
    if point.x < -0.28:
        return 0
    if point.x > 0.28:
        return 2
    return 1


def band_color(point):
    colors = (
        (224, 58, 50),
        (64, 184, 92),
        (54, 118, 230),
        (242, 190, 64),
    )
    return colors[band_index(point)]


def face_palette_colors():
    return (
        (230 / 255.0, 77 / 255.0, 61 / 255.0, 1.0),
        (73 / 255.0, 188 / 255.0, 103 / 255.0, 1.0),
        (62 / 255.0, 126 / 255.0, 236 / 255.0, 1.0),
        (243 / 255.0, 197 / 255.0, 67 / 255.0, 1.0),
        (161 / 255.0, 94 / 255.0, 255 / 255.0, 1.0),
        (48 / 255.0, 198 / 255.0, 210 / 255.0, 1.0),
    )


def dominant_axis_index(world_normal):
    axis = max(range(3), key=lambda idx: abs(world_normal[idx]))
    sign = 0 if world_normal[axis] >= 0 else 1
    return axis * 2 + sign


def setup_materials(mesh_objects, material_mode):
    # edit this: position_bands makes appearance/color optimization visible in
    # synthetic Stage 1 tests; gray remains the default geometry-only check.
    if material_mode == "gray":
        material = make_material("stage1_gray_material", (0.72, 0.72, 0.72, 1.0))
        for obj in mesh_objects:
            obj.data.materials.clear()
            obj.data.materials.append(material)
        return

    if material_mode == "position_bands":
        material_colors = [
            (224 / 255.0, 58 / 255.0, 50 / 255.0, 1.0),
            (64 / 255.0, 184 / 255.0, 92 / 255.0, 1.0),
            (54 / 255.0, 118 / 255.0, 230 / 255.0, 1.0),
            (242 / 255.0, 190 / 255.0, 64 / 255.0, 1.0),
        ]
    else:
        material_colors = list(face_palette_colors())

    materials = [
        make_material(f"stage1_material_{idx}", color)
        for idx, color in enumerate(material_colors)
    ]
    for obj in mesh_objects:
        obj.data.materials.clear()
        for material in materials:
            obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            if material_mode == "position_bands":
                world_center = obj.matrix_world @ polygon.center
                polygon.material_index = band_index(world_center)
            else:
                world_normal = (obj.matrix_world.to_3x3() @ polygon.normal).normalized()
                polygon.material_index = dominant_axis_index(world_normal)


def setup_scene(resolution, white_background, lens):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.color = (1.0, 1.0, 1.0) if white_background else (0.0, 0.0, 0.0)

    light_data = bpy.data.lights.new("stage1_key_light", type="AREA")
    light_data.energy = 400
    light_data.size = 5.0
    light = bpy.data.objects.new("stage1_key_light", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (2.5, -3.0, 4.0)

    camera_data = bpy.data.cameras.new("stage1_camera")
    camera = bpy.data.objects.new("stage1_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera_data.lens = lens
    camera_data.sensor_width = 32
    return camera


def fibonacci_sphere(num_views, elevation_min, elevation_max):
    views = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    min_z = math.sin(math.radians(elevation_min))
    max_z = math.sin(math.radians(elevation_max))
    for idx in range(num_views):
        t = (idx + 0.5) / float(num_views)
        z = min_z + (max_z - min_z) * t
        radius_xy = math.sqrt(max(0.0, 1.0 - z * z))
        theta = idx * golden_angle
        views.append(mathutils.Vector((math.cos(theta) * radius_xy, math.sin(theta) * radius_xy, z)))
    return views


def look_at(camera, location, target):
    camera.location = location
    direction = target - location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def camera_angle_x(camera):
    return float(camera.data.angle_x)


def render_view(scene, camera, image_path, mask_path):
    scene.render.filepath = str(image_path)
    bpy.ops.render.render(write_still=True)

    # edit this: Blender 5 can expose an empty Render Result buffer after write_still;
    # reload the saved RGBA image so alpha-derived masks are stable across versions.
    rendered = bpy.data.images.load(str(image_path))
    width, height = rendered.size
    pixels = list(rendered.pixels)
    mask = bpy.data.images.new(f"mask_{image_path.stem}", width=width, height=height, alpha=False)
    mask_pixels = [0.0] * (width * height * 4)
    background_is_white = bool(scene.world.color[0] > 0.9 and scene.world.color[1] > 0.9 and scene.world.color[2] > 0.9)
    for idx in range(width * height):
        red = pixels[idx * 4 + 0]
        green = pixels[idx * 4 + 1]
        blue = pixels[idx * 4 + 2]
        alpha = pixels[idx * 4 + 3]
        if alpha > 0.01 and alpha < 0.99:
            value = 1.0
        elif background_is_white:
            value = 0.0 if (red > 0.97 and green > 0.97 and blue > 0.97) else 1.0
        else:
            value = 1.0 if (red > 0.03 or green > 0.03 or blue > 0.03) else 0.0
        mask_pixels[idx * 4 + 0] = value
        mask_pixels[idx * 4 + 1] = value
        mask_pixels[idx * 4 + 2] = value
        mask_pixels[idx * 4 + 3] = 1.0
    mask.pixels.foreach_set(mask_pixels)
    mask.filepath_raw = str(mask_path)
    mask.file_format = "PNG"
    mask.save()
    bpy.data.images.remove(rendered)
    bpy.data.images.remove(mask)


def triangulated_world_meshes(mesh_objects):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    meshes = []
    for obj in mesh_objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
        mesh.transform(obj.matrix_world)
        mesh.calc_loop_triangles()
        meshes.append(mesh)
    return meshes


def sample_points(mesh_objects, point_count, seed):
    random.seed(seed)
    meshes = triangulated_world_meshes(mesh_objects)
    triangles = []
    total_area = 0.0
    for mesh in meshes:
        vertices = mesh.vertices
        for tri in mesh.loop_triangles:
            p0 = vertices[tri.vertices[0]].co.copy()
            p1 = vertices[tri.vertices[1]].co.copy()
            p2 = vertices[tri.vertices[2]].co.copy()
            area = 0.5 * (p1 - p0).cross(p2 - p0).length
            if area > 0:
                total_area += area
                triangles.append((total_area, p0, p1, p2))

    if not triangles:
        raise ValueError("Imported mesh has no sampleable triangles.")

    points = []
    for _ in range(point_count):
        target = random.random() * total_area
        lo, hi = 0, len(triangles) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if triangles[mid][0] < target:
                lo = mid + 1
            else:
                hi = mid
        _, p0, p1, p2 = triangles[lo]
        r1 = math.sqrt(random.random())
        r2 = random.random()
        point = (1.0 - r1) * p0 + r1 * (1.0 - r2) * p1 + r1 * r2 * p2
        points.append(point)

    for mesh in meshes:
        bpy.data.meshes.remove(mesh)
    return points


def write_points_ply(path, points, material_mode):
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point in points:
            if material_mode == "position_bands":
                red, green, blue = band_color(point)
            elif material_mode == "face_palette":
                palette = (
                    (230, 77, 61),
                    (73, 188, 103),
                    (62, 126, 236),
                    (243, 197, 67),
                    (161, 94, 255),
                    (48, 198, 210),
                )
                axis = dominant_axis_index(point.normalized() if point.length > 1e-8 else mathutils.Vector((0.0, 0.0, 1.0)))
                red, green, blue = palette[axis]
            else:
                red, green, blue = (184, 184, 184)
            f.write(f"{point.x:.8f} {point.y:.8f} {point.z:.8f} 0 0 0 {red} {green} {blue}\n")


def ensure_dirs(output_path):
    for split in ("train", "test"):
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "masks" / split).mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    mesh_path = Path(args.mesh_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    object_name = args.object_name.strip() if args.object_name.strip() else mesh_path.stem
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    output_path.mkdir(parents=True, exist_ok=True)
    ensure_dirs(output_path)

    reset_scene()
    mesh_objects = import_mesh(mesh_path)
    normalization = normalize_objects(mesh_objects)
    setup_materials(mesh_objects, args.material_mode)
    camera = setup_scene(args.resolution, args.white_background, args.lens)

    views = fibonacci_sphere(args.num_views, args.elevation_min, args.elevation_max)
    radius = args.radius_scale
    target = mathutils.Vector((0.0, 0.0, 0.0))
    transforms = {
        "train": {"camera_angle_x": camera_angle_x(camera), "frames": []},
        "test": {"camera_angle_x": camera_angle_x(camera), "frames": []},
    }

    for idx, view_dir in enumerate(views):
        split = "test" if args.test_hold > 0 and idx % args.test_hold == 0 else "train"
        stem = f"{idx:06d}"
        look_at(camera, view_dir * radius, target)
        bpy.context.view_layer.update()

        image_path = output_path / "images" / split / f"{stem}.png"
        mask_path = output_path / "masks" / split / f"{stem}.png"
        render_view(bpy.context.scene, camera, image_path, mask_path)

        transforms[split]["camera_angle_x"] = camera_angle_x(camera)
        transforms[split]["frames"].append(
            {
                "file_path": f"./images/{split}/{stem}",
                "mask_path": f"./masks/{split}/{stem}.png",
                "transform_matrix": [[float(value) for value in row] for row in camera.matrix_world],
            }
        )

    for split in ("train", "test"):
        with open(output_path / f"transforms_{split}.json", "w", encoding="utf-8") as f:
            json.dump(transforms[split], f, indent=2)

    points = sample_points(mesh_objects, args.point_count, args.seed)
    write_points_ply(output_path / "points3d.ply", points, args.material_mode)

    summary = {
        "object_name": str(object_name),
        "physics_shape": str(args.physics_shape),
        "mesh_path": str(mesh_path),
        "output_path": str(output_path),
        "num_views": int(args.num_views),
        "num_train_views": len(transforms["train"]["frames"]),
        "num_test_views": len(transforms["test"]["frames"]),
        "resolution": int(args.resolution),
        "point_count": int(args.point_count),
        "material_mode": str(args.material_mode),
        "normalization": normalization,
        "physics_prior": {
            "density": float(args.density),
            "friction": float(args.friction),
            "restitution": float(args.restitution),
        },
    }
    with open(output_path / "stage1_dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    object_asset = {
        "object_name": str(object_name),
        "physics_shape": str(args.physics_shape),
        "mesh_path": str(mesh_path),
        "stage1_dataset_path": str(output_path),
        "stage1_points_ply": str(output_path / "points3d.ply"),
        "stage1_summary_path": str(output_path / "stage1_dataset_summary.json"),
        "normalization": normalization,
        "physics_prior": summary["physics_prior"],
        "material_mode": str(args.material_mode),
    }
    with open(output_path / "object_asset.json", "w", encoding="utf-8") as f:
        json.dump(object_asset, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
