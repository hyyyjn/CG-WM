import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


STANDARD_PREFIX = ("x", "y", "z", "nx", "ny", "nz")
STANDARD_SUFFIX = ("opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")


def numbered_fields(names, prefix):
    return sorted((name for name in names if name.startswith(prefix)), key=lambda name: int(name.split("_")[-1]))


def export_sibr_ply(input_path: Path, output_path: Path):
    ply = PlyData.read(str(input_path))
    vertices = ply["vertex"].data
    names = vertices.dtype.names

    # edit this: SG-GS stores geometry-only feature channels in the PLY.
    # The SIBR Gaussian viewer expects the standard 3DGS layout, so export a
    # viewer-only copy without f_geo_* and object_id fields.
    output_names = []
    output_names.extend(name for name in STANDARD_PREFIX if name in names)
    output_names.extend(numbered_fields(names, "f_dc_"))
    output_names.extend(numbered_fields(names, "f_rest_"))
    output_names.extend(name for name in STANDARD_SUFFIX if name in names)

    missing = [name for name in (*STANDARD_PREFIX, *STANDARD_SUFFIX) if name not in names]
    if missing:
        raise ValueError(f"Input PLY is missing required fields: {missing}")

    output_dtype = [(name, vertices.dtype.fields[name][0]) for name in output_names]
    output_vertices = np.empty(vertices.shape[0], dtype=output_dtype)
    for name in output_names:
        output_vertices[name] = vertices[name]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(output_vertices, "vertex")], text=False).write(str(output_path))
    print(f"Wrote SIBR-compatible PLY: {output_path}")
    print(f"Kept {len(output_names)} fields, removed {len(names) - len(output_names)} SG-GS-only fields.")


def main():
    parser = argparse.ArgumentParser(description="Export a standard 3DGS PLY for the SIBR Gaussian viewer.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export_sibr_ply(args.input, args.output)


if __name__ == "__main__":
    main()
