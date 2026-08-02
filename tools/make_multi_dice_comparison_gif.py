from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a side-by-side MuJoCo GT vs Stage2 contact-graph comparison GIF."
    )
    parser.add_argument("--rgb_dir", required=True, type=Path)
    parser.add_argument("--contact_graph_frames", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max_frames", default=120, type=int)
    parser.add_argument("--fps", default=12, type=int)
    return parser.parse_args()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def scene_bounds(reports: list[dict]) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for report in reports:
        for die in report["dice"]:
            xs.append(float(die["position"][0]))
            ys.append(float(die["position"][1]))
    pad = 0.18
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def project_xy(
    position: list[float],
    bounds: tuple[float, float, float, float],
    panel: tuple[int, int, int, int],
) -> tuple[float, float]:
    min_x, max_x, min_y, max_y = bounds
    left, top, width, height = panel
    x, y = float(position[0]), float(position[1])
    px = left + 18 + (x - min_x) / max(max_x - min_x, 1e-6) * (width - 36)
    py = top + height - 24 - (y - min_y) / max(max_y - min_y, 1e-6) * (height - 48)
    return px, py


def draw_stage2_overlay(base: Image.Image, report: dict, bounds: tuple[float, float, float, float]) -> Image.Image:
    image = base.copy().convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, _ = image.size
    panel = (width - 318, 18, 300, 205)
    left, top, panel_width, panel_height = panel
    draw.rounded_rectangle(
        (left, top, left + panel_width, top + panel_height),
        radius=8,
        fill=(255, 255, 255, 220),
        outline=(20, 20, 20, 90),
    )

    comparison = report.get("contact_comparison", {})
    graph_pairs = {tuple(pair) for pair in comparison.get("graph_pairs", [])}
    mujoco_pairs = {tuple(pair) for pair in comparison.get("mujoco_pairs", [])}
    true_positive = graph_pairs & mujoco_pairs
    false_positive = graph_pairs - mujoco_pairs
    false_negative = mujoco_pairs - graph_pairs
    positions = [die["position"] for die in report["dice"]]

    def draw_pairs(pairs, color, line_width):
        for body_i, body_j in pairs:
            draw.line(
                (
                    *project_xy(positions[body_i], bounds, panel),
                    *project_xy(positions[body_j], bounds, panel),
                ),
                fill=color,
                width=line_width,
            )

    draw_pairs(false_positive, (230, 130, 25, 190), 3)
    draw_pairs(false_negative, (40, 80, 220, 230), 4)
    draw_pairs(true_positive, (220, 45, 45, 240), 5)

    colors = [
        (220, 40, 44),
        (32, 95, 210),
        (35, 150, 80),
        (230, 165, 35),
        (120, 75, 190),
        (20, 160, 170),
    ]
    for die in report["dice"]:
        die_index = int(die["die"])
        px, py = project_xy(die["position"], bounds, panel)
        color = colors[die_index % len(colors)]
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(*color, 235), outline=(0, 0, 0, 180), width=1)
        draw.text((px + 9, py - 8), str(die_index), fill=(25, 25, 25, 255))

    draw.text((left + 14, top + 10), f"frame {int(report['frame_index']):03d}", fill=(20, 20, 20, 255))
    draw.text((left + 14, top + 31), f"MuJoCo {len(mujoco_pairs)} | Stage2 graph {len(graph_pairs)}", fill=(20, 20, 20, 255))
    draw.text((left + 14, top + 52), f"TP {len(true_positive)}  FP {len(false_positive)}  FN {len(false_negative)}", fill=(20, 20, 20, 255))
    draw.text((left + 14, top + 76), "red=match  orange=graph only  blue=miss", fill=(70, 70, 70, 255))
    return image


def make_comparison_frame(
    rgb_dir: Path,
    report: dict,
    bounds: tuple[float, float, float, float],
) -> Image.Image:
    frame_index = int(report["frame_index"])
    base = Image.open(rgb_dir / f"{frame_index:06d}.png").convert("RGB")
    stage2 = draw_stage2_overlay(base, report, bounds)
    left = base.resize((480, 270), Image.Resampling.LANCZOS)
    right = stage2.resize((480, 270), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (960, 308), (248, 248, 244))
    canvas.paste(left, (0, 38))
    canvas.paste(right, (480, 38))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 960, 38), fill=(30, 30, 30))
    draw.text((18, 11), "MuJoCo GT render", fill=(255, 255, 255))
    draw.text((498, 11), "Stage1 asset -> Stage2 contact graph comparison", fill=(255, 255, 255))
    return canvas


def main() -> None:
    args = parse_args()
    reports = read_json(args.contact_graph_frames.resolve())
    if args.max_frames > 0:
        reports = reports[: int(args.max_frames)]
    if not reports:
        raise ValueError("No reports to render.")
    bounds = scene_bounds(reports)
    frames = [
        np.asarray(make_comparison_frame(args.rgb_dir.resolve(), report, bounds))
        for report in reports
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output.resolve(), frames, duration=1.0 / max(1, int(args.fps)))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
