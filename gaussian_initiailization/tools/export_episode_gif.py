import argparse
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Export an episode RGB folder to a shareable GIF.")
    parser.add_argument("--rgb_dir", required=True, type=str)
    parser.add_argument("--output_gif", required=True, type=str)
    parser.add_argument("--fps", default=15, type=int)
    parser.add_argument("--max_frames", default=0, type=int, help="0 means use all frames.")
    parser.add_argument("--resize_width", default=0, type=int, help="0 means keep original size.")
    return parser.parse_args()


def main():
    args = parse_args()
    rgb_dir = Path(args.rgb_dir).expanduser().resolve()
    output_gif = Path(args.output_gif).expanduser().resolve()
    frames = sorted(path for path in rgb_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames found in {rgb_dir}")

    if args.max_frames > 0:
        frames = frames[: int(args.max_frames)]

    pil_frames = []
    for frame_path in frames:
        image = Image.open(frame_path).convert("RGB")
        if args.resize_width > 0:
            width, height = image.size
            scale = float(args.resize_width) / float(width)
            image = image.resize((int(args.resize_width), max(1, int(round(height * scale)))), Image.BILINEAR)
        pil_frames.append(image)

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / float(args.fps))))
    pil_frames[0].save(
        output_gif,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(str(output_gif))


if __name__ == "__main__":
    main()
