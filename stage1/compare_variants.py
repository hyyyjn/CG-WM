import ast
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image


def parse_cfg_args(cfg_path):
    if not cfg_path.exists():
        return {}
    text = cfg_path.read_text().strip()
    if not text.startswith("Namespace(") or not text.endswith(")"):
        return {}
    payload = text[len("Namespace("):-1]
    if not payload:
        return {}
    try:
        fake_call = ast.parse(f"f({payload})", mode="eval").body
    except SyntaxError:
        return {}
    parsed = {}
    for keyword in fake_call.keywords:
        parsed[keyword.arg] = ast.literal_eval(keyword.value)
    return parsed


def parse_training_args(training_args_path):
    if not training_args_path.exists():
        return {}
    try:
        return json.loads(training_args_path.read_text())
    except Exception:
        return {}


def compute_psnr(render_path, gt_path):
    render = np.asarray(Image.open(render_path).convert("RGB"), dtype=np.float32) / 255.0
    gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
    mse = np.mean((render - gt) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(-10.0 * np.log10(mse))


def summarize_split(model_dir, split_name):
    split_dir = model_dir / split_name
    if not split_dir.exists():
        return None

    method_dirs = sorted([path for path in split_dir.iterdir() if path.is_dir()])
    if not method_dirs:
        return None

    method_dir = method_dirs[-1]
    render_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    if not render_dir.exists() or not gt_dir.exists():
        return None

    render_files = sorted(render_dir.glob("*.png"))
    if not render_files:
        return None

    psnrs = []
    for render_file in render_files:
        gt_file = gt_dir / render_file.name
        if not gt_file.exists():
            continue
        psnrs.append(compute_psnr(render_file, gt_file))

    if not psnrs:
        return None

    return {
        "method": method_dir.name,
        "num_views": len(psnrs),
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_min": float(np.min(psnrs)),
        "psnr_max": float(np.max(psnrs)),
    }


def load_results_json(model_dir):
    results_path = model_dir / "results.json"
    if not results_path.exists():
        return None


def load_densification_stats(model_dir):
    stats_path = model_dir / "densification_stats.jsonl"
    if not stats_path.exists():
        return None

    rows = []
    for line in stats_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

    if not rows:
        return None

    last = rows[-1]
    return {
        "num_events": len(rows),
        "last_iteration": last.get("iteration"),
        "last_visible_points": last.get("visible_points"),
        "last_grad_mean": last.get("grad_mean"),
        "last_grad_max": last.get("grad_max"),
        "last_clone_selected": last.get("clone_selected"),
        "last_split_selected": last.get("split_selected"),
        "last_pruned": last.get("pruned"),
        "last_net_new_points": last.get("net_new_points"),
        "last_geometry_feature_loss": last.get("geometry_feature_loss"),
        "mean_clone_selected": float(np.mean([row.get("clone_selected", 0) for row in rows])),
        "mean_split_selected": float(np.mean([row.get("split_selected", 0) for row in rows])),
        "mean_pruned": float(np.mean([row.get("pruned", 0) for row in rows])),
        "mean_net_new_points": float(np.mean([row.get("net_new_points", 0) for row in rows])),
    }
    try:
        return json.loads(results_path.read_text())
    except Exception:
        return None


def latest_iteration(model_dir):
    point_cloud_root = model_dir / "point_cloud"
    if not point_cloud_root.exists():
        return None
    iterations = []
    for path in point_cloud_root.iterdir():
        if not path.is_dir() or not path.name.startswith("iteration_"):
            continue
        try:
            iterations.append(int(path.name.split("_")[-1]))
        except ValueError:
            pass
    if not iterations:
        return None
    return max(iterations)


def summarize_model(model_path):
    model_dir = Path(model_path).expanduser().resolve()
    cfg = parse_cfg_args(model_dir / "cfg_args")
    training_args = parse_training_args(model_dir / "training_args.json")
    merged_args = cfg.copy()
    merged_args.update(training_args)
    test_summary = summarize_split(model_dir, "test")
    train_summary = summarize_split(model_dir, "train")
    results_json = load_results_json(model_dir)
    densification_stats = load_densification_stats(model_dir)

    summary = {
        "model_path": str(model_dir),
        "model_name": model_dir.name,
        "latest_iteration": latest_iteration(model_dir),
        "sam_feature_weight": merged_args.get("sam_feature_weight"),
        "sam_feature_normalization": merged_args.get("sam_feature_normalization"),
        "joint_optimization": merged_args.get("joint_optimization"),
        "alternating_optimization": merged_args.get("alternating_optimization"),
        "test": test_summary,
        "train": train_summary,
        "results_json": results_json,
        "densification": densification_stats,
    }
    return summary


def print_summary_table(summaries):
    headers = [
        "model",
        "iter",
        "sam_w",
        "sam_norm",
        "mode",
        "test_psnr",
        "train_psnr",
        "densify_it",
        "events",
        "clone",
        "split",
        "pruned",
        "net_new",
        "avg_clone",
        "avg_split",
        "avg_pruned",
        "avg_net",
        "feat_loss",
    ]
    rows = []
    for item in summaries:
        mode = "joint" if item["joint_optimization"] else "alt" if item["alternating_optimization"] else "baseline"
        rows.append(
            [
                item["model_name"],
                "-" if item["latest_iteration"] is None else str(item["latest_iteration"]),
                "-" if item["sam_feature_weight"] is None else str(item["sam_feature_weight"]),
                "-" if item["sam_feature_normalization"] is None else str(item["sam_feature_normalization"]),
                mode,
                "-" if item["test"] is None else f'{item["test"]["psnr_mean"]:.4f}',
                "-" if item["train"] is None else f'{item["train"]["psnr_mean"]:.4f}',
                "-" if item["densification"] is None or item["densification"]["last_iteration"] is None else str(item["densification"]["last_iteration"]),
                "-" if item["densification"] is None else str(item["densification"]["num_events"]),
                "-" if item["densification"] is None else str(item["densification"]["last_clone_selected"]),
                "-" if item["densification"] is None else str(item["densification"]["last_split_selected"]),
                "-" if item["densification"] is None else str(item["densification"]["last_pruned"]),
                "-" if item["densification"] is None else str(item["densification"]["last_net_new_points"]),
                "-" if item["densification"] is None else f'{item["densification"]["mean_clone_selected"]:.1f}',
                "-" if item["densification"] is None else f'{item["densification"]["mean_split_selected"]:.1f}',
                "-" if item["densification"] is None else f'{item["densification"]["mean_pruned"]:.1f}',
                "-" if item["densification"] is None else f'{item["densification"]["mean_net_new_points"]:.1f}',
                "-" if item["densification"] is None or item["densification"]["last_geometry_feature_loss"] is None else f'{item["densification"]["last_geometry_feature_loss"]:.4f}',
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    fmt = "  ".join(f"{{:{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * width for width in widths]))
    for row in rows:
        print(fmt.format(*row))


if __name__ == "__main__":
    parser = ArgumentParser(description="Compare training variants from existing model folders.")
    parser.add_argument("--model_paths", "-m", nargs="+", required=True, help="Model output directories to compare.")
    parser.add_argument("--output_json", default="", type=str, help="Optional path to save the comparison summary JSON.")
    args = parser.parse_args()

    summaries = [summarize_model(model_path) for model_path in args.model_paths]
    print_summary_table(summaries)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved comparison summary to {output_path}")
