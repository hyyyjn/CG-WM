"""Per-episode trajectory plots (x, y, z and rotation error) for a video physics fit report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = [("train", row) for row in report["results"]["train"]] + [
        ("holdout", row) for row in report["results"]["holdout"]]
    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 2.6 * len(rows)), squeeze=False)
    for (group, row), ax_row in zip(rows, axes):
        gt = np.asarray(row["ground_truth_body_positions"])
        pred = np.asarray(row["predicted_body_positions"])
        frames = np.arange(len(gt))
        for axis, label in enumerate("xyz"):
            ax = ax_row[axis]
            ax.plot(frames, gt[:, axis], color="black", lw=2, label="MuJoCo GT")
            ax.plot(frames, pred[:, axis], color="tab:green", lw=1.5, ls="--", label="Stage2 prediction")
            ax.set_ylabel(f"{label} [m]")
            if axis == 0:
                ax.set_title(f"{group} {row['episode']}", loc="left", fontsize=9)
        ax = ax_row[3]
        ax.plot(frames, np.asarray(row["per_frame_translation_m"]) * 100, color="tab:red", label="translation [cm]")
        ax.plot(frames, row["per_frame_rotation_deg"], color="tab:blue", label="rotation [deg]")
        ax.set_title(f"mean {row['translation_mean_m'] * 100:.2f} cm / {row['rotation_mean_deg']:.2f} deg", fontsize=9)
        ax.legend(fontsize=7)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("frame")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=110)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
