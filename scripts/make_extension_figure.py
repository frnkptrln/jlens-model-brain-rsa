"""Plot the exploratory prompt-by-depth J-lens-minus-raw RSA trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.table)
    frame["layer"] = frame["location"].str.removeprefix("layer_").astype(int)
    styles = {
        "raw": {"label": "Raw caption", "color": "#2563eb", "marker": "o"},
        "imagine": {"label": "Imagine prompt", "color": "#dc2626", "marker": "s"},
    }

    fig, axis = plt.subplots(figsize=(8.1, 4.8), constrained_layout=True)
    axis.axhline(0, color="#6b7280", linewidth=1, linestyle="--", zorder=0)
    for prompt, style in styles.items():
        subset = frame[frame["prompt"] == prompt].sort_values("layer")
        axis.errorbar(
            subset["layer"],
            subset["mean_delta"],
            yerr=subset["sem_delta"],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=6,
            capsize=3,
        )
    axis.set_xticks([9, 18, 27, 34])
    axis.set_xlabel("Qwen3-4B layer")
    axis.set_ylabel("Mean RSA difference (J-lens − raw)")
    axis.set_title("Dense Jacobian transport is prompt- and depth-dependent")
    axis.legend(frameon=False, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
