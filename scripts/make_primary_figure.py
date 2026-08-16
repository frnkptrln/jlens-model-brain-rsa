"""Plot participant-level J-lens-minus-raw RSA contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROI_ORDER = ("ventral", "lateral", "parietal")
LAYER_ORDER = ("layer_18", "layer_27")


def make_figure(analysis_dir: Path, output: Path) -> None:
    paired = pd.read_csv(analysis_dir / "paired_contrasts.csv")
    primary = pd.read_csv(analysis_dir / "primary_subject_contrasts.csv")
    primary_test = json.loads((analysis_dir / "primary_test.json").read_text())

    series: list[tuple[str, np.ndarray]] = []
    for roi in ROI_ORDER:
        for layer in LAYER_ORDER:
            values = (
                paired.loc[
                    (paired["family"] == "confirmatory_component")
                    & (paired["roi"] == roi)
                    & (paired["location"] == layer),
                    ["subject", "delta_jlens_minus_raw"],
                ]
                .sort_values("subject")["delta_jlens_minus_raw"]
                .to_numpy(float)
            )
            if values.size != 8:
                raise ValueError(f"expected 8 participants for {roi}/{layer}")
            series.append((f"{roi.title()}\nL{layer.removeprefix('layer_')}", values))
    aggregate = primary.sort_values("subject")["mean_delta"].to_numpy(float)
    if aggregate.size != 8:
        raise ValueError("expected 8 participant-level primary contrasts")
    series.append(("Primary\nmean", aggregate))

    figure, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    jitter = np.linspace(-0.15, 0.15, 8)
    for index, (_label, values) in enumerate(series):
        point_color = "#287c8e" if index < len(series) - 1 else "#d07a26"
        axis.scatter(
            np.full(values.size, index) + jitter,
            values,
            s=34,
            color=point_color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
        axis.errorbar(
            index,
            mean,
            yerr=sem,
            fmt="D",
            markersize=6,
            color="#1b1b1b",
            capsize=4,
            linewidth=1.5,
            zorder=4,
        )

    axis.axhline(0, color="#666666", linewidth=1, linestyle="--", zorder=1)
    axis.set_xticks(range(len(series)), [label for label, _ in series])
    axis.set_ylabel("RSA difference (J-lens - raw)")
    axis.set_title("Participant-level Jacobian-transport contrasts")
    axis.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.99,
        0.02,
        f"Points: NSD participants; diamonds: mean +/- SEM. "
        f"Primary exact two-sided sign-flip p = {primary_test['p_signflip_two_sided']:.4f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_figure(args.analysis_dir, args.output)


if __name__ == "__main__":
    main()
