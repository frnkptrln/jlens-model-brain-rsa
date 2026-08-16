"""Regenerate the three-panel figure used in the sprint paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figure-1.png"

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 8.4,
        "axes.titlesize": 9.4,
        "axes.labelsize": 8.4,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

study_a = pd.read_csv(ROOT / "results" / "study_a" / "layer_summary.csv")
primary = pd.read_csv(
    ROOT / "results" / "study_b" / "confirmatory" / "primary_subject_contrasts.csv"
)
trajectory = pd.read_csv(
    ROOT / "results" / "study_b" / "extension" / "layer_trajectory_group.csv"
)

layers_a = study_a["layer"].to_numpy()
delta_a = study_a["delta_j_minus_raw"].to_numpy()
ci_lo = study_a["ci95_low"].to_numpy()
ci_hi = study_a["ci95_high"].to_numpy()
yerr = np.vstack([delta_a - ci_lo, ci_hi - delta_a])

fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.0), constrained_layout=True)

ax = axes[0]
colors = ["#2D6F72" if q < 0.05 else "#8B8B8B" for q in study_a["q_fdr"]]
ax.errorbar(
    layers_a,
    delta_a,
    yerr=yerr,
    fmt="none",
    ecolor="#505050",
    elinewidth=1.1,
    capsize=3,
    zorder=1,
)
ax.scatter(layers_a, delta_a, c=colors, s=34, zorder=2)
ax.axhline(0, color="#5A5A5A", linewidth=0.8)
ax.set_xticks(layers_a)
ax.set_xlabel("Qwen3.5-4B layer")
ax.set_ylabel("J - raw RSA")
ax.set_title("A  Searchlight study")
ax.text(
    0.02,
    0.98,
    "Late-layer gain\n(95% CI)",
    transform=ax.transAxes,
    va="top",
    ha="left",
    color="#4A4A4A",
)

ax = axes[1]
x = np.arange(1, len(primary) + 1)
vals = primary["mean_delta"].to_numpy()
ax.bar(x, vals, color="#A64B4B", width=0.72)
ax.axhline(0, color="#5A5A5A", linewidth=0.8)
ax.axhline(vals.mean(), color="#5E2020", linewidth=1.2, linestyle="--")
ax.set_xticks(x)
ax.set_xlabel("Participant")
ax.set_ylabel("J - raw RSA")
ax.set_title("B  ROI confirmatory study")
ax.text(
    0.03,
    0.05,
    f"mean = {vals.mean():.5f}\n8/8 negative",
    transform=ax.transAxes,
    va="bottom",
    ha="left",
    color="#5E2020",
)

ax = axes[2]
for prompt, color, label in [
    ("raw", "#2F5D8A", "Raw caption"),
    ("imagine", "#C47A32", "Imagery prompt"),
]:
    data = trajectory[trajectory["prompt"] == prompt].copy()
    data["layer"] = data["location"].str.extract(r"(\d+)").astype(int)
    data = data.sort_values("layer")
    ax.errorbar(
        data["layer"],
        data["mean_delta"],
        yerr=data["sem_delta"],
        marker="o",
        linewidth=1.5,
        markersize=4.2,
        capsize=2.5,
        color=color,
        label=label,
    )
ax.axhline(0, color="#5A5A5A", linewidth=0.8)
ax.set_xticks([9, 18, 27, 34])
ax.set_xlabel("Qwen3-4B layer")
ax.set_ylabel("J - raw RSA")
ax.set_title("C  Prompt x depth extension")
ax.legend(frameon=False, fontsize=7.4, loc="lower right")

fig.suptitle(
    "Dense Jacobian transport does not have a pipeline-invariant brain-alignment effect",
    fontsize=10.5,
    fontweight="bold",
    color="#3F3F3F",
)
fig.savefig(OUT, dpi=260, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(OUT)
