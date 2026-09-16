from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/e3"
FIGURES = ROOT / "outputs/figures"
ROUTES = ["top", "mid", "bot"]
COLORS = {"top": "#E69F00", "mid": "#009E73", "bot": "#0072B2"}
SHAPES = {"top": "D", "mid": "^", "bot": "s"}
METRICS = ["vt_rmse_3d_cm", "visual_dispersion_3d_cm", "kt_rmse_3d_cm", "kalman_pos_std_3d_cm", "rpy_std_deg"]
LABELS = ["V-T 3D\nRMSE", "Visual position\ndispersion", "K-T 3D\nRMSE", "Kalman position\ndispersion", "Attitude\ndispersion"]
PRIMARY = "original_static"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 10, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})


def save(fig, stem):
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(FIGURES / (stem + "." + suffix), dpi=300, bbox_inches="tight", pad_inches=.05)
    plt.close(fig)

def heatmap(markers):
    positions = [[1, 2, 3, 4, 5], [6, None, 7, None, 8], [10, 11, 12, 13, 14]]
    fig, axes = plt.subplots(2, 1, figsize=(6.9, 4.8))
    fig.subplots_adjust(left=.13, right=.87, bottom=.13, top=.94, hspace=.55)
    panel_labels = []
    for k, (ax, metric, heading) in enumerate(zip(axes, METRICS[:2], ["V-T 3D RMSE", "Visual position dispersion"])):
        values = np.full((3, 5), np.nan)
        for row in range(3):
            for col, marker in enumerate(positions[row]):
                if marker is not None:
                    values[row, col] = markers.loc[marker, metric]
        cmap = plt.get_cmap("YlOrRd").copy()
        cmap.set_bad("#E8E8E8")
        lo, hi = np.floor(np.nanmin(values) * 2) / 2, np.ceil(np.nanmax(values) * 2) / 2
        im = ax.imshow(values, cmap=cmap, vmin=lo, vmax=hi, aspect="auto", interpolation="none")
        ax.set_yticks(range(3), ["TOP route", "MID route", "BOT route"])
        ax.set_xticks([0, 2, 4], ["left", "center", "right"])
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", length=0)
        for row in range(3):
            for col, marker in enumerate(positions[row]):
                if marker is not None:
                    value = values[row, col]
                    color = "white" if (value - lo) / (hi - lo) > .66 else "#151515"
                    ax.text(col, row, f"M{marker}\n{value:.2f}", ha="center", va="center", color=color, fontsize=9)
        ax.set_title(heading, fontsize=11, pad=6)
        cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.025)
        cb.set_label("cm")
        panel_labels.append(ax.text(.5, -.19 if k == 0 else -.30, f"({chr(97+k)})", transform=ax.transAxes, ha="center", va="top", fontweight="bold"))
    axes[1].set_xlabel("Marker position in the tunnel schematic", labelpad=3)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert not panel_labels[0].get_window_extent(renderer).overlaps(axes[1].title.get_window_extent(renderer))
    assert not panel_labels[1].get_window_extent(renderer).overlaps(axes[1].xaxis.label.get_window_extent(renderer))
    save(fig, "F8_Marker_Level_Visual_Outcomes")

def correlation_plot(windows, correlations):
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.9), sharex=True, sharey="col")
    fig.subplots_adjust(left=.12, right=.97, top=.87, bottom=.12, hspace=.65, wspace=.30)
    handles = [Line2D([], [], marker="o", linestyle="none", color=COLORS[r], label=r.upper()+" route") for r in ROUTES]
    handles += [Line2D([], [], marker="s", linestyle="none", color="#606A73", label="Compared locations"),
                Line2D([], [], marker="o", linestyle="none", color="#606A73", label="Reference locations")]
    fig.legend(handles=[handles[k] for k in [0, 3, 1, 4, 2]], loc="upper center", bbox_to_anchor=(.52, 1.0), ncol=3, frameon=False, fontsize=9)
    for i, mode in enumerate(["original_static", "attitude_absolute"]):
        frame = windows[windows["mode"].eq(mode)]
        for j, metric in enumerate(METRICS[3:]):
            ax = axes[i, j]
            for (route, role), part in frame.groupby(["level", "comparison_role"]):
                ax.scatter(part.visual_dispersion_3d_cm, part[metric], s=25,
                           c=COLORS[route], marker="s" if role == "compared" else "o",
                           edgecolors="white", linewidths=.35, alpha=.9, zorder=3)
            x, y = frame.visual_dispersion_3d_cm.to_numpy(), frame[metric].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            xx = np.array([x.min(), x.max()])
            ax.plot(xx, slope*xx+intercept, color="#646D78", linestyle="--", linewidth=1, zorder=2)
            row = correlations[correlations["mode"].eq(mode) & correlations.scope.eq("pooled") & correlations.metric.eq(metric)].iloc[0]
            ax.text(.04, .96, f"$r = {row.pearson_r:.3f}$", transform=ax.transAxes, va="top", fontsize=10)
            ax.grid(color="#E3E7EB", linewidth=.6)
            ax.set_ylabel("Kalman position\ndispersion (cm)" if j == 0 else "Attitude dispersion (deg)")
            ax.set_title("Static orientation" if i == 0 else "Recorded-attitude adjustment", fontsize=10)
            ax.set_xlabel("Visual position dispersion (cm)")
            ax.tick_params(labelbottom=True)
            ax.text(.5, -.29, f"({chr(97+2*i+j)})", transform=ax.transAxes, ha="center", va="top", fontweight="bold")
    save(fig, "F10_Visual_and_Flight_State_Concordance")

if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    markers = pd.read_csv(DATA / "e3_marker_means.csv")
    heatmap(markers[markers["mode"].eq(PRIMARY)].set_index("marker"))
    correlation_plot(pd.read_csv(DATA / "e3_all_window_outcomes.csv"), pd.read_csv(DATA / "e3_correlations.csv"))
