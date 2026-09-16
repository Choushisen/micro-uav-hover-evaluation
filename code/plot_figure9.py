"""Redraw existing within-flight differences using tighter route-specific x axes.

Run: python plot_figure9.py
Means, bootstrap intervals, and all individual differences are read unchanged.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd


# EDIT HERE: change limits to zoom; values and interval endpoints remain checked.
FIGSIZE = (10.8, 5.3)
FONT_SIZE = 10
FONT_FAMILY = "DejaVu Serif"
DPI = 300
POSITION_XLIMS = {"top": (-0.12, 2.65), "mid": (-0.98, 0.40), "bot": (-0.22, 1.48)}
ATTITUDE_XLIMS = {"top": (-0.20, 5.20), "mid": (-3.50, 0.20), "bot": (-0.10, 1.20)}
POSITION_TICK_STEPS = {"top": 0.5, "mid": 0.25, "bot": 0.4}
ATTITUDE_TICK_STEPS = {"top": 1.0, "mid": 1.0, "bot": 0.25}
# Set True to use identical horizontal scales across routes for each measurement type.
USE_COMMON_X_LIMITS = False
COMMON_POSITION_XLIM = (-1.0, 2.65)
COMMON_ATTITUDE_XLIM = (-3.5, 5.2)
COLORS = {"top": "#E69F00", "mid": "#009E73", "bot": "#0072B2"}
SHAPES = {"top": "D", "mid": "^", "bot": "s"}
FLIGHT_POINT_SIZE = 24
MEAN_MARKER_SIZE = 7
INTERVAL_LINEWIDTH = 2
JITTER_HEIGHT = 0.105
LEFTS = [0.185, 0.465, 0.745]
PANEL_WIDTH = 0.237
POSITION_BOTTOM_HEIGHT = (0.39, 0.43)
ATTITUDE_BOTTOM_HEIGHT = (0.13, 0.12)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/e3"
OUT = ROOT / "outputs/figures"
ROUTES = ["top", "mid", "bot"]
METRICS = ["vt_rmse_3d_cm", "visual_dispersion_3d_cm", "kt_rmse_3d_cm", "kalman_pos_std_3d_cm", "rpy_std_deg"]
LABELS = ["V-T 3D\nRMSE", "Visual position\ndispersion", "K-T 3D\nRMSE",
          "Kalman position\ndispersion", "Attitude\ndispersion"]


def compact_tick(value, _position):
    if abs(value) < 1e-10:
        return "0"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def main():
    OUT.mkdir(exist_ok=True)
    summary = pd.read_csv(DATA / "e3_route_comparisons.csv")
    flights = pd.read_csv(DATA / "e3_flight_differences.csv")
    summary = summary[summary["mode"].eq("original_static") & summary.contrast.eq("group")]
    flights = flights[flights["mode"].eq("original_static") & flights.contrast.eq("group")]
    assert len(summary) == 15 and len(flights) == 75
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": FONT_SIZE,
                         "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE - 1,
                         "ytick.labelsize": FONT_SIZE, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=FIGSIZE)
    handles = [Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor="#66717C",
                      linestyle="none", label="Flight difference"),
               Line2D([], [], marker="D", color="#66717C", linewidth=2,
                      label="Mean and 95% bootstrap interval")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.57, 0.995),
               frameon=False, ncol=2, fontsize=FONT_SIZE)
    if not USE_COMMON_X_LIMITS:
        fig.text(0.57, 0.91, "Horizontal scales differ by route", ha="center", va="center",
                 fontsize=FONT_SIZE - 1, color="#555555")
    audit = []
    for j, route in enumerate(ROUTES):
        pos = fig.add_axes([LEFTS[j], POSITION_BOTTOM_HEIGHT[0], PANEL_WIDTH, POSITION_BOTTOM_HEIGHT[1]])
        att = fig.add_axes([LEFTS[j], ATTITUDE_BOTTOM_HEIGHT[0], PANEL_WIDTH, ATTITUDE_BOTTOM_HEIGHT[1]])
        for axis, metrics, kind in [(pos, METRICS[:4], "position"), (att, METRICS[4:], "attitude")]:
            if kind == "position":
                limits = COMMON_POSITION_XLIM if USE_COMMON_X_LIMITS else POSITION_XLIMS[route]
                step = 0.8 if USE_COMMON_X_LIMITS else POSITION_TICK_STEPS[route]
            else:
                limits = COMMON_ATTITUDE_XLIM if USE_COMMON_X_LIMITS else ATTITUDE_XLIMS[route]
                step = 2.0 if USE_COMMON_X_LIMITS else ATTITUDE_TICK_STEPS[route]
            assert limits[0] < 0 < limits[1], "The zero reference must remain visible."
            for i, metric in enumerate(metrics):
                y = len(metrics) - 1 - i
                row = summary[summary.route.eq(route) & summary.metric.eq(metric)].iloc[0]
                d = flights[flights.route.eq(route) & flights.metric.eq(metric)].sort_values("test").difference.to_numpy()
                assert len(d) == 5 and np.isfinite(d).all()
                np.testing.assert_allclose(d.mean(), row.mean_difference, rtol=0, atol=1e-10)
                visible = np.r_[d, row.mean_difference, row.ci95_low, row.ci95_high, 0]
                assert visible.min() > limits[0] and visible.max() < limits[1], (route, metric, limits)
                axis.scatter(d, y + np.linspace(-JITTER_HEIGHT, JITTER_HEIGHT, 5), s=FLIGHT_POINT_SIZE,
                             facecolors="white", edgecolors=COLORS[route], linewidths=1.1, zorder=4)
                axis.errorbar(row.mean_difference, y,
                              xerr=[[row.mean_difference - row.ci95_low], [row.ci95_high - row.mean_difference]],
                              marker=SHAPES[route], color=COLORS[route], markersize=MEAN_MARKER_SIZE,
                              markeredgecolor="white", linewidth=INTERVAL_LINEWIDTH, capsize=4, zorder=5)
                audit.append(dict(route=route, metric=metric, xmin=limits[0], xmax=limits[1],
                                  observed_min=float(d.min()), observed_max=float(d.max()),
                                  mean=row.mean_difference, ci_low=row.ci95_low, ci_high=row.ci95_high))
            axis.set_xlim(*limits)
            axis.set_ylim(-0.5, len(metrics) - 0.5)
            ticks = np.arange(np.ceil(limits[0] / step), np.floor(limits[1] / step) + 1) * step
            axis.xaxis.set_major_locator(FixedLocator(ticks))
            axis.xaxis.set_major_formatter(FuncFormatter(compact_tick))
            axis.grid(axis="x", color="#DCE1E6", linewidth=0.7)
            axis.axvline(0, color="#737C86", linewidth=1, linestyle="--", zorder=2)
            axis.spines["left"].set_visible(False)
            axis.spines["bottom"].set_color("#858E99")
            axis.tick_params(axis="y", length=0, pad=8)
            axis.set_yticks(range(len(metrics)))
            labels = list(reversed(LABELS[:4])) if kind == "position" else [LABELS[4]]
            axis.set_yticklabels(labels if j == 0 else [""] * len(metrics))
            axis.set_xlabel("Difference (cm)" if kind == "position" else "Difference (deg)", labelpad=4)
        pos.set_title(route.upper() + " route", fontsize=FONT_SIZE + 2, fontweight="bold", color=COLORS[route], pad=12)
        fig.text(LEFTS[j] + PANEL_WIDTH / 2, 0.015, f"({chr(97 + j)})", ha="center", va="bottom", fontweight="bold")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.width and box.height:
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, text.get_text()
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"F9_Matched_Configuration_Differences.{extension}", dpi=DPI, facecolor="white")
    plt.close(fig)
    pd.DataFrame(audit).to_csv(OUT / "F9_axis_and_value_checks.csv", index=False)
    report = dict(route_specific_x_axes=not USE_COMMON_X_LIMITS, flight_points=75, summary_intervals=15,
                  all_data_and_interval_endpoints_visible=True, zero_visible_in_all_panels=True,
                  statistics_recomputed=False, source_sha256={name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
                                                              for name in ["e3_route_comparisons.csv", "e3_flight_differences.csv"]})
    (OUT / "F9_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
