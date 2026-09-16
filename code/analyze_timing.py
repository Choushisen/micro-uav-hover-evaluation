from timing_functions import *

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/timing"

def save(frame, name):
    frame.to_csv(OUT / name, index=False, float_format="%.15g")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(ROOT / "data/timing/e3_shift_window_metrics.csv")
    detail, summaries, corrs, pooled = [], [], [], []
    for (shift, mode, route), g in full.groupby(["shift_s", "mode", "level"], sort=True):
        for metric in METRICS:
            p = g.pivot(index="test", columns="marker", values=metric)
            a, b = p[GROUPS[route][0]].mean(axis=1), p[GROUPS[route][1]].mean(axis=1)
            d = a-b
            summaries.append(dict(shift_s=shift, mode=mode, route=route, metric=metric,
                compared_mean=a.mean(), reference_mean=b.mean(), mean_difference=d.mean(),
                sd_difference=d.std(ddof=1), positive_flights=int((d>0).sum()), min_difference=d.min()))
            for test in p.index:
                detail.append(dict(shift_s=shift, mode=mode, route=route, test=test, metric=metric,
                                   difference=d.loc[test]))
    for (shift, mode), g in full.groupby(["shift_s", "mode"]):
        pooled.append(dict(shift_s=shift, mode=mode, n_samples=g.n_samples.sum(),
            pooled_vk_rmse_cm=np.sqrt(np.average(g.vk_rmse_3d_cm**2, weights=g.n_samples))))
        for metric in ["kalman_pos_std_3d_cm", "rpy_std_deg"]:
            corrs.append(dict(shift_s=shift, mode=mode, metric=metric,
                pooled_r=g[["visual_dispersion_3d_cm", metric]].corr().iloc[0, 1],
                six_group_centered_r=centered_corr(g, metric)))
    summary = pd.DataFrame(summaries)
    save(summary, "e3_shift_route_comparisons.csv")
    save(pd.DataFrame(detail), "e3_shift_flight_differences.csv")
    save(pd.DataFrame(corrs), "e3_shift_correlations.csv")
    save(pd.DataFrame(pooled), "e3_shift_pooled_agreement.csv")
    ranges = []
    for extent in [.05, .10, .15, .20]:
        part = summary[summary.shift_s.abs().le(extent+1e-9)]
        for (mode, route, metric), g in part.groupby(["mode", "route", "metric"]):
            ranges.append(dict(extent_s=extent, mode=mode, route=route, metric=metric,
                min_mean=g.mean_difference.min(), max_mean=g.mean_difference.max(),
                min_positive_flights=g.positive_flights.min(), max_positive_flights=g.positive_flights.max(),
                min_individual_difference=g.min_difference.min()))
    save(pd.DataFrame(ranges), "e3_shift_comparison_ranges.csv")

    print("All-flight time sensitivity summaries recomputed.")

if __name__ == "__main__":
    main()
