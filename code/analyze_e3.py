from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

KEYS = ["test", "level", "run", "segment", "marker"]

METRICS = ["vt_rmse_3d_cm", "visual_dispersion_3d_cm", "kt_rmse_3d_cm",
           "kalman_pos_std_3d_cm", "rpy_std_deg"]

ROUTES = ["top", "mid", "bot"]

GROUPS = {"top": ([2, 4], [1, 3, 5]), "mid": ([6, 7], [8]),
          "bot": ([11, 13], [10, 12, 14])}

MODES = ["original_static", "attitude_absolute", "attitude_run_centered"]

B = 20000

DIFF_SEED = 20260722

CORR_SEED = 20260907

def summarize_difference(values):
    d = np.asarray(values, dtype=float)
    assert len(d) == 5 and np.isfinite(d).all()
    rng = np.random.default_rng(DIFF_SEED)
    distribution = rng.choice(d, (B, len(d))).mean(axis=1)
    ci = np.quantile(distribution, [.025, .975])
    null = np.array(list(product([-1, 1], repeat=len(d)))) @ d / len(d)
    return dict(mean_difference=d.mean(), sd_difference=d.std(ddof=1),
                ci95_low=ci[0], ci95_high=ci[1], positive_flights=int((d > 0).sum()),
                n_flights=len(d), sign_flip_p=float(np.mean(np.abs(null) >= abs(d.mean()) - 1e-12)),
                bootstrap_resamples=B, bootstrap_seed=DIFF_SEED)

def weighted_correlation(x, y, weights, groups=None):
    if groups is None:
        groups = np.zeros(len(x))
    xx, yy, xy = weights @ (x * x), weights @ (y * y), weights @ (x * y)
    for label in np.unique(groups):
        mask = groups == label
        w = weights[:, mask]
        n = w.sum(axis=1)
        sx, sy = w @ x[mask], w @ y[mask]
        xx -= sx * sx / n
        yy -= sy * sy / n
        xy -= sx * sy / n
    denominator = np.sqrt(np.maximum(xx, 0) * np.maximum(yy, 0))
    return np.divide(xy, denominator, out=np.full_like(xy, np.nan), where=denominator > 1e-10)

def bootstrap_flights(frame):
    weights = np.zeros((B, len(frame)))
    rng = np.random.default_rng(CORR_SEED)
    for route in sorted(frame.level.unique()):
        flights = sorted(frame.loc[frame.level.eq(route), "test"].unique())
        assert len(flights) == 5
        draws = rng.integers(0, 5, (B, 5))
        for i, flight in enumerate(flights):
            weights[:, frame.test.eq(flight).to_numpy()] = (draws == i).sum(axis=1)[:, None]
    assert np.all(weights.sum(axis=1) == len(frame))
    return weights

def correlations(full):
    rows = []
    for mode, frame in full.groupby("mode"):
        frame = frame.sort_values(KEYS).reset_index(drop=True)
        weights = bootstrap_flights(frame)
        groups = (frame.level + "|" + frame.comparison_role).to_numpy()
        scopes = [("pooled", np.ones(len(frame), bool), False),
                  ("six_group_centered", np.ones(len(frame), bool), True)]
        scopes.extend(("route_" + route, frame.level.eq(route).to_numpy(), False) for route in ROUTES)
        scopes.extend((name, groups == name, False) for name in sorted(set(groups)))
        for scope, mask, centered in scopes:
            g = frame.loc[mask]
            w = weights[:, mask]
            labels = groups[mask] if centered else None
            x = g.visual_dispersion_3d_cm.to_numpy()
            for metric in METRICS[3:]:
                y = g[metric].to_numpy()
                r = weighted_correlation(x, y, np.ones((1, len(x))), labels)[0]
                draws = weighted_correlation(x, y, w, labels)
                finite = draws[np.isfinite(draws)]
                low, high = np.quantile(finite, [.025, .975])
                xx, yy = x.copy(), y.copy()
                if centered:
                    for label in np.unique(labels):
                        sel = labels == label
                        xx[sel] -= xx[sel].mean()
                        yy[sel] -= yy[sel].mean()
                assert abs(np.corrcoef(xx, yy)[0, 1] - r) < 1e-10
                rows.append(dict(mode=mode, scope=scope, metric=metric, pearson_r=r,
                                 ci95_low=low, ci95_high=high, n_windows=len(g),
                                 n_flights=g.test.nunique(), resamples=B, valid_resamples=len(finite),
                                 seed=CORR_SEED, resampling_unit="whole flight within route"))
    return pd.DataFrame(rows)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/statistics"

def save(frame, name):
    frame.to_csv(OUT / name, index=False, float_format="%.15g")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(ROOT / "data/e3/e3_all_window_outcomes.csv")
    baseline = full[full["mode"].eq("original_static")]
    counts = baseline.groupby("level").agg(flights=("test", "nunique"), windows=("test", "size"),
                                          paired_samples=("n_samples", "sum"),
                                          min_samples_per_window=("n_samples", "min"),
                                          max_samples_per_window=("n_samples", "max"))
    assert counts.flights.sum() == 15 and counts.windows.sum() == 65 and counts.paired_samples.sum() == 10679
    save(counts.reset_index(), "e3_sample_structure.csv")
    marker_means = full.groupby(["mode", "level", "marker", "comparison_role", "geometry"])[METRICS].mean().reset_index()
    save(marker_means, "e3_marker_means.csv")
    rows, details = [], []
    for (mode, route), group in full.groupby(["mode", "level"]):
        for metric in METRICS:
            pivot = group.pivot(index="test", columns="marker", values=metric).sort_index()
            comparisons = [("group", *GROUPS[route])]
            if route == "mid":
                comparisons += [("M6_minus_M8", [6], [8]), ("M7_minus_M8", [7], [8]),
                                ("M6_minus_M7", [6], [7])]
            for contrast, first, reference in comparisons:
                a, b = pivot[first].mean(axis=1), pivot[reference].mean(axis=1)
                d = a - b
                row = dict(mode=mode, route=route, contrast=contrast, metric=metric,
                           compared_mean=a.mean(), reference_mean=b.mean(), **summarize_difference(d))
                rows.append(row)
                for test in pivot.index:
                    details.append(dict(mode=mode, route=route, contrast=contrast, metric=metric,
                                        test=test, compared_mean=a.loc[test], reference_mean=b.loc[test],
                                        difference=d.loc[test]))
    summary, differences = pd.DataFrame(rows), pd.DataFrame(details)
    save(summary, "e3_route_comparisons.csv")
    save(differences, "e3_flight_differences.csv")

    save(correlations(full), "e3_correlations.csv")
    print("Recomputed E3 statistics for all 15 flights and 65 windows.")

if __name__ == "__main__":
    main()
