"""Recompute attitude variants and metrics from the selected aligned trajectories."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from compensation import apply_compensation, compensate, summarize_windows, summarize_units

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/examples"


def compare(frame, expected, keys, columns, tolerance=1e-7):
    paired = frame.merge(expected[keys + columns], on=keys, suffixes=("", "_reference"), validate="one_to_one")
    assert len(paired) == len(frame)
    errors = {name: float(np.max(np.abs(paired[name] - paired[name + "_reference"]))) for name in columns}
    assert max(errors.values()) < tolerance, errors
    return errors


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    e3 = pd.read_csv(ROOT / "examples/processed/e3_selected_samples.csv")
    baseline = pd.read_csv(ROOT / "examples/processed/baseline_selected_samples.csv")
    positions = [prefix + "_" + axis for prefix in ("absolute", "centered") for axis in "xyz"]
    e3_new = apply_compensation(e3)
    baseline_new = compensate(baseline)
    report = {"e3_compensated_position_max_error_m": compare(e3_new, e3, ["test", "segment", "time"], positions),
              "baseline_compensated_position_max_error_m": compare(baseline_new, baseline, ["unit_id", "timestamp"], positions)}
    e3_metrics = summarize_windows(e3_new)
    baseline_metrics = summarize_units(baseline_new)
    report["e3_metric_max_error_cm"] = compare(e3_metrics, pd.read_csv(ROOT / "data/e3/e3_all_window_outcomes.csv"),
                                              ["test", "segment", "mode"],
                                              ["vt_rmse_3d_cm", "visual_dispersion_3d_cm", "vk_rmse_3d_cm"])
    e3_metrics.to_csv(OUT / "e3_recomputed_window_metrics.csv", index=False)
    baseline_metrics.to_csv(OUT / "baseline_example_unit_metrics.csv", index=False)
    e3.groupby(["test", "level", "run", "segment", "marker"]).agg(
        first_retained_time=("time", "min"), last_retained_time=("time", "max"), retained_samples=("time", "size")
    ).reset_index().to_csv(OUT / "e3_retained_sample_intervals.csv", index=False)
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Selected trajectory positions and E3 metrics verified; E1/E2 metrics cover the supplied examples only.")


if __name__ == "__main__":
    main()
