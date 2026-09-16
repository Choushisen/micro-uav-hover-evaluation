"""Aggregate archived detection counts and screen-recording clock readings."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/supporting"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    flight = pd.read_csv(ROOT / "data/detection/e3_flight_detection_frame_proportions.csv")
    window = pd.read_csv(ROOT / "data/detection/window_results.csv")
    readings = pd.read_csv(ROOT / "data/latency/screen_latency_samples.csv")
    lag = pd.to_numeric(readings.display_lag_ms, errors="coerce").dropna()
    report = {
        "whole_video": {"flights": len(flight), "detected_frames": int(flight.detection_records.sum()),
                        "total_frames": int(flight.video_frames.sum()),
                        "detected_frame_fraction": float(flight.detection_records.sum()/flight.video_frames.sum())},
        "within_observed_window_endpoints": {"windows": len(window), "boxed_frames": int(window.boxed_frames.sum()),
                                             "unboxed_frames": int(window.unboxed_frames.sum()), "span_frames": int(window.span_frames.sum()),
                                             "boxed_frame_fraction": float(window.boxed_frames.sum()/window.span_frames.sum())},
        "screen_recording_display_lag_ms": {"readings": len(lag), "mean": float(lag.mean()), "sd": float(lag.std(ddof=1)),
                                            "minimum": float(lag.min()), "median": float(lag.median()),
                                            "p95": float(lag.quantile(.95)), "maximum": float(lag.max())}}
    assert report["within_observed_window_endpoints"]["unboxed_frames"] == 613
    assert (window.boxed_frames + window.unboxed_frames == window.span_frames).all()
    for _, row in readings.dropna(subset=["direct_clock_s", "returned_clock_s", "display_lag_ms"]).iterrows():
        assert abs((row.direct_clock_s-row.returned_clock_s)*1000-row.display_lag_ms) < 1e-7
    (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, keys in {
        "e3_shift_route_comparisons.csv": ["shift_s", "mode", "route", "metric"],
        "e3_shift_flight_differences.csv": ["shift_s", "mode", "route", "test", "metric"],
        "e3_shift_pooled_agreement.csv": ["shift_s", "mode"],
        "e3_shift_correlations.csv": ["shift_s", "mode", "metric"],
        "e3_shift_comparison_ranges.csv": ["extent_s", "mode", "route", "metric"],
    }.items():
        actual = pd.read_csv(ROOT / "outputs/timing" / name).sort_values(keys).reset_index(drop=True)
        expected = pd.read_csv(ROOT / "data/timing" / name).sort_values(keys).reset_index(drop=True)
        assert len(actual) == len(expected) and actual[keys].equals(expected[keys])
        for column in expected.select_dtypes(include="number").columns:
            assert np.allclose(actual[column], expected[column], atol=1e-8, rtol=0, equal_nan=True), (name, column)
    print("Detection, display-lag, and complete time-sensitivity summaries verified.")


if __name__ == "__main__":
    main()
