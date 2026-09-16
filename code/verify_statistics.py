"""Compare recomputed full E3 statistics with the manuscript reference tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    "e3_sample_structure.csv": ["level"],
    "e3_marker_means.csv": ["mode", "level", "marker", "comparison_role", "geometry"],
    "e3_route_comparisons.csv": ["mode", "route", "contrast", "metric"],
    "e3_flight_differences.csv": ["mode", "route", "contrast", "metric", "test"],
    "e3_correlations.csv": ["mode", "scope", "metric"],
}


def main():
    report = {}
    for name, keys in TABLES.items():
        actual = pd.read_csv(ROOT / "outputs/statistics" / name).sort_values(keys).reset_index(drop=True)
        expected = pd.read_csv(ROOT / "data/e3" / name).sort_values(keys).reset_index(drop=True)
        assert len(actual) == len(expected)
        assert actual[keys].equals(expected[keys])
        columns = expected.select_dtypes(include="number").columns
        maximum = 0.0
        for column in columns:
            assert np.allclose(actual[column], expected[column], atol=1e-8, rtol=0, equal_nan=True), (name, column)
            delta = np.abs(actual[column] - expected[column])
            maximum = max(maximum, float(delta.max()))
        report[name] = {"rows": len(actual), "maximum_numeric_difference": maximum}
    (ROOT / "outputs/statistics/verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("All five full E3 statistical tables agree with the supplied references.")


if __name__ == "__main__":
    main()
