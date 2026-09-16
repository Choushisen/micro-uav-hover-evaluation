# Data and Analysis Code

Associated manuscript: **Screening Tunnel Configurations for Reduced Micro-UAV Hover Performance Using Marker-Based Visual Measurements**.

This package provides selected flight records from Experiments I-III and complete numerical inputs for the main Experiment III analyses. Experiment I/II full-study summary tables are not included. Raw-record examples and all-flight statistical inputs are stored separately.

## Contents

| Directory | Scope |
| --- | --- |
| `examples/E3/` | Archived numeric records for test1 and test2 on each of TOP, MID, and BOT: six individual flights. |
| `examples/E1_fixed_1p2/`, `examples/E1_fixed_1p5/` | One archived example at each E1 target height. |
| `examples/E2_vertical/`, `examples/E2_horizontal/` | One archived example for each E2 task. |
| `examples/processed/` | Calibrated, aligned trajectory samples for these ten examples, including attitude, targets, and marker coordinates. |
| `examples/interpolation_support/` | The timestamp/attitude interpolation support used for the six E3 examples. |
| `data/e3/` | Complete results for 15 flights and 65 hover windows, including location summaries, within-flight differences, bootstrap intervals, and correlations. |
| `data/timing/` | All-flight window metrics and summaries for offsets of 0, +/-0.05, +/-0.10, +/-0.15, and +/-0.20 s. |
| `data/detection/` | Whole-video detection counts and within-window boxed/unboxed-frame counts. |
| `data/latency/` | Clock readings and derived display-lag counts from the screen recording. |
| `data/practical/` | E3 position-error decomposition tables. |
| `code/` | Portable analysis and plotting scripts. |

E3 test1 and test2 were selected by repetition number, not by measured performance. Each test number identifies one flight on each route; the three routes are not one continuous flight. E1/E2 examples are identified in `examples/selected_flights.csv`. All archived rows of each supplied numeric flight file are retained. Full videos, detector-training datasets, model weights, and raw records for the remaining flights are outside this package.

## Reproduction

The analysis scripts were prepared for Python 3.9 with NumPy 1.21.5, pandas 1.4.4, and Matplotlib 3.5.2. These versions refer to the analysis environment, not the detector-training environment.

```bash
python -m pip install -r requirements.txt
python code/run_all.py
```

Use `python code/run_all.py --skip-plots` to run numerical checks only. Outputs are written to `outputs/`; input files are not changed. No network access or GPU is required after dependencies have been installed. Scripts use paths relative to the package, not the original author's computer.

The reproduction starting point for visual-position calculations is the supplied calibrated and aligned trajectory table. `process_examples.py` recomputes both attitude variants and window metrics for the selected examples. Its E1/E2 output contains example results only, not full-study summaries. `analyze_timing_examples.py` interpolates the shared position and attitude records at the specified offsets. `analyze_e3.py` recomputes full-study configuration statistics from all 65 window records. `analyze_timing.py` aggregates the all-flight offset tables. Detector inference and the original camera acquisition program are not dependencies of these scripts.

The two E3 repetitions serve as examples of the processing procedure. The manuscript's full-study comparisons use `data/e3/e3_all_window_outcomes.csv`, which includes all five flights per route. Figures 8-10 use the complete E3 tables. Figure 7 and the E1/E2 full-study aggregation scripts are not included. Plotting scripts retain the existing figure definitions and expose sizing parameters for editing.

`verify_statistics.py` checks reproduced E3 tables against the provided reference tables. The other analysis scripts also check their results against the corresponding shared summaries. `manifest.csv` lists package files and their SHA-256 hashes.

## Interpretation

Position records denoted `mocap_*` in legacy column names are motion-capture-aided onboard Kalman positions. V-T is visual-to-target RMSE; V-K is visual-to-Kalman agreement. Position dispersion is computed about the trajectory's own mean, with sample standard deviation. These quantities have different references.

Timing offsets query the Kalman and attitude records at visual host timestamp plus offset. They are a time-sensitivity diagnostic. Detection-frame proportions are counts over recorded video frames, not annotated object-detection recall. The within-window denominator spans the first to last detected frame in each retained hover interval.

See `DATA_DICTIONARY.md` for identifiers, units, coordinate fields, and mode definitions.

## License

The analysis and plotting code is licensed under the MIT License
(see LICENSE).

The experimental data in data/ are licensed under CC BY 4.0
(see data/LICENSE).

Please cite the associated paper when using these materials
in research.
