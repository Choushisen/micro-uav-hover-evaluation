# Data Dictionary

## Identifiers and units

`test`/`flight_id` identify one flight execution. `level`/`route` identify TOP, MID, or BOT. `run` is the repetition number. `segment` and `unit_id` identify the archived analysis interval within a flight. `marker` is the location label used in the manuscript. These labels are assigned by the planned route.

Raw and processed positions, targets, and marker coordinates are in metres. Summary RMSE and position-dispersion columns ending in `_cm` are in centimetres. Angles and attitude dispersion are in degrees. Host timestamps and offsets are in seconds. Raw CSV files have no header; processed files and summary tables have headers.

## Archived numeric flight files

| Filename token | Columns in file order |
| --- | --- |
| `_pos_` | Host timestamp, onboard Kalman X, Y, Z position. |
| `_att_` | Host timestamp, onboard roll, pitch, yaw. |
| `_posref_` | Host timestamp, logged position-reference X, Y, Z, where present in the archived flight. |
| `_det_` | Host timestamp, detected box center X, center Y, box width, box height (pixels). E2/E3 files include a sixth, second host-side timestamp. |

Files retain their original naming and rows. E1 files are supplied as preserved in the existing archive, including the archived trimming of position/detection records. E1/E2 reference records are provided when present in the selected archived flight. For E3, the targets used in analysis are stored explicitly in the processed sample table. `examples/selected_flights.csv` links examples to their position, attitude, and detection files. The detection logs use host timestamps, not hardware exposure timestamps; use the processed trajectory table as the starting point for the packaged calibrated-position calculations.

## Processed examples

`e3_selected_samples.csv` contains accepted samples from E3 test1/test2, with original visual position `vision_x/y/z`, interpolated Kalman position `mocap_x/y/z_interp`, target and marker coordinates, aligned roll/pitch/yaw, and the computed attitude variants. `time` is the visual host timestamp used for pairing. `time_basis` records the archived timing convention.

`baseline_selected_samples.csv` uses `original_x/y/z` for visual position and `reference_x/y/z` for Lighthouse-aided Kalman position. `timestamp` is the pairing time. `nominal_heading_deg` retains the nominal heading transformation used by the baseline analysis, including the E2-B 90-degree heading. The compensation functions preserve the archived coordinate transformation.

`absolute_x/y/z` and `centered_x/y/z` are the two attitude-adjusted trajectories. `run_reference_roll/pitch/yaw_deg` contain the flight reference angles used by the centered variant. These references are computed across the accepted samples of the flight and remain fixed in the time-offset diagnostic.

Each interpolation-support CSV has four columns: timestamp, roll, pitch, yaw. Angles are unwrapped and timestamps have been prepared for interpolation. The original support is supplied to preserve the treatment of repeated rounded timestamps in the archived processing.

`outputs/examples/e3_retained_sample_intervals.csv`, produced by the scripts, records the first and last retained observation and the sample count. These endpoints describe observed samples, not the commanded arrival/departure times. No analysis window is reselected by the package. `outputs/examples/baseline_example_unit_metrics.csv` contains metrics for the four selected E1/E2 example flights only. Full-study E1/E2 summaries are not included.

## Metrics and comparisons

| Field | Definition |
| --- | --- |
| `vt_rmse_3d_cm` | Square root of mean squared Euclidean distance from visual position to the target. |
| `kt_rmse_3d_cm` | Corresponding Kalman-to-target RMSE. |
| `vk_rmse_3d_cm` | Square root of mean squared visual-to-Kalman position difference. |
| `visual_dispersion_3d_cm` | Euclidean norm of the three sample standard deviations of visual position. |
| `kalman_pos_std_3d_cm` | Corresponding Kalman position dispersion. |
| `rpy_std_deg` | Euclidean norm of roll, pitch, and yaw sample standard deviations in the recorded attitude interval. |
| `n_samples`/`n_sync`/`pairs` | Number of accepted position pairs in the relevant interval or aggregate. |
| `comparison_role` | Compared or reference location within the same route. |

`original_static` is the primary visual trajectory. `attitude_absolute` applies the recorded-attitude adjustment; `attitude_run_centered` uses the flight-median-reference adjustment. All three modes are retained in the tables, so 195 E3 outcome rows represent 65 windows, not 195 independent observations. Complementary Kalman/attitude outcomes in `data/e3/` are reused across visual modes. The time-offset diagnostic additionally re-queries the reference streams and shifts the native attitude interval.

Configuration groups follow the manuscript: TOP compares M2/M4 with M1/M3/M5; MID compares M6/M7 with M8; BOT compares M11/M13 with M10/M12/M14. The MID table also includes the recorded individual-location contrasts. `difference` is the compared-location mean minus the reference-location mean within one flight. `positive_flights` counts positive differences among the five flights of that route.

Difference intervals use 20,000 bootstrap resamples of the five flight differences. Correlation intervals resample whole flights within each route, retaining their windows together. Seeds and sample sizes are included in the statistical tables and code. A pooled RMSE uses sample-count weights on squared RMSE before taking the square root. A mean of flight/window RMSE values gives equal weight to those units. The observed minimum/maximum range is not a confidence interval.

## Timing, detection, and supporting tables

`shift_s` is added to the visual host time before reference interpolation. Negative offsets query earlier states. `extent_s` describes the largest absolute offset included in a summary. Attitude adjustments retain the original flight reference angles. Static V-T and static visual dispersion do not change when only the reference time is shifted.

Whole-video detection tables use `detection_records / video_frames`, with one detection record corresponding to one recorded frame. Within-window tables use `boxed_frames / span_frames`; `span_frames` includes both detected endpoint frames. `unboxed_frames` counts frames without a box between those endpoints. `gap_events.csv` records contiguous unboxed intervals. These stored frame-audit results are provided without the source videos.

Screen-recording `display_lag_ms` equals `(direct_clock_s - returned_clock_s) * 1000` for readable clock pairs. Clock-pair image labels refer to the source audit; those images are not included. The measured display lag and the diagnostic timestamp offsets are distinct quantities.

Practical tables decompose E3 mean squared target error into squared mean offset plus mean squared fluctuation, with axis-wise values. E3 commanded holds are 30 s per planned hover location; accepted sample intervals are stored separately. The original position-reference logs for the selected E1/E2 examples are retained, but the full-study reference-hold and visit-sequence summary tables are not included.
