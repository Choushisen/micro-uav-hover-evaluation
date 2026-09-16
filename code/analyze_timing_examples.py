"""Apply the archived time-offset diagnostic to the six shared E3 flights."""
import json
from timing_functions import *

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/timing_examples"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(ROOT / "examples/processed/e3_selected_samples.csv")
    files = pd.read_csv(ROOT / "examples/selected_flights.csv").set_index("flight_id")
    rows = []
    for test, flight in samples.groupby("test", sort=True):
        entry = files.loc[test]
        native = pd.read_csv(ROOT / entry.attitude_csv, header=None).sort_values(0, kind="stable")
        native[[1, 2, 3]] = np.rad2deg(np.unwrap(np.deg2rad(native[[1, 2, 3]].to_numpy()), axis=0))
        attitude = pd.read_csv(ROOT / entry.interpolation_support_csv, header=None, float_precision="round_trip")
        pos = pd.read_csv(ROOT / entry.position_csv, header=None).sort_values(0, kind="stable").drop_duplicates(0, keep="last")
        ref_angles = flight[["run_reference_roll_deg", "run_reference_pitch_deg", "run_reference_yaw_deg"]].iloc[0].to_numpy()
        ref = rotation(ref_angles[None, :])[0]
        for key, window in flight.groupby(KEYS, sort=True):
            t = window.time.to_numpy()
            marker = window[["marker_x", "marker_y", "marker_z"]].to_numpy()
            visual = window[["vision_x", "vision_y", "vision_z"]].to_numpy()
            target = window[["target_x", "target_y", "target_z"]].to_numpy()
            for shift in SHIFTS:
                query = t + shift
                kalman, angles = interp(pos, query), interp(attitude, query)
                rot = rotation(angles)
                positions = [visual, marker-np.einsum("nij,nj->ni", rot, marker-visual),
                             marker-np.einsum("nij,nj->ni", rot @ ref.T, marker-visual)]
                selected = native[(native[0] >= query.min()) & (native[0] <= query.max())]
                for mode, xyz in zip(MODES, positions):
                    rows.append(dict(zip(KEYS, key), shift_s=float(shift), mode=mode, n_samples=len(t),
                                     n_native_attitude=len(selected), vt_rmse_3d_cm=100*rms(xyz-target),
                                     visual_dispersion_3d_cm=100*dispersion(xyz), vk_rmse_3d_cm=100*rms(xyz-kalman),
                                     kt_rmse_3d_cm=100*rms(kalman-target), kalman_pos_std_3d_cm=100*dispersion(kalman),
                                     rpy_std_deg=dispersion(selected[[1, 2, 3]].to_numpy())))
    result = pd.DataFrame(rows)
    expected = pd.read_csv(ROOT / "data/timing/e3_shift_window_metrics.csv")
    check = result.merge(expected, on=KEYS+["mode", "shift_s"], suffixes=("", "_reference"), validate="one_to_one")
    assert len(check) == len(result)
    errors = {metric: float((check[metric]-check[metric+"_reference"]).abs().max()) for metric in METRICS}
    assert max(errors.values()) < 1e-7, errors
    result.to_csv(OUT / "e3_selected_shift_window_metrics.csv", index=False)
    (OUT / "verification.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    print("Nine-offset diagnostic reproduced for all six shared E3 flights.")


if __name__ == "__main__":
    main()
