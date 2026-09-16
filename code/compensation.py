import numpy as np
import pandas as pd

KEYS = ["test", "level", "run", "segment", "marker"]

MARKERS = {
    "top": {
        1: np.array([1.2702, -1.7705, 2.21]),
        2: np.array([1.2705, -1.2405, 2.21]),
        3: np.array([1.2709, -0.6205, 2.22]),
        4: np.array([1.2714, 0.0895, 2.20]),
        5: np.array([1.2718, 0.6895, 2.21]),
    },
    "mid": {
        1: np.array([1.2701, -1.7805, 1.35]),
        2: np.array([1.2709, -0.6305, 1.35]),
        3: np.array([1.2718, 0.6795, 1.35]),
    },
    "bot": {
        1: np.array([1.2702, -1.7605, 0.41]),
        2: np.array([1.2705, -1.2455, 0.405]),
        3: np.array([1.2709, -0.6205, 0.41]),
        4: np.array([1.2714, 0.0995, 0.41]),
        5: np.array([1.2718, 0.6895, 0.415]),
    },
}

def rotation_zyx(roll_deg: np.ndarray, pitch_deg: np.ndarray, yaw_deg: np.ndarray) -> np.ndarray:
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rotations = np.empty((len(roll), 3, 3), dtype=float)
    rotations[:, 0, 0] = cy * cp
    rotations[:, 0, 1] = cy * sp * sr - sy * cr
    rotations[:, 0, 2] = cy * sp * cr + sy * sr
    rotations[:, 1, 0] = sy * cp
    rotations[:, 1, 1] = sy * sp * sr + cy * cr
    rotations[:, 1, 2] = sy * sp * cr - cy * sr
    rotations[:, 2, 0] = -sp
    rotations[:, 2, 1] = cp * sr
    rotations[:, 2, 2] = cp * cr
    return rotations

def marker_coordinate(level: str, segment: int) -> np.ndarray:
    return MARKERS[level][int(segment)]

def apply_compensation(samples: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for test, group in samples.groupby("test", sort=True):
        current = group.copy()
        marker_xyz = np.vstack(
            [
                marker_coordinate(str(level), int(segment))
                for level, segment in zip(current["level"], current["segment"])
            ]
        )
        original = current[["vision_x", "vision_y", "vision_z"]].to_numpy(dtype=float)
        body_to_marker_nominal = marker_xyz - original

        absolute_rotation = rotation_zyx(
            current["roll_deg"].to_numpy(dtype=float),
            current["pitch_deg"].to_numpy(dtype=float),
            current["yaw_deg"].to_numpy(dtype=float),
        )
        absolute_vector = np.einsum(
            "nij,nj->ni", absolute_rotation, body_to_marker_nominal
        )
        absolute_position = marker_xyz - absolute_vector

        reference = current[["roll_deg", "pitch_deg", "yaw_deg"]].median()
        reference_rotation = rotation_zyx(
            np.array([reference["roll_deg"]]),
            np.array([reference["pitch_deg"]]),
            np.array([reference["yaw_deg"]]),
        )[0]
        relative_rotation = np.einsum(
            "nij,jk->nik", absolute_rotation, reference_rotation.T
        )
        centered_vector = np.einsum(
            "nij,nj->ni", relative_rotation, body_to_marker_nominal
        )
        centered_position = marker_xyz - centered_vector

        current[["marker_x", "marker_y", "marker_z"]] = marker_xyz
        current[["absolute_x", "absolute_y", "absolute_z"]] = absolute_position
        current[["centered_x", "centered_y", "centered_z"]] = centered_position
        current["run_reference_roll_deg"] = reference["roll_deg"]
        current["run_reference_pitch_deg"] = reference["pitch_deg"]
        current["run_reference_yaw_deg"] = reference["yaw_deg"]
        parts.append(current)
    return pd.concat(parts, ignore_index=True)

def vector_rmse(errors: np.ndarray) -> tuple[float, float, float, float]:
    axis = np.sqrt(np.mean(errors**2, axis=0)) * 100.0
    combined = np.sqrt(np.mean(np.sum(errors**2, axis=1))) * 100.0
    return float(axis[0]), float(axis[1]), float(axis[2]), float(combined)

def vector_dispersion(positions: np.ndarray) -> tuple[float, float, float, float]:
    axis = np.std(positions, axis=0, ddof=1) * 100.0
    return float(axis[0]), float(axis[1]), float(axis[2]), float(np.linalg.norm(axis))

def summarize_windows(samples: pd.DataFrame) -> pd.DataFrame:
    position_columns = {
        "original_static": ["vision_x", "vision_y", "vision_z"],
        "attitude_absolute": ["absolute_x", "absolute_y", "absolute_z"],
        "attitude_run_centered": ["centered_x", "centered_y", "centered_z"],
    }
    rows = []
    for keys, group in samples.groupby(KEYS, sort=True):
        target = group[["target_x", "target_y", "target_z"]].to_numpy(dtype=float)
        kalman = group[["mocap_x_interp", "mocap_y_interp", "mocap_z_interp"]].to_numpy(dtype=float)
        for mode, columns in position_columns.items():
            position = group[columns].to_numpy(dtype=float)
            vt = vector_rmse(position - target)
            vk = vector_rmse(position - kalman)
            dispersion = vector_dispersion(position)
            rows.append(
                {
                    **dict(zip(KEYS, keys)),
                    "mode": mode,
                    "n_samples": len(group),
                    "vt_rmse_x_cm": vt[0],
                    "vt_rmse_y_cm": vt[1],
                    "vt_rmse_z_cm": vt[2],
                    "vt_rmse_3d_cm": vt[3],
                    "vk_rmse_x_cm": vk[0],
                    "vk_rmse_y_cm": vk[1],
                    "vk_rmse_z_cm": vk[2],
                    "vk_rmse_3d_cm": vk[3],
                    "visual_dispersion_x_cm": dispersion[0],
                    "visual_dispersion_y_cm": dispersion[1],
                    "visual_dispersion_z_cm": dispersion[2],
                    "visual_dispersion_3d_cm": dispersion[3],
                }
            )
    return pd.DataFrame(rows)

def compensate(samples: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for flight_id, group in samples.groupby("flight_id", sort=True):
        current = group.copy()
        original = current[["original_x", "original_y", "original_z"]].to_numpy(dtype=float)
        marker = current[["marker_x", "marker_y", "marker_z"]].to_numpy(dtype=float)
        nominal_vector = marker - original
        attitude_rotation = rotation_zyx(
            current["roll_deg"].to_numpy(dtype=float),
            current["pitch_deg"].to_numpy(dtype=float),
            current["yaw_deg"].to_numpy(dtype=float),
        )
        heading = float(current["nominal_heading_deg"].iloc[0])
        nominal_rotation = rotation_zyx(
            np.array([0.0]), np.array([0.0]), np.array([heading])
        )[0]
        absolute_delta = np.einsum(
            "ij,njk,kl->nil", nominal_rotation, attitude_rotation, nominal_rotation.T
        )
        absolute_position = marker - np.einsum(
            "nij,nj->ni", absolute_delta, nominal_vector
        )

        median_angles = current[["roll_deg", "pitch_deg", "yaw_deg"]].median()
        median_rotation = rotation_zyx(
            np.array([median_angles["roll_deg"]]),
            np.array([median_angles["pitch_deg"]]),
            np.array([median_angles["yaw_deg"]]),
        )[0]
        local_relative = np.einsum(
            "nij,jk->nik", attitude_rotation, median_rotation.T
        )
        centered_delta = np.einsum(
            "ij,njk,kl->nil", nominal_rotation, local_relative, nominal_rotation.T
        )
        centered_position = marker - np.einsum(
            "nij,nj->ni", centered_delta, nominal_vector
        )

        current[["absolute_x", "absolute_y", "absolute_z"]] = absolute_position
        current[["centered_x", "centered_y", "centered_z"]] = centered_position
        current["run_reference_roll_deg"] = median_angles["roll_deg"]
        current["run_reference_pitch_deg"] = median_angles["pitch_deg"]
        current["run_reference_yaw_deg"] = median_angles["yaw_deg"]
        parts.append(current)
    return pd.concat(parts, ignore_index=True)

def summarize_units(samples: pd.DataFrame) -> pd.DataFrame:
    position_columns = {
        "original_static": ["original_x", "original_y", "original_z"],
        "attitude_absolute": ["absolute_x", "absolute_y", "absolute_z"],
        "attitude_run_centered": ["centered_x", "centered_y", "centered_z"],
    }
    rows = []
    for (stage, flight_id, unit_id), group in samples.groupby(
        ["stage", "flight_id", "unit_id"], sort=True
    ):
        reference = group[["reference_x", "reference_y", "reference_z"]].to_numpy(dtype=float)
        target = group[["target_x", "target_y", "target_z"]].to_numpy(dtype=float)
        for mode, columns in position_columns.items():
            position = group[columns].to_numpy(dtype=float)
            vk = vector_rmse(position - reference)
            vt = vector_rmse(position - target)
            dispersion = vector_dispersion(position)
            rows.append(
                {
                    "stage": stage,
                    "flight_id": flight_id,
                    "unit_id": unit_id,
                    "mode": mode,
                    "n_samples": len(group),
                    "vk_rmse_x_cm": vk[0],
                    "vk_rmse_y_cm": vk[1],
                    "vk_rmse_z_cm": vk[2],
                    "vk_rmse_3d_cm": vk[3],
                    "vt_rmse_x_cm": vt[0],
                    "vt_rmse_y_cm": vt[1],
                    "vt_rmse_z_cm": vt[2],
                    "vt_rmse_3d_cm": vt[3],
                    "visual_dispersion_x_cm": dispersion[0],
                    "visual_dispersion_y_cm": dispersion[1],
                    "visual_dispersion_z_cm": dispersion[2],
                    "visual_dispersion_3d_cm": dispersion[3],
                }
            )
    return pd.DataFrame(rows)
