from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["test", "level", "run", "segment", "marker"]

GROUPS = {"top": ([2, 4], [1, 3, 5]), "mid": ([6, 7], [8]),
          "bot": ([11, 13], [10, 12, 14])}

METRICS = ["vt_rmse_3d_cm", "visual_dispersion_3d_cm", "vk_rmse_3d_cm",
           "kt_rmse_3d_cm", "kalman_pos_std_3d_cm", "rpy_std_deg"]

SHIFTS = np.array([-.20, -.15, -.10, -.05, 0., .05, .10, .15, .20])

MODES = ["original_static", "attitude_absolute", "attitude_run_centered"]

def rotation(a):
    r, p, y = np.deg2rad(a).T
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    m = np.empty((len(a), 3, 3))
    m[:, 0, 0], m[:, 0, 1], m[:, 0, 2] = cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr
    m[:, 1, 0], m[:, 1, 1], m[:, 1, 2] = sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr
    m[:, 2, 0], m[:, 2, 1], m[:, 2, 2] = -sp, cp*sr, cp*cr
    return m

def interp(data, t):
    assert t.min() >= data[0].min() and t.max() <= data[0].max()
    return np.column_stack([np.interp(t, data[0], data[c]) for c in [1, 2, 3]])

def rms(a):
    return float(np.sqrt(np.mean(np.sum(a*a, axis=1))))

def dispersion(a):
    return float(np.linalg.norm(np.std(a, axis=0, ddof=1)))

def centered_corr(frame, metric):
    g = frame.level + "|" + frame.marker.astype(int).map(
        lambda m: "compared" if m in [2, 4, 6, 7, 11, 13] else "reference")
    x, y = frame.visual_dispersion_3d_cm.copy(), frame[metric].copy()
    return np.corrcoef(x-x.groupby(g).transform("mean"), y-y.groupby(g).transform("mean"))[0, 1]
