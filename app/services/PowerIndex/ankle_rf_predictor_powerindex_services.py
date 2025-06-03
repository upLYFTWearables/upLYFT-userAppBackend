import os
import numpy as np
import pandas as pd
from joblib import load
from scipy.signal       import butter, filtfilt, find_peaks
from scipy.ndimage      import gaussian_filter1d

# ──────────────────────────────────────────────────────────────
# II.  FILTERING
# ──────────────────────────────────────────────────────────────
def heavy_smooth(sig, fs,
                 lp_cut=12.0, lp_order=4,
                 gauss_sigma_ms=50.0, gauss_passes=2):
    b, a = butter(lp_order, lp_cut/(0.5*fs), btype='low')
    out  = filtfilt(b, a, sig)

    sigma = gauss_sigma_ms * 1e-3 * fs
    for _ in range(gauss_passes):
        out = gaussian_filter1d(out, sigma=sigma, mode='nearest')
    return out


# ──────────────────────────────────────────────────────────────
# III.  PEAK PICKING
# ──────────────────────────────────────────────────────────────
def detect_big_positive_peaks(power_smoothed, fs,
                              min_prominence=None,
                              min_distance_s=1.0):
    if min_prominence is None:
        rng = np.percentile(power_smoothed, 95) - np.percentile(power_smoothed, 5)
        min_prominence = 0.15 * rng

    idx, _ = find_peaks(power_smoothed,
                        prominence=min_prominence,
                        distance=int(min_distance_s*fs))
    if len(idx) < 3:
        raise RuntimeError("Not enough peaks – adjust thresholds.")
    return idx


# ──────────────────────────────────────────────────────────────
# IV.  WINDOW-BASED AVERAGING  ← NEW
# ──────────────────────────────────────────────────────────────
def extract_windows_around_peaks(sig, peak_idx, fs,
                                 half_window_s=1.25,
                                 require_isolated=True):
    """
    Cut out ± half_window_s around every peak.

    A window is kept only if:
      • it fits inside the signal boundaries
      • (optionally) *no other peak* lies inside that window
    Returns list of 1-D numpy arrays (all same length).
    """
    hw = int(round(half_window_s * fs))
    win_len = 2 * hw + 1             # samples per window
    windows = []

    for p in peak_idx:
        start = p - hw
        end   = p + hw + 1           # Python slice is exclusive on the right
        if start < 0 or end > len(sig):
            continue                 # window would run off the array

        if require_isolated:
            others = peak_idx[(peak_idx != p) &
                               (peak_idx > start) &
                               (peak_idx < end)]
            if len(others):
                continue             # another peak contaminates the window

        windows.append(sig[start:end])

    if not windows:
        raise RuntimeError("No clean windows extracted – relax isolation rules?")
    return np.stack(windows)         # shape (n_windows, win_len)


def average_windows(windows, num_points=200):
    """
    Resample each window (2.5 s) to num_points with the peak landing at 50 %,
    then average across windows.
    """
    n_win, win_len = windows.shape
    x_old = np.linspace(0, 100, win_len)
    x_new = np.linspace(0, 100, num_points)

    resampled = [np.interp(x_new, x_old, w) for w in windows]
    return np.mean(resampled, axis=0)


def compute_propulsion_absorption(curve):
    pos = np.where(curve > 0, curve, 0)
    neg = np.where(curve < 0, curve, 0)
    return np.trapz(pos), np.trapz(neg)

MODEL_PATH = "models/ankle_rf_model.joblib"
RF_FEATURES = [
    'foot_AccelX','foot_AccelY','foot_AccelZ','foot_GyroX','foot_GyroY','foot_GyroZ',
    'shank_AccelX','shank_AccelY','shank_AccelZ','shank_GyroX','shank_GyroY','shank_GyroZ'
]

def predict_ankle_power(foot_csv_path, shank_csv_path, fs=100.0):
    rf = load(MODEL_PATH)

    foot = pd.read_csv(foot_csv_path)
    shank = pd.read_csv(shank_csv_path)

    foot.rename(columns={k: f"foot_{k}" for k in ['AccelX','AccelY','AccelZ','GyroX','GyroY','GyroZ']}, inplace=True)
    shank.rename(columns={k: f"shank_{k}" for k in ['AccelX','AccelY','AccelZ','GyroX','GyroY','GyroZ']}, inplace=True)

    foot.sort_values("Timestamp_us", inplace=True)
    shank.sort_values("Timestamp_us", inplace=True)
    merged = pd.merge_asof(foot, shank, on="Timestamp_us", direction="nearest")
    df = merged.loc[merged[RF_FEATURES].notna().all(axis=1)].reset_index(drop=True)

    y_pred = rf.predict(df[RF_FEATURES].values)
    y_smooth = heavy_smooth(y_pred, fs=fs)

    peaks = detect_big_positive_peaks(y_smooth, fs=fs)
    windows = extract_windows_around_peaks(y_smooth, peaks, fs=fs, half_window_s=1.25)
    avg_curve = average_windows(windows, num_points=200)

    peak_power = float(np.nanmax(avg_curve))
    avg_power = float(np.nanmean(avg_curve))
    propulsion, absorption = compute_propulsion_absorption(avg_curve)

    return {
        "peak_power": peak_power,
        "average_power": avg_power,
        "propulsion_integral": propulsion,
        "absorption_integral": absorption
    }
