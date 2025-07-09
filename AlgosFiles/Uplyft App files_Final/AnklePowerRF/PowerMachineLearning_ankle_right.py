#!/usr/bin/env python3
"""
Random-Forest ankle-power estimator + window-based averaging

For every large positive ankle-power peak:
   take 1.25 s before and 1.25 s after (2.5 s window) *if* that window
   contains no other positive peak.  Resample all accepted windows so the
   peak sits at 50 % of the vector, then average.

────────────────────────────────────────────────────────────────────
Required third-party packages:
    numpy, pandas, scipy, scikit-learn, matplotlib
"""

import os, glob
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.ensemble   import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics    import r2_score
from scipy.signal       import butter, filtfilt, find_peaks
from scipy.ndimage      import gaussian_filter1d


# ──────────────────────────────────────────────────────────────
# I.  DATA-LOADING HELPERS  (unchanged)
# ──────────────────────────────────────────────────────────────
def load_data_for_rf(data_dir):
    jp_p  = os.path.join(data_dir, "AB*", "*", "levelground", "jp", "*.mat")
    imu_p = os.path.join(data_dir, "AB*", "*", "levelground", "imu", "*.mat")
    gc_p  = os.path.join(data_dir, "AB*", "*", "levelground", "gcRight", "*.mat")

    jp_files, imu_files, gc_files = map(glob.glob, (jp_p, imu_p, gc_p))
    data_dict = {'jp': {}, 'imu': {}, 'gcRight': {}}

    for fp in jp_files:      # JP
        d = sio.loadmat(fp)
        if {"dataMatrix", "varNames"} <= d.keys():
            fid = os.path.basename(fp)
            data_dict['jp'][fid] = {
                'dataMatrix': d["dataMatrix"],
                'varNames'  : [str(v).strip("[]'") for v in d["varNames"].flatten()],
                'filepath'  : fp
            }
    for fp in imu_files:     # IMU
        d = sio.loadmat(fp)
        if {"dataMatrix", "varNames"} <= d.keys():
            fid = os.path.basename(fp)
            data_dict['imu'][fid] = {
                'dataMatrix': d["dataMatrix"],
                'varNames'  : [str(v).strip("[]'") for v in d["varNames"].flatten()],
                'filepath'  : fp
            }
    for fp in gc_files:      # GC
        d = sio.loadmat(fp)
        if {"dataMatrix", "varNames"} <= d.keys():
            fid = os.path.basename(fp)
            data_dict['gcRight'][fid] = {
                'dataMatrix': d["dataMatrix"],
                'varNames'  : [str(v).strip("[]'") for v in d["varNames"].flatten()],
                'filepath'  : fp
            }
    return data_dict


def pair_files(data_dict):
    out = []
    for fid, jp in data_dict['jp'].items():
        if fid in data_dict['imu'] and fid in data_dict['gcRight']:
            out.append((jp, data_dict['imu'][fid], data_dict['gcRight'][fid]))
    return out


def extract_features_and_target(jp, imu):
    imu_names, imu_mat = imu['varNames'], imu['dataMatrix']
    jp_names , jp_mat  = jp ['varNames'], jp ['dataMatrix']

    tgt_idx   = jp_names.index('ankle_angle_r_power')
    feat_cols = [c for c in imu_names
                 if c != 'Header' and (c.startswith('foot_') or c.startswith('shank_'))]
    feat_idx  = [imu_names.index(c) for c in feat_cols]

    N = min(len(imu_mat), len(jp_mat))
    X = imu_mat[:N, feat_idx]
    y = jp_mat [:N, tgt_idx]
    hdr = jp_mat[:N, jp_names.index('Header')]

    valid = ~np.isnan(y)
    return X[valid], y[valid], hdr[valid]


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


# ──────────────────────────────────────────────────────────────
# V.  MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) train RF on lab data  ─────────────────────────────────
    DATA_DIR = "AlgosFiles/Raw data/RFdata/Data3"    # ← edit
    triples  = pair_files(load_data_for_rf(DATA_DIR))
    if not triples:
        raise RuntimeError("No paired JP/IMU/gcRight .mat triples found!")

    jp, imu, _ = triples[0]          # use first triple
    X, y, _    = extract_features_and_target(jp, imu)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42)

    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_tr, y_tr)

    print("\n──────── RF PERFORMANCE ────────")
    print(f"Train  R²  {r2_score(y_tr, rf.predict(X_tr)):.3f}")
    print(f"Test   R²  {r2_score(y_te, rf.predict(X_te)):.3f}")
    print("─────────────────────────────────\n")
#   Save the model
    from joblib import dump

    # Train the Random Forest model
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_tr, y_tr)

    # Save the trained model to disk
    MODEL_PATH = "models/ankle_rf_model.joblib"
    os.makedirs("models", exist_ok=True)
    dump(rf, MODEL_PATH)

    print(f"Model saved at: {MODEL_PATH}")

    # 2) load new IMU recording  ───────────────────────────────
    FOOT_PATH  = "AlgosFiles/Raw data/foot.txt"    # ← edit
    SHANK_PATH = "AlgosFiles/Raw data/shank.txt"   # ← edit
    FS_NEW     = 100.0   # Hz

    foot  = pd.read_csv(FOOT_PATH , delimiter=",", header=0)
    shank = pd.read_csv(SHANK_PATH, delimiter=",", header=0)

    foot.rename(columns={
        'AccelX':'foot_AccelX','AccelY':'foot_AccelY','AccelZ':'foot_AccelZ',
        'GyroX' :'foot_GyroX' ,'GyroY' :'foot_GyroY' ,'GyroZ' :'foot_GyroZ'
    }, inplace=True)
    shank.rename(columns={
        'AccelX':'shank_AccelX','AccelY':'shank_AccelY','AccelZ':'shank_AccelZ',
        'GyroX' :'shank_GyroX' ,'GyroY' :'shank_GyroY' ,'GyroZ' :'shank_GyroZ'
    }, inplace=True)

    foot .sort_values('Timestamp_us', inplace=True)
    shank.sort_values('Timestamp_us', inplace=True)
    merged = pd.merge_asof(foot, shank,
                           on='Timestamp_us',
                           direction='nearest')

    imu_cols = [
        'foot_AccelX','foot_AccelY','foot_AccelZ','foot_GyroX','foot_GyroY','foot_GyroZ',
        'shank_AccelX','shank_AccelY','shank_AccelZ','shank_GyroX','shank_GyroY','shank_GyroZ'
    ]
    df = merged.loc[merged[imu_cols].notna().all(axis=1)].reset_index(drop=True)

    y_pred = rf.predict(df[imu_cols].values)
    print(f"New IMU samples fed to RF: {len(y_pred)}")

    # 3) smooth & pick peaks  ─────────────────────────────────
    time_s = df['Timestamp_us'].values * 1e-6
    y_smooth = heavy_smooth(y_pred, fs=FS_NEW)

    peaks = detect_big_positive_peaks(y_smooth, fs=FS_NEW,
                                      min_prominence=None,
                                      min_distance_s=1.0)

    # 4) fixed-window averaging  ──────────────────────────────
    windows = extract_windows_around_peaks(
        y_smooth, peaks, fs=FS_NEW,
        half_window_s=1.25,          # ±1.25 s
        require_isolated=True)       # no other peak inside window

    avg_curve = average_windows(windows, num_points=200)
    peak_power            = float(np.nanmax(avg_curve))
    avg_power             = float(np.nanmean(avg_curve))
    propulsion, absorption = compute_propulsion_absorption(avg_curve)
    print("\n──── Power metrics (averaged predicted curve) ────")
    print(f"Peak power      : {peak_power:8.2f}  W")
    print(f"Average power   : {avg_power:8.2f}  W")
    print(f"Propulsion (∫ +): {propulsion:8.2f}  W·s")
    print(f"Absorption (∫ −): {absorption:8.2f}  W·s")
    print("──────────────────────────────────────────────────\n")

    
    # 5) plots  ───────────────────────────────────────────────
    # (a) time-series with peaks
    plt.figure(figsize=(10,4))
    plt.plot(time_s, y_smooth, label='Smoothed power')
    plt.scatter(time_s[peaks], y_smooth[peaks], marker='x', s=40, label='Big +ve peaks')
    plt.xlabel('Time (s)'); plt.ylabel('Predicted ankle power')
    plt.title('Predicted ankle-power vs. time')
    plt.legend(); plt.tight_layout(); plt.show()

    # (b) averaged window (0–100 %, peak at 50 %)
    pct = np.linspace(0, 100, len(avg_curve))
    plt.figure(figsize=(8,4))
    plt.plot(pct, avg_curve, 'r')
    plt.axvline(50, ls='--', alpha=.4)
    plt.title('Predicted ankle power – ±1.25 s window averaged')
    plt.xlabel('Window (%: 0 % = −1.25 s, 50 % = peak, 100 % = +1.25 s)')
    plt.ylabel('Predicted Power')
    plt.tight_layout(); plt.show()

    prop, absb = compute_propulsion_absorption(avg_curve)
    print(f"Propulsion (∫ positive):  {prop:8.2f}")
    print(f"Absorption (∫ negative): {absb:8.2f}")
