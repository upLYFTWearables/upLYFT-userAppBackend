#!/usr/bin/env python3
"""PowerMetrics – 3‑IMU pipeline (Phases A–F)
================================================

Implemented phases
------------------
A  Load & clock‑align CSVs (left foot, right foot, pelvis).
B  Rotate sensor‑frame accel → global frame; subtract gravity.
C  Integrate to velocity with foot ZUPT and pelvis pseudo‑ZUPT.
D  Detect heel‑strike / toe‑off from each foot’s Stationary flag.
E  **Stride‑window fusion** – build alternating HS sequence and slice
   strides (Left‑to‑Left & Right‑to‑Right windows).
F  **CoM excursion per stride** – integrate pelvis velocity → position,
   high‑pass detrend each stride, then take min/max of p<sub>z</sub>, p<sub>y</sub>.

Console output
--------------
After running, the script prints

* Right‑ and Left‑foot mean **GCT**
* Mean **stance** and **swing** times
* **Cadence** (steps min⁻¹)
* Global **min / max COM vertical** position (m)
* Global **min / max COM lateral** position (m)

Dependencies: numpy, pandas, scipy. Quaternion rotation is imported from
your original *PowerMetrics.py* via `rotate_acceleration_to_global`.
"""

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
GRAVITY   = 9.81   # m s⁻²
ACC_THR   = 0.35   # |a|-g < 0.35 m s⁻²  ⇢  stationary
GYRO_THR  = 2.0    # |ω|  < 2 deg/s      ⇢  stationary
MIN_GCT   = 0.05   # s  discard contacts shorter than this
HP_CUTOFF = 0.3    # Hz high‑pass for drift removal (per stride)
LEFT_PATH   = Path("AlgosFiles/Raw data/our own data/raw data_100Hz/walking_mahad_100Hz_141degSE/left leg low/data_3.txt")
RIGHT_PATH  = Path("AlgosFiles/Raw data/our own data/raw data_100Hz/walking_mahad_100Hz_141degSE/right leg low/data_3.txt")
PELVIS_PATH = Path("AlgosFiles/Raw data/our own data/raw data_100Hz/walking_mahad_100Hz_141degSE/pelvis/data_2.txt")
OUT_DIR     = Path("./out")  # change if you want a different folder

def rotate_acceleration_to_global(data):
    """Rotate local [AccelX,Y,Z] into global frame using quats."""
    w, x, y, z = (data[c].values for c in [' QuatW',' QuatX',' QuatY',' QuatZ'])
    ax, ay, az = (data[c].values for c in [' AccelX',' AccelY',' AccelZ'])
    N          = len(data) 
    gacc       = np.zeros((N,3))
    for i in range(N):
        qw, qx, qy, qz = w[i], x[i], y[i], z[i]
        r11 = qw*qw + qx*qx - qy*qy - qz*qz
        r12 = 2*(qx*qy - qw*qz)
        r13 = 2*(qx*qz + qw*qy)
        r21 = 2*(qx*qy + qw*qz)
        r22 = qw*qw - qx*qx + qy*qy - qz*qz
        r23 = 2*(qy*qz - qw*qx)
        r31 = 2*(qx*qz - qw*qy)
        r32 = 2*(qy*qz + qw*qx)
        r33 = qw*qw - qx*qx - qy*qy + qz*qz
        gacc[i] = [
            r11*ax[i] + r12*ay[i] + r13*az[i],
            r21*ax[i] + r22*ay[i] + r23*az[i],
            r31*ax[i] + r32*ay[i] + r33*az[i]
        ]
    return gacc
# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

def _detect_stationary(df: pd.DataFrame,
                       acc_thr: float = ACC_THR,
                       gyro_thr: float = GYRO_THR) -> np.ndarray:
    acc_vec  = df[[' AccelX', ' AccelY', ' AccelZ']].values
    gyro_vec = df[[' GyroX', ' GyroY', ' GyroZ']].abs().values
    acc_err  = np.abs(np.linalg.norm(acc_vec, axis=1) - GRAVITY)
    gyro_mag = np.linalg.norm(gyro_vec, axis=1)
    return (acc_err < acc_thr) & (gyro_mag < gyro_thr)


def _compute_velocity(global_acc: np.ndarray,
                      ts_sec: np.ndarray,
                      stationary: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(global_acc)
    for i in range(1, len(ts_sec)):
        dt = ts_sec[i] - ts_sec[i-1]
        vel[i] = vel[i-1] + global_acc[i-1] * dt
        if stationary[i]:
            vel[i] = 0.0
    return vel


def _compute_position(vel: np.ndarray, ts_sec: np.ndarray) -> np.ndarray:
    pos = np.zeros_like(vel)
    for i in range(1, len(ts_sec)):
        dt = ts_sec[i] - ts_sec[i-1]
        pos[i] = pos[i-1] + vel[i-1] * dt
    return pos


def _highpass(sig: np.ndarray, fs: float, cutoff_hz: float = HP_CUTOFF):
    b, a = butter(1, cutoff_hz / (0.5 * fs), btype='high', analog=False)
    return filtfilt(b, a, sig, axis=0)


def _gait_events_from_stationary(df: pd.DataFrame,
                                 min_contact_s: float = MIN_GCT) -> pd.DataFrame:
    stat = df['Stationary'].values.astype(int)
    t    = df['Time_s'].values
    diff = np.diff(stat, prepend=stat[0])
    hs_idx = np.where(diff == 1)[0]
    to_idx = np.where(diff == -1)[0]
    if to_idx.size and to_idx[0] < hs_idx[0]:
        to_idx = to_idx[1:]
    n = min(len(hs_idx), len(to_idx))
    hs_idx, to_idx = hs_idx[:n], to_idx[:n]
    gct = t[to_idx] - t[hs_idx]
    mask = gct >= min_contact_s
    return pd.DataFrame({'HS_time': t[hs_idx][mask],
                         'TO_time': t[to_idx][mask],
                         'GCT': gct[mask]})

# ─────────────────────────────────────────────────────────────
# Per‑sensor processing
# ─────────────────────────────────────────────────────────────

def process_sensor(csv_path: Path,
                   out_dir: Path,
                   name: str,
                   stationary_mask_from_feet: np.ndarray | None = None):
    df = pd.read_csv(csv_path)
    if 'Timestamp' not in df.columns:
        raise ValueError("CSV missing Timestamp_us column → " + str(csv_path))

    ts = (df['Timestamp'] - df['Timestamp'].iloc[0]) / 1e3
    df['Time_s'] = ts

    g_acc = rotate_acceleration_to_global(df)
    g_acc[:, 2] -= GRAVITY
    df[['GlobalAx', 'GlobalAy', 'GlobalAz']] = g_acc


    if name == 'pelvis' and stationary_mask_from_feet is not None:
    # Make the foot mask the same length as the pelvis DataFrame:
        m = len(stationary_mask_from_feet)
        n = len(df)
        if n <= m:
            stationary = stationary_mask_from_feet[:n]
        else:
            # if somehow pelvis is longer, pad with False
            pad = np.zeros(n - m, dtype=bool)
            stationary = np.concatenate([stationary_mask_from_feet, pad])
    else:
        stationary = _detect_stationary(df)

    df['Stationary'] = stationary

    vel = _compute_velocity(g_acc, ts.values, stationary)
    df[['VelX', 'VelY', 'VelZ']] = vel

    # Only pelvis needs position for CoM work
    if name == 'pelvis':
        pos = _compute_position(vel, ts.values)
        df[['PosX', 'PosY', 'PosZ']] = pos

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}_proc.csv", index=False)
    print(f"✓  {name.capitalize():6s} » saved")
    return df, stationary

# ─────────────────────────────────────────────────────────────
# Stride‑level metrics (Phases D–F)
# ─────────────────────────────────────────────────────────────

def _stride_level_metrics(events_L: pd.DataFrame,
                          events_R: pd.DataFrame,
                          total_duration_s: float):
    gct_L, gct_R = events_L['GCT'], events_R['GCT']
    def swings(ev):
        if len(ev) < 2:
            return np.array([])
        return ev['HS_time'].values[1:] - ev['TO_time'].values[:-1]
    swing_L, swing_R = swings(events_L), swings(events_R)
    stance_all = np.concatenate([gct_L, gct_R])
    swing_all  = np.concatenate([swing_L, swing_R]) if swing_L.size else np.array([np.nan])
    cadence = (len(gct_L) + len(gct_R)) / total_duration_s * 60
    return {
        'GCT_left_ms':  np.nanmean(gct_L)  * 1000,
        'GCT_right_ms': np.nanmean(gct_R) * 1000,
        'Stance_ms':    np.nanmean(stance_all) * 1000,
        'Swing_ms':     np.nanmean(swing_all)  * 1000,
        'Cadence_spm':  cadence
    }


def _build_stride_windows(events: pd.DataFrame):
    """Return list of (start, end) from consecutive HS."""
    times = events['HS_time'].values
    return list(zip(times[:-1], times[1:]))


def _com_metrics(pelvis_df: pd.DataFrame,
                 windows_L, windows_R):
    t      = pelvis_df['Time_s'].values
    p_y    = pelvis_df['PosY'].values
    p_z    = pelvis_df['PosZ'].values
    fs     = 1.0 / np.median(np.diff(t))
    global_min_y, global_max_y = np.inf, -np.inf
    global_min_z, global_max_z = np.inf, -np.inf

    for w in windows_L + windows_R:
        mask = (t >= w[0]) & (t < w[1])
        if mask.sum() < 5:
            continue
        seg_y = _highpass(p_y[mask], fs)
        seg_z = _highpass(p_z[mask], fs)
        global_min_y = min(global_min_y, seg_y.min())
        global_max_y = max(global_max_y, seg_y.max())
        global_min_z = min(global_min_z, seg_z.min())
        global_max_z = max(global_max_z, seg_z.max())

    return global_min_z, global_max_z, global_min_y, global_max_y

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    # Phase A–C: process left, right, pelvis
    df_L, stat_L = process_sensor(LEFT_PATH,  OUT_DIR, 'left')
    df_R, stat_R = process_sensor(RIGHT_PATH, OUT_DIR, 'right')

    # build pelvis stationary mask = (left OR right) stationary, truncated to shortest
    min_len   = min(len(stat_L), len(stat_R))
    stat_feet = stat_L[:min_len] | stat_R[:min_len]
    df_P, _   = process_sensor(PELVIS_PATH, OUT_DIR, 'pelvis', stat_feet)

    # Phase D: gait events from Stationary flag
    events_L = _gait_events_from_stationary(df_L)
    events_R = _gait_events_from_stationary(df_R)
    duration = max(df_L['Time_s'].iloc[-1], df_R['Time_s'].iloc[-1])
    stride_metrics = _stride_level_metrics(events_L, events_R, duration)

    # Phase E: build stride windows from alternating HS
    windows_L = _build_stride_windows(events_L)
    windows_R = _build_stride_windows(events_R)

    # Phase F: CoM excursion on pelvis
    min_z, max_z, min_y, max_y = _com_metrics(df_P, windows_L, windows_R)

    # Print summary
    print("\n———— Stride metrics ————")
    print(f"Right GCT   : {stride_metrics['GCT_right_ms']:.1f} ms")
    print(f"Left  GCT   : {stride_metrics['GCT_left_ms'] :.1f} ms")
    print(f"Stance time : {stride_metrics['Stance_ms']   :.1f} ms")
    print(f"Swing  time : {stride_metrics['Swing_ms']    :.1f} ms")
    print(f"Cadence     : {stride_metrics['Cadence_spm']:.1f} steps·min⁻¹")

    print("\n———— CoM position (global) ————")
    print(f"Vertical min : {min_z:.3f} m")
    print(f"Vertical max : {max_z:.3f} m")
    print(f"Lateral  min : {min_y:.3f} m")
    print(f"Lateral  max : {max_y:.3f} m")
    vertical_osc   = max_z - min_z
    horizontal_osc = max_y - min_y

    print(f"Vertical Oscillation   : {vertical_osc:.3f} m")
    print(f"Horizontal Oscillation : {horizontal_osc:.3f} m")


if __name__ == "__main__":
    # ensure the hard-coded paths actually exist:
    for p in (LEFT_PATH, RIGHT_PATH, PELVIS_PATH):
        if not p.is_file():
            sys.exit(f"❌  File not found: {p}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()