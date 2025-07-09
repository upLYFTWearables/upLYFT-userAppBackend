#!/usr/bin/env python3
"""
────────────────────────
Computes segment power PLUS temporal & spatial gait metrics:

    • Ground-contact time (GCT)
    • Swing time
    • Stride time
    • Cadence
    • Stride length
    • Load (peak vertical force)

All original functionality (translational / rotational / total power,
mean-cycle plots, etc.) is preserved.
"""
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal    import butter, filtfilt, find_peaks
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────────────────────
# 1)  GAIT-EVENT DETECTION (Heel Strikes & Toe Offs)
# ─────────────────────────────────────────────────────────────
def apply_lowpass_filter(signal, fs=100.0, cutoff=10.0, order=4):
    nyquist = 0.5 * fs
    normal  = cutoff / nyquist
    b, a    = butter(order, normal, btype="low", analog=False)
    return  filtfilt(b, a, signal)


def detect_gait_events_gyro_kmeans(
    filepaths,
    min_interval_us_strike=200_000,
    min_interval_us_toe=200_000,
    plot_peaks=False,
    fs=100.0,
    apply_filter=True,
    cutoff=10.0,
    filter_order=4,
    initial_prominence=0.5
):
    """
    Detect gait events (heel strikes & toe offs) using:
      • low-pass filtering of GyroY
      • negative-peak detection
      • k-means clustering into two classes
    Returns two DataFrames (heel_strikes, toe_offs) plus raw signal.
    """
    combined, total_offset = [], 0
    for fp in filepaths:
        df = pd.read_csv(fp)
        if {'Timestamp_us', 'GyroY'}.difference(df.columns):
            raise ValueError(f"{fp} missing Timestamp_us or GyroY")
        start_t = df['Timestamp_us'].min()
        df['Timestamp_us'] = df['Timestamp_us'] - start_t + total_offset
        total_offset = df['Timestamp_us'].max() + min_interval_us_strike
        combined.append(df)

    data = pd.concat(combined, ignore_index=True).sort_values('Timestamp_us')
    gyroY = data['GyroY'].values
    ts    = data['Timestamp_us'].values

    if apply_filter:
        gyroY = apply_lowpass_filter(gyroY, fs, cutoff, filter_order)

    neg_peaks, _ = find_peaks(-gyroY, prominence=initial_prominence)
    if not len(neg_peaks):
        print("No candidate peaks found.")
        return pd.DataFrame(), pd.DataFrame(), ts, gyroY

    cand = pd.DataFrame({
        'Timestamp_us': ts[neg_peaks],
        'PeakGyroY'   : gyroY[neg_peaks]
    })

    # 2-cluster k-means on peak amplitudes
    labels = KMeans(n_clusters=2, random_state=42).fit_predict(
        cand['PeakGyroY'].values.reshape(-1, 1)
    )
    cand['Cluster'] = labels
    means = cand.groupby('Cluster')['PeakGyroY'].mean()
    toe_cluster  = means.idxmin()             # more negative → toe-off
    heel_cluster = 1 - toe_cluster

    toe_cand  = cand[cand['Cluster'] == toe_cluster].sort_values('Timestamp_us')
    heel_cand = cand[cand['Cluster'] == heel_cluster].sort_values('Timestamp_us')

    # enforce min-time spacing
    def keep_min_interval(df_events, min_us):
        keep, last_t = [], -math.inf
        for _, r in df_events.iterrows():
            if r['Timestamp_us'] - last_t > min_us:
                keep.append(r)
                last_t = r['Timestamp_us']
        return pd.DataFrame(keep)

    toe_offs      = keep_min_interval(toe_cand , min_interval_us_toe  )
    heel_strikes  = keep_min_interval(heel_cand, min_interval_us_strike)

    # optional plot
    if plot_peaks:
        plt.figure(figsize=(10,5))
        plt.plot(ts/1e6, gyroY, label='GyroY')
        plt.scatter(toe_offs ['Timestamp_us']/1e6, toe_offs ['PeakGyroY'],
                    c='g', label='Toe-off')
        plt.scatter(heel_strikes['Timestamp_us']/1e6, heel_strikes['PeakGyroY'],
                    c='r', label='Heel-strike')
        plt.xlabel('Time (s)'); plt.ylabel('GyroY (deg/s)')
        plt.title('Gait events'); plt.legend(); plt.grid(); plt.show()

    return heel_strikes, toe_offs, ts, gyroY

# ─────────────────────────────────────────────────────────────
# 2)  ROTATION & FILTER HELPERS (unchanged)
# ─────────────────────────────────────────────────────────────
def rotate_acceleration_to_global(data):
    """Rotate local [AccelX,Y,Z] into global frame using quats."""
    w, x, y, z = (data[c].values for c in ['QuatW','QuatX','QuatY','QuatZ'])
    ax, ay, az = (data[c].values for c in ['AccelX','AccelY','AccelZ'])
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


def butter_lowpass_3d(acc3, fs=100.0, cutoff=10.0, order=4):
    nyq   = 0.5*fs
    b, a  = butter(order, cutoff/nyq, btype='low', analog=False)
    flt   = np.zeros_like(acc3)
    for i in range(3):
        flt[:,i] = filtfilt(b, a, acc3[:,i])
    return flt


def compute_rotational_torque_and_power(data, inertia_diag, fs=100.0):
    """ τ = I·α + ω×(I·ω)  ;  P = τ·ω """
    omega_deg = data[['GyroX','GyroY','GyroZ']].to_numpy()
    omega     = np.deg2rad(omega_deg)
    dt        = 1.0/fs
    alpha     = np.gradient(omega, dt, axis=0)
    I         = np.diag(inertia_diag)
    torque    = np.cross(omega, (I @ omega.T).T) + (I @ alpha.T).T
    rot_power = np.einsum('ij,ij->i', torque, omega)
    return torque, rot_power, omega


def compute_velocity_with_zupt(global_acc, ts_us, heel_times):
    """Simple forward Euler integration with zero-velocity update at HS."""
    N      = len(global_acc)
    vel    = np.zeros((N,3))
    i_hs   = 0
    for i in range(1, N):
        dt = (ts_us[i] - ts_us[i-1]) / 1e6
        vel[i] = vel[i-1] + global_acc[i-1]*dt
        # reset at heel-strike indices
        while i_hs < len(heel_times) and ts_us[i] >= heel_times[i_hs]:
            vel[i] = 0.0
            i_hs  += 1
    return vel


def compute_translational_power(mass, acc_g, vel):
    """ P = (m·a) · v """
    return np.einsum('ij,ij->i', mass*acc_g, vel)


def compute_mean_cycle(results_df, heel_times, power_col='TotalPower', num_points=101):
    """Interpolate power for each gait cycle onto 0-100 % grid, return mean ± SD."""
    cycles = []
    ht = heel_times
    for i in range(len(ht)-1):
        start, end = ht[i], ht[i+1]
        mask   = (results_df['Timestamp_us']>=start)&(results_df['Timestamp_us']<=end)
        if not mask.any(): continue
        seg    = results_df.loc[mask, ['Timestamp_us', power_col]].to_numpy()
        pct    = 100*(seg[:,0]-start)/(end-start)
        grid   = np.linspace(0,100,num_points)
        interp = np.interp(grid, pct, seg[:,1])
        cycles.append(interp)
    if not cycles:
        return None, None, None
    arr = np.vstack(cycles)
    return grid, arr.mean(0), arr.std(0)

# ─────────────────────────────────────────────────────────────
# 3)  MAIN: POWER + NEW GAIT METRICS + LOAD
# ─────────────────────────────────────────────────────────────
def compute_power_metrics_with_zupt(
    filepaths,
    foot_mass=1.0,
    foot_inertia_diag=(0.0010,0.0053,0.0060),
    use_quaternion=True,
    remove_gravity=True,
    gravity=9.81,
    apply_filter=True,
    fs=100.0,
    cutoff=10.0,
    filter_order=4,
    stationary_time_s=2.0,
    body_mass=65.0,
    plot_results=True
):
    # (A) Gait events -------------------------------------------------
    heel_df, toe_df, gyro_ts, gyroY = detect_gait_events_gyro_kmeans(
        filepaths, 200_000, 200_000, False,
        fs, apply_filter, cutoff, filter_order, 0.5
    )
    heel_times = heel_df['Timestamp_us'].to_numpy() if not heel_df.empty else np.array([])
    toe_times  = toe_df ['Timestamp_us'].to_numpy() if not toe_df .empty else np.array([])

    # (B) Load & concatenate -----------------------------------------
    combined, offset = [], 0
    for fp in filepaths:
        df = pd.read_csv(fp)
        t0 = df['Timestamp_us'].min()
        df['Timestamp_us'] = df['Timestamp_us'] - t0 + offset
        offset = df['Timestamp_us'].max() + int(1_000_000//fs)
        combined.append(df)
    data = (pd.concat(combined, ignore_index=True)
              .sort_values('Timestamp_us')
              .reset_index(drop=True))
    ts_us = data['Timestamp_us'].to_numpy()

    # (C) Optional raw accel plot ------------------------------------
    if plot_results:
        plt.figure(figsize=(10,4))
        plt.plot(ts_us/1e6, data['AccelX'], label='AccelX')
        plt.plot(ts_us/1e6, data['AccelY'], label='AccelY')
        plt.plot(ts_us/1e6, data['AccelZ'], label='AccelZ')
        plt.title('Raw Acceleration'); plt.xlabel('Time (s)'); plt.ylabel('m/s²')
        plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    # (D) Rotate to global -------------------------------------------
    if use_quaternion and {'QuatW','QuatX','QuatY','QuatZ'}.issubset(data.columns):
        global_acc = rotate_acceleration_to_global(data)
    else:
        global_acc = data[['AccelX','AccelY','AccelZ']].to_numpy()

    # (E) Remove gravity ---------------------------------------------
    if remove_gravity:
        mask_st = ts_us <= stationary_time_s*1e6
        g_vec   = global_acc[mask_st].mean(0) if mask_st.any() else np.array([0,0,gravity])
        global_acc = global_acc - g_vec

    # (F) Low-pass filter --------------------------------------------
    if apply_filter:
        global_acc = butter_lowpass_3d(global_acc, fs, cutoff, filter_order)

    # (G) Velocity & power -------------------------------------------
    velocity            = compute_velocity_with_zupt(global_acc, ts_us, heel_times)
    translational_power = compute_translational_power(foot_mass, global_acc, velocity)
    torque, rotational_power, omega = compute_rotational_torque_and_power(
        data, foot_inertia_diag, fs
    )
    total_power = translational_power + rotational_power

    # ——— NEW: compute vertical load (N) ————————————————
    # re-add gravity in Z to get true vertical accel, then F = m·a
    Fz       = body_mass * (global_acc[:,2] + gravity)
    pk_load  = np.max(Fz)

    # (H) Original power stats ---------------------------------------
    pk_trans, pk_rot, pk_tot = map(np.max, (translational_power,
                                            rotational_power,
                                            total_power))
    avg_trans, avg_rot, avg_tot = map(np.mean, (translational_power,
                                                rotational_power,
                                                total_power))

    # (I) New temporal metrics ---------------------------------------
    # Ground-contact time
    if heel_times.size and toe_times.size:
        toe_idx     = np.searchsorted(toe_times, heel_times)
        valid_gct   = toe_idx < toe_times.size
        gct_s       = (toe_times[toe_idx[valid_gct]] - heel_times[valid_gct]) / 1e6
    else:
        gct_s = np.array([])

    # Swing time = next heel-strike − current toe-off
    if toe_times.size >= 1 and heel_times.size >= 2:
        hs_next  = heel_times[1:len(toe_times)]
        swing_s  = (hs_next - toe_times[:len(hs_next)]) / 1e6
    else:
        swing_s = np.array([])

    # Stride (same-foot heel-strike-to-heel-strike)
    stride_s = np.diff(heel_times) / 1e6 if heel_times.size >= 2 else np.array([])
    cadence_spm = 120.0/stride_s.mean() if stride_s.size else np.nan  # 2 steps / stride

    # (J) Stride length ----------------------------------------------
    stride_len = []
    if heel_times.size >= 2:
        for i in range(len(heel_times)-1):
            seg = (ts_us >= heel_times[i]) & (ts_us <= heel_times[i+1])
            if seg.sum()<2: continue
            dt  = np.diff(ts_us[seg])/1e6
            vx  = velocity[seg,0]
            disp = np.sum(0.5*(vx[1:]+vx[:-1])*dt)
            stride_len.append(disp)
    stride_len = np.array(stride_len)

    # (K) Pack sample-wise DataFrame ---------------------------------
    results_df = pd.DataFrame({
        'Timestamp_us' : ts_us,
        'AccX_global'  : global_acc[:,0],
        'AccY_global'  : global_acc[:,1],
        'AccZ_global'  : global_acc[:,2],
        'VelX'         : velocity[:,0],
        'VelY'         : velocity[:,1],
        'VelZ'         : velocity[:,2],
        'TransPower'   : translational_power,
        'TorqueX'      : torque[:,0],
        'TorqueY'      : torque[:,1],
        'TorqueZ'      : torque[:,2],
        'OmegaX_rad'   : omega[:,0],
        'OmegaY_rad'   : omega[:,1],
        'OmegaZ_rad'   : omega[:,2],
        'RotPower'     : rotational_power,
        'TotalPower'   : total_power,
        'TotalPower_PW': total_power/body_mass
    })

    # (L) Print summary ----------------------------------------------
    print("────────────────────────────────────────────────────────")
    print(f"TRANSLATIONAL POWER  peak {pk_trans:8.2f} W   mean {avg_trans:8.2f} W")
    print(f"ROTATIONAL POWER     peak {pk_rot :8.2f} W   mean {avg_rot :8.2f} W")
    print(f"TOTAL POWER          peak {pk_tot :8.2f} W   mean {avg_tot :8.2f} W")
    if stride_s.size:
        print(f"STRIDE TIME        {stride_s.mean()*1000:7.1f} ± {stride_s.std()*1000:5.1f} ms")
        print(f"CADENCE                 {cadence_spm:7.1f} steps/min")
    if gct_s.size:
        print(f"GCT               {gct_s.mean()*1000:7.1f} ± {gct_s.std()*1000:5.1f} ms")
    if swing_s.size:
        print(f"SWING             {swing_s.mean()*1000:7.1f} ± {swing_s.std()*1000:5.1f} ms")
    if stride_len.size:
        print(f"STRIDE LENGTH          {stride_len.mean():5.2f} ± {stride_len.std():4.2f} m")
    # ——— Print peak load in Newtons ——————————————
    print(f"PEAK LOAD            {pk_load:8.2f} N")
    print("────────────────────────────────────────────────────────")

    # (M) Extra metrics dict -----------------------------------------
    extra_metrics = {
        'HeelStrike_us'  : heel_times,
        'ToeOff_us'      : toe_times,
        'StrideTime_s'   : stride_s,
        'Cadence_spm'    : cadence_spm,
        'GCT_s'          : gct_s,
        'Swing_s'        : swing_s,
        'StrideLength_m' : stride_len,
        'Load_N'         : pk_load
    }

    # (N) Plots (kept identical, now use results_df) -----------------
    if plot_results:
        time_s = ts_us/1e6
        # Power time-series
        fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(12,10),sharex=True)
        ax1.plot(time_s, results_df['TransPower']); ax1.set_ylabel('W')
        ax1.set_title('Translational Power (m·a)·v'); ax1.grid()
        ax2.plot(time_s, results_df['RotPower']   , color='r'); ax2.set_ylabel('W')
        ax2.set_title('Rotational Power τ·ω'); ax2.grid()
        ax3.plot(time_s, results_df['TotalPower'] , color='g'); ax3.set_ylabel('W')
        ax3.set_title('Total Power'); ax3.set_xlabel('Time (s)'); ax3.grid()
        plt.tight_layout(); plt.show()

        # Mean gait-cycle power profile
        if heel_times.size >= 2:
            grid, mean_p, std_p = compute_mean_cycle(results_df, heel_times,
                                                     'TotalPower', 101)
            if grid is not None:
                plt.figure(figsize=(10,5))
                plt.plot(grid, mean_p, label='Mean Total Power')
                plt.fill_between(grid, mean_p-std_p, mean_p+std_p, alpha=.3)
                plt.xlabel('% Gait cycle'); plt.ylabel('W');
                plt.title('Mean Gait-Cycle Power'); plt.legend(); plt.grid()
                plt.show()

        # Power-to-weight
        plt.figure(figsize=(10,4))
        plt.plot(time_s, results_df['TotalPower_PW'])
        plt.title('Instantaneous Total Power / Weight'); plt.xlabel('Time (s)')
        plt.ylabel('W/kg'); plt.grid(); plt.tight_layout(); plt.show()

        # Rolling average of total power
        results_df['RollingAvgPower'] = results_df['TotalPower'].rolling(
            window=100, min_periods=1).mean()
        plt.figure(figsize=(10,4))
        plt.plot(time_s, results_df['RollingAvgPower'])
        plt.title('Rolling Average Total Power'); plt.xlabel('Time (s)')
        plt.ylabel('W'); plt.grid(); plt.tight_layout(); plt.show()

    return results_df, extra_metrics

# ─────────────────────────────────────────────────────────────
# 4)  Usage example
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ►►► EDIT THIS PATH ◄◄◄
    test_file = "AlgosFiles/Raw data/foot.txt"

    df, metrics = compute_power_metrics_with_zupt(
        filepaths         =[test_file],
        foot_mass         =1.0,                    # kg
        foot_inertia_diag =(0.0010,0.0053,0.0060), # kg·m²
        use_quaternion    =True,
        remove_gravity    =True,
        apply_filter      =True,
        fs                =100.0,
        cutoff            =10.0,
        filter_order      =4,
        stationary_time_s =2.0,
        body_mass         =65.0,
        plot_results      =True
    )
