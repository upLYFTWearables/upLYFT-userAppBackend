import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy.interpolate import interp1d

###############################################################################
# 1) Quaternion and Rotation-Related Helpers
###############################################################################
def quaternion_to_matrix(q):
    """
    Convert quaternion (w, x, y, z) to a 3x3 rotation matrix.
    """
    w, x, y, z = q
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-12:
        return np.eye(3)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),         1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x*x + y*y)]
    ])


def average_rotation_matrix(quaternions):
    """
    Given a list of quaternions, convert each to a rotation matrix,
    sum them, and use SVD to extract the best orthonormal approximation.
    """
    if len(quaternions) == 0:
        return np.eye(3)

    R_sum = np.zeros((3, 3))
    for q in quaternions:
        R_q = quaternion_to_matrix(q)
        R_sum += R_q

    R_sum /= len(quaternions)
    U, _, Vt = np.linalg.svd(R_sum)
    R_avg = U @ Vt
    return R_avg


def matrix_to_euler_ZYX(R):
    """
    Extract Euler angles from R using Z-Y-X sequence:
      R = Rz(alpha_z)*Ry(beta_y)*Rx(gamma_x).
    Returns (alpha_z, beta_y, gamma_x) in radians.
    """
    beta_y = -math.asin(R[2, 0])
    cos_beta = math.cos(beta_y)
    if abs(cos_beta) < 1e-6:
        # Gimbal lock fallback
        alpha_z = math.atan2(-R[0, 1], R[1, 1])
        gamma_x = 0.0
    else:
        alpha_z = math.atan2(R[1, 0], R[0, 0])
        gamma_x = math.atan2(R[2, 1], R[2, 2])

    return alpha_z, beta_y, gamma_x

###############################################################################
# 2) Main Analysis Function with Gait-Cycle Extraction
###############################################################################
def analyze_two_imus(
    filename_thigh,
    filename_shank,
    standing_time_s=5.0,
    ignore_initial_s=5.0,
    lowpass_cutoff=0.3,
    margin_before=10,
    margin_after=10,
    required_increase_ratio=1.2,
    cycle_length=101,
    cycle_to_remove=None
):
    """
    Reads two IMU CSV files (thigh & shank) and computes
    relative knee angles (Flex/Ext, Varus/Valgus, Int/Ext),
    identifies gait cycles, and generates plots.
    """
    # --- [A] Read & Calibrate ---
    df_thigh = pd.read_csv(filename_thigh, header=0, sep=',', skipinitialspace=True)
    t_thigh_us = df_thigh["Timestamp_us"].values
    Wt, Xt, Yt, Zt = df_thigh["QuatW"], df_thigh["QuatX"], df_thigh["QuatY"], df_thigh["QuatZ"]
    stand_thresh_thigh_us = t_thigh_us[0] + standing_time_s * 1e6
    mask_thigh = (t_thigh_us <= stand_thresh_thigh_us)
    quats_thigh_stand = list(zip(Wt[mask_thigh], Xt[mask_thigh], Yt[mask_thigh], Zt[mask_thigh]))
    R0_thigh = (average_rotation_matrix(quats_thigh_stand)
                if len(quats_thigh_stand) >= 10 else np.eye(3))

    df_shank = pd.read_csv(filename_shank, header=0, sep=',', skipinitialspace=True)
    t_shank_us = df_shank["Timestamp_us"].values
    Ws, Xs, Ys, Zs = df_shank["QuatW"], df_shank["QuatX"], df_shank["QuatY"], df_shank["QuatZ"]
    stand_thresh_shank_us = t_shank_us[0] + standing_time_s * 1e6
    mask_shank = (t_shank_us <= stand_thresh_shank_us)
    quats_shank_stand = list(zip(Ws[mask_shank], Xs[mask_shank], Ys[mask_shank], Zs[mask_shank]))
    R0_shank = (average_rotation_matrix(quats_shank_stand)
                if len(quats_shank_stand) >= 10 else np.eye(3))

    df_thigh["time_s"] = (df_thigh["Timestamp_us"] - df_thigh["Timestamp_us"].iloc[0]) * 1e-6
    df_shank["time_s"] = (df_shank["Timestamp_us"] - df_shank["Timestamp_us"].iloc[0]) * 1e-6
    df_thigh_rounded = df_thigh.copy(); df_shank_rounded = df_shank.copy()
    df_thigh_rounded["time_s"] = df_thigh_rounded["time_s"].round(2)
    df_shank_rounded["time_s"] = df_shank_rounded["time_s"].round(2)
    df_merged = pd.merge(df_thigh_rounded, df_shank_rounded, on="time_s", how="inner",
                         suffixes=("_thigh", "_shank"))
    time_s = df_merged["time_s"].values

    # --- [B] Compute Relative Rotations & Euler Angles ---
    angles_flex_ext = []  # Flexion/Extension
    angles_var_val = []   # Varus/Valgus
    angles_int_ext = []   # Internal/External rotation

    for i in range(len(df_merged)):
        R_thigh_prime = R0_thigh.T @ quaternion_to_matrix(
            (df_merged["QuatW_thigh"].iat[i], df_merged["QuatX_thigh"].iat[i],
             df_merged["QuatY_thigh"].iat[i], df_merged["QuatZ_thigh"].iat[i]))
        R_shank_prime = R0_shank.T @ quaternion_to_matrix(
            (df_merged["QuatW_shank"].iat[i], df_merged["QuatX_shank"].iat[i],
             df_merged["QuatY_shank"].iat[i], df_merged["QuatZ_shank"].iat[i]))
        R_rel = R_thigh_prime.T @ R_shank_prime
        alpha_z, beta_y, gamma_x = matrix_to_euler_ZYX(R_rel)

        angles_flex_ext.append(math.degrees(beta_y))
        angles_var_val.append(-math.degrees(alpha_z))
        angles_int_ext.append(math.degrees(gamma_x))

    angles_flex_ext = np.array(angles_flex_ext)
    angles_var_val  = np.array(angles_var_val)
    angles_int_ext  = np.array(angles_int_ext)

    # --- [C] Low-Pass Filter ---
    b, a = butter(1, lowpass_cutoff, btype='low')
    flex_ext_filt = filtfilt(b, a, angles_flex_ext)
    var_val_filt  = filtfilt(b, a, angles_var_val)
    int_ext_filt  = filtfilt(b, a, angles_int_ext)

    # --- [D] Plot #1: Raw vs Filtered ---
    fig, ax = plt.subplots(3, 1, figsize=(10,10), sharex=True)
    ax[0].plot(time_s, angles_flex_ext, label='Flex/Ext (raw)', alpha=0.4)
    ax[0].plot(time_s, flex_ext_filt,    label='Flex/Ext (filtered)', lw=2)
    ax[0].set_title('Knee Flexion/Extension vs. Time')

    ax[1].plot(time_s, angles_var_val, label='Varus/Valgus (raw)', alpha=0.4)
    ax[1].plot(time_s, var_val_filt,  label='Varus/Valgus (filtered)', lw=2)
    ax[1].set_title('Knee Varus/Valgus vs. Time')

    ax[2].plot(time_s, angles_int_ext, label='Int/Ext (raw)', alpha=0.4)
    ax[2].plot(time_s, int_ext_filt,  label='Int/Ext (filtered)', lw=2)
    ax[2].set_title('Knee Internal/External Rotation vs. Time')

    for a_sub in ax:
        a_sub.set_ylabel('Angle (deg)'); a_sub.grid(True, linestyle='--', alpha=0.7); a_sub.legend()
    ax[2].set_xlabel('Time (s)')
    #plt.tight_layout(); plt.show()

    # --- [E] Detect Gait Cycles in Flex/Ext ---
    mask_walk = (time_s >= ignore_initial_s)
    walk_idxs = np.where(mask_walk)[0]
    if len(walk_idxs) < 2:
        print(f"Warning: no data beyond {ignore_initial_s}s to detect cycles.")
        return
    flex_walk = flex_ext_filt[mask_walk]
    sm_pk, _ = find_peaks(flex_walk, prominence=1, distance=50, height=(-10,20))
    lg_pk, _ = find_peaks(flex_walk, prominence=5, distance=50, height=5)
    small_peaks = walk_idxs[sm_pk]; large_peaks = walk_idxs[lg_pk]

    cycle_indices = []
    cycles_flex_ext = []
    cycles_var_val  = []
    cycles_int_ext  = []

    for sp in small_peaks:
        valid_l = large_peaks[large_peaks > sp]
        if len(valid_l)==0 or flex_ext_filt[valid_l[0]] <= flex_ext_filt[sp]*required_increase_ratio:
            continue
        lp = valid_l[0]
        start, end = max(sp-margin_before,0), min(lp+margin_after, len(flex_ext_filt)-1)
        cycle_indices.append((start, end))
        old_x = np.linspace(0,1, end-start+1)
        new_x = np.linspace(0,1, cycle_length)
        cycles_flex_ext.append(np.interp(new_x, old_x, flex_ext_filt[start:end+1]))
        cycles_var_val.append(np.interp(new_x, old_x, var_val_filt[start:end+1]))
        cycles_int_ext.append(np.interp(new_x, old_x, int_ext_filt[start:end+1]))

    cycles_flex_ext = np.array(cycles_flex_ext)
    cycles_var_val  = np.array(cycles_var_val)
    cycles_int_ext  = np.array(cycles_int_ext)

    if cycles_flex_ext.size == 0:
        print("No valid gait cycles found.")
        return
    if cycle_to_remove is not None and 0<=cycle_to_remove<len(cycles_flex_ext):
        cycles_flex_ext = np.delete(cycles_flex_ext, cycle_to_remove, 0)
        cycles_var_val  = np.delete(cycles_var_val,  cycle_to_remove, 0)
        cycles_int_ext  = np.delete(cycles_int_ext,  cycle_to_remove, 0)
        cycle_indices.pop(cycle_to_remove)

    # --- [G] Mean & Std, Filter & Re-zero ---
    mean_flex_ext = np.mean(cycles_flex_ext, axis=0)
    std_flex_ext  = np.std(cycles_flex_ext, axis=0)
    mean_var_val  = np.mean(cycles_var_val, axis=0)
    std_var_val   = np.std(cycles_var_val, axis=0)
    mean_int_ext  = np.mean(cycles_int_ext, axis=0)
    std_int_ext   = np.std(cycles_int_ext, axis=0)

    b2, a2 = butter(2, 0.2, btype='low')
    mean_flex_ext = filtfilt(b2, a2, mean_flex_ext)
    std_flex_ext  = filtfilt(b2, a2, std_flex_ext)
    mean_var_val  = filtfilt(b2, a2, mean_var_val)
    std_var_val   = filtfilt(b2, a2, std_var_val)
    mean_int_ext  = filtfilt(b2, a2, mean_int_ext)
    std_int_ext   = filtfilt(b2, a2, std_int_ext)

    # Shift the main Flex/Ext mean curve so its global min is 0
    global_min_idx = np.argmin(mean_flex_ext)
    offset_value   = mean_flex_ext[global_min_idx]
    mean_flex_ext   -= offset_value
    mean_flex_ext[global_min_idx] = 0.0

    # Shift the other two by the same offset for consistency
    mean_var_val -= offset_value
    mean_int_ext -= offset_value


    # --- INSERT PRINT BLOCK BELOW ---
    # total cycle durations from start/end indices
    durations = [time_s[end] - time_s[start] for start, end in cycle_indices]
    avg_cycle_dur = np.mean(durations)

    # flex/ext is in mean_flex_ext
    idx_min = np.argmin(mean_flex_ext)
    idx_max = np.argmax(mean_flex_ext)
    phase_time = avg_cycle_dur * abs(idx_max - idx_min) / (cycle_length - 1)
    flexion_phase_dur   = phase_time
    extension_phase_dur = avg_cycle_dur - phase_time

    # varus/valgus peaks from mean_var_val
    varus_peak  = np.max(mean_var_val)
    valgus_peak = np.min(mean_var_val)

    # int/ext rotation peaks from mean_int_ext
    introt_peak = np.max(mean_int_ext)
    extrot_peak = np.min(mean_int_ext)

    print(f"Average gait cycle duration:       {avg_cycle_dur:.2f} s")
    print(f"  • Flexion phase duration:        {flexion_phase_dur:.2f} s")
    print(f"  • Extension phase duration:      {extension_phase_dur:.2f} s")
    print(f"Varus/Valgus (mean cycle) peaks:   +{varus_peak:.2f}° / {valgus_peak:.2f}°")
    print(f"Int/Ext rotation (mean cycle) peaks:+{introt_peak:.2f}° / {extrot_peak:.2f}°")


    # --- [H] Plot #2: Flex/Ext with shaded cycles ---
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(time_s, flex_ext_filt, label='Flex/Ext (filtered)')
    ax2.set_title(f'Flex/Ext with Detected Gait Cycles (Ignoring first {ignore_initial_s:.1f}s)')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Angle (deg)'); ax2.grid(True, linestyle='--', alpha=0.7)
    for i,(start,end) in enumerate(cycle_indices):
        t_slice = time_s[start:end+1]; a_slice = flex_ext_filt[start:end+1]
        c = f"C{i%10}"
        ax2.fill_between(t_slice, a_slice, y2=0, color=c, alpha=0.2)
        ax2.axvline(time_s[start], color=c, linestyle='--', alpha=0.7)
        ax2.axvline(time_s[end],   color=c, linestyle='--', alpha=0.7)
    #ax2.legend(); plt.tight_layout(); plt.show()

    # --- [I] Plot #3: Mean ± STD for each angle ---
    x_cycle = np.linspace(0,100,cycle_length)
    fig3, ax3 = plt.subplots(3,1,figsize=(10,10),sharex=True)
    ax3[0].plot(x_cycle, mean_flex_ext, label='Mean Flex/Ext')
    ax3[0].fill_between(x_cycle, mean_flex_ext-std_flex_ext, mean_flex_ext+std_flex_ext,
                       alpha=0.2, label='±1 STD')
    ax3[0].set_title(f'Flex/Ext Mean Cycle (N={len(durations)} cycles)')
    ax3[1].plot(x_cycle, mean_var_val, label='Mean Varus/Valgus')
    ax3[1].fill_between(x_cycle, mean_var_val-std_var_val, mean_var_val+std_var_val,
                       alpha=0.2, label='±1 STD')
    ax3[1].set_title('Varus/Valgus Mean Cycle')
    ax3[2].plot(x_cycle, mean_int_ext, label='Mean Int/Ext')
    ax3[2].fill_between(x_cycle, mean_int_ext-std_int_ext, mean_int_ext+std_int_ext,
                       alpha=0.2, label='±1 STD')
    ax3[2].set_title('Int/Ext Mean Cycle')
    for a_sub in ax3:
        a_sub.grid(True, linestyle='--', alpha=0.7); a_sub.set_ylabel('Angle (deg)'); a_sub.legend()
    ax3[2].set_xlabel('Gait Cycle (%)')
    #plt.tight_layout(); plt.show()

    return (
        time_s,
        (angles_flex_ext, angles_var_val, angles_int_ext),
        (flex_ext_filt, var_val_filt, int_ext_filt),
        cycles_flex_ext, cycles_var_val, cycles_int_ext,
        mean_flex_ext, mean_var_val, mean_int_ext,
        std_flex_ext, std_var_val, std_int_ext
    )

###############################################################################
# Example Usage
###############################################################################
# if __name__ == "__main__":
#     filename_thigh_imu = '/Users/mahadparwaiz/Desktop/foot/test1/shank.txt'
#     filename_shank_imu = '/Users/mahadparwaiz/Desktop/foot/test1/foot.txt'

#     results = analyze_two_imus(
#         filename_thigh_imu,
#         filename_shank_imu,
#         standing_time_s=5.0,    # first 5s used to define neutral
#         ignore_initial_s=5.0,   # skip first 5s for peak detection
#         lowpass_cutoff=0.15,    # e.g. ~7.5 Hz if sampling at 50 Hz
#         margin_before=15,
#         margin_after=30,
#         required_increase_ratio=1.2,
#         cycle_length=101,
#         cycle_to_remove=None
#     )