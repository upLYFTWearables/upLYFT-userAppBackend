###############################################################################
# Hip-angle analysis from two IMUs: pelvis (top) & thigh (bottom)
###############################################################################
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
    Convert quaternion (w, x, y, z) to a 3×3 rotation matrix.
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
    Average a list of quaternions by converting to matrices,
    summing, and orthonormalising with SVD.
    """
    if len(quaternions) == 0:
        return np.eye(3)

    R_sum = np.zeros((3, 3))
    for q in quaternions:
        R_sum += quaternion_to_matrix(q)

    R_sum /= len(quaternions)
    U, _, Vt = np.linalg.svd(R_sum)
    return U @ Vt


def matrix_to_euler_ZYX(R):
    """
    Extract Z-Y-X (yaw-pitch-roll) Euler angles (rad) from a
    rotation matrix. Returns (alpha_z, beta_y, gamma_x).
    """
    beta_y = -math.asin(R[2, 0])
    cos_beta = math.cos(beta_y)
    if abs(cos_beta) < 1e-6:
        # Gimbal–lock fallback
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
    filename_pelvis,              # ← TOP IMU (was “thigh”)
    filename_thigh,               # ← BOTTOM IMU (was “shank”)
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
    Reads two IMU CSV files (pelvis & thigh) and computes relative hip angles:
      • Flexion/Extension
      • Abduction/Adduction
      • Internal/External rotation

    It identifies gait cycles and produces summary plots/metrics.
    The mathematics is identical to the original knee script;
    only the labels and variable names have been renamed for clarity.
    """
    # --- [A] Read & Calibrate -------------------------------------------------
    # Pelvis (top) ------------------------------------------------------------
    df_pelvis = pd.read_csv(filename_pelvis, header=0, sep=',', skipinitialspace=True)
    t_pelvis_us = df_pelvis["Timestamp_us"].values
    Wp, Xp, Yp, Zp = df_pelvis["QuatW"], df_pelvis["QuatX"], df_pelvis["QuatY"], df_pelvis["QuatZ"]
    stand_thresh_pelvis_us = t_pelvis_us[0] + standing_time_s * 1e6
    mask_pelvis = (t_pelvis_us <= stand_thresh_pelvis_us)
    quats_pelvis_stand = list(zip(Wp[mask_pelvis], Xp[mask_pelvis], Yp[mask_pelvis], Zp[mask_pelvis]))
    R0_pelvis = (average_rotation_matrix(quats_pelvis_stand)
                 if len(quats_pelvis_stand) >= 10 else np.eye(3))

    # Thigh (bottom) ----------------------------------------------------------
    df_thigh = pd.read_csv(filename_thigh, header=0, sep=',', skipinitialspace=True)
    t_thigh_us = df_thigh["Timestamp_us"].values
    Wt, Xt, Yt, Zt = df_thigh["QuatW"], df_thigh["QuatX"], df_thigh["QuatY"], df_thigh["QuatZ"]
    stand_thresh_thigh_us = t_thigh_us[0] + standing_time_s * 1e6
    mask_thigh = (t_thigh_us <= stand_thresh_thigh_us)
    quats_thigh_stand = list(zip(Wt[mask_thigh], Xt[mask_thigh], Yt[mask_thigh], Zt[mask_thigh]))
    R0_thigh = (average_rotation_matrix(quats_thigh_stand)
                if len(quats_thigh_stand) >= 10 else np.eye(3))

    # Align timestamps --------------------------------------------------------
    df_pelvis["time_s"] = (df_pelvis["Timestamp_us"] - df_pelvis["Timestamp_us"].iloc[0]) * 1e-6
    df_thigh["time_s"] = (df_thigh["Timestamp_us"] - df_thigh["Timestamp_us"].iloc[0]) * 1e-6
    df_pelvis_r = df_pelvis.copy(); df_thigh_r = df_thigh.copy()
    df_pelvis_r["time_s"] = df_pelvis_r["time_s"].round(2)
    df_thigh_r["time_s"] = df_thigh_r["time_s"].round(2)
    df_merged = pd.merge(df_pelvis_r, df_thigh_r, on="time_s", how="inner",
                         suffixes=("_pelvis", "_thigh"))
    time_s = df_merged["time_s"].values

    # --- [B] Compute Relative Rotations & Euler Angles -----------------------
    angles_flex_ext = []   # Hip Flexion / Extension
    angles_abd_add = []    # Hip Abduction / Adduction
    angles_int_ext = []    # Hip Internal / External rotation

    for i in range(len(df_merged)):
        R_pelvis_prime = R0_pelvis.T @ quaternion_to_matrix(
            (df_merged["QuatW_pelvis"].iat[i], df_merged["QuatX_pelvis"].iat[i],
             df_merged["QuatY_pelvis"].iat[i], df_merged["QuatZ_pelvis"].iat[i]))
        R_thigh_prime = R0_thigh.T @ quaternion_to_matrix(
            (df_merged["QuatW_thigh"].iat[i], df_merged["QuatX_thigh"].iat[i],
             df_merged["QuatY_thigh"].iat[i], df_merged["QuatZ_thigh"].iat[i]))
        R_rel = R_pelvis_prime.T @ R_thigh_prime
        alpha_z, beta_y, gamma_x = matrix_to_euler_ZYX(R_rel)

        angles_flex_ext.append(math.degrees(beta_y))
        angles_abd_add.append(-math.degrees(alpha_z))   # sign preserved from knee script
        angles_int_ext.append(math.degrees(gamma_x))

    angles_flex_ext = np.array(angles_flex_ext)
    angles_abd_add = np.array(angles_abd_add)
    angles_int_ext = np.array(angles_int_ext)

    # --- [C] Low-Pass Filter --------------------------------------------------
    b, a = butter(1, lowpass_cutoff, btype='low')
    flex_ext_filt = filtfilt(b, a, angles_flex_ext)
    abd_add_filt  = filtfilt(b, a, angles_abd_add)
    int_ext_filt  = filtfilt(b, a, angles_int_ext)

    # --- [D] Plot #1: Raw vs Filtered ----------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(10,10), sharex=True)
    ax[0].plot(time_s, angles_flex_ext, label='Flex/Ext (raw)', alpha=0.4)
    ax[0].plot(time_s, flex_ext_filt,    label='Flex/Ext (filtered)', lw=2)
    ax[0].set_title('Hip Flexion / Extension vs. Time')

    ax[1].plot(time_s, angles_abd_add, label='Abd/Add (raw)', alpha=0.4)
    ax[1].plot(time_s, abd_add_filt,   label='Abd/Add (filtered)', lw=2)
    ax[1].set_title('Hip Abduction / Adduction vs. Time')

    ax[2].plot(time_s, angles_int_ext, label='Int/Ext (raw)', alpha=0.4)
    ax[2].plot(time_s, int_ext_filt,   label='Int/Ext (filtered)', lw=2)
    ax[2].set_title('Hip Internal / External Rotation vs. Time')

    for a_sub in ax:
        a_sub.set_ylabel('Angle (deg)')
        a_sub.grid(True, linestyle='--', alpha=0.7)
        a_sub.legend()
    ax[2].set_xlabel('Time (s)')
    plt.tight_layout(); plt.show()

    #NEED TO ADD GAIT EVENT DETECTION JUST FOR ANKLES, HAVE NOT WRITTEN CODE FOR IT YET. 
    # --- [E] Detect Gait Cycles in Flex/Ext ----------------------------------
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
    cycles_abd_add  = []
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
        cycles_abd_add.append(np.interp(new_x, old_x,  abd_add_filt[start:end+1]))
        cycles_int_ext.append(np.interp(new_x, old_x,   int_ext_filt[start:end+1]))

    cycles_flex_ext = np.array(cycles_flex_ext)
    cycles_abd_add  = np.array(cycles_abd_add)
    cycles_int_ext  = np.array(cycles_int_ext)

    if cycles_flex_ext.size == 0:
        print("No valid gait cycles found.")
        return
    if cycle_to_remove is not None and 0 <= cycle_to_remove < len(cycles_flex_ext):
        cycles_flex_ext = np.delete(cycles_flex_ext, cycle_to_remove, 0)
        cycles_abd_add  = np.delete(cycles_abd_add,  cycle_to_remove, 0)
        cycles_int_ext  = np.delete(cycles_int_ext,  cycle_to_remove, 0)
        cycle_indices.pop(cycle_to_remove)

    # --- [G] Mean & Std, Filter, Re-zero -------------------------------------
    mean_flex_ext = np.mean(cycles_flex_ext, axis=0)
    std_flex_ext  = np.std(cycles_flex_ext,  axis=0)
    mean_abd_add  = np.mean(cycles_abd_add,  axis=0)
    std_abd_add   = np.std(cycles_abd_add,   axis=0)
    mean_int_ext  = np.mean(cycles_int_ext,  axis=0)
    std_int_ext   = np.std(cycles_int_ext,   axis=0)

    b2, a2 = butter(2, 0.2, btype='low')
    mean_flex_ext = filtfilt(b2, a2, mean_flex_ext)
    std_flex_ext  = filtfilt(b2, a2, std_flex_ext)
    mean_abd_add  = filtfilt(b2, a2, mean_abd_add)
    std_abd_add   = filtfilt(b2, a2, std_abd_add)
    mean_int_ext  = filtfilt(b2, a2, mean_int_ext)
    std_int_ext   = filtfilt(b2, a2, std_int_ext)

    # Re-zero at global minimum of Flex/Ext
    global_min_idx = np.argmin(mean_flex_ext)
    offset_value   = mean_flex_ext[global_min_idx]
    mean_flex_ext   -= offset_value
    mean_flex_ext[global_min_idx] = 0.0
    mean_abd_add   -= offset_value   # preserve relative offset
    mean_int_ext   -= offset_value

    # --- [H] Metrics ---------------------------------------------------------
    durations = [time_s[end] - time_s[start] for start, end in cycle_indices]
    avg_cycle_dur = np.mean(durations)

    idx_min = np.argmin(mean_flex_ext)
    idx_max = np.argmax(mean_flex_ext)
    phase_time = avg_cycle_dur * abs(idx_max - idx_min) / (cycle_length - 1)
    flexion_phase_dur   = phase_time
    extension_phase_dur = avg_cycle_dur - phase_time

    abd_peak  = np.max(mean_abd_add)
    add_peak  = np.min(mean_abd_add)

    introt_peak = np.max(mean_int_ext)
    extrot_peak = np.min(mean_int_ext)

    print(f"Average gait cycle duration:           {avg_cycle_dur:.2f} s")
    print(f"  • Flexion phase duration:            {flexion_phase_dur:.2f} s")
    print(f"  • Extension phase duration:          {extension_phase_dur:.2f} s")
    print(f"Varus/Valgus (mean cycle) peaks:            +{abd_peak:.2f}° / {add_peak:.2f}°")
    print(f"Int/Ext rotation (mean cycle) peaks:   +{introt_peak:.2f}° / {extrot_peak:.2f}°")

    # --- [I] Plot #2: Flex/Ext with shaded cycles ----------------------------
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(time_s, flex_ext_filt, label='Flex/Ext (filtered)')
    ax2.set_title(f'Hip Flex/Ext with Detected Gait Cycles (Ign. first {ignore_initial_s:.1f}s)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (deg)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    for i, (start, end) in enumerate(cycle_indices):
        t_slice = time_s[start:end+1]
        a_slice = flex_ext_filt[start:end+1]
        c = f"C{i%10}"
        ax2.fill_between(t_slice, a_slice, y2=0, color=c, alpha=0.2)
        ax2.axvline(time_s[start], color=c, linestyle='--', alpha=0.7)
        ax2.axvline(time_s[end],   color=c, linestyle='--', alpha=0.7)
    ax2.legend(); plt.tight_layout(); plt.show()

    # --- [J] Plot #3: Mean ± STD for each DOF --------------------------------
    x_cycle = np.linspace(0, 100, cycle_length)
    fig3, ax3 = plt.subplots(3, 1, figsize=(10,10), sharex=True)
    ax3[0].plot(x_cycle, mean_flex_ext, label='Mean Flex/Ext')
    ax3[0].fill_between(x_cycle, mean_flex_ext-std_flex_ext, mean_flex_ext+std_flex_ext,
                        alpha=0.2, label='±1 STD')
    ax3[0].set_title(f'Flex/Ext Mean Cycle (N = {len(durations)} cycles)')

    ax3[1].plot(x_cycle, mean_abd_add, label='Mean Abd/Add')
    ax3[1].fill_between(x_cycle, mean_abd_add-std_abd_add, mean_abd_add+std_abd_add,
                        alpha=0.2, label='±1 STD')
    ax3[1].set_title('Abduction / Adduction Mean Cycle')

    ax3[2].plot(x_cycle, mean_int_ext, label='Mean Int/Ext')
    ax3[2].fill_between(x_cycle, mean_int_ext-std_int_ext, mean_int_ext+std_int_ext,
                        alpha=0.2, label='±1 STD')
    ax3[2].set_title('Internal / External Rotation Mean Cycle')

    for a_sub in ax3:
        a_sub.grid(True, linestyle='--', alpha=0.7)
        a_sub.set_ylabel('Angle (deg)')
        a_sub.legend()
    ax3[2].set_xlabel('Gait Cycle (%)')
    plt.tight_layout(); plt.show()

    # Return values (structure unchanged except for names)
    return (
        time_s,
        (angles_flex_ext, angles_abd_add, angles_int_ext),
        (flex_ext_filt,  abd_add_filt, int_ext_filt),
        cycles_flex_ext, cycles_abd_add, cycles_int_ext,
        mean_flex_ext,  mean_abd_add,  mean_int_ext,
        std_flex_ext,   std_abd_add,   std_int_ext
    )

###############################################################################
# Example Usage
###############################################################################
if __name__ == "__main__":
    filename_pelvis_imu = "AlgosFiles/Raw data/shank.txt" 
    filename_thigh_imu  = "AlgosFiles/Raw data/foot.txt" 

    analyze_two_imus(
        filename_pelvis_imu,
        filename_thigh_imu,
        standing_time_s=5.0,    # first 5 s used to define neutral
        ignore_initial_s=5.0,   # skip first 5 s for peak detection
        lowpass_cutoff=0.15,    # e.g. ~7.5 Hz if sampling at 50 Hz
        margin_before=15,
        margin_after=30,
        required_increase_ratio=1.2,
        cycle_length=101,
        cycle_to_remove=None
    )
