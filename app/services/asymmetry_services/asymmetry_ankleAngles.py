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
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
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
    # ---- Cycle detection parameters ----
    margin_before=10,
    margin_after=10,
    required_increase_ratio=1.2,
    cycle_length=101,
    cycle_to_remove=None,
):
    """
    Reads two IMU CSV files (shank & foot) with Timestamp_us in microseconds,
    calibrates each in the first `standing_time_s` seconds, and computes the
    **RELATIVE ankle angles**:

        1. Dorsi‑/Plantar‑flexion (previously Flex/Ext)
        2. Inversion/Eversion (previously Abd/Add)
        3. Internal/External Rotation (unchanged)

    The underlying maths **is unchanged** from the knee version – only the
    *nomenclature* and visual labelling have been updated for the ankle.

    In addition to all previous plots and returns, the function now prints the
    peak (max & min) values of the three filtered angle traces to the terminal.
    """

    ###########################################################################
    # (A) Read & Calibrate – identical to the knee version
    ###########################################################################
    df_thigh = pd.read_csv(filename_thigh, header=0, sep=",", skipinitialspace=True)
    t_thigh_us = df_thigh["Timestamp_us"].values
    Wt, Xt, Yt, Zt = (
        df_thigh["QuatW"],
        df_thigh["QuatX"],
        df_thigh["QuatY"],
        df_thigh["QuatZ"],
    )

    # Calibrate shank (was thigh) in first standing_time_s seconds
    stand_thresh_thigh_us = t_thigh_us[0] + standing_time_s * 1e6
    mask_thigh = t_thigh_us <= stand_thresh_thigh_us
    quats_thigh_stand = list(zip(Wt[mask_thigh], Xt[mask_thigh], Yt[mask_thigh], Zt[mask_thigh]))
    if len(quats_thigh_stand) < 10:
        print("Warning: not enough standing data for IMU 1. Using identity as neutral.")
        R0_thigh = np.eye(3)
    else:
        R0_thigh = average_rotation_matrix(quats_thigh_stand)

    # Read foot CSV (was shank)
    df_shank = pd.read_csv(filename_shank, header=0, sep=",", skipinitialspace=True)
    t_shank_us = df_shank["Timestamp_us"].values
    Ws, Xs, Ys, Zs = (
        df_shank["QuatW"],
        df_shank["QuatX"],
        df_shank["QuatY"],
        df_shank["QuatZ"],
    )

    # Calibrate foot IMU
    stand_thresh_shank_us = t_shank_us[0] + standing_time_s * 1e6
    mask_shank = t_shank_us <= stand_thresh_shank_us
    quats_shank_stand = list(zip(Ws[mask_shank], Xs[mask_shank], Ys[mask_shank], Zs[mask_shank]))
    if len(quats_shank_stand) < 10:
        print("Warning: not enough standing data for IMU 2. Using identity as neutral.")
        R0_shank = np.eye(3)
    else:
        R0_shank = average_rotation_matrix(quats_shank_stand)

    # Convert to a common time base (seconds) for alignment
    df_thigh["time_s"] = (df_thigh["Timestamp_us"] - df_thigh["Timestamp_us"].iloc[0]) * 1e-6
    df_shank["time_s"] = (df_shank["Timestamp_us"] - df_shank["Timestamp_us"].iloc[0]) * 1e-6

    # Merge by rounded time
    df_thigh_rounded = df_thigh.copy()
    df_shank_rounded = df_shank.copy()
    df_thigh_rounded["time_s"] = df_thigh_rounded["time_s"].round(2)
    df_shank_rounded["time_s"] = df_shank_rounded["time_s"].round(2)

    df_merged = pd.merge(
        df_thigh_rounded,
        df_shank_rounded,
        on="time_s",
        how="inner",
        suffixes=("_thigh", "_shank"),
    )
    time_s = df_merged["time_s"].values

    # Extract merged quaternions
    Wt_m = df_merged["QuatW_thigh"].values
    Xt_m = df_merged["QuatX_thigh"].values
    Yt_m = df_merged["QuatY_thigh"].values
    Zt_m = df_merged["QuatZ_thigh"].values

    Ws_m = df_merged["QuatW_shank"].values
    Xs_m = df_merged["QuatX_shank"].values
    Ys_m = df_merged["QuatY_shank"].values
    Zs_m = df_merged["QuatZ_shank"].values

    ###########################################################################
    # (B) Compute Relative Rotations & Euler Angles – no maths changed
    ###########################################################################
    angles_abd = []  # will hold Dorsi/Plantar‑flexion
    angles_rot = []  # will hold Inversion/Eversion
    angles_flex = []  # will hold Internal/External rotation

    for i in range(len(df_merged)):
        # IMU 1
        q_thigh = (Wt_m[i], Xt_m[i], Yt_m[i], Zt_m[i])
        R_thigh = quaternion_to_matrix(q_thigh)
        R_thigh_prime = R0_thigh.T @ R_thigh

        # IMU 2
        q_shank = (Ws_m[i], Xs_m[i], Ys_m[i], Zs_m[i])
        R_shank = quaternion_to_matrix(q_shank)
        R_shank_prime = R0_shank.T @ R_shank

        # Relative rotation: IMU1' -> IMU2'
        R_rel = R_thigh_prime.T @ R_shank_prime
        alpha_z, beta_y, gamma_x = matrix_to_euler_ZYX(R_rel)

        # Map to ankle nomenclature
        angles_abd.append(math.degrees(beta_y))  # Dorsi/Plantar‑flexion
        angles_rot.append(-math.degrees(alpha_z))  # Inversion/Eversion
        angles_flex.append(math.degrees(gamma_x))  # Internal/External rotation

    angles_abd = np.array(angles_abd)
    angles_rot = np.array(angles_rot)
    angles_flex = np.array(angles_flex)

    ###########################################################################
    # (C) Low‑Pass Filter (unchanged)
    ###########################################################################
    b, a = butter(1, lowpass_cutoff, btype="low")
    abd_filt = filtfilt(b, a, angles_abd)
    rot_filt = filtfilt(b, a, angles_rot)
    flex_filt = filtfilt(b, a, angles_flex)

    # -----------------------------------------------------------
    # *** PRINT PEAK VALUES (new feature) ***
    # -----------------------------------------------------------
    peak_dpf_max = np.max(abd_filt)
    peak_dpf_min = np.min(abd_filt)
    peak_inv_max = np.max(rot_filt)
    peak_inv_min = np.min(rot_filt)
    peak_i_er_max = np.max(flex_filt)
    peak_i_er_min = np.min(flex_filt)

    print("\n========= PEAK ANKLE ANGLES =========")
    print(f"Dorsi/Plantar‑flexion   : max = {peak_dpf_max: .2f}°, min = {peak_dpf_min: .2f}°")
    print(f"Inversion/Eversion      : max = {peak_inv_max: .2f}°, min = {peak_inv_min: .2f}°")
    print(f"Internal/External Rot.  : max = {peak_i_er_max: .2f}°, min = {peak_i_er_min: .2f}°")
    print("====================================\n")

    ###########################################################################
    # (D) Plot #1: Raw vs. Filtered Angles – labels updated for ankle
    ###########################################################################
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Dorsi/Plantar‑flexion
    ax[0].plot(time_s, angles_abd, label="Dorsi/Plantar (raw)", alpha=0.4)
    ax[0].plot(time_s, abd_filt, label="Dorsi/Plantar (filtered)", lw=2)
    ax[0].set_ylabel("Angle (deg)")
    ax[0].set_title("Ankle Dorsi/Plantar‑flexion vs. Time")
    ax[0].grid(True, linestyle="--", alpha=0.7)
    ax[0].legend()

    # Inversion/Eversion
    ax[1].plot(time_s, angles_rot, label="Inv/Ev (raw)", alpha=0.4)
    ax[1].plot(time_s, rot_filt, label="Inv/Ev (filtered)", lw=2)
    ax[1].set_ylabel("Angle (deg)")
    ax[1].set_title("Ankle Inversion/Eversion vs. Time")
    ax[1].grid(True, linestyle="--", alpha=0.7)
    ax[1].legend()

    # Internal/External Rotation
    ax[2].plot(time_s, angles_flex, label="Int/Ext Rot (raw)", alpha=0.4)
    ax[2].plot(time_s, flex_filt, label="Int/Ext Rot (filtered)", lw=2)
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Angle (deg)")
    ax[2].set_title("Ankle Internal/External Rotation vs. Time")
    ax[2].grid(True, linestyle="--", alpha=0.7)
    ax[2].legend()

    #plt.tight_layout()
    #plt.show()

    #NEED TO ADD GAIT EVENT DETECTION JUST FOR ANKLES, HAVE NOT WRITTEN CODE FOR IT YET. 

    ###########################################################################
    # (J) Return – unchanged
    ###########################################################################
    return (
        time_s,
        (angles_abd, angles_rot, angles_flex),
        (abd_filt, rot_filt, flex_filt),
        None, None, None,  # placeholders if later sections removed here
        None, None, None,
        None, None, None,
    )


###############################################################################
# Example Usage (paths unchanged – replace with your own)
###############################################################################
# if __name__ == "__main__":
#     filename_shank_imu = "AlgosFiles/Raw data/shank.txt"  # IMU on shank
#     filename_foot_imu = "AlgosFiles/Raw data/foot.txt"    # IMU on foot

#     analyze_two_imus(
#         filename_shank_imu,
#         filename_foot_imu,
#         standing_time_s=5.0,
#         ignore_initial_s=5.0,
#         lowpass_cutoff=0.15,
#         margin_before=15,
#         margin_after=30,
#         required_increase_ratio=1.2,
#         cycle_length=101,
#         cycle_to_remove=None,
#     )
