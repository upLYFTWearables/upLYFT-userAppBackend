import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

###############################################################################
# 1) Quaternion & Matrix Helpers
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
    Compute an approximate average rotation matrix from a list of quaternions
    by converting each to a rotation matrix, summing, then using SVD to
    extract the best orthonormal approximation.
    """
    if len(quaternions) == 0:
        return np.eye(3)

    R_sum = np.zeros((3,3))
    for q in quaternions:
        R_sum += quaternion_to_matrix(q)

    R_sum /= len(quaternions)
    U, _, Vt = np.linalg.svd(R_sum)
    R_avg = U @ Vt
    return R_avg

def matrix_to_euler_ZYX(R):
    """
    Extract Euler angles from a rotation matrix R using a Z-Y-X sequence:
        R = Rz(yaw) * Ry(pitch) * Rx(roll)

    Returns (yaw [alpha_z], pitch [beta_y], roll [gamma_x]) in radians.
    """
    beta_y = -math.asin(R[2, 0])       # pitch
    cos_beta = math.cos(beta_y)
    if abs(cos_beta) < 1e-6:
        # Gimbal lock fallback
        alpha_z = math.atan2(-R[0, 1], R[1, 1])  # yaw
        gamma_x = 0.0                            # roll
    else:
        alpha_z = math.atan2(R[1, 0], R[0, 0])   # yaw
        gamma_x = math.atan2(R[2, 1], R[2, 2])   # roll

    return alpha_z, beta_y, gamma_x

###############################################################################
# 2) Main Function to Analyze Pelvis IMU
###############################################################################
def analyze_pelvis_imu(
    filename,
    standing_time_s=5.0,
    lowpass_cutoff=0.3
):
    """
    Reads a single-IMU CSV (pelvis-mounted), calibrates using the first
    'standing_time_s' seconds, and computes 3 pelvic angles over time:
      - Pelvic Tilt (pitch)
      - Pelvic Obliquity (roll)
      - Pelvic Rotation (yaw)

    Parameters
    ----------
    filename : str
        CSV file with at least the following columns:
          [Timestamp_us, QuatW, QuatX, QuatY, QuatZ]
    standing_time_s : float
        How many seconds from the start to consider "neutral stance"
    lowpass_cutoff : float
        Normalized cutoff frequency (fraction of Nyquist) for
        the Butterworth low-pass filter.

    Returns
    -------
    time_s : np.array
        Time vector in seconds, aligned to the start of recording.
    (pelvis_tilt, pelvis_obl, pelvis_rot) : tuple of np.arrays
        Unfiltered angle traces in degrees.
        - pelvis_tilt        ~ pitch (forward/backward tilt)
        - pelvis_obl         ~ roll (drop on left vs. right side)
        - pelvis_rot         ~ yaw  (transverse rotation)
    """
    # A) Load Data
    df = pd.read_csv(filename, header=0, sep=',', skipinitialspace=True)
    t_us = df["Timestamp_us"].values
    W = df["QuatW"].values
    X = df["QuatX"].values
    Y = df["QuatY"].values
    Z = df["QuatZ"].values

    # Convert microseconds to seconds
    time_s = (t_us - t_us[0]) * 1e-6

    # B) Calibrate in first standing_time_s seconds
    stand_end_us = t_us[0] + standing_time_s * 1e6
    stand_mask = (t_us <= stand_end_us)
    quats_stand = list(zip(W[stand_mask], X[stand_mask], Y[stand_mask], Z[stand_mask]))

    if len(quats_stand) < 10:
        print("Warning: not enough standing data. Using identity for calibration.")
        R0 = np.eye(3)
    else:
        R0 = average_rotation_matrix(quats_stand)  # "neutral" orientation

    # C) Compute Pelvic Angles for Each Sample
    pelvis_tilt = []  # pitch  (forward/backward tilt)
    pelvis_obl  = []  # roll   (obliquity in frontal plane)
    pelvis_rot  = []  # yaw    (rotation in transverse plane)

    for i in range(len(df)):
        # Current quaternion -> matrix
        q = (W[i], X[i], Y[i], Z[i])
        R_current = quaternion_to_matrix(q)

        # Transform by R0^T so that the standing orientation is identity
        R_cal = R0.T @ R_current

        # Euler angles from the calibrated rotation
        alpha_z, beta_y, gamma_x = matrix_to_euler_ZYX(R_cal)

        # alpha_z => yaw    => pelvic rotation
        # beta_y  => pitch  => pelvic tilt
        # gamma_x => roll   => pelvic obliquity
        pelvis_rot.append(math.degrees(alpha_z))
        pelvis_tilt.append(math.degrees(-beta_y))
        pelvis_obl.append(math.degrees(gamma_x))

    pelvis_tilt = np.array(pelvis_tilt)
    pelvis_obl  = np.array(pelvis_obl)
    pelvis_rot  = np.array(pelvis_rot)

    # D) Low-Pass Filter (Optional)
    b, a = butter(2, lowpass_cutoff, btype='low')
    tilt_filt = filtfilt(b, a, pelvis_tilt)
    obl_filt  = filtfilt(b, a, pelvis_obl)
    rot_filt  = filtfilt(b, a, pelvis_rot)

    # E) Plot Results
    fig, ax = plt.subplots(3, 1, figsize=(10,10), sharex=True)

    # 1) Pelvic Tilt
   # ax[0].plot(time_s, pelvis_tilt, label='Tilt (raw)', alpha=0.4)
    ax[0].plot(time_s, tilt_filt, label='Tilt (filtered)', lw=2)
    ax[0].set_ylabel('Angle (deg)')
    ax[0].set_title('Pelvic Tilt vs. Time')
    ax[0].grid(True, linestyle='--', alpha=0.7)
    ax[0].legend()

    # 2) Pelvic Obliquity
    #ax[1].plot(time_s, pelvis_obl, label='Obliquity (raw)', alpha=0.4)
    ax[1].plot(time_s, obl_filt, label='Rotation (filtered)', lw=2)
    ax[1].set_ylabel('Angle (deg)')
    ax[1].set_title('Pelvic Rotation vs. Time')
    ax[1].grid(True, linestyle='--', alpha=0.7)
    ax[1].legend()

    # 3) Pelvic Rotation
   # ax[2].plot(time_s, pelvis_rot, label='Rotation (raw)', alpha=0.4)
    ax[2].plot(time_s, rot_filt, label='Obliquity (filtered)', lw=2)
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Angle (deg)')
    ax[2].set_title('Pelvic Obliquity vs. Time')
    ax[2].grid(True, linestyle='--', alpha=0.7)
    ax[2].legend()

    plt.tight_layout()
    plt.show()

    # F) Return the data
    return time_s, (pelvis_tilt, pelvis_obl, pelvis_rot)


###############################################################################
# Example usage (uncomment if you want to run this script directly)
###############################################################################
if __name__ == "__main__":
    filename_single_imu = "/Users/mahadparwaiz/Desktop/HIp isolated/walk4/data_65.txt"  # update
    standing_time_s = 4.0
    lowpass_cutoff = 0.2

    time_s, (tilt_deg, obl_deg, rot_deg) = analyze_pelvis_imu(
        filename_single_imu,
        standing_time_s=standing_time_s, 
        lowpass_cutoff=lowpass_cutoff
    )

    # tilt_deg, obl_deg, rot_deg contain the unfiltered angles in degrees.
    # The script will also produce plots of raw vs. filtered angles.
