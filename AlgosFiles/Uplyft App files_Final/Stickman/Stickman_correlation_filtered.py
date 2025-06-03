import numpy as np
import pandas as pd
import scipy.interpolate
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, correlate
import imumocap  # Your IMU motion capture library

# =============================================================================
# 1. Helper Functions
# =============================================================================

def read_imu_csv(filepath):
    """
    Reads a CSV file with columns: Timestamp, QuatW, QuatX, QuatY, QuatZ.
    Converts timestamp to seconds and normalizes quaternions.
    """
    try:
        df = pd.read_csv(filepath)  # Assuming comma-separated
        # Remove any leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()
        time = df['Timestamp'].values / 1000.0  # Convert ms -> seconds
        quaternions = df[['QuatW', 'QuatX', 'QuatY', 'QuatZ']].values
        
        # Prevent division by zero in normalization
        norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
        norms[norms == 0] = 1
        quaternions /= norms  # Normalize quaternions
        
        return time, quaternions
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        raise
    except KeyError as e:
        print(f"Missing column in {filepath}: {e}")
        raise

def smooth_quaternions(quats, window_length=5, polyorder=2):
    """
    Apply Savitzky-Golay filter to each quaternion component for noise reduction,
    then re-normalize to maintain unit quaternions.
    
    - quats: (N, 4) array of quaternion samples over time.
    - window_length, polyorder: parameters for Savitzky-Golay.
      Make sure window_length is odd and <= number of samples.
    """
    quats_smoothed = np.copy(quats)
    n_samples = quats.shape[0]

    # If the data is too short for the default window_length, reduce it
    if window_length > n_samples:
        window_length = max(3, n_samples if n_samples % 2 == 1 else n_samples - 1)

    for i in range(4):
        quats_smoothed[:, i] = savgol_filter(
            quats[:, i], window_length=window_length, polyorder=polyorder
        )

    # Normalize to restore unit magnitude
    norms = np.linalg.norm(quats_smoothed, axis=1, keepdims=True)
    norms[norms == 0] = 1
    quats_smoothed /= norms
    return quats_smoothed

def cross_correlate_time_shift(ref_time, ref_quats, cmp_time, cmp_quats, max_shift=2.0, dt=0.01):
    """
    Estimate the time shift between two quaternion signals by cross-correlating
    the w-component (quats[:,0]) after both are resampled onto a uniform time grid.
    
    - ref_time, ref_quats: reference time/quaternion arrays
    - cmp_time, cmp_quats: comparison time/quaternion arrays
    - max_shift: maximum +/- shift in seconds to consider
    - dt: resampling interval for cross-correlation
    
    Returns: time_shift (float), meaning cmp_time should be shifted by this amount
    to best align with ref_time.
    """
    if len(ref_time) < 2 or len(cmp_time) < 2:
        # Not enough data to correlate
        return 0.0

    # Build a uniform time axis that covers the entire range of both signals
    all_start = min(ref_time[0], cmp_time[0])
    all_end   = max(ref_time[-1], cmp_time[-1])
    uniform_t = np.arange(all_start, all_end, dt)

    # Interpolate w-components onto that uniform grid
    ref_w = np.interp(uniform_t, ref_time, ref_quats[:, 0])
    cmp_w = np.interp(uniform_t, cmp_time, cmp_quats[:, 0])

    # Limit correlation lag to ±(max_shift / dt) samples
    max_lag_samples = int(np.round(max_shift / dt))

    # Detrend by subtracting mean to reduce bias
    ref_w -= np.mean(ref_w)
    cmp_w -= np.mean(cmp_w)

    # Full cross-correlation
    corr = correlate(ref_w, cmp_w, mode='full')

    # Lags array: from -(len(cmp_w)-1) to +(len(ref_w)-1)
    lags = np.arange(-len(cmp_w)+1, len(ref_w))

    # Keep only the valid portion of lags within ± max_lag_samples
    valid_idx = np.where((lags >= -max_lag_samples) & (lags <= max_lag_samples))[0]
    corr_valid = corr[valid_idx]
    lags_valid = lags[valid_idx]

    # Best shift is the lag where |corr| is largest
    best_shift_samples = lags_valid[np.argmax(np.abs(corr_valid))]

    # Convert shift in samples -> shift in seconds
    time_shift = best_shift_samples * dt

    return time_shift

def shift_timestamps(time_array, shift):
    """
    Shifts all timestamps by `shift` seconds (can be positive or negative).
    """
    return time_array + shift

# =============================================================================
# 2. Load All IMU Data
# =============================================================================

# Paths to IMU data files (same as before)
pelvis_path = "/Users/mahadparwaiz/Desktop/test/raw data_100Hz/walking_mahad_100Hz_141degSE/pelvis/data_2.txt"
left_upper_leg_path = "/Users/mahadparwaiz/Desktop/test/raw data_100Hz/walking_mahad_100Hz_141degSE/left leg up/data_2.txt"
left_lower_leg_path = "/Users/mahadparwaiz/Desktop/test/raw data_100Hz/walking_mahad_100Hz_141degSE/left leg low/data_3.txt"
right_upper_leg_path = "/Users/mahadparwaiz/Desktop/test/raw data_100Hz/walking_mahad_100Hz_141degSE/right leg up/data_6.txt"
right_lower_leg_path = "/Users/mahadparwaiz/Desktop/test/raw data_100Hz/walking_mahad_100Hz_141degSE/right leg low/data_3.txt"

# Load raw data (time in seconds, quaternions in w,x,y,z)
time_pelvis, quat_pelvis = read_imu_csv(pelvis_path)
time_left_upper, quat_left_upper = read_imu_csv(left_upper_leg_path)
time_left_lower, quat_left_lower = read_imu_csv(left_lower_leg_path)
time_right_upper, quat_right_upper = read_imu_csv(right_upper_leg_path)
time_right_lower, quat_right_lower = read_imu_csv(right_lower_leg_path)

# =============================================================================
# 3. Cross-Correlation for Time Offset Alignment
# =============================================================================
# Let's pick PELVIS as our reference. We will align all other IMUs to pelvis.

# 3.a) Estimate time shifts relative to pelvis
max_shift_seconds = 2.0  # If you think IMUs might be offset up to +/-2s
dt_for_cc = 0.01         # Step size for correlation resampling

shift_left_upper  = cross_correlate_time_shift(time_pelvis, quat_pelvis,
                                               time_left_upper, quat_left_upper,
                                               max_shift=max_shift_seconds, dt=dt_for_cc)

shift_left_lower  = cross_correlate_time_shift(time_pelvis, quat_pelvis,
                                               time_left_lower, quat_left_lower,
                                               max_shift=max_shift_seconds, dt=dt_for_cc)

shift_right_upper = cross_correlate_time_shift(time_pelvis, quat_pelvis,
                                               time_right_upper, quat_right_upper,
                                               max_shift=max_shift_seconds, dt=dt_for_cc)

shift_right_lower = cross_correlate_time_shift(time_pelvis, quat_pelvis,
                                               time_right_lower, quat_right_lower,
                                               max_shift=max_shift_seconds, dt=dt_for_cc)

print(f"[INFO] Time shift L_U: {shift_left_upper:.3f} s")
print(f"[INFO] Time shift L_L: {shift_left_lower:.3f} s")
print(f"[INFO] Time shift R_U: {shift_right_upper:.3f} s")
print(f"[INFO] Time shift R_L: {shift_right_lower:.3f} s")

# 3.b) Shift the timestamps so that they (hopefully) align to pelvis
time_left_upper  = shift_timestamps(time_left_upper,  shift_left_upper)
time_left_lower  = shift_timestamps(time_left_lower,  shift_left_lower)
time_right_upper = shift_timestamps(time_right_upper, shift_right_upper)
time_right_lower = shift_timestamps(time_right_lower, shift_right_lower)

# =============================================================================
# 4. (Optional) Apply Smoothing to Each Quaternion Time Series

# =============================================================================
"""
quat_pelvis      = smooth_quaternions(quat_pelvis,      window_length=7, polyorder=2)
quat_left_upper  = smooth_quaternions(quat_left_upper,  window_length=7, polyorder=2)
quat_left_lower  = smooth_quaternions(quat_left_lower,  window_length=7, polyorder=2)
quat_right_upper = smooth_quaternions(quat_right_upper, window_length=7, polyorder=2)
quat_right_lower = smooth_quaternions(quat_right_lower, window_length=7, polyorder=2)
"""
# =============================================================================
# 5. Resample Data onto a Common Time Axis
# =============================================================================
# After alignment + smoothing, define your common timeline & interpolate
start_time = max(time_pelvis[0], time_left_upper[0], time_left_lower[0],
                 time_right_upper[0], time_right_lower[0])
end_time = min(time_pelvis[-1], time_left_upper[-1], time_left_lower[-1],
               time_right_upper[-1], time_right_lower[-1])

FPS = 100  # Frame rate for visualization
common_time = np.arange(start_time, end_time, 1.0 / FPS)

def interpolate_quaternion(time_in, quats_in, time_out):
    """Interpolates quaternions linearly to match a common time axis."""
    # Using linear interpolation for each component
    qw = scipy.interpolate.interp1d(time_in, quats_in[:, 0], kind='linear', fill_value="extrapolate")(time_out)
    qx = scipy.interpolate.interp1d(time_in, quats_in[:, 1], kind='linear', fill_value="extrapolate")(time_out)
    qy = scipy.interpolate.interp1d(time_in, quats_in[:, 2], kind='linear', fill_value="extrapolate")(time_out)
    qz = scipy.interpolate.interp1d(time_in, quats_in[:, 3], kind='linear', fill_value="extrapolate")(time_out)
    quats_out = np.vstack([qw, qx, qy, qz]).T
    
    # Re-normalize to maintain unit quaternions
    norms = np.linalg.norm(quats_out, axis=1, keepdims=True)
    norms[norms == 0] = 1
    quats_out /= norms
    return quats_out

# Now resample
quat_pelvis_resampled = interpolate_quaternion(time_pelvis,      quat_pelvis,      common_time)
quat_left_upper_resampled  = interpolate_quaternion(time_left_upper,  quat_left_upper,  common_time)
quat_left_lower_resampled  = interpolate_quaternion(time_left_lower,  quat_left_lower,  common_time)
quat_right_upper_resampled = interpolate_quaternion(time_right_upper, quat_right_upper, common_time)
quat_right_lower_resampled = interpolate_quaternion(time_right_lower, quat_right_lower, common_time)

# =============================================================================
# 6. Create Kinematic Model (Same as Original)
# =============================================================================
PELVIS_LENGTH = 1.0
PELVIS_WIDTH  = 2.0
UPPER_LEG_LENGTH = 4.0
LOWER_LEG_LENGTH = 4.0
FOOT_LENGTH  = 1.0  # Placeholder

# Left Leg Links
left_foot = imumocap.Link("left_foot", FOOT_LENGTH)  # Placeholder
left_lower_leg = imumocap.Link("left_lower_leg", LOWER_LEG_LENGTH,
    [(left_foot, imumocap.Link.matrix(pitch=-90, x=LOWER_LEG_LENGTH))])
left_upper_leg = imumocap.Link("left_upper_leg", UPPER_LEG_LENGTH,
    [(left_lower_leg, imumocap.Link.matrix(x=UPPER_LEG_LENGTH))])

# Right Leg Links
right_foot = imumocap.Link("right_foot", FOOT_LENGTH)  # Placeholder
right_lower_leg = imumocap.Link("right_lower_leg", LOWER_LEG_LENGTH,
    [(right_foot, imumocap.Link.matrix(pitch=-90, x=LOWER_LEG_LENGTH))])
right_upper_leg = imumocap.Link("right_upper_leg", UPPER_LEG_LENGTH,
    [(right_lower_leg, imumocap.Link.matrix(x=UPPER_LEG_LENGTH))])

# Pelvis Link with Both Legs Attached
pelvis = imumocap.Link("pelvis", PELVIS_LENGTH,
    [
        (left_upper_leg,  imumocap.Link.matrix(y=PELVIS_WIDTH / 2, roll=180, yaw=180)),
        (right_upper_leg, imumocap.Link.matrix(y=-PELVIS_WIDTH / 2, roll=180, yaw=180))
    ])

# =============================================================================
# 7. Calibration (Align Initial Pose)
# =============================================================================
pelvis.joint = imumocap.Link.matrix(pitch=-90)  # Example upright orientation

# Set IMU global orientations using the first quaternion sample
pelvis.set_imu_global(imumocap.Link.matrix(quaternion=quat_pelvis_resampled[0]))
left_upper_leg.set_imu_global(imumocap.Link.matrix(quaternion=quat_left_upper_resampled[0]))
left_lower_leg.set_imu_global(imumocap.Link.matrix(quaternion=quat_left_lower_resampled[0]))
right_upper_leg.set_imu_global(imumocap.Link.matrix(quaternion=quat_right_upper_resampled[0]))
right_lower_leg.set_imu_global(imumocap.Link.matrix(quaternion=quat_right_lower_resampled[0]))

# =============================================================================
# 8. Build Frames for Animation
# =============================================================================
frames = []
for i in range(len(common_time)):
    pelvis.set_joint_from_imu_global(imumocap.Link.matrix(quaternion=quat_pelvis_resampled[i]))
    left_upper_leg.set_joint_from_imu_global(imumocap.Link.matrix(quaternion=quat_left_upper_resampled[i]))
    left_lower_leg.set_joint_from_imu_global(imumocap.Link.matrix(quaternion=quat_left_lower_resampled[i]))
    right_upper_leg.set_joint_from_imu_global(imumocap.Link.matrix(quaternion=quat_right_upper_resampled[i]))
    right_lower_leg.set_joint_from_imu_global(imumocap.Link.matrix(quaternion=quat_right_lower_resampled[i]))
    frames.append([l.joint.copy() for l in imumocap.Link.flatten(pelvis)])

# =============================================================================
# 9. Visualization
# =============================================================================
# Static Pose Visualization
imumocap.plot(pelvis, elev=10, azim=45, figsize=(10, 6))
plt.show()

# Animation Visualization and GIF Export
imumocap.plot(
    pelvis,
    frames,
    fps=FPS,
    elev=10,
    azim=45,
    file_name="dual_leg_animation2.gif",
    figsize=(10, 6),
    dpi=120
)
