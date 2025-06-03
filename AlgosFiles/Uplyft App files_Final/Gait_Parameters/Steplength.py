import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

def load_imu_file(filepath, delimiter=","):
    """
    Load an IMU .txt file with columns:
    Timestamp, AccelX, AccelY, AccelZ, GyroX, GyroY, GyroZ, MagX, MagY, MagZ, QuatW, QuatX, QuatY, QuatZ.
    
    :param filepath: Path to the IMU .txt file.
    :param delimiter: Delimiter used in the .txt file (default is tab).
    :return: Pandas DataFrame with columns named accordingly.
    """
    # Adjust header row or skip rows as needed depending on your file structure.
    # If there's a header line with column names, use header=0. 
    # If there is no header, pass header=None and define column names manually.
    col_names = [
        "Timestamp", " AccelX", " AccelY", " AccelZ", 
        " GyroX", " GyroY", " GyroZ", 
        " MagX", " MagY", " MagZ", 
        " QuatW", " QuatX", " QuatY", " QuatZ"
    ]
    
    df = pd.read_csv(filepath, delimiter=delimiter, names=col_names, header=0)
    
    return df

def butter_filter(data, cutoff, fs, order=4, btype='low'):
    """
    Apply a Butterworth filter (low-pass by default) to the data.
    
    :param data: 1D array of data to filter.
    :param cutoff: cutoff frequency (Hz) for the filter.
    :param fs: sampling frequency (Hz).
    :param order: order of the Butterworth filter.
    :param btype: filter type ('low', 'high', 'bandpass', etc.).
    :return: filtered data.
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype=btype, analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def detect_steps(accel_signal, timestamps, height_threshold=1.0, distance=100):
    """
    Detect steps based on peak detection of an acceleration signal.

    :param accel_signal: 1D array of the acceleration magnitude (or any relevant signal).
    :param timestamps: 1D array of timestamps (in seconds or milliseconds).
    :param height_threshold: Minimum height for a valid peak.
    :param distance: Minimum distance between peaks (in samples).
    :return: indices of detected step peaks, array of step timestamps.
    """
    # find_peaks returns an array of indices where peaks occur
    peaks, properties = find_peaks(accel_signal, height=height_threshold, distance=distance)
    
    # Convert indices to timestamps if needed
    step_times = timestamps[peaks]
    
    return peaks, step_times

def process_imu_data(df, sensor_name="Pelvis", fs=None, 
                     filter_cutoff=5.0, height_threshold=1.0, distance=100):
    """
    Process the IMU DataFrame to detect step count and step cadence.

    :param df: Pandas DataFrame with columns [Timestamp, AccelX, AccelY, AccelZ, ...]
    :param sensor_name: For labeling plots or debug messages.
    :param fs: Sampling frequency in Hz (if known). If not known, it will be estimated from timestamps.
    :param filter_cutoff: cutoff frequency for optional low-pass filter (Hz).
    :param height_threshold: threshold for peak detection (adjust as needed).
    :param distance: minimum distance (in samples) between peaks for peak detection.
    :return: step_count, step_cadence (steps/min), (peaks, step_times).
    """
    
    # 1) Convert Timestamp from ms to seconds (optional, for convenience)
    #    If you prefer to keep it in ms, skip the below line and adapt accordingly.
    df["TimeSec"] = df["Timestamp"] / 1000.0
    
    # 2) Estimate sampling frequency if not provided
    if fs is None:
        # Average difference between consecutive timestamps
        time_diffs = np.diff(df["TimeSec"])
        avg_time_diff = np.mean(time_diffs)  # in seconds
        if avg_time_diff > 0:
            fs = 1.0 / avg_time_diff
        else:
            raise ValueError("Could not estimate sampling frequency (fs). Check timestamps.")
    
    # 3) Compute the magnitude of acceleration
    #    Optionally, you can remove gravity by subtracting mean or applying a high-pass filter.
    accel_x = df[" AccelX"].values
    accel_y = df[" AccelY"].values
    accel_z = df[" AccelZ"].values
    
    accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    
    # 4) (Optional) Filter the acceleration magnitude to reduce noise
    filtered_accel = butter_filter(accel_magnitude, cutoff=filter_cutoff, fs=fs, order=4, btype='low')
    
    # 5) Detect steps using peak detection
    peaks, step_times = detect_steps(
        filtered_accel,
        df["TimeSec"].values,
        height_threshold=height_threshold,
        distance=distance
    )
    
    # 6) Compute step count
    step_count = len(peaks)
    
    # 7) Compute total time (in seconds) and step cadence (steps per minute)
    total_time_sec = df["TimeSec"].iloc[-1] - df["TimeSec"].iloc[0]
    step_cadence = (step_count / total_time_sec) * 60.0  # steps/min
    
    print(f"--- {sensor_name} ---")
    print(f"Estimated Sampling Frequency: {fs:.2f} Hz")
    print(f"Step Count: {step_count}")
    print(f"Total Duration: {total_time_sec:.2f} s")
    print(f"Step Cadence: {step_cadence:.2f} steps/min")
    
    return step_count, step_cadence, (peaks, step_times)

def main():
    """
    Main function to read multiple IMU files, process them, and optionally plot the results.
    Modify the file paths, sampling frequency (if known), filter parameters, 
    and detection thresholds as needed.
    """
    # Example file paths (adjust to your own)
    imu_files = {
        "Left_Below_Knee": "/Users/mahadparwaiz/Desktop/test/Raw data/our own data/raw data_100Hz/walking_mahad_100Hz_141degSE/left leg low/data_3.txt",
        "Left_Above_Knee": "/Users/mahadparwaiz/Desktop/test/Raw data/our own data/raw data_100Hz/walking_mahad_100Hz_141degSE/left leg up/data_2.txt",
        "Right_Below_Knee": "/Users/mahadparwaiz/Desktop/test/Raw data/our own data/raw data_100Hz/walking_mahad_100Hz_141degSE/right leg low/data_3.txt",
        "Right_Above_Knee": "/Users/mahadparwaiz/Desktop/test/Raw data/our own data/raw data_100Hz/walking_mahad_100Hz_141degSE/right leg up/data_6.txt",
        "Pelvis": "/Users/mahadparwaiz/Desktop/test/Raw data/our own data/raw data_100Hz/walking_mahad_100Hz_141degSE/pelvis/data_2.txt"
    }
    
    # Known or approximate sampling frequency in Hz (if you know it).
    # If you don't know it, set it to None and the script will estimate from data.
    fs = None
    
    # Parameters for detection and filtering
    filter_cutoff = 5.0      # (Hz) typical low-pass filter cutoff for walking
    height_threshold = 1.0   # Adjust based on your data amplitude
    distance = 100           # Minimum number of samples between peaks (depends on walking speed, fs)
    
    results = {}
    
    for sensor_name, filepath in imu_files.items():
        if not os.path.isfile(filepath):
            print(f"File not found: {filepath}")
            continue
        
        # Load data
        df = load_imu_file(filepath)
        
        # Process data
        step_count, step_cadence, (peaks, step_times) = process_imu_data(
            df, 
            sensor_name=sensor_name,
            fs=fs,
            filter_cutoff=filter_cutoff,
            height_threshold=height_threshold,
            distance=distance
        )
        
        # Store results
        results[sensor_name] = {
            "step_count": step_count,
            "step_cadence": step_cadence,
            "peaks": peaks,
            "step_times": step_times
        }
        
        # Optionally, plot the results for debugging
        # Magnitude of acceleration
        accel_x = df[" AccelX"].values
        accel_y = df[" AccelY"].values
        accel_z = df[" AccelZ"].values
        accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        plt.figure(figsize=(10, 4))
        plt.plot(df["TimeSec"], accel_magnitude, label="Raw Accel Magnitude", alpha=0.5)
        plt.plot(df["TimeSec"].iloc[results[sensor_name]["peaks"]],
                 accel_magnitude[results[sensor_name]["peaks"]], 
                 'rx', label='Detected Steps')
        plt.title(f"Detected Steps - {sensor_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration Magnitude (m/s^2 or G)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Example of how to access results programmatically
    for sensor_name in results:
        print(f"\nSensor: {sensor_name}")
        print(f" - Step Count: {results[sensor_name]['step_count']}")
        print(f" - Step Cadence: {results[sensor_name]['step_cadence']:.2f} steps/min")

if __name__ == "__main__":
    main()
