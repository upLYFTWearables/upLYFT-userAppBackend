import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from scipy.interpolate import interp1d

def load_data_for_rf(data_dir):
    """
    Recursively loads .mat files from Data3/ABxx/<date>/levelground/(jp|imu|gcRight).
    Each file is expected to contain 'dataMatrix' and 'varNames'.
    
    Returns
    -------
    data_dict : dict
        Keys: 'jp', 'imu', 'gcRight'
        Each maps to { filename -> {dataMatrix, varNames, filepath} }
    """
    jp_pattern  = os.path.join(data_dir, "AB*", "*", "levelground", "jp", "*.mat")
    imu_pattern = os.path.join(data_dir, "AB*", "*", "levelground", "imu", "*.mat")
    gc_pattern  = os.path.join(data_dir, "AB*", "*", "levelground", "gcRight", "*.mat")
    
    jp_files  = glob.glob(jp_pattern)
    imu_files = glob.glob(imu_pattern)
    gc_files  = glob.glob(gc_pattern)
    
    data_dict = {
        'jp': {},
        'imu': {},
        'gcRight': {}
    }
    
    # Load JP files
    for fpath in jp_files:
        mat_contents = sio.loadmat(fpath)
        if "dataMatrix" in mat_contents and "varNames" in mat_contents:
            data_matrix = mat_contents["dataMatrix"]
            var_names   = [str(name).strip("[]'") for name in mat_contents["varNames"].flatten()]
            file_id     = os.path.basename(fpath)
            data_dict['jp'][file_id] = {
                'dataMatrix': data_matrix,
                'varNames': var_names,
                'filepath': fpath
            }
    
    # Load IMU files
    for fpath in imu_files:
        mat_contents = sio.loadmat(fpath)
        if "dataMatrix" in mat_contents and "varNames" in mat_contents:
            data_matrix = mat_contents["dataMatrix"]
            var_names   = [str(name).strip("[]'") for name in mat_contents["varNames"].flatten()]
            file_id     = os.path.basename(fpath)
            data_dict['imu'][file_id] = {
                'dataMatrix': data_matrix,
                'varNames': var_names,
                'filepath': fpath
            }
    
    # Load gcRight files
    for fpath in gc_files:
        mat_contents = sio.loadmat(fpath)
        if "dataMatrix" in mat_contents and "varNames" in mat_contents:
            data_matrix = mat_contents["dataMatrix"]
            var_names   = [str(name).strip("[]'") for name in mat_contents["varNames"].flatten()]
            file_id     = os.path.basename(fpath)
            data_dict['gcRight'][file_id] = {
                'dataMatrix': data_matrix,
                'varNames': var_names,
                'filepath': fpath
            }
    
    return data_dict

def pair_files(data_dict):
    """
    Pairs JP, IMU, and gcRight files based on matching basenames.
    Returns a list of (jp_data, imu_data, gc_data).
    """
    pairs = []
    for file_id, jp_info in data_dict['jp'].items():
        if file_id in data_dict['imu'] and file_id in data_dict['gcRight']:
            imu_info = data_dict['imu'][file_id]
            gc_info  = data_dict['gcRight'][file_id]
            pairs.append((jp_info, imu_info, gc_info))
    return pairs

def extract_features_and_target(jp_data, imu_data):
    """
    Extract features (IMU signals) and target (knee_angle_r_power) from matched JP/IMU data.
    
    Features: columns in IMU varNames that start with 'shank_' or 'thigh_' (excluding 'Header').
    Target:   'knee_angle_r_power' in JP.
    
    Returns
    -------
    X : (N, n_features)  Feature array
    y : (N,)             Knee power
    jp_header : (N,)     Time stamps for JP (aligned to X,y)
    """
    # --- IMU ---
    imu_var_names = imu_data['varNames']
    imu_matrix    = imu_data['dataMatrix']
    
    try:
        imu_header_idx = imu_var_names.index('Header')
    except ValueError:
        raise ValueError("No 'Header' column found in IMU varNames!")
    
    # Select columns that start with 'shank_' or 'thigh_'
    sensor_prefixes = ['shank_', 'thigh_']
    imu_feature_cols = [col for col in imu_var_names 
                        if col != 'Header' and any(col.startswith(prefix) for prefix in sensor_prefixes)]
    imu_feature_idxs = [imu_var_names.index(col) for col in imu_feature_cols]
    
    imu_header = imu_matrix[:, imu_header_idx]
    X          = imu_matrix[:, imu_feature_idxs]
    
    # --- JP ---
    jp_var_names = jp_data['varNames']
    jp_matrix    = jp_data['dataMatrix']
    
    try:
        jp_header_idx = jp_var_names.index('Header')
    except ValueError:
        raise ValueError("No 'Header' column found in JP varNames!")
    
    # Target: knee_angle_r_power
    try:
        target_idx = jp_var_names.index('knee_angle_r_power')
    except ValueError:
        raise ValueError("Column 'knee_angle_r_power' not found in JP data!")
    
    jp_header = jp_matrix[:, jp_header_idx]
    y         = jp_matrix[:, target_idx]
    
    # Synchronize by trimming to minimum length
    min_len = min(len(imu_header), len(jp_header))
    imu_header = imu_header[:min_len]
    X          = X[:min_len, :]
    jp_header  = jp_header[:min_len]
    y          = y[:min_len]
    
    # Remove any rows with NaN target
    valid_idx  = ~np.isnan(y)
    X          = X[valid_idx, :]
    y          = y[valid_idx]
    jp_header  = jp_header[valid_idx]
    imu_header = imu_header[valid_idx]
    
    return X, y, jp_header

def segment_gait_cycles_using_gc(gc_data, jp_header, y, y_pred):
    """
    Segment y and y_pred into gait cycles based on 'HeelStrike' == 0 from gcRight data.
    Resample each cycle to 100 points, then average.
    
    Returns
    -------
    avg_cycle_true : (100,) Average true knee power over a gait cycle (0-100%)
    avg_cycle_pred : (100,) Average predicted knee power over a gait cycle (0-100%)
    """
    gc_matrix    = gc_data['dataMatrix']
    gc_var_names = gc_data['varNames']
    
    # Identify Header and HeelStrike columns in gcRight
    try:
        header_idx     = gc_var_names.index('Header')
        heelstrike_idx = gc_var_names.index('HeelStrike')
    except ValueError:
        raise ValueError("Expected 'Header' and 'HeelStrike' columns in gcRight data!")
    
    gc_time       = gc_matrix[:, header_idx]
    gc_heelstrike = gc_matrix[:, heelstrike_idx]
    
    # Helper: map a given time to the closest index in jp_header
    def time_to_jp_index(t):
        return np.argmin(np.abs(jp_header - t))
    
    # Identify heel-strike times (assumed when HeelStrike == 0)
    hs_times = gc_time[gc_heelstrike == 0]
    
    if len(hs_times) < 2:
        print("Not enough heel-strike events in gcRight to form a gait cycle.")
        return None, None
    
    true_cycles = []
    pred_cycles = []
    
    for i in range(len(hs_times) - 1):
        t_start = hs_times[i]
        t_end   = hs_times[i + 1]
        if t_end <= t_start:
            continue
        
        start_idx = time_to_jp_index(t_start)
        end_idx   = time_to_jp_index(t_end)
        if end_idx <= start_idx:
            continue
        
        cycle_true = y[start_idx:end_idx]
        cycle_pred = y_pred[start_idx:end_idx]
        
        if len(cycle_true) < 2:
            continue
        
        # Resample the cycle to 100 points (representing 0-100% of the gait cycle)
        x_original = np.linspace(0, 100, num=len(cycle_true))
        x_new      = np.linspace(0, 100, num=100)
        
        interp_true = interp1d(x_original, cycle_true, kind='linear')
        interp_pred = interp1d(x_original, cycle_pred, kind='linear')
        
        true_cycles.append(interp_true(x_new))
        pred_cycles.append(interp_pred(x_new))
    
    if len(true_cycles) == 0:
        print("No valid cycles found after segmentation.")
        return None, None
    
    true_cycles = np.array(true_cycles)
    pred_cycles = np.array(pred_cycles)
    
    avg_cycle_true = np.mean(true_cycles, axis=0)
    avg_cycle_pred = np.mean(pred_cycles, axis=0)
    
    return avg_cycle_true, avg_cycle_pred

if __name__ == "__main__":
    # Adjust the path to your Data3 folder
    data_dir = "/Users/mahadparwaiz/Desktop/RFdata/Data3"
    
    # 1) Load JP, IMU, and gcRight files
    data_dict = load_data_for_rf(data_dir)
    
    # 2) Pair files by matching basenames
    pairs = pair_files(data_dict)
    print(f"Found {len(pairs)} matched sets of JP, IMU, and gcRight.")
    
    if not pairs:
        raise RuntimeError("No JP/IMU/gcRight files found with matching filenames.")
    
    # For demonstration, select one trial (for example, pairs[5])
    jp_data, imu_data, gc_data = pairs[6]
    
    # 3) Extract features (using shank and thigh IMU data) and target (knee_angle_r_power)
    X, y, jp_header = extract_features_and_target(jp_data, imu_data)
    print(f"IMU feature shape: {X.shape}, Target shape: {y.shape}")
    
    # 4) Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 5) Train a RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    
    # 6) Evaluate on training and test sets
    y_pred_train = rf.predict(X_train)
    y_pred_test  = rf.predict(X_test)
    print(f"Train R^2: {r2_score(y_train, y_pred_train):.3f}, MSE: {mean_squared_error(y_train, y_pred_train):.3f}")
    print(f"Test  R^2: {r2_score(y_test, y_pred_test):.3f}, MSE: {mean_squared_error(y_test, y_pred_test):.3f}")
    
    # 7) Predict over the entire time series for the trial
    y_pred_full = rf.predict(X)
    
    # 8) Plot raw knee power vs. time (true and predicted)
    plt.figure(figsize=(10,4))
    plt.plot(jp_header, y, label='True Knee Power (Right)', color='blue')
    plt.plot(jp_header, y_pred_full, label='Predicted Knee Power (Right)', color='red', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Knee Power")
    plt.title("Right Knee Power: Raw Over Time")
    plt.legend()
    plt.show()
    
    # 9) Segment the trial into gait cycles using gcRight (heel-strike events)
    avg_cycle_true, avg_cycle_pred = segment_gait_cycles_using_gc(gc_data, jp_header, y, y_pred_full)
    
    # 10) Plot the averaged 0-100% gait cycle (true vs. predicted)
    if avg_cycle_true is not None and avg_cycle_pred is not None:
        plt.figure(figsize=(10,5))
        gc_axis = np.linspace(0, 100, 100)
        plt.plot(gc_axis, avg_cycle_true, label='True Knee Power (Right)', color='blue')
        plt.plot(gc_axis, avg_cycle_pred, label='Predicted Knee Power (Right)', color='red', linestyle='--')
        plt.title("Average Gait Cycle: Right Knee Power")
        plt.xlabel("Gait Cycle (%)")
        plt.ylabel("Knee Power")
        plt.legend()
        plt.show()
