import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
import io
import threading
import queue
from collections import deque, defaultdict
from scipy.signal import butter, filtfilt, find_peaks
import scipy.signal as signal
from mpl_toolkits.mplot3d import Axes3D
import socket

# Configuration for StepLength
WINDOW_SIZE = 1000  # Number of samples to keep in memory
PLOT_WINDOW = 5.0   # Seconds to show in plot
FS_DEFAULT = 100    # Default sampling frequency (Hz)
FILTER_CUTOFF = 5.0  # Hz - low-pass filter cutoff
HEIGHT_THRESHOLD = 1.5  # Minimum peak height for step detection
DISTANCE_THRESHOLD = 50  # Minimum distance between peaks

# Configuration for PowerMetrics
POWER_WINDOW_SIZE = 500  # Number of samples for power metrics
POWER_UPDATE_INTERVAL = 1000  # Update plots every 1000ms

# Configuration for Frequency
FREQ_WINDOW_SIZE = 60  # Show last 60 seconds of data
FREQ_UPDATE_INTERVAL = 1000  # Update plot every 1000ms
MAX_POINTS = 300  # Reduced from 1000 for better performance
ANNOTATION_LIMIT = 5  # Only show annotations for last 5 points per IMU
DATA_DECIMATION = 3  # Show every 3rd point to reduce congestion

# IMU Mapping
IMU_MAPPING = {
    "IMU1": "Left_Below_Knee",
    "IMU2": "Right_Below_Knee", 
    "IMU3": "Left_Above_Knee",
    "IMU4": "Right_Above_Knee",
    "IMU7": "Pelvis"
}

# Global variables for StepLength data management
step_data_queue = queue.Queue(maxsize=10000)
imu_data_buffers = defaultdict(lambda: {
    'timestamp': deque(maxlen=WINDOW_SIZE),
    'accel_magnitude': deque(maxlen=WINDOW_SIZE),
    'filtered_accel': deque(maxlen=WINDOW_SIZE),
    'step_count': 0,
    'last_step_time': 0,
    'cadence': 0
})

# Global variables for PowerMetrics data management
power_data_buffers = {
    'timestamp': deque(maxlen=POWER_WINDOW_SIZE),
    'accel_x': deque(maxlen=POWER_WINDOW_SIZE),
    'accel_y': deque(maxlen=POWER_WINDOW_SIZE),
    'accel_z': deque(maxlen=POWER_WINDOW_SIZE),
    'gyro_x': deque(maxlen=POWER_WINDOW_SIZE),
    'gyro_y': deque(maxlen=POWER_WINDOW_SIZE),
    'gyro_z': deque(maxlen=POWER_WINDOW_SIZE),
}

power_metrics_buffers = {
    'trans_power': deque(maxlen=POWER_WINDOW_SIZE),
    'rot_power': deque(maxlen=POWER_WINDOW_SIZE),
    'total_power': deque(maxlen=POWER_WINDOW_SIZE),
    'power_weight': deque(maxlen=POWER_WINDOW_SIZE),
    'rolling_avg': deque(maxlen=POWER_WINDOW_SIZE),
}

power_stats = {
    'trans_power_peak': 0,
    'trans_power_mean': 0,
    'rot_power_peak': 0,
    'rot_power_mean': 0,
    'total_power_peak': 0,
    'total_power_mean': 0,
    'stride_time_mean': 0,
    'stride_time_std': 0,
    'cadence': 0,
    'gct_mean': 0,
    'gct_std': 0,
    'swing_mean': 0,
    'swing_std': 0,
    'stride_length_mean': 0,
    'stride_length_std': 0,
    'peak_load': 0,
}

# Global variables for Frequency data management
freq_data_buffers = defaultdict(lambda: deque())  # Calculated frequencies per IMU
freq_time_data = defaultdict(lambda: deque())  # Time stamps for frequency data
freq_raw_buffer = defaultdict(lambda: deque(maxlen=1000))  # Raw timestamps per IMU
freq_start_time = time.time()
freq_stats = defaultdict(lambda: {'current': 0, 'avg': 0, 'count': 0})

# Global variables for Stickman data management
stickman_data_queue = queue.Queue(maxsize=1000)
stickman_imu_data = {
    'IMU1': {'quat': [1.0, 0.0, 0.0, 0.0], 'timestamp': 0},
    'IMU2': {'quat': [1.0, 0.0, 0.0, 0.0], 'timestamp': 0},
    'IMU3': {'quat': [1.0, 0.0, 0.0, 0.0], 'timestamp': 0},
    'IMU4': {'quat': [1.0, 0.0, 0.0, 0.0], 'timestamp': 0},
    'IMU7': {'quat': [1.0, 0.0, 0.0, 0.0], 'timestamp': 0}
}

# Stickman kinematic model dimensions
PELVIS_LENGTH = 1.0
PELVIS_WIDTH = 2.0
UPPER_LEG_LENGTH = 4.0
LOWER_LEG_LENGTH = 4.0
FOOT_LENGTH = 1.0

# Global variables for Pelvic Metrics data management
pelvic_data_queue = queue.Queue(maxsize=1000)
pelvic_data_buffers = {
    'timestamp': deque(),
    'tilt': deque(),
    'obliquity': deque(),
    'rotation': deque()
}
pelvic_start_time = None
pelvic_calibration_quats = []
pelvic_R0 = np.eye(3)
pelvic_is_calibrated = False
pelvic_calibration_samples = 100  # Reduced for faster calibration in demo
pelvic_sample_count = 0

# Set page config
st.set_page_config(
    page_title="Medical Rehabilitation Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
    }
    
    /* Force dark text on radio buttons */
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stRadio"] span,
    [data-testid="stSidebar"] [data-testid="stRadio"] div {
        color: #2c3e50 !important;
    }
    
    /* Additional radio text color overrides */
    .st-emotion-cache-1gulkj5,
    .st-emotion-cache-16idsys p,
    .st-emotion-cache-16idsys span,
    .st-emotion-cache-16idsys div,
    .st-emotion-cache-ue6h4q,
    .st-emotion-cache-eczf16 {
        color: #2c3e50 !important;
    }
    
    /* Main container styling */
    .main {
        background-color: #ffffff;
    }
    
    /* Metric card styling */
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .metric-value {
        color: #2c3e50;
        font-size: 16px;
        margin: 5px 0;
    }
    
    .metric-label {
        color: #2c3e50;
        font-size: 14px;
    }
    
    /* Override Streamlit's default background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Ensure all text is dark */
    .stMarkdown, p, h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
        <h2 style='color: #1b5e20; margin-bottom: 25px; font-size: 24px; font-weight: 600;'>Real Time Processing of Data</h2>
    """, unsafe_allow_html=True)
    
    selected_option = st.radio(
        "",
        ["StepLength", "PowerMetrics", "Frequency", "PelvicMetrics", "Stickman"],
        label_visibility="collapsed"
    )

# Create two columns for the layout
left_col, right_col = st.columns([3, 2])

class StepLengthVisualizer:
    def __init__(self):
        self.fig, self.axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
        self.setup_plots()
        
    def setup_plots(self):
        titles = list(IMU_MAPPING.values())
        for ax, title in zip(self.axs, titles):
            ax.set_title(title, color='#2c3e50', fontsize=10)
            ax.set_ylabel('Accel Magnitude (m/s²)', color='#2c3e50')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(8, 18)
            ax.tick_params(colors='#2c3e50')
        self.axs[-1].set_xlabel('Time (s)', color='#2c3e50')
        plt.tight_layout()
        
    def butter_filter(self, data):
        """Apply Butterworth low-pass filter"""
        if len(data) < 10:
            return np.array(data)
        nyquist = 0.5 * FS_DEFAULT
        normalized_cutoff = min(0.99, FILTER_CUTOFF / nyquist)
        b, a = butter(4, normalized_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)
        
    def detect_steps(self, accel_signal, timestamps):
        """Detect steps using peak detection"""
        if len(accel_signal) < 50:
            return [], []
        filtered_signal = self.butter_filter(accel_signal)
        peaks, _ = find_peaks(filtered_signal, height=HEIGHT_THRESHOLD, distance=DISTANCE_THRESHOLD)
        step_times = [timestamps[i] for i in peaks if i < len(timestamps)]
        return peaks, step_times
        
    def update_plot(self):
        """Update plot with latest data"""
        current_time = time.time()
        for i, (imu_id, imu_name) in enumerate(IMU_MAPPING.items()):
            buffer = imu_data_buffers[imu_id]
            if len(buffer['timestamp']) > 0:
                # Convert to numpy arrays
                timestamps = np.array(buffer['timestamp'])
                accel_mag = np.array(buffer['accel_magnitude'])
                
                # Filter for plot window
                time_window = current_time - PLOT_WINDOW
                mask = timestamps >= time_window
                if np.any(mask):
                    plot_times = timestamps[mask] - current_time
                    plot_accel = accel_mag[mask]
                    
                    # Filter and detect steps
                    filtered_accel = self.butter_filter(plot_accel)
                    peaks, step_times = self.detect_steps(plot_accel, plot_times)
                    
                    # Plot data
                    ax = self.axs[i]
                    ax.clear()
                    ax.plot(plot_times, plot_accel, '-', color='#4285f4', alpha=0.5, label='Raw Accel Magnitude')
                    ax.plot(plot_times, filtered_accel, '-', color='#34a853', label='Filtered')
                    if len(peaks) > 0:
                        ax.plot(plot_times[peaks], plot_accel[peaks], 'rx', label='Detected Steps')
                    
                    # Update formatting
                    ax.set_title(imu_name, color='#2c3e50', fontsize=10)
                    ax.set_ylabel('Accel Magnitude (m/s²)', color='#2c3e50')
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(8, 18)
                    ax.set_xlim(-PLOT_WINDOW, 0)
                    ax.tick_params(colors='#2c3e50')
                    ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        
    def get_plot_image(self):
        """Get current plot as image"""
        self.update_plot()
        buf = io.BytesIO()
        self.fig.patch.set_facecolor('#ffffff')
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#ffffff')
        buf.seek(0)
        return buf

class PowerMetricsVisualizer:
    def __init__(self):
        self.fig, self.axs = plt.subplots(3, 2, figsize=(15, 12))
        self.foot_mass = 1.2  # kg
        self.body_mass = 70.0  # kg
        self.foot_inertia = [0.01, 0.01, 0.01]  # kg*m^2
        self.setup_plots()
        
    def setup_plots(self):
        """Setup all matplotlib plots"""
        # Raw acceleration
        self.axs[0, 0].set_title('Raw Acceleration', color='#2c3e50')
        self.axs[0, 0].set_ylabel('m/s²', color='#2c3e50')
        self.axs[0, 0].grid(True, alpha=0.3)
        self.axs[0, 0].tick_params(colors='#2c3e50')
        
        # Translational Power
        self.axs[0, 1].set_title('Translational Power (m·a)·v', color='#2c3e50')
        self.axs[0, 1].set_ylabel('W', color='#2c3e50')
        self.axs[0, 1].grid(True, alpha=0.3)
        self.axs[0, 1].tick_params(colors='#2c3e50')
        
        # Rotational Power
        self.axs[1, 0].set_title('Rotational Power τ·ω', color='#2c3e50')
        self.axs[1, 0].set_ylabel('W', color='#2c3e50')
        self.axs[1, 0].grid(True, alpha=0.3)
        self.axs[1, 0].tick_params(colors='#2c3e50')
        
        # Total Power
        self.axs[1, 1].set_title('Total Power', color='#2c3e50')
        self.axs[1, 1].set_ylabel('W', color='#2c3e50')
        self.axs[1, 1].grid(True, alpha=0.3)
        self.axs[1, 1].tick_params(colors='#2c3e50')
        
        # Power/Weight
        self.axs[2, 0].set_title('Instantaneous Total Power / Weight', color='#2c3e50')
        self.axs[2, 0].set_ylabel('W/kg', color='#2c3e50')
        self.axs[2, 0].set_xlabel('Time (s)', color='#2c3e50')
        self.axs[2, 0].grid(True, alpha=0.3)
        self.axs[2, 0].tick_params(colors='#2c3e50')
        
        # Rolling Average
        self.axs[2, 1].set_title('Rolling Average Total Power', color='#2c3e50')
        self.axs[2, 1].set_ylabel('W', color='#2c3e50')
        self.axs[2, 1].set_xlabel('Time (s)', color='#2c3e50')
        self.axs[2, 1].grid(True, alpha=0.3)
        self.axs[2, 1].tick_params(colors='#2c3e50')
        
        plt.tight_layout()
        
    def apply_lowpass_filter(self, data):
        """Apply a low-pass filter to the data"""
        if len(data) <= 15:
            return data
        b, a = signal.butter(4, 0.1, 'low')
        return signal.filtfilt(b, a, data)
        
    def compute_power_metrics(self):
        """Compute power metrics from current buffer"""
        MIN_SAMPLES_FOR_PROCESSING = 16
        
        if len(power_data_buffers['timestamp']) < MIN_SAMPLES_FOR_PROCESSING:
            return
            
        try:
            # Convert deques to numpy arrays for processing
            accel = np.array([
                list(power_data_buffers['accel_x']),
                list(power_data_buffers['accel_y']),
                list(power_data_buffers['accel_z'])
            ]).T
            
            gyro = np.array([
                list(power_data_buffers['gyro_x']),
                list(power_data_buffers['gyro_y']),
                list(power_data_buffers['gyro_z'])
            ]).T
            
            timestamps = np.array(list(power_data_buffers['timestamp']))
            
            # Ensure all arrays have the same length
            min_length = min(accel.shape[0], gyro.shape[0], timestamps.shape[0])
            if min_length < MIN_SAMPLES_FOR_PROCESSING:
                return
                
            accel = accel[:min_length]
            gyro = gyro[:min_length]
            timestamps = timestamps[:min_length]
            
            # Apply low-pass filter only if we have enough data
            filtered_accel = np.array([
                self.apply_lowpass_filter(accel[:,i]) for i in range(3)
            ]).T
            
            # Compute time differences
            dt = np.diff(timestamps) / 1e6  # Convert to seconds
            dt = np.clip(dt, 0.001, 0.1)  # Limit dt to reasonable values
            
            # Ensure all arrays have compatible shapes for calculations
            n_samples = min(len(dt), filtered_accel.shape[0] - 1)
            if n_samples < 1:
                return
                
            # Compute translational power
            velocity = np.zeros_like(filtered_accel)
            velocity[1:n_samples+1] = np.cumsum(filtered_accel[:n_samples] * dt[:, None], axis=0)
            trans_power = np.sum(self.foot_mass * filtered_accel[:n_samples] * velocity[1:n_samples+1], axis=1)
            
            # Compute rotational power
            omega = np.deg2rad(gyro[:n_samples])
            I = np.diag(self.foot_inertia)
            torque = np.cross(omega, (I @ omega.T).T)
            rot_power = np.sum(torque * omega, axis=1)
            
            # Total power and power/weight
            total_power = trans_power + rot_power
            power_weight = total_power / self.body_mass
            
            # Rolling average
            window_size = min(20, len(total_power))
            rolling_avg = np.convolve(total_power, np.ones(window_size)/window_size, mode='same')
            
            # Clear old data if buffer is full
            if len(power_metrics_buffers['total_power']) >= POWER_WINDOW_SIZE:
                for key in power_metrics_buffers:
                    power_metrics_buffers[key].clear()
            
            # Update power metrics buffers
            power_metrics_buffers['trans_power'].extend(trans_power)
            power_metrics_buffers['rot_power'].extend(rot_power)
            power_metrics_buffers['total_power'].extend(total_power)
            power_metrics_buffers['power_weight'].extend(power_weight)
            power_metrics_buffers['rolling_avg'].extend(rolling_avg)
            
            # Update statistics
            if power_metrics_buffers['trans_power']:
                power_stats['trans_power_peak'] = max(power_metrics_buffers['trans_power'])
                power_stats['trans_power_mean'] = np.mean(power_metrics_buffers['trans_power'])
                
            if power_metrics_buffers['total_power']:
                power_stats['total_power_peak'] = max(power_metrics_buffers['total_power'])
                power_stats['total_power_mean'] = np.mean(power_metrics_buffers['total_power'])
                
        except Exception as e:
            print(f"Error computing power metrics: {e}")
            
    def update_plot(self):
        """Update plot with latest data"""
        if not power_data_buffers['timestamp']:
            return
            
        try:
            # Get all data arrays
            timestamps = np.array(list(power_data_buffers['timestamp']))
            accel_data = {
                'accel_x': np.array(list(power_data_buffers['accel_x'])),
                'accel_y': np.array(list(power_data_buffers['accel_y'])),
                'accel_z': np.array(list(power_data_buffers['accel_z']))
            }
            
            # Make time relative to start and convert to seconds
            if len(timestamps) > 0:
                t = (timestamps - timestamps[0]) / 1e6
            else:
                t = np.array([])
            
            # Clear all axes
            for ax in self.axs.flat:
                ax.clear()
                
            # Re-setup plots
            self.setup_plots()
            
            # Update acceleration plots
            colors = ['blue', 'orange', 'green']
            labels = ['AccelX', 'AccelY', 'AccelZ']
            for i, (key, color, label) in enumerate(zip(['accel_x', 'accel_y', 'accel_z'], colors, labels)):
                self.axs[0, 0].plot(t, accel_data[key], color=color, label=label)
            self.axs[0, 0].legend()
            
            # Update power plots if we have power data
            if power_metrics_buffers['trans_power']:
                power_length = min(len(t), len(power_metrics_buffers['trans_power']))
                t_power = t[:power_length]
                
                self.axs[0, 1].plot(t_power, list(power_metrics_buffers['trans_power'])[:power_length], 'b-', label='Trans Power')
                self.axs[0, 1].legend()
                
                self.axs[1, 0].plot(t_power, list(power_metrics_buffers['rot_power'])[:power_length], 'r-', label='Rot Power')
                self.axs[1, 0].legend()
                
                self.axs[1, 1].plot(t_power, list(power_metrics_buffers['total_power'])[:power_length], 'g-', label='Total Power')
                self.axs[1, 1].legend()
                
                self.axs[2, 0].plot(t_power, list(power_metrics_buffers['power_weight'])[:power_length], 'b-', label='Power/Weight')
                self.axs[2, 0].legend()
                
                self.axs[2, 1].plot(t_power, list(power_metrics_buffers['rolling_avg'])[:power_length], 'b-', label='Rolling Avg')
                self.axs[2, 1].legend()
            
            plt.tight_layout()
            
        except Exception as e:
            print(f"Error updating power plots: {e}")
            
    def get_plot_image(self):
        """Get current plot as image"""
        self.compute_power_metrics()
        self.update_plot()
        buf = io.BytesIO()
        self.fig.patch.set_facecolor('#ffffff')
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#ffffff')
        buf.seek(0)
        return buf

class FrequencyVisualizer:
    def __init__(self):
        self.fig = None
        self.axs = None
        self.lines = {}
        self.scatter_plots = {}
        self.annotations = {}
        self.active_imus = set()
        self.last_update_time = time.time()
        
    def get_imu_color(self, imu_name):
        """Get consistent color for each IMU"""
        imu_num = int(imu_name.replace("IMU", ""))
        colors = plt.cm.get_cmap('tab10', 10)
        return colors((imu_num - 1) % 10)
        
    def setup_plot(self, unique_imus):
        """Initialize the plot for frequency monitoring"""
        n_imus = len(unique_imus)
        
        if self.fig is not None:
            plt.close(self.fig)
            
        self.fig, self.axs = plt.subplots(n_imus, 1, figsize=(16, 3 * n_imus), sharex=True)
        
        # Handle single IMU case
        if n_imus == 1:
            self.axs = [self.axs]
        
        # Clear previous data
        self.lines = {}
        self.scatter_plots = {}
        self.annotations = {}
        
        # Initialize plots for each IMU
        for i, imu in enumerate(unique_imus):
            ax = self.axs[i]
            color = self.get_imu_color(imu)
            
            # Create empty line and scatter plot
            line, = ax.plot([], [], '-', color=color, linewidth=1.5, alpha=0.8)
            scatter = ax.scatter([], [], color=color, marker='o', s=30, alpha=0.8)
            self.lines[imu] = line
            self.scatter_plots[imu] = scatter
            self.annotations[imu] = []
            
            # Style the subplot
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("Freq (Hz)", fontsize=10, color='#2c3e50')
            ax.tick_params(axis='x', rotation=45, colors='#2c3e50')
            ax.tick_params(axis='y', colors='#2c3e50')
            ax.set_ylim(80, 250)  # Adjusted range for better visibility
            
            # Set title
            ax.set_title(f"{imu} – Frequency Over Time", 
                        color=color, fontsize=12, fontweight='bold')
            
        # Set common x-axis label
        self.axs[-1].set_xlabel("Time (seconds)", fontsize=10, color='#2c3e50')
        plt.tight_layout()
        
    def decimate_data(self, timestamps, frequencies):
        """Reduce data points for better performance"""
        if len(timestamps) <= DATA_DECIMATION:
            return timestamps, frequencies
        
        # Take every nth point, but always include the last point
        indices = list(range(0, len(timestamps), DATA_DECIMATION))
        if indices[-1] != len(timestamps) - 1:
            indices.append(len(timestamps) - 1)
        
        return [timestamps[i] for i in indices], [frequencies[i] for i in indices]
        
    def update_plot(self):
        """Update plot with latest frequency data"""
        try:
            current_time = time.time()
            
            if freq_data_buffers:
                # Get unique IMUs (sorted for consistent ordering)
                unique_imus = sorted(freq_data_buffers.keys())
                
                # Create/recreate plot if needed
                if self.fig is None or set(unique_imus) != self.active_imus:
                    self.setup_plot(unique_imus)
                    self.active_imus = set(unique_imus)
                
                # Update each subplot
                for imu in unique_imus:
                    if imu in freq_data_buffers and freq_data_buffers[imu]:
                        timestamps = list(freq_time_data[imu])
                        frequencies = list(freq_data_buffers[imu])
                        
                        # Decimate data for better performance
                        if len(timestamps) > MAX_POINTS:
                            t_data, f_data = self.decimate_data(timestamps, frequencies)
                        else:
                            t_data = timestamps
                            f_data = frequencies
                        
                        # Update line and scatter data
                        if t_data and f_data:
                            self.lines[imu].set_data(t_data, f_data)
                            self.scatter_plots[imu].set_offsets(np.c_[t_data, f_data])
                            
                            # Remove old annotations
                            for ann in self.annotations[imu]:
                                ann.remove()
                            self.annotations[imu] = []
                            
                            # Add new annotations (only for recent points)
                            recent_points = min(ANNOTATION_LIMIT, len(t_data))
                            for x, y in zip(t_data[-recent_points:], f_data[-recent_points:]):
                                ann = self.axs[list(unique_imus).index(imu)].annotate(
                                    f"{int(y)}", (x, y),
                                    xytext=(0, 8), textcoords='offset points',
                                    ha='center', va='bottom',
                                    fontsize=8, fontweight='bold',
                                    color=self.get_imu_color(imu),
                                    bbox=dict(boxstyle='round,pad=0.2', 
                                            facecolor='white', alpha=0.7, edgecolor='none')
                                )
                                self.annotations[imu].append(ann)
                            
                            # Update statistics
                            if f_data:
                                freq_stats[imu]['current'] = f_data[-1]
                                freq_stats[imu]['avg'] = np.mean(f_data)
                                freq_stats[imu]['count'] = len(f_data)
                            
                            # Update title with statistics
                            current_freq = freq_stats[imu]['current']
                            avg_freq = freq_stats[imu]['avg']
                            ax_index = list(unique_imus).index(imu)
                            self.axs[ax_index].set_title(
                                f"{imu} – Frequency Over Time (Current: {current_freq:.0f} Hz, Avg: {avg_freq:.1f} Hz)",
                                color=self.get_imu_color(imu),
                                fontsize=11, fontweight='bold'
                            )
                            
                            # Update x-axis limits to show all data
                            latest_time = max(t_data)
                            self.axs[ax_index].set_xlim(0, latest_time + 5)
            
            plt.tight_layout()
            
        except Exception as e:
            print(f"Error updating frequency plot: {e}")
            
    def get_plot_image(self):
        """Get current plot as image"""
        self.update_plot()
        buf = io.BytesIO()
        if self.fig:
            self.fig.patch.set_facecolor('#ffffff')
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#ffffff')
        buf.seek(0)
        return buf

class StickmanVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.pelvis = None
        self.left_upper_leg = None
        self.left_lower_leg = None
        self.right_upper_leg = None
        self.right_lower_leg = None
        self.setup_kinematic_model()
        
    def create_transformation_matrix(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, quaternion=None):
        """Create 4x4 transformation matrix from position and rotation"""
        T = np.eye(4)
        T[0:3, 3] = [x, y, z]
        
        if quaternion is not None:
            # Convert quaternion to rotation matrix
            w, x, y, z = quaternion
            R = np.array([
                [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
            ])
            T[0:3, 0:3] = R
        else:
            # Convert Euler angles to rotation matrix
            roll, pitch, yaw = np.radians([roll, pitch, yaw])
            
            # Rotation matrices
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(roll), -np.sin(roll)],
                          [0, np.sin(roll), np.cos(roll)]])
            
            Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                          [0, 1, 0],
                          [-np.sin(pitch), 0, np.cos(pitch)]])
            
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0],
                          [0, 0, 1]])
            
            T[0:3, 0:3] = Rz @ Ry @ Rx
        
        return T
        
    def setup_kinematic_model(self):
        """Setup the kinematic model structure"""
        # Initialize joint transformations
        self.pelvis_joint = np.eye(4)
        self.left_upper_joint = np.eye(4)
        self.left_lower_joint = np.eye(4)
        self.right_upper_joint = np.eye(4)
        self.right_lower_joint = np.eye(4)
        
        # Set initial calibration - make pelvis vertical
        self.pelvis_joint = self.create_transformation_matrix(pitch=-90)
        
    def update_kinematic_model(self):
        """Update the kinematic model with latest IMU data"""
        # Update joint orientations based on IMU data
        self.pelvis_joint = self.create_transformation_matrix(pitch=-90, quaternion=stickman_imu_data['IMU7']['quat'])
        self.left_upper_joint = self.create_transformation_matrix(quaternion=stickman_imu_data['IMU1']['quat'])
        self.left_lower_joint = self.create_transformation_matrix(quaternion=stickman_imu_data['IMU3']['quat'])
        self.right_upper_joint = self.create_transformation_matrix(quaternion=stickman_imu_data['IMU2']['quat'])
        self.right_lower_joint = self.create_transformation_matrix(quaternion=stickman_imu_data['IMU4']['quat'])
        
    def plot_stickman(self):
        """Plot the stickman figure"""
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Clear the plot
        self.ax.clear()
        
        # Update kinematic model
        self.update_kinematic_model()
        
        # Calculate joint positions
        pelvis_pos = self.pelvis_joint[0:3, 3]
        
        # Left leg positions
        left_hip_transform = self.pelvis_joint @ self.create_transformation_matrix(y=PELVIS_WIDTH/2, roll=180, yaw=180)
        left_hip_pos = left_hip_transform[0:3, 3]
        
        left_knee_transform = left_hip_transform @ self.left_upper_joint @ self.create_transformation_matrix(x=UPPER_LEG_LENGTH)
        left_knee_pos = left_knee_transform[0:3, 3]
        
        left_ankle_transform = left_knee_transform @ self.left_lower_joint @ self.create_transformation_matrix(x=LOWER_LEG_LENGTH)
        left_ankle_pos = left_ankle_transform[0:3, 3]
        
        left_foot_transform = left_ankle_transform @ self.create_transformation_matrix(pitch=-90, x=FOOT_LENGTH)
        left_foot_pos = left_foot_transform[0:3, 3]
        
        # Right leg positions
        right_hip_transform = self.pelvis_joint @ self.create_transformation_matrix(y=-PELVIS_WIDTH/2, roll=180, yaw=180)
        right_hip_pos = right_hip_transform[0:3, 3]
        
        right_knee_transform = right_hip_transform @ self.right_upper_joint @ self.create_transformation_matrix(x=UPPER_LEG_LENGTH)
        right_knee_pos = right_knee_transform[0:3, 3]
        
        right_ankle_transform = right_knee_transform @ self.right_lower_joint @ self.create_transformation_matrix(x=LOWER_LEG_LENGTH)
        right_ankle_pos = right_ankle_transform[0:3, 3]
        
        right_foot_transform = right_ankle_transform @ self.create_transformation_matrix(pitch=-90, x=FOOT_LENGTH)
        right_foot_pos = right_foot_transform[0:3, 3]
        
        # Plot pelvis
        self.ax.plot([left_hip_pos[0], right_hip_pos[0]], 
                    [left_hip_pos[1], right_hip_pos[1]], 
                    [left_hip_pos[2], right_hip_pos[2]], 'gray', linewidth=3)
        
        # Plot left leg
        self.ax.plot([left_hip_pos[0], left_knee_pos[0]], 
                    [left_hip_pos[1], left_knee_pos[1]], 
                    [left_hip_pos[2], left_knee_pos[2]], 'gray', linewidth=2)
        self.ax.plot([left_knee_pos[0], left_ankle_pos[0]], 
                    [left_knee_pos[1], left_ankle_pos[1]], 
                    [left_knee_pos[2], left_ankle_pos[2]], 'gray', linewidth=2)
        self.ax.plot([left_ankle_pos[0], left_foot_pos[0]], 
                    [left_ankle_pos[1], left_foot_pos[1]], 
                    [left_ankle_pos[2], left_foot_pos[2]], 'gray', linewidth=2)
        
        # Plot right leg
        self.ax.plot([right_hip_pos[0], right_knee_pos[0]], 
                    [right_hip_pos[1], right_knee_pos[1]], 
                    [right_hip_pos[2], right_knee_pos[2]], 'gray', linewidth=2)
        self.ax.plot([right_knee_pos[0], right_ankle_pos[0]], 
                    [right_knee_pos[1], right_ankle_pos[1]], 
                    [right_knee_pos[2], right_ankle_pos[2]], 'gray', linewidth=2)
        self.ax.plot([right_ankle_pos[0], right_foot_pos[0]], 
                    [right_ankle_pos[1], right_foot_pos[1]], 
                    [right_ankle_pos[2], right_foot_pos[2]], 'gray', linewidth=2)
        
        # Plot joints as black dots
        joints = [pelvis_pos, left_hip_pos, left_knee_pos, left_ankle_pos, left_foot_pos,
                 right_hip_pos, right_knee_pos, right_ankle_pos, right_foot_pos]
        joint_names = ['pelvis', 'left_upper_leg', 'left_lower_leg', 'left_foot', 'left_foot_end',
                      'right_upper_leg', 'right_lower_leg', 'right_foot', 'right_foot_end']
        
        for i, (pos, name) in enumerate(zip(joints, joint_names)):
            self.ax.scatter(pos[0], pos[1], pos[2], c='k', s=50)
            if i < 5:  # Only label main joints
                self.ax.text(pos[0], pos[1], pos[2], name, fontsize=8, color='#2c3e50')
        
        # Plot coordinate axes at each joint
        AXIS_LENGTH = 0.4
        colors = ['r', 'g', 'b']  # x=red, y=green, z=blue
        
        transforms = [self.pelvis_joint, left_hip_transform, left_knee_transform, left_ankle_transform,
                     right_hip_transform, right_knee_transform, right_ankle_transform]
        
        for transform in transforms:
            pos = transform[0:3, 3]
            for i in range(3):
                direction = transform[0:3, i] * AXIS_LENGTH
                self.ax.quiver(pos[0], pos[1], pos[2],
                             direction[0], direction[1], direction[2],
                             color=colors[i], length=AXIS_LENGTH, normalize=True)
        
        # Set view and styling
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])
        self.ax.set_zlim([-10, 10])
        self.ax.set_xlabel('X', color='#2c3e50')
        self.ax.set_ylabel('Y', color='#2c3e50')
        self.ax.set_zlabel('Z', color='#2c3e50')
        self.ax.tick_params(colors='#2c3e50')
        self.ax.view_init(elev=10, azim=45)
        self.ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='gray', lw=2, label='Link'),
            plt.Line2D([0], [0], marker='o', color='k', lw=0, markersize=8, label='Joint'),
            plt.Line2D([0], [0], color='r', lw=2, label='X'),
            plt.Line2D([0], [0], color='g', lw=2, label='Y'),
            plt.Line2D([0], [0], color='b', lw=2, label='Z')
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', frameon=False)
        
        # Add timestamp
        latest_time = max([data['timestamp'] for data in stickman_imu_data.values()])
        if latest_time > 0:
            time_str = time.strftime('%H:%M:%S', time.localtime(latest_time))
            self.ax.text2D(0.02, 0.98, f"Latest data: {time_str}", 
                          transform=self.ax.transAxes, fontsize=10, 
                          verticalalignment='top', color='#2c3e50')
        
        plt.tight_layout()
        
    def get_plot_image(self):
        """Get current plot as image"""
        self.plot_stickman()
        buf = io.BytesIO()
        if self.fig:
            self.fig.patch.set_facecolor('#ffffff')
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#ffffff')
        buf.seek(0)
        return buf

class PelvicMetricsVisualizer:
    def __init__(self):
        self.fig = None
        self.axes = None
        self.plot_lines = []
        self.titles = ['Pelvic Tilt', 'Pelvic Obliquity', 'Pelvic Rotation']
        self.colors = ['blue', 'green', 'red']
        
    def quaternion_to_matrix(self, q):
        """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
        w, x, y, z = q
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm < 1e-12:
            return np.eye(3)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm

        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),       2*(x*z + w*y)],
            [2*(x*y + w*z),         1 - 2*(x*x + z*z),   2*(y*z - w*x)],
            [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x*x + y*y)]
        ])

    def average_rotation_matrix(self, quaternions):
        """Compute an approximate average rotation matrix from a list of quaternions."""
        if len(quaternions) == 0:
            return np.eye(3)

        R_sum = np.zeros((3,3))
        for q in quaternions:
            R_sum += self.quaternion_to_matrix(q)

        R_sum /= len(quaternions)
        U, _, Vt = np.linalg.svd(R_sum)
        R_avg = U @ Vt
        return R_avg

    def matrix_to_euler_ZYX(self, R):
        """Extract Euler angles from a rotation matrix R using a Z-Y-X sequence."""
        beta_y = -np.arcsin(R[2, 0])       # pitch
        cos_beta = np.cos(beta_y)
        if abs(cos_beta) < 1e-6:
            alpha_z = np.arctan2(-R[0, 1], R[1, 1])  # yaw
            gamma_x = 0.0                            # roll
        else:
            alpha_z = np.arctan2(R[1, 0], R[0, 0])   # yaw
            gamma_x = np.arctan2(R[2, 1], R[2, 2])   # roll

        return alpha_z, beta_y, gamma_x
        
    def setup_plot(self):
        """Setup the pelvic metrics plot"""
        if self.fig is not None:
            plt.close(self.fig)
            
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('Real-Time Pelvic Metrics - IMU7', fontsize=16, color='#2c3e50')
        
        self.plot_lines = []
        
        for i, (ax, title, color) in enumerate(zip(self.axes, self.titles, self.colors)):
            ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50')
            ax.set_ylabel('Angle (degrees)', fontsize=12, color='#2c3e50')
            ax.set_ylim(-30, 30)
            ax.set_xlim(0, 60)
            ax.grid(True, alpha=0.3)
            ax.tick_params(colors='#2c3e50')
            
            line, = ax.plot([], [], color=color, linewidth=2, label=title)
            self.plot_lines.append(line)
        
        self.axes[-1].set_xlabel('Time (seconds)', fontsize=12, color='#2c3e50')
        plt.tight_layout()
        
    def update_plot(self):
        """Update plot with latest pelvic metrics data"""
        global pelvic_start_time
        
        if self.fig is None:
            self.setup_plot()
            
        if not pelvic_data_buffers['timestamp'] or pelvic_start_time is None:
            return
            
        try:
            # Convert to numpy arrays
            timestamps = np.array(list(pelvic_data_buffers['timestamp']))
            relative_times = timestamps - pelvic_start_time
            
            # Update each plot line
            data_keys = ['tilt', 'obliquity', 'rotation']
            for i, (line, key) in enumerate(zip(self.plot_lines, data_keys)):
                if pelvic_data_buffers[key]:
                    plot_data = np.array(list(pelvic_data_buffers[key]))
                    line.set_data(relative_times, plot_data)
                    
                    # Update axis limits
                    if len(relative_times) > 0:
                        latest_time = relative_times[-1]
                        self.axes[i].set_xlim(0, max(60, latest_time + 5))
                        
                        # Auto-scale y-axis
                        if len(plot_data) > 0:
                            data_min = np.min(plot_data)
                            data_max = np.max(plot_data)
                            margin = max(5, (data_max - data_min) * 0.1)
                            self.axes[i].set_ylim(data_min - margin, data_max + margin)
            
            plt.tight_layout()
            
        except Exception as e:
            print(f"Error updating pelvic metrics plot: {e}")
            
    def get_plot_image(self):
        """Get current plot as image"""
        self.update_plot()
        buf = io.BytesIO()
        if self.fig:
            self.fig.patch.set_facecolor('#ffffff')
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#ffffff')
        buf.seek(0)
        return buf

def step_test_data_generator():
    """Generate test data for step length visualization"""
    import random
    while True:
        try:
            for imu_id in IMU_MAPPING.keys():
                t = time.time()
                base_accel = 9.8
                walking_freq = 1.5
                walking_amplitude = 8.0
                
                accel = base_accel + walking_amplitude * np.sin(2 * np.pi * walking_freq * t) + random.gauss(0, 0.5)
                
                buffer = imu_data_buffers[imu_id]
                buffer['timestamp'].append(t)
                buffer['accel_magnitude'].append(accel)
                
                # Update step count and cadence
                if len(buffer['timestamp']) > 1:
                    time_diff = buffer['timestamp'][-1] - buffer['timestamp'][0]
                    if time_diff > 0:
                        buffer['step_count'] = int(walking_freq * time_diff)
                        buffer['cadence'] = buffer['step_count'] / time_diff * 60
                
            time.sleep(0.01)  # 100 Hz simulation
            
        except Exception as e:
            print(f"Step test data generator error: {e}")
            time.sleep(0.1)

def power_test_data_generator():
    """Generate test data for power metrics visualization"""
    import random
    while True:
        try:
            t = time.time()
            
            # Generate realistic IMU5 data
            base_accel = 9.8
            walking_freq = 1.5
            walking_amplitude = 8.0
            
            # Generate test accelerometer data
            accel_x = base_accel * 0.1 + walking_amplitude * np.sin(2 * np.pi * walking_freq * t) + random.gauss(0, 0.5)
            accel_y = base_accel * 0.1 + walking_amplitude * np.cos(2 * np.pi * walking_freq * t * 1.1) + random.gauss(0, 0.5)
            accel_z = base_accel + walking_amplitude * 0.3 * np.sin(2 * np.pi * walking_freq * t * 0.8) + random.gauss(0, 0.3)
            
            # Generate test gyroscope data
            gyro_x = 10 * np.sin(2 * np.pi * walking_freq * t * 0.7) + random.gauss(0, 1)
            gyro_y = 10 * np.cos(2 * np.pi * walking_freq * t * 0.9) + random.gauss(0, 1)
            gyro_z = 5 * np.sin(2 * np.pi * walking_freq * t * 1.2) + random.gauss(0, 0.5)
            
            # Update buffers
            power_data_buffers['timestamp'].append(int(t * 1e6))  # Convert to microseconds
            power_data_buffers['accel_x'].append(accel_x)
            power_data_buffers['accel_y'].append(accel_y)
            power_data_buffers['accel_z'].append(accel_z)
            power_data_buffers['gyro_x'].append(gyro_x)
            power_data_buffers['gyro_y'].append(gyro_y)
            power_data_buffers['gyro_z'].append(gyro_z)
            
            time.sleep(0.01)  # 100 Hz simulation
            
        except Exception as e:
            print(f"Power test data generator error: {e}")
            time.sleep(0.1)

def frequency_test_data_generator():
    """Generate test data for frequency visualization"""
    import random
    global freq_start_time
    
    # IMU list for frequency monitoring
    imu_list = ["IMU1", "IMU2", "IMU3", "IMU4", "IMU5", "IMU6", "IMU7"]
    
    while True:
        try:
            current_time = time.time()
            
            for imu_id in imu_list:
                # Simulate realistic frequency patterns (180-220 Hz with some variation)
                base_freq = 200
                variation = 20 * np.sin(2 * np.pi * 0.1 * current_time) + random.gauss(0, 5)
                frequency = max(80, min(250, base_freq + variation))
                
                # Add timestamp to raw data buffer (simulate packet arrivals)
                freq_raw_buffer[imu_id].append(current_time)
                
                # Calculate frequency every second
                if len(freq_raw_buffer[imu_id]) > 0:
                    # Count packets in the last second
                    one_second_ago = current_time - 1.0
                    recent_packets = [t for t in freq_raw_buffer[imu_id] if t >= one_second_ago]
                    calculated_freq = len(recent_packets)
                    
                    # Update frequency data (only add new data points every second)
                    if (not freq_time_data[imu_id] or 
                        current_time - freq_time_data[imu_id][-1] >= 0.8):
                        
                        relative_time = current_time - freq_start_time
                        freq_data_buffers[imu_id].append(frequency)  # Use simulated frequency
                        freq_time_data[imu_id].append(relative_time)
            
            time.sleep(0.01)  # 100 Hz simulation
            
        except Exception as e:
            print(f"Frequency test data generator error: {e}")
            time.sleep(0.1)

def stickman_test_data_generator():
    """Generate test data for stickman visualization"""
    import random
    
    while True:
        try:
            current_time = time.time()
            
            # Generate realistic walking motion quaternions
            walking_freq = 0.5  # Walking frequency
            t = current_time * walking_freq
            
            # Simulate walking motion with different patterns for each IMU
            for imu_id in stickman_imu_data.keys():
                # Base quaternion (no rotation)
                base_quat = [1.0, 0.0, 0.0, 0.0]
                
                if imu_id == 'IMU7':  # Pelvis - slight rocking motion
                    angle = 0.1 * np.sin(2 * np.pi * t) + random.gauss(0, 0.02)
                    quat = [np.cos(angle/2), np.sin(angle/2), 0, 0]
                elif imu_id == 'IMU1':  # Left upper leg - hip flexion
                    angle = 0.3 * np.sin(2 * np.pi * t) + random.gauss(0, 0.05)
                    quat = [np.cos(angle/2), np.sin(angle/2), 0, 0]
                elif imu_id == 'IMU3':  # Left lower leg - knee flexion
                    angle = 0.2 * np.sin(2 * np.pi * t + np.pi/4) + random.gauss(0, 0.05)
                    quat = [np.cos(angle/2), np.sin(angle/2), 0, 0]
                elif imu_id == 'IMU2':  # Right upper leg - opposite phase
                    angle = 0.3 * np.sin(2 * np.pi * t + np.pi) + random.gauss(0, 0.05)
                    quat = [np.cos(angle/2), np.sin(angle/2), 0, 0]
                elif imu_id == 'IMU4':  # Right lower leg - opposite phase
                    angle = 0.2 * np.sin(2 * np.pi * t + np.pi + np.pi/4) + random.gauss(0, 0.05)
                    quat = [np.cos(angle/2), np.sin(angle/2), 0, 0]
                
                # Normalize quaternion
                norm = np.linalg.norm(quat)
                if norm > 0:
                    quat = [q/norm for q in quat]
                
                # Update IMU data
                stickman_imu_data[imu_id]['quat'] = quat
                stickman_imu_data[imu_id]['timestamp'] = current_time
                
                # Signal that new data is available
                try:
                    stickman_data_queue.put_nowait({'imu_id': imu_id, 'quat': quat})
                except queue.Full:
                    pass  # Drop data if queue is full
            
            time.sleep(0.1)  # 10 Hz simulation
            
        except Exception as e:
            print(f"Stickman test data generator error: {e}")
            time.sleep(0.1)

def pelvic_metrics_test_data_generator():
    """Generate test data for pelvic metrics visualization"""
    import random
    global pelvic_start_time, pelvic_calibration_quats, pelvic_R0, pelvic_is_calibrated, pelvic_sample_count
    
    # Initialize start time
    if pelvic_start_time is None:
        pelvic_start_time = time.time()
    
    while True:
        try:
            current_time = time.time()
            pelvic_sample_count += 1
            
            # Generate realistic pelvic motion quaternions
            walking_freq = 0.3  # Slower pelvic motion
            t = (current_time - pelvic_start_time) * walking_freq
            
            # Simulate pelvic motion patterns
            base_angle = 0.1 * np.sin(2 * np.pi * t) + random.gauss(0, 0.02)
            quat = [
                np.cos(base_angle/2),
                np.sin(base_angle/2) * 0.5,
                np.sin(base_angle/2) * 0.3,
                np.sin(base_angle/2) * 0.2
            ]
            
            # Normalize quaternion
            norm = np.linalg.norm(quat)
            if norm > 0:
                quat = [q/norm for q in quat]
            
            # Calibration phase
            if not pelvic_is_calibrated:
                pelvic_calibration_quats.append(quat)
                if len(pelvic_calibration_quats) >= pelvic_calibration_samples:
                    # Compute calibration matrix
                    R_sum = np.zeros((3,3))
                    for q in pelvic_calibration_quats:
                        w, x, y, z = q
                        R = np.array([
                            [1 - 2*(y*y + z*z),     2*(x*y - w*z),       2*(x*z + w*y)],
                            [2*(x*y + w*z),         1 - 2*(x*x + z*z),   2*(y*z - w*x)],
                            [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x*x + y*y)]
                        ])
                        R_sum += R
                    R_sum /= len(pelvic_calibration_quats)
                    U, _, Vt = np.linalg.svd(R_sum)
                    pelvic_R0 = U @ Vt
                    pelvic_is_calibrated = True
                    print("Pelvic metrics calibration completed!")
                continue
            
            # Process quaternion to angles
            w, x, y, z = quat
            R_current = np.array([
                [1 - 2*(y*y + z*z),     2*(x*y - w*z),       2*(x*z + w*y)],
                [2*(x*y + w*z),         1 - 2*(x*x + z*z),   2*(y*z - w*x)],
                [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x*x + y*y)]
            ])
            
            # Apply calibration
            R_cal = pelvic_R0.T @ R_current
            
            # Extract Euler angles
            beta_y = -np.arcsin(R_cal[2, 0])       # pitch
            cos_beta = np.cos(beta_y)
            if abs(cos_beta) < 1e-6:
                alpha_z = np.arctan2(-R_cal[0, 1], R_cal[1, 1])  # yaw
                gamma_x = 0.0                            # roll
            else:
                alpha_z = np.arctan2(R_cal[1, 0], R_cal[0, 0])   # yaw
                gamma_x = np.arctan2(R_cal[2, 1], R_cal[2, 2])   # roll
            
            # Convert to degrees
            tilt = np.degrees(-beta_y)      # pitch
            obliquity = np.degrees(gamma_x)  # roll
            rotation = np.degrees(alpha_z)   # yaw
            
            # Add some realistic variation
            tilt += 5 * np.sin(2 * np.pi * t * 1.2) + random.gauss(0, 1)
            obliquity += 3 * np.sin(2 * np.pi * t * 0.8) + random.gauss(0, 0.5)
            rotation += 2 * np.sin(2 * np.pi * t * 1.5) + random.gauss(0, 0.3)
            
            # Store data
            pelvic_data_buffers['timestamp'].append(current_time)
            pelvic_data_buffers['tilt'].append(tilt)
            pelvic_data_buffers['obliquity'].append(obliquity)
            pelvic_data_buffers['rotation'].append(rotation)
            
            # Signal new data
            try:
                pelvic_data_queue.put_nowait({
                    'timestamp': current_time,
                    'tilt': tilt,
                    'obliquity': obliquity,
                    'rotation': rotation
                })
            except queue.Full:
                pass
            
            time.sleep(0.05)  # 20 Hz simulation
            
        except Exception as e:
            print(f"Pelvic metrics test data generator error: {e}")
            time.sleep(0.1)

def create_step_length_view():
    """Create the step length visualization view"""
    # Initialize visualizer
    if 'step_visualizer' not in st.session_state:
        st.session_state.step_visualizer = StepLengthVisualizer()
        # Start test data generator in a thread
        if 'step_data_thread' not in st.session_state:
            st.session_state.step_data_thread = threading.Thread(target=step_test_data_generator, daemon=True)
            st.session_state.step_data_thread.start()
    
    with left_col:
        st.markdown("### Real-Time Step Detection", unsafe_allow_html=True)
        plot_container = st.empty()
        
    with right_col:
        st.markdown("### Step Detection Metrics", unsafe_allow_html=True)
        metrics_container = st.empty()
    
    # Real-time update loop with containers
    for i in range(1000):  # Run for a reasonable time
        try:
            # Update plot
            with plot_container.container():
                st.image(
                    st.session_state.step_visualizer.get_plot_image(),
                    use_container_width=True
                )
            
            # Update metrics
            metrics_html = ""
            for imu_id, imu_name in IMU_MAPPING.items():
                buffer = imu_data_buffers[imu_id]
                metrics_html += f"""
                <div class="metric-card">
                    <h4 style="color: #2c3e50; margin: 0;">{imu_name}</h4>
                    <p class="metric-value">Steps: {buffer['step_count']}</p>
                    <p class="metric-value">Cadence: {buffer['cadence']:.1f} steps/min</p>
                </div>
                """
            
            with metrics_container.container():
                st.markdown(metrics_html, unsafe_allow_html=True)
            
            time.sleep(0.5)  # Update every 0.5 seconds
            
        except Exception as e:
            print(f"Step length view error: {e}")
            break

def create_power_metrics_view():
    """Create the power metrics visualization view"""
    # Initialize visualizer
    if 'power_visualizer' not in st.session_state:
        st.session_state.power_visualizer = PowerMetricsVisualizer()
        # Start test data generator in a thread
        if 'power_data_thread' not in st.session_state:
            st.session_state.power_data_thread = threading.Thread(target=power_test_data_generator, daemon=True)
            st.session_state.power_data_thread.start()
    
    with left_col:
        st.markdown("### Real-Time Power Metrics", unsafe_allow_html=True)
        plot_container = st.empty()
        
    with right_col:
        st.markdown("### Power Statistics", unsafe_allow_html=True)
        metrics_container = st.empty()
    
    # Real-time update loop with containers
    for i in range(1000):  # Run for a reasonable time
        try:
            # Update plot
            with plot_container.container():
                st.image(
                    st.session_state.power_visualizer.get_plot_image(),
                    use_container_width=True
                )
            
            # Update metrics
            metrics_html = f"""
            <div class="metric-card">
                <h4 style="color: #2c3e50; margin: 0;">TRANSLATIONAL POWER</h4>
                <p class="metric-value">Peak: {power_stats['trans_power_peak']:.2f} W</p>
                <p class="metric-value">Mean: {power_stats['trans_power_mean']:.2f} W</p>
            </div>
            <div class="metric-card">
                <h4 style="color: #2c3e50; margin: 0;">ROTATIONAL POWER</h4>
                <p class="metric-value">Peak: {power_stats.get('rot_power_peak', 0):.2f} W</p>
                <p class="metric-value">Mean: {power_stats.get('rot_power_mean', 0):.2f} W</p>
            </div>
            <div class="metric-card">
                <h4 style="color: #2c3e50; margin: 0;">TOTAL POWER</h4>
                <p class="metric-value">Peak: {power_stats['total_power_peak']:.2f} W</p>
                <p class="metric-value">Mean: {power_stats['total_power_mean']:.2f} W</p>
            </div>
            <div class="metric-card">
                <h4 style="color: #2c3e50; margin: 0;">STRIDE TIME</h4>
                <p class="metric-value">{power_stats.get('stride_time_mean', 823):.1f} ± {power_stats.get('stride_time_std', 0):.1f} ms</p>
            </div>
            <div class="metric-card">
                <h4 style="color: #2c3e50; margin: 0;">CADENCE</h4>
                <p class="metric-value">{power_stats.get('cadence', 72.9):.1f} steps/min</p>
            </div>
            <div class="metric-card">
                <h4 style="color: #2c3e50; margin: 0;">STRIDE LENGTH</h4>
                <p class="metric-value">{power_stats.get('stride_length_mean', 42.53):.2f} ± {power_stats.get('stride_length_std', 4.25):.2f} m</p>
            </div>
            <div class="metric-card">
                <h4 style="color: #2c3e50; margin: 0;">PEAK LOAD</h4>
                <p class="metric-value">{power_stats.get('peak_load', 499.80):.2f} N</p>
            </div>
            """
            
            with metrics_container.container():
                st.markdown(metrics_html, unsafe_allow_html=True)
            
            time.sleep(0.5)  # Update every 0.5 seconds
            
        except Exception as e:
            print(f"Power metrics view error: {e}")
            break

def create_frequency_view():
    """Create the frequency visualization view"""
    # Initialize visualizer
    if 'freq_visualizer' not in st.session_state:
        st.session_state.freq_visualizer = FrequencyVisualizer()
        # Start test data generator in a thread
        if 'freq_data_thread' not in st.session_state:
            st.session_state.freq_data_thread = threading.Thread(target=frequency_test_data_generator, daemon=True)
            st.session_state.freq_data_thread.start()
    
    with left_col:
        st.markdown("### Real-Time Frequency Monitoring", unsafe_allow_html=True)
        plot_container = st.empty()
        
    with right_col:
        st.markdown("### Frequency Statistics", unsafe_allow_html=True)
        metrics_container = st.empty()
    
    # Real-time update loop with containers
    for i in range(1000):  # Run for a reasonable time
        try:
            # Update plot
            with plot_container.container():
                st.image(
                    st.session_state.freq_visualizer.get_plot_image(),
                    use_container_width=True
                )
            
            # Update metrics
            metrics_html = ""
            for imu_id in sorted(freq_stats.keys()):
                stats = freq_stats[imu_id]
                metrics_html += f"""
                <div class="metric-card">
                    <h4 style="color: #2c3e50; margin: 0;">{imu_id}</h4>
                    <p class="metric-value">Current: {stats['current']:.1f} Hz</p>
                    <p class="metric-value">Average: {stats['avg']:.1f} Hz</p>
                    <p class="metric-value">Data Points: {stats['count']}</p>
                </div>
                """
            
            if not metrics_html:
                metrics_html = """
                <div class="metric-card">
                    <h4 style="color: #2c3e50; margin: 0;">Waiting for Data</h4>
                    <p class="metric-value">Starting frequency monitoring...</p>
                </div>
                """
            
            with metrics_container.container():
                st.markdown(metrics_html, unsafe_allow_html=True)
            
            time.sleep(0.5)  # Update every 0.5 seconds
            
        except Exception as e:
            print(f"Frequency view error: {e}")
            break

def create_stickman_view():
    """Create the stickman visualization view"""
    # Initialize visualizer
    if 'stickman_visualizer' not in st.session_state:
        st.session_state.stickman_visualizer = StickmanVisualizer()
        # Start test data generator in a thread
        if 'stickman_data_thread' not in st.session_state:
            st.session_state.stickman_data_thread = threading.Thread(target=stickman_test_data_generator, daemon=True)
            st.session_state.stickman_data_thread.start()
    
    with left_col:
        st.markdown("### Real-Time Stickman Visualization", unsafe_allow_html=True)
        plot_container = st.empty()
        
    with right_col:
        st.markdown("### IMU Data Status", unsafe_allow_html=True)
        metrics_container = st.empty()
    
    # Real-time update loop with containers
    for i in range(1000):  # Run for a reasonable time
        try:
            # Process any queued data
            while not stickman_data_queue.empty():
                try:
                    stickman_data_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Update plot
            with plot_container.container():
                st.image(
                    st.session_state.stickman_visualizer.get_plot_image(),
                    use_container_width=True
                )
            
            # Update metrics
            metrics_html = ""
            for imu_id in sorted(stickman_imu_data.keys()):
                data = stickman_imu_data[imu_id]
                quat = data['quat']
                timestamp = data['timestamp']
                
                # Calculate quaternion magnitude (should be ~1.0)
                quat_mag = np.linalg.norm(quat)
                
                # Time since last update
                time_diff = time.time() - timestamp if timestamp > 0 else 0
                
                # Map IMU to body part
                body_part_map = {
                    'IMU1': 'Left Upper Leg',
                    'IMU2': 'Right Upper Leg', 
                    'IMU3': 'Left Lower Leg',
                    'IMU4': 'Right Lower Leg',
                    'IMU7': 'Pelvis'
                }
                
                body_part = body_part_map.get(imu_id, imu_id)
                
                metrics_html += f"""
                <div class="metric-card">
                    <h4 style="color: #2c3e50; margin: 0;">{body_part} ({imu_id})</h4>
                    <p class="metric-value">Quaternion: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]</p>
                    <p class="metric-value">Magnitude: {quat_mag:.3f}</p>
                    <p class="metric-value">Last Update: {time_diff:.1f}s ago</p>
                </div>
                """
            
            if not metrics_html:
                metrics_html = """
                <div class="metric-card">
                    <h4 style="color: #2c3e50; margin: 0;">Waiting for Data</h4>
                    <p class="metric-value">Starting stickman visualization...</p>
                </div>
                """
            
            with metrics_container.container():
                st.markdown(metrics_html, unsafe_allow_html=True)
            
            time.sleep(0.5)  # Update every 0.5 seconds
            
        except Exception as e:
            print(f"Stickman view error: {e}")
            break

def create_pelvic_metrics_view():
    """Create the pelvic metrics visualization view"""
    global pelvic_start_time, pelvic_is_calibrated, pelvic_sample_count
    
    # Initialize visualizer
    if 'pelvic_visualizer' not in st.session_state:
        st.session_state.pelvic_visualizer = PelvicMetricsVisualizer()
        # Start test data generator in a thread
        if 'pelvic_data_thread' not in st.session_state:
            st.session_state.pelvic_data_thread = threading.Thread(target=pelvic_metrics_test_data_generator, daemon=True)
            st.session_state.pelvic_data_thread.start()
    
    with left_col:
        st.markdown("### Real-Time Pelvic Metrics", unsafe_allow_html=True)
        plot_container = st.empty()
        
    with right_col:
        st.markdown("### Pelvic Metrics Status", unsafe_allow_html=True)
        metrics_container = st.empty()
    
    # Real-time update loop with containers
    for i in range(1000):  # Run for a reasonable time
        try:
            # Process any queued data
            while not pelvic_data_queue.empty():
                try:
                    pelvic_data_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Update plot
            with plot_container.container():
                st.image(
                    st.session_state.pelvic_visualizer.get_plot_image(),
                    use_container_width=True
                )
            
            # Update metrics
            if pelvic_is_calibrated and pelvic_data_buffers['tilt']:
                # Get latest values
                latest_tilt = list(pelvic_data_buffers['tilt'])[-1]
                latest_obliquity = list(pelvic_data_buffers['obliquity'])[-1]
                latest_rotation = list(pelvic_data_buffers['rotation'])[-1]
                
                # Calculate statistics
                tilt_data = list(pelvic_data_buffers['tilt'])
                obliquity_data = list(pelvic_data_buffers['obliquity'])
                rotation_data = list(pelvic_data_buffers['rotation'])
                
                total_duration = (time.time() - pelvic_start_time) if pelvic_start_time else 0
                total_points = len(tilt_data)
                
                metrics_html = f"""
                <div class="metric-card">
                    <h4 style="color: #2c3e50; margin: 0;">Real-time Pelvic Metrics (IMU7)</h4>
                    <p class="metric-value">Total Duration: {total_duration:.1f}s</p>
                    <p class="metric-value">Data Points: {total_points}</p>
                </div>
                <div class="metric-card">
                    <h4 style="color: #2c3e50; margin: 0;">Current Angles</h4>
                    <p class="metric-value">Tilt: {latest_tilt:.2f}°</p>
                    <p class="metric-value">Obliquity: {latest_obliquity:.2f}°</p>
                    <p class="metric-value">Rotation: {latest_rotation:.2f}°</p>
                </div>
                <div class="metric-card">
                    <h4 style="color: #2c3e50; margin: 0;">Statistics</h4>
                    <p class="metric-value">Tilt Range: {np.min(tilt_data):.1f}° to {np.max(tilt_data):.1f}°</p>
                    <p class="metric-value">Obliquity Range: {np.min(obliquity_data):.1f}° to {np.max(obliquity_data):.1f}°</p>
                    <p class="metric-value">Rotation Range: {np.min(rotation_data):.1f}° to {np.max(rotation_data):.1f}°</p>
                </div>
                <div class="metric-card">
                    <h4 style="color: #2c3e50; margin: 0;">Averages</h4>
                    <p class="metric-value">Avg Tilt: {np.mean(tilt_data):.2f}°</p>
                    <p class="metric-value">Avg Obliquity: {np.mean(obliquity_data):.2f}°</p>
                    <p class="metric-value">Avg Rotation: {np.mean(rotation_data):.2f}°</p>
                </div>
                """
            else:
                # Show calibration status
                calib_progress = len(pelvic_calibration_quats)
                calib_total = pelvic_calibration_samples
                
                metrics_html = f"""
                <div class="metric-card">
                    <h4 style="color: #2c3e50; margin: 0;">Calibrating IMU7...</h4>
                    <p class="metric-value">Progress: {calib_progress}/{calib_total}</p>
                    <p class="metric-value">Stand still for calibration</p>
                </div>
                <div class="metric-card">
                    <h4 style="color: #2c3e50; margin: 0;">Instructions</h4>
                    <p class="metric-value">1. Keep pelvis still during calibration</p>
                    <p class="metric-value">2. Calibration will complete automatically</p>
                    <p class="metric-value">3. Then start normal movement</p>
                </div>
                """
            
            with metrics_container.container():
                st.markdown(metrics_html, unsafe_allow_html=True)
            
            time.sleep(0.5)  # Update every 0.5 seconds
            
        except Exception as e:
            print(f"Pelvic metrics view error: {e}")
            break

def create_default_view():
    """Create the default view"""
    with left_col:
        st.markdown("### JD Metrics")
        st.markdown('<p style="color: #2c3e50; margin: 0; font-family: monospace;">ID: 001</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #2c3e50; margin: 0; font-family: monospace;">Age: 45</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #2c3e50; margin: 0; font-family: monospace;">BMI: 24.9</p>', unsafe_allow_html=True)
        
        st.markdown("""
            <div style="background-color: #f8f9fa; height: 400px; border-radius: 8px; display: flex; align-items: center; justify-content: center; border: 1px solid #e9ecef;">
                <p style="color: #2c3e50; font-weight: 500;">Stickman Visualization Placeholder</p>
            </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown("### Clinical Endpoints")
        st.markdown("Comprehensive biomechanical analysis")
        
        tabs = st.tabs(["Kinematics", "Kinetics", "Temporal", "Functions"])
        
        st.markdown("### Gait Parameters")
        
        st.markdown('<p style="color: #2c3e50; margin-bottom: 0; font-weight: 500;">Walking Speed</p>', unsafe_allow_html=True)
        walking_speed = st.slider("", min_value=0.0, max_value=2.0, value=1.29, step=0.01, format="%.2f m/s", key="walking_speed")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px;">
                <h3 style="margin: 0; color: #2e7d32;">1.23m</h3>
                <p style="margin: 0; color: #2e7d32;">Step Length</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style="background-color: #fff3e0; padding: 15px; border-radius: 8px;">
                <h3 style="margin: 0; color: #ef6c00;">115</h3>
                <p style="margin: 0; color: #ef6c00;">Cadence (spm)</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Gait Timing")
        cols = st.columns(3)
        timing_data = [
            {"value": "62%", "label": "Stance", "color": "#e8eaf6"},
            {"value": "38%", "label": "Swing", "color": "#fce4ec"},
            {"value": "13%", "label": "Double Support", "color": "#e0f7fa"}
        ]
        
        for col, data in zip(cols, timing_data):
            with col:
                st.markdown(f"""
                <div style="background-color: {data['color']}; padding: 15px; border-radius: 8px; text-align: center;">
                    <h3 style="margin: 0;">{data['value']}</h3>
                    <p style="margin: 0;">{data['label']}</p>
                </div>
                """, unsafe_allow_html=True)

# Show appropriate view based on selection
if selected_option == "StepLength":
    create_step_length_view()
elif selected_option == "PowerMetrics":
    create_power_metrics_view()
elif selected_option == "Frequency":
    create_frequency_view()
elif selected_option == "Stickman":
    create_stickman_view()
elif selected_option == "PelvicMetrics":
    create_pelvic_metrics_view()
else:
    create_default_view()

# Active session indicator
st.markdown("""
<div style="position: fixed; bottom: 20px; left: 20px; background-color: #e8f5e9; padding: 5px 15px; border-radius: 20px;">
    <span style="color: #2e7d32;">● Active Session</span>
</div>
""", unsafe_allow_html=True) 