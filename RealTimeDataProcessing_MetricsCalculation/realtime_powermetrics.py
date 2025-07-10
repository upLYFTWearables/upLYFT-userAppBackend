#!/usr/bin/env python3
"""
Real-time Power Metrics Processing and Visualization
Adapted from PowerMetrics.py for dynamic data processing from IMU5 foot sensor
"""
import socket
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import csv
from datetime import datetime
import scipy.signal as signal
from scipy.signal import find_peaks
import pandas as pd
import traceback

# Constants for IMU data processing
WINDOW_SIZE = 500  # Number of samples to keep in buffer
UPDATE_INTERVAL = 1000  # Update plots every 1000ms
UDP_PORT = 12345
BUFFER_SIZE = 1024

class RealTimePowerMetrics:
    def __init__(self):
        # Physical parameters
        self.foot_mass = 1.2  # kg
        self.body_mass = 70.0  # kg
        self.foot_inertia = [0.01, 0.01, 0.01]  # kg*m^2
        
        # Debug timing
        self.last_imu_debug = 0
        
        # Data buffers
        self.data_buffer = {
            'timestamp': deque(maxlen=WINDOW_SIZE),
            'accel_x': deque(maxlen=WINDOW_SIZE),
            'accel_y': deque(maxlen=WINDOW_SIZE),
            'accel_z': deque(maxlen=WINDOW_SIZE),
            'gyro_x': deque(maxlen=WINDOW_SIZE),
            'gyro_y': deque(maxlen=WINDOW_SIZE),
            'gyro_z': deque(maxlen=WINDOW_SIZE),
        }
        
        # Power metrics buffers
        self.power_metrics = {
            'trans_power': deque(maxlen=WINDOW_SIZE),
            'rot_power': deque(maxlen=WINDOW_SIZE),
            'total_power': deque(maxlen=WINDOW_SIZE),
            'power_weight': deque(maxlen=WINDOW_SIZE),
            'rolling_avg': deque(maxlen=WINDOW_SIZE),
        }
        
        # Gait analysis buffers
        self.gait_metrics = {
            'stride_times': [],
            'gct_times': [],
            'swing_times': [],
            'stride_lengths': [],
            'cadence': 0,
            'peak_load': 0,
        }
        
        # Statistics
        self.stats = {
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
        
        # Setup plots
        self.setup_plots()
        
        # UDP socket setup with larger buffer for multiple IMUs
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Set socket buffer size to handle multiple IMU streams
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        self.sock.bind(('0.0.0.0', UDP_PORT))  # Listen on all interfaces
        self.running = False
        
        # Timing
        self.last_plot_update = 0
        self.last_stats_update = 0
        self.stats_update_interval = 1.0  # Update stats every 2 seconds
        
        # Gait detection
        self.last_peak_time = 0
        self.current_stride_start = 0
        self.in_stance = False
        
    def setup_plots(self):
        """Setup all matplotlib plots"""
        self.fig = plt.figure(figsize=(15, 12))
        
        # Raw acceleration
        self.ax1 = plt.subplot(3, 2, 1)
        self.ax1.set_title('Raw Acceleration')
        self.ax1.set_ylabel('m/s²')
        self.ax1.grid(True)
        self.accel_lines = []
        colors = ['blue', 'orange', 'green']
        labels = ['AccelX', 'AccelY', 'AccelZ']
        for i, (color, label) in enumerate(zip(colors, labels)):
            line, = self.ax1.plot([], [], color=color, label=label)
            self.accel_lines.append(line)
        self.ax1.legend()
        
        # Translational Power
        self.ax2 = plt.subplot(3, 2, 2)
        self.ax2.set_title('Translational Power (m·a)·v')
        self.ax2.set_ylabel('W')
        self.ax2.grid(True)
        self.trans_power_line, = self.ax2.plot([], [], 'b-', label='Trans Power')
        self.ax2.legend()
        
        # Rotational Power
        self.ax3 = plt.subplot(3, 2, 3)
        self.ax3.set_title('Rotational Power τ·ω')
        self.ax3.set_ylabel('W')
        self.ax3.grid(True)
        self.rot_power_line, = self.ax3.plot([], [], 'r-', label='Rot Power')
        self.ax3.legend()
        
        # Total Power
        self.ax4 = plt.subplot(3, 2, 4)
        self.ax4.set_title('Total Power')
        self.ax4.set_ylabel('W')
        self.ax4.grid(True)
        self.total_power_line, = self.ax4.plot([], [], 'g-', label='Total Power')
        self.ax4.legend()
        
        # Power/Weight
        self.ax5 = plt.subplot(3, 2, 5)
        self.ax5.set_title('Instantaneous Total Power / Weight')
        self.ax5.set_ylabel('W/kg')
        self.ax5.set_xlabel('Time (s)')
        self.ax5.grid(True)
        self.power_weight_line, = self.ax5.plot([], [], 'b-', label='Power/Weight')
        self.ax5.legend()
        
        # Rolling Average
        self.ax6 = plt.subplot(3, 2, 6)
        self.ax6.set_title('Rolling Average Total Power')
        self.ax6.set_ylabel('W')
        self.ax6.set_xlabel('Time (s)')
        self.ax6.grid(True)
        self.rolling_avg_line, = self.ax6.plot([], [], 'b-', label='Rolling Avg')
        self.ax6.legend()
        
        plt.tight_layout()
        
    def apply_lowpass_filter(self, data):
        """Apply a low-pass filter to the data"""
        if len(data) <= 15:  # Check if we have enough data for the filter
            return data  # Return unfiltered data if not enough samples
            
        # Apply Butterworth filter
        b, a = signal.butter(4, 0.1, 'low')
        return signal.filtfilt(b, a, data)
        
    def parse_imu_data(self, data_str):
        """Parse IMU data from UDP packet and extract IMU5 data"""
        try:
            lines = data_str.strip().split('\n')
            imu5_data = []
            other_imus = set()  # Track other IMUs for debugging
            
            for line in lines:
                if not line.strip():
                    continue
                    
                parts = line.split(',')
                if len(parts) >= 13:  # Ensure we have enough data fields
                    imu_id = parts[2].strip()
                    
                    # Track which IMUs we're seeing
                    if imu_id != 'IMU5':
                        other_imus.add(imu_id)
                    
                    if imu_id == 'IMU5':
                        try:
                            # Convert time format to microseconds
                            time_str = parts[1].strip()
                            try:
                                # Try parsing as microseconds first
                                timestamp = int(time_str)
                            except ValueError:
                                # If that fails, parse as time string
                                try:
                                    # Parse time string and convert to microseconds
                                    h, m, s = time_str.split(':')
                                    s, ms = s.split('.')
                                    total_seconds = int(h) * 3600 + int(m) * 60 + int(s)
                                    timestamp = total_seconds * 1000000 + int(ms) * 1000
                                except Exception as e:
                                    print(f"Error parsing timestamp {time_str}: {e}")
                                    continue

                            data_point = {
                                'timestamp': timestamp,
                                'accel_x': float(parts[3]),
                                'accel_y': float(parts[4]),
                                'accel_z': float(parts[5]),
                                'gyro_x': float(parts[6]),
                                'gyro_y': float(parts[7]),
                                'gyro_z': float(parts[8]),
                            }
                            imu5_data.append(data_point)
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing IMU5 data: {e}")
                            continue
            
            # Print debug info about other IMUs occasionally
            if other_imus and time.time() - self.last_imu_debug > 5.0:  # Every 5 seconds
                print(f"\nDetected IMUs: {', '.join(sorted(other_imus))}")
                self.last_imu_debug = time.time()
            
            return imu5_data
            
        except Exception as e:
            print(f"Error parsing data: {e}")
            return []
      
    def update_plots(self):
        """Update all plot lines with current data"""
        # Limit update rate
        current_time = time.time()
        if current_time - self.last_plot_update < UPDATE_INTERVAL/1000.0:
            return
            
        if not self.data_buffer['timestamp']:
            return
            
        try:
            # Get all data arrays
            timestamps = np.array(list(self.data_buffer['timestamp']))
            accel_data = {
                'accel_x': np.array(list(self.data_buffer['accel_x'])),
                'accel_y': np.array(list(self.data_buffer['accel_y'])),
                'accel_z': np.array(list(self.data_buffer['accel_z']))
            }
            trans_power_data = np.array(list(self.power_metrics['trans_power']))
            rot_power_data = np.array(list(self.power_metrics['rot_power']))
            total_power_data = np.array(list(self.power_metrics['total_power']))
            power_weight_data = np.array(list(self.power_metrics['power_weight']))
            rolling_avg_data = np.array(list(self.power_metrics['rolling_avg']))
            
            # Find minimum length to ensure all arrays match
            min_length = min(
                timestamps.shape[0],
                min(arr.shape[0] for arr in accel_data.values()),
                trans_power_data.shape[0] if trans_power_data.size > 0 else float('inf'),
                rot_power_data.shape[0] if rot_power_data.size > 0 else float('inf'),
                total_power_data.shape[0] if total_power_data.size > 0 else float('inf'),
                power_weight_data.shape[0] if power_weight_data.size > 0 else float('inf'),
                rolling_avg_data.shape[0] if rolling_avg_data.size > 0 else float('inf')
            )
            
            if min_length < 2:  # Need at least 2 points to plot
                return
                
            # Trim all arrays to same length
            timestamps = timestamps[:min_length]
            # Make time relative to start and convert to seconds
            if len(timestamps) > 0:
                t = (timestamps - timestamps[0]) / 1e6  # Convert to seconds
            else:
                t = np.array([])
            
            # Update acceleration plots
            for i, key in enumerate(['accel_x', 'accel_y', 'accel_z']):
                self.accel_lines[i].set_data(t, accel_data[key][:min_length])
            
            # Update power plots if we have power data
            if trans_power_data.size > 0:
                power_length = min(min_length, trans_power_data.shape[0])
                t_power = t[:power_length]
                self.trans_power_line.set_data(t_power, trans_power_data[:power_length])
                self.rot_power_line.set_data(t_power, rot_power_data[:power_length])
                self.total_power_line.set_data(t_power, total_power_data[:power_length])
                self.power_weight_line.set_data(t_power, power_weight_data[:power_length])
                self.rolling_avg_line.set_data(t_power, rolling_avg_data[:power_length])
            
            # Adjust plot limits
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
                ax.relim()
                ax.autoscale_view()
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.last_plot_update = current_time
            
        except Exception as e:
            print(f"Error updating plots: {e}")
            traceback.print_exc()  # Print full traceback for debugging
            
    def detect_gait_events(self, accel_data, timestamps):
        """Detect gait events from acceleration data"""
        try:
            # Use vertical acceleration (Z-axis) for gait detection
            accel_z = accel_data[:, 2]
            
            # Find peaks in acceleration (heel strikes)
            peaks, _ = find_peaks(accel_z, height=np.mean(accel_z) + 0.5 * np.std(accel_z), distance=50)
            
            current_time = timestamps[-1] / 1e6  # Convert to seconds
            
            for peak_idx in peaks:
                peak_time = timestamps[peak_idx] / 1e6
                
                # Calculate stride time
                if self.last_peak_time > 0:
                    stride_time = (peak_time - self.last_peak_time) * 1000  # Convert to ms
                    if 500 < stride_time < 2000:  # Reasonable stride time range
                        self.gait_metrics['stride_times'].append(stride_time)
                        
                        # Keep only recent stride times
                        if len(self.gait_metrics['stride_times']) > 20:
                            self.gait_metrics['stride_times'].pop(0)
                
                self.last_peak_time = peak_time
                
        except Exception as e:
            print(f"Error in gait detection: {e}")
            
    def calculate_statistics(self):
        """Calculate and update statistics"""
        try:
            # Power statistics
            if self.power_metrics['trans_power']:
                trans_power = list(self.power_metrics['trans_power'])
                self.stats['trans_power_peak'] = max(trans_power)
                self.stats['trans_power_mean'] = np.mean(trans_power)
                
            if self.power_metrics['rot_power']:
                rot_power = list(self.power_metrics['rot_power'])
                self.stats['rot_power_peak'] = max(rot_power)
                self.stats['rot_power_mean'] = np.mean(rot_power)
                
            if self.power_metrics['total_power']:
                total_power = list(self.power_metrics['total_power'])
                self.stats['total_power_peak'] = max(total_power)
                self.stats['total_power_mean'] = np.mean(total_power)
            
            # Gait statistics
            if self.gait_metrics['stride_times']:
                stride_times = self.gait_metrics['stride_times']
                self.stats['stride_time_mean'] = np.mean(stride_times)
                self.stats['stride_time_std'] = np.std(stride_times)
                self.stats['cadence'] = 60000 / np.mean(stride_times)  # steps per minute
                
                # Estimate GCT and swing times (simplified)
                self.stats['gct_mean'] = np.mean(stride_times) * 0.6  # Typical 60% stance
                self.stats['gct_std'] = np.std(stride_times) * 0.6
                self.stats['swing_mean'] = np.mean(stride_times) * 0.4  # Typical 40% swing
                self.stats['swing_std'] = np.std(stride_times) * 0.4
                
                # Estimate stride length (simplified)
                self.stats['stride_length_mean'] = self.stats['cadence'] * 70 / 60 / 2  # Rough estimate
                self.stats['stride_length_std'] = self.stats['stride_length_mean'] * 0.1
            
            # Peak load estimate from acceleration
            if self.data_buffer['accel_z']:
                accel_z = list(self.data_buffer['accel_z'])
                self.stats['peak_load'] = max(accel_z) * self.body_mass  # Rough estimate
                
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            
    def print_statistics(self):
        """Print current statistics to console"""
        print("\n" + "="*50)
        print(f"TRANSLATIONAL POWER  peak  {self.stats['trans_power_peak']:8.2f} W   mean  {self.stats['trans_power_mean']:8.2f} W")
        print(f"ROTATIONAL POWER     peak  {self.stats['rot_power_peak']:8.2f} W   mean  {self.stats['rot_power_mean']:8.2f} W")
        print(f"TOTAL POWER          peak  {self.stats['total_power_peak']:8.2f} W   mean  {self.stats['total_power_mean']:8.2f} W")
        print(f"STRIDE TIME         {self.stats['stride_time_mean']:6.1f} ± {self.stats['stride_time_std']:5.1f} ms")
        print(f"CADENCE                    {self.stats['cadence']:6.1f} steps/min")
        print(f"GCT                {self.stats['gct_mean']:6.1f} ± {self.stats['gct_std']:5.1f} ms")
        print(f"SWING              {self.stats['swing_mean']:6.1f} ± {self.stats['swing_std']:5.1f} ms")
        print(f"STRIDE LENGTH           {self.stats['stride_length_mean']:4.2f} ± {self.stats['stride_length_std']:4.2f} m")
        print(f"PEAK LOAD              {self.stats['peak_load']:6.2f} N")
        print("="*50)
            
    def compute_power_metrics(self):
        """Compute power metrics from current buffer"""
        MIN_SAMPLES_FOR_PROCESSING = 16  # Minimum samples needed for filtering
        
        if len(self.data_buffer['timestamp']) < MIN_SAMPLES_FOR_PROCESSING:
            return
            
        try:
            # Convert deques to numpy arrays for processing
            accel = np.array([
                list(self.data_buffer['accel_x']),
                list(self.data_buffer['accel_y']),
                list(self.data_buffer['accel_z'])
            ]).T
            
            gyro = np.array([
                list(self.data_buffer['gyro_x']),
                list(self.data_buffer['gyro_y']),
                list(self.data_buffer['gyro_z'])
            ]).T
            
            timestamps = np.array(list(self.data_buffer['timestamp']))
            
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
            if len(self.power_metrics['total_power']) >= WINDOW_SIZE:
                for key in self.power_metrics:
                    self.power_metrics[key].clear()
            
            # Update power metrics buffers
            self.power_metrics['trans_power'].extend(trans_power)
            self.power_metrics['rot_power'].extend(rot_power)
            self.power_metrics['total_power'].extend(total_power)
            self.power_metrics['power_weight'].extend(power_weight)
            self.power_metrics['rolling_avg'].extend(rolling_avg)
            
            # Detect gait events
            self.detect_gait_events(filtered_accel, timestamps)
            
            # Update statistics periodically
            current_time = time.time()
            if current_time - self.last_stats_update > self.stats_update_interval:
                self.calculate_statistics()
                self.print_statistics()
                self.last_stats_update = current_time
            
        except Exception as e:
            print(f"Error computing metrics: {e}")
            
    def update_data_buffers(self, imu5_data):
        """Update data buffers with new IMU5 data"""
        if not imu5_data:
            return
            
        try:
            # Sort data by timestamp to ensure proper order
            imu5_data.sort(key=lambda x: x['timestamp'])
            
            # Clear buffers if they're full
            if len(self.data_buffer['timestamp']) >= WINDOW_SIZE:
                for key in self.data_buffer:
                    self.data_buffer[key].clear()
            
            # Update buffers with new data
            for data in imu5_data:
                for key in self.data_buffer:
                    self.data_buffer[key].append(data[key])
                    
        except Exception as e:
            print(f"Error updating buffers: {e}")
    
    def start_data_collection(self):
        """Start collecting data from UDP socket"""
        self.running = True
        print(f"Listening for all IMU data on port {UDP_PORT}")
        print("Extracting IMU5 data in real-time...")
        print("Real-time statistics will appear every 2 seconds...")
        
        while self.running:
            try:
                data, addr = self.sock.recvfrom(BUFFER_SIZE)
                data_str = data.decode('utf-8')
                
                # Parse IMU5 data from mixed stream
                imu5_data = self.parse_imu_data(data_str)
                
                if imu5_data:
                    # Update data buffers
                    self.update_data_buffers(imu5_data)
                    
                    # Compute power metrics
                    self.compute_power_metrics()
                    
                    # Update plots
                    self.update_plots()
                    
            except Exception as e:
                print(f"Error in data collection: {e}")
                
    def run(self):
        """Run the real-time power metrics system"""
        # Start data collection in a separate thread
        data_thread = threading.Thread(target=self.start_data_collection)
        data_thread.daemon = True
        data_thread.start()
        
        # Show plots
        plt.show()
        
        # Cleanup
        self.running = False
        self.sock.close()

if __name__ == "__main__":
    app = RealTimePowerMetrics()
    app.run() 