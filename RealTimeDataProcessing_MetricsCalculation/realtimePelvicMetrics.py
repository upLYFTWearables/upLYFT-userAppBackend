import numpy as np
import math
from collections import deque
from scipy.signal import butter, filtfilt
import time
import csv
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import queue
import threading
import signal

# Add project root to Python path so we can import UDP from workspace root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Configuration
WINDOW_SIZE = 10000  # Increased buffer size to hold more data for complete graph
PLOT_WINDOW = 60.0  # Seconds to show in plot (1 minute sliding window)
TARGET_IMU = 'IMU7'  # Only process IMU7 data

# Global variables
data_queue = queue.Queue(maxsize=10000)
running = True
data_received = False
start_time = time.time()

###############################################################################
# 1) Quaternion & Matrix Helpers
###############################################################################
def quaternion_to_matrix(q):
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
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
    """Compute an approximate average rotation matrix from a list of quaternions."""
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
    """Extract Euler angles from a rotation matrix R using a Z-Y-X sequence."""
    beta_y = -math.asin(R[2, 0])       # pitch
    cos_beta = math.cos(beta_y)
    if abs(cos_beta) < 1e-6:
        alpha_z = math.atan2(-R[0, 1], R[1, 1])  # yaw
        gamma_x = 0.0                            # roll
    else:
        alpha_z = math.atan2(R[1, 0], R[0, 0])   # yaw
        gamma_x = math.atan2(R[2, 1], R[2, 2])   # roll

    return alpha_z, beta_y, gamma_x

###############################################################################
# 2) Real-time Pelvic Metrics Processor
###############################################################################
class RealtimePelvicMetrics:
    def __init__(self, 
                 buffer_size=100,
                 calibration_samples=500,
                 lowpass_cutoff=0.3):
        """Initialize the real-time pelvic metrics processor."""
        self.buffer_size = buffer_size
        self.calibration_samples = calibration_samples
        self.lowpass_cutoff = lowpass_cutoff
        
        # Initialize buffers
        self.calibration_quats = []
        self.angle_buffer = {
            'tilt': deque(maxlen=buffer_size),
            'obliquity': deque(maxlen=buffer_size),
            'rotation': deque(maxlen=buffer_size)
        }
        
        # Calibration matrix (identity until calibrated)
        self.R0 = np.eye(3)
        self.is_calibrated = False
        
        # Create low-pass filter
        self.b, self.a = butter(2, lowpass_cutoff, btype='low')
        
    def calibrate(self, quaternion):
        """Add calibration sample and compute calibration if enough samples."""
        if not self.is_calibrated:
            self.calibration_quats.append(quaternion)
            if len(self.calibration_quats) >= self.calibration_samples:
                self.R0 = average_rotation_matrix(self.calibration_quats)
                self.is_calibrated = True
                print("✅ Calibration completed!")
                return True
        return False

    def process_quaternion(self, quaternion):
        """Process a new quaternion measurement and return the current angles."""
        # During calibration phase
        if not self.is_calibrated:
            if self.calibrate(quaternion):
                return None  # Calibration just completed
            return None  # Still calibrating
            
        # Convert current quaternion to rotation matrix
        R_current = quaternion_to_matrix(quaternion)
        
        # Apply calibration
        R_cal = self.R0.T @ R_current
        
        # Get Euler angles
        alpha_z, beta_y, gamma_x = matrix_to_euler_ZYX(R_cal)
        
        # Convert to degrees
        tilt = math.degrees(-beta_y)      # pitch
        obliquity = math.degrees(gamma_x)  # roll
        rotation = math.degrees(alpha_z)   # yaw
        
        # Add to buffers
        self.angle_buffer['tilt'].append(tilt)
        self.angle_buffer['obliquity'].append(obliquity)
        self.angle_buffer['rotation'].append(rotation)
        
        # Apply filtering if we have enough samples
        if len(self.angle_buffer['tilt']) >= self.buffer_size:
            tilt_filtered = filtfilt(self.b, self.a, list(self.angle_buffer['tilt']))[-1]
            obl_filtered = filtfilt(self.b, self.a, list(self.angle_buffer['obliquity']))[-1]
            rot_filtered = filtfilt(self.b, self.a, list(self.angle_buffer['rotation']))[-1]
        else:
            tilt_filtered = tilt
            obl_filtered = obliquity
            rot_filtered = rotation
            
        return {
            'tilt': tilt_filtered,
            'obliquity': obl_filtered,
            'rotation': rot_filtered
        }

###############################################################################
# 3) Data Logger
###############################################################################
class DataLogger:
    def __init__(self, output_dir="logs"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create a new log file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(output_dir, f"pelvic_metrics_{timestamp}.csv")
        
        # Create and write header
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ',
                           'QuatW', 'QuatX', 'QuatY', 'QuatZ', 'Tilt', 'Obliquity', 'Rotation'])
    
    def log_data(self, timestamp, raw_data, processed_angles):
        """Log both raw IMU data and processed angles."""
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp] + raw_data + 
                          [processed_angles['tilt'], 
                           processed_angles['obliquity'],
                           processed_angles['rotation']])

###############################################################################
# 4) Real-time Pelvic Metrics Calculator with Visualization
###############################################################################
class RealTimePelvicMetricsCalculator:
    def __init__(self):
        # Initialize the metrics processor
        self.metrics_processor = RealtimePelvicMetrics()
        self.logger = DataLogger()
        
        # Data buffers for visualization - NO maxlen to keep ALL data
        self.data_buffer = {
            'timestamp': deque(),  # Keep ALL timestamps
            'tilt': deque(),       # Keep ALL tilt data
            'obliquity': deque(),  # Keep ALL obliquity data
            'rotation': deque()    # Keep ALL rotation data
        }
        
        self.start_time = None
        self.sample_count = 0
        
        # Create the plot
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('Real-Time Pelvic Metrics - IMU7', fontsize=16)
        
        # Setup subplots
        self.titles = ['Pelvic Tilt', 'Pelvic Obliquity', 'Pelvic Rotation']
        self.colors = ['blue', 'green', 'red']
        self.plot_lines = []
        
        for i, (ax, title, color) in enumerate(zip(self.axes, self.titles, self.colors)):
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel('Angle (degrees)', fontsize=12)
            ax.set_ylim(-60, 60)  # Initial range, will auto-scale
            ax.set_xlim(0, 60)    # Initial window, will expand as data comes in
            ax.grid(True, alpha=0.3)
            
            line, = ax.plot([], [], color=color, linewidth=2, label=title)
            self.plot_lines.append(line)
        
        self.axes[-1].set_xlabel('Time (seconds)', fontsize=12)
        plt.tight_layout()
        
        # Status text
        self.status_text = self.fig.text(0.02, 0.02, 'Waiting for IMU7 data...', fontsize=10,
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        print("📊 Graph window created and ready!")

    def update_plot(self, frame):
        """Update plots with new data - called by matplotlib animation"""
        global data_received, start_time
        
        try:
            # Process all available data from queue
            new_data_available = False
            while not data_queue.empty():
                try:
                    data_point = data_queue.get_nowait()
                    if self.process_data_point(data_point):
                        new_data_available = True
                except queue.Empty:
                    break
            
            # Check if we should show auto-start message
            if not data_received and not hasattr(self, 'timeout_shown'):
                elapsed_time = time.time() - start_time
                if elapsed_time > 10:  # Show message after 10 seconds
                    print(f"\n⏰ No IMU7 data received after {elapsed_time:.0f} seconds.")
                    print("📡 Make sure UDP.py is running and IMU7 is sending data.")
                    self.timeout_shown = True
            
            if not new_data_available:
                return self.plot_lines
            
            # Update plots
            current_time = time.time()
            if self.start_time is None:
                return self.plot_lines
                
            # Convert to numpy arrays for plotting
            if len(self.data_buffer['timestamp']) < 2:
                return self.plot_lines
                
            timestamps = np.array(self.data_buffer['timestamp'])
            
            # Convert to relative time from start (show ALL data from beginning)
            relative_timestamps = timestamps - self.start_time
            
            # Show ALL data from the beginning - NO filtering, NO sliding window
            plot_times = relative_timestamps  # Use all timestamps
            
            # Update each plot line with ALL historical data
            data_keys = ['tilt', 'obliquity', 'rotation']
            for i, (line, key) in enumerate(zip(self.plot_lines, data_keys)):
                if len(self.data_buffer[key]) > 0:
                    plot_data = np.array(self.data_buffer[key])  # ALL data points
                    line.set_data(plot_times, plot_data)
                    
                    # Always show complete time range from 0 to current time
                    if len(plot_times) > 0:
                        latest_time = plot_times[-1]
                        # Expanding window: always show from 0 to current time + small buffer
                        self.axes[i].set_xlim(0, latest_time + 5)
                        
                        # Auto-scale y-axis to fit ALL data points
                        if len(plot_data) > 0:
                            data_min = np.min(plot_data)
                            data_max = np.max(plot_data)
                            margin = max(5, (data_max - data_min) * 0.1)  # 10% margin or 5° minimum
                            self.axes[i].set_ylim(data_min - margin, data_max + margin)
                    else:
                        # Initial state
                        self.axes[i].set_xlim(0, 60)
                        self.axes[i].set_ylim(-60, 60)
            
            # Update status text
            if self.metrics_processor.is_calibrated:
                latest_angles = {
                    'tilt': list(self.data_buffer['tilt'])[-1] if self.data_buffer['tilt'] else 0,
                    'obliquity': list(self.data_buffer['obliquity'])[-1] if self.data_buffer['obliquity'] else 0,
                    'rotation': list(self.data_buffer['rotation'])[-1] if self.data_buffer['rotation'] else 0
                }
                
                # Calculate total duration
                total_duration = (time.time() - self.start_time) if self.start_time else 0
                total_points = len(self.data_buffer['tilt'])
                
                status_text = (f"Real-time Pelvic Metrics (IMU7)\n"
                             f"Total Duration: {total_duration:.1f}s\n"
                             f"Data Points: {total_points}\n"
                             f"Current Angles:\n"
                             f"Tilt: {latest_angles['tilt']:.2f}°\n"
                             f"Obliquity: {latest_angles['obliquity']:.2f}°\n"
                             f"Rotation: {latest_angles['rotation']:.2f}°")
            else:
                calib_progress = len(self.metrics_processor.calibration_quats)
                calib_total = self.metrics_processor.calibration_samples
                status_text = (f"Calibrating IMU7...\n"
                             f"Progress: {calib_progress}/{calib_total}\n"
                             f"Stand still for calibration")
            
            self.status_text.set_text(status_text)
            
            return self.plot_lines
            
        except Exception as e:
            print(f"Plot update error: {e}")
            return self.plot_lines

    def process_data_point(self, data_point):
        """Process a single data point from UDP stream"""
        global data_received
        
        try:
            # Only process IMU7 data
            if data_point.get('IMU') != TARGET_IMU:
                return False
                
            # Mark that we received IMU7 data
            data_received = True
            
            # Initialize start time on first sample
            if self.start_time is None:
                self.start_time = time.time()
                print(f"✅ First IMU7 sample received! Starting processing...")
                
            current_time = time.time()
            relative_time = current_time - self.start_time
            self.sample_count += 1
            
            # Extract quaternion data
            try:
                quat = [
                    float(data_point['QuatW']),
                    float(data_point['QuatX']),
                    float(data_point['QuatY']),
                    float(data_point['QuatZ'])
                ]
            except (ValueError, KeyError) as e:
                print(f"Error parsing quaternion data: {e}")
                return False
            
            # Process quaternion
            angles = self.metrics_processor.process_quaternion(quat)
            
            if angles is None:
                # Still calibrating
                if self.sample_count % 100 == 0:  # Print every 100 samples
                    calib_progress = len(self.metrics_processor.calibration_quats)
                    calib_total = self.metrics_processor.calibration_samples
                    print(f"📊 Calibrating... {calib_progress}/{calib_total}")
                return False
            
            # Store data for visualization
            self.data_buffer['timestamp'].append(current_time)
            self.data_buffer['tilt'].append(angles['tilt'])
            self.data_buffer['obliquity'].append(angles['obliquity'])
            self.data_buffer['rotation'].append(angles['rotation'])
            
            # Log data
            try:
                accel = [float(data_point['AccelX']), float(data_point['AccelY']), float(data_point['AccelZ'])]
                gyro = [float(data_point['GyroX']), float(data_point['GyroY']), float(data_point['GyroZ'])]
                raw_data = accel + gyro + quat
                self.logger.log_data(relative_time, raw_data, angles)
            except Exception as e:
                print(f"Logging error: {e}")
            
            # Console output (less frequent)
            if self.sample_count % 20 == 0:
                print(f"\rTime: {relative_time:6.2f}s | Tilt: {angles['tilt']:6.2f}° | "
                      f"Obliquity: {angles['obliquity']:6.2f}° | Rotation: {angles['rotation']:6.2f}°", end='')
            
            return True
            
        except Exception as e:
            print(f"Data processing error: {e}")
            return False

###############################################################################
# 5) UDP Data Receiver - Direct UDP Socket Approach
###############################################################################
import socket

def udp_data_receiver():
    """Receive UDP data directly from socket"""
    global running
    
    try:
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)
        
        # Listen on all interfaces, port 12345
        UDP_IP = "0.0.0.0"
        UDP_PORT = 12345
        sock.bind((UDP_IP, UDP_PORT))
        
        print(f"📡 UDP receiver started on {UDP_IP}:{UDP_PORT}")
        
        while running:
            try:
                data, addr = sock.recvfrom(4096)
                raw_data = data.decode('utf-8').strip()
                
                # Parse multiple lines (multiple IMU readings)
                lines = raw_data.split('\n')
                for line in lines:
                    if not line.strip():
                        continue
                    
                    # Skip header line
                    if line.startswith('Count,Timestamp'):
                        continue
                    
                    fields = line.strip().split(',')
                    if len(fields) >= 15:
                        try:
                            # Parse the CSV format: Count,Timestamp,IMU,AccelX,AccelY,AccelZ,GyroX,GyroY,GyroZ,MagX,MagY,MagZ,QuatW,QuatX,QuatY,QuatZ
                            data_dict = {
                                "Count": fields[0],
                                "Timestamp": fields[1],
                                "IMU": fields[2],
                                "AccelX": fields[3],
                                "AccelY": fields[4],
                                "AccelZ": fields[5],
                                "GyroX": fields[6],
                                "GyroY": fields[7],
                                "GyroZ": fields[8],
                                "MagX": fields[9],
                                "MagY": fields[10],
                                "MagZ": fields[11],
                                "QuatW": fields[12],
                                "QuatX": fields[13],
                                "QuatY": fields[14],
                                "QuatZ": fields[15] if len(fields) > 15 else "0.0"
                            }
                            
                            # Only process IMU7 data
                            if data_dict["IMU"] == TARGET_IMU:
                                # Add to processing queue
                                if not data_queue.full():
                                    data_queue.put_nowait(data_dict)
                                    global data_received
                                    data_received = True
                            
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing line: {line[:50]}... Error: {e}")
                        
            except socket.timeout:
                continue
            except Exception as e:
                print(f"UDP receiver error: {e}")
                
    except Exception as e:
        print(f"UDP socket error: {e}")
    finally:
        try:
            sock.close()
        except:
            pass

def main():
    """Main function to start real-time pelvic metrics calculation"""
    global running, data_received, start_time
    
    print("=== Real-Time Pelvic Metrics Analyzer ===")
    print("🎯 Target: IMU7 (Pelvis)")
    print("📊 Metrics: Tilt, Obliquity, Rotation")
    print("📡 Data source: Direct UDP socket (port 12345)")
    print("📈 Display: Complete historical data (from start to stop)")
    print("⏱️  Window: Expanding view - keeps ALL data points")
    print()
    
    # Start UDP data receiver thread
    udp_thread = threading.Thread(target=udp_data_receiver, daemon=True)
    udp_thread.start()
    
    # Reset start time
    start_time = time.time()
    
    # Create calculator and start animation
    print("🚀 Initializing visualization...")
    calculator = RealTimePelvicMetricsCalculator()
    
    print("📈 Starting real-time analysis...")
    print("⏳ Waiting for IMU7 data...")
    print("💡 Make sure your IMU system is sending data to port 12345!")
    print("\nPress Ctrl+C to stop")
    
    # Start matplotlib animation
    ani = animation.FuncAnimation(
        calculator.fig,
        calculator.update_plot,
        interval=50,  # Update every 50ms (20 FPS)
        blit=False,
        cache_frame_data=False
    )
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        running = False
        
    print("✅ Real-time pelvic metrics analysis stopped.")

if __name__ == "__main__":
    main() 