import socket
import time
from datetime import datetime
import threading
import queue
import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque, defaultdict
import tkinter as tk
from tkinter import ttk

# Configuration
UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 12345
BUFFER_SIZE = 4096
WINDOW_SIZE = 1000  # Number of samples to keep in memory for processing
PLOT_WINDOW = 5.0   # Seconds to show in plot
FS_DEFAULT = 100    # Default sampling frequency (Hz)

# IMU Mapping
IMU_MAPPING = {
    "IMU1": "Left_Below_Knee",
    "IMU2": "Right_Below_Knee", 
    "IMU3": "Left_Above_Knee",
    "IMU4": "Right_Above_Knee",
    "IMU7": "Pelvis"
}

# Step detection parameters
FILTER_CUTOFF = 5.0      # Hz - low-pass filter cutoff
HEIGHT_THRESHOLD = 1.5   # Minimum peak height for step detection
DISTANCE_THRESHOLD = 50  # Minimum distance between peaks (samples)

# Global variables
data_queue = queue.Queue(maxsize=10000)
running = True
debug_counter = 0  # For periodic debugging
data_received = False  # Track if any real UDP data has been received
start_time = time.time()  # Track when application started
imu_data_buffers = defaultdict(lambda: {
    'timestamp': deque(maxlen=WINDOW_SIZE),
    'accel_x': deque(maxlen=WINDOW_SIZE),
    'accel_y': deque(maxlen=WINDOW_SIZE),
    'accel_z': deque(maxlen=WINDOW_SIZE),
    'accel_magnitude': deque(maxlen=WINDOW_SIZE),
    'filtered_accel': deque(maxlen=WINDOW_SIZE),
    'step_count': 0,
    'last_step_time': 0,
    'cadence': 0
})

class RealTimeStepLengthCalculator:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('Real-Time Step Detection - Multiple IMU Sensors', fontsize=16)
        
        # Flatten axes for easier indexing
        self.axes = self.axes.flatten()
        
        # Store plot lines for each IMU
        self.plot_lines = {}
        self.step_markers = {}
        
        # Setup plots for each IMU
        imu_names = list(IMU_MAPPING.values())
        for i, imu_name in enumerate(imu_names):
            if i < len(self.axes):
                ax = self.axes[i]
                ax.set_title(f'{imu_name}')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Accel Magnitude (m/s²)')
                ax.set_ylim(0, 20)  # Initial range, will auto-adjust
                ax.grid(True, alpha=0.3)
                
                # Create plot lines
                line1, = ax.plot([], [], 'b-', alpha=0.6, label='Raw Accel Magnitude')
                line2, = ax.plot([], [], 'g-', linewidth=2, label='Filtered')
                line3, = ax.plot([], [], 'rx', markersize=8, label='Detected Steps')
                
                self.plot_lines[imu_name] = {
                    'raw': line1,
                    'filtered': line2,
                    'steps': line3
                }
                ax.legend(fontsize=8)
        
        # Hide unused subplots
        for i in range(len(imu_names), len(self.axes)):
            self.axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Status text
        self.status_text = self.fig.text(0.02, 0.02, '', fontsize=10, 
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    def butter_filter(self, data, cutoff=FILTER_CUTOFF, fs=FS_DEFAULT, order=4):
        """Apply Butterworth low-pass filter"""
        if len(data) < 10:  # Need minimum samples for filtering
            return np.array(data)
        
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff / nyquist
        
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.99
            
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def detect_steps_realtime(self, accel_signal, timestamps, imu_name):
        """Detect steps in real-time using peak detection"""
        if len(accel_signal) < 50:  # Need minimum samples
            return [], []
        
        # Apply filtering
        filtered_signal = self.butter_filter(accel_signal)
        
        # Detect peaks
        peaks, _ = find_peaks(filtered_signal, 
                            height=HEIGHT_THRESHOLD, 
                            distance=DISTANCE_THRESHOLD)
        
        if len(peaks) > 0:
            step_times = [timestamps[i] for i in peaks if i < len(timestamps)]
            return peaks, step_times
        
        return [], []

    def update_plot(self, frame):
        """Update plots with new data"""
        global debug_counter, data_received, start_time
        try:
            # Check if we should auto-start test data (30 second timeout)
            if not data_received and not hasattr(self, 'test_data_started'):
                elapsed_time = time.time() - start_time
                if elapsed_time > 30:  # 30 seconds timeout
                    print(f"\n⏰ No real UDP data received after {elapsed_time:.0f} seconds.")
                    print("🚀 Auto-starting test data generator...")
                    self.test_data_started = True
                    test_thread = threading.Thread(target=test_data_generator, daemon=True)
                    test_thread.start()
            
            # Process all available data from queue
            new_data_available = False
            while not data_queue.empty():
                try:
                    data_point = data_queue.get_nowait()
                    self.process_data_point(data_point)
                    new_data_available = True
                except queue.Empty:
                    break
            
            # Debug: Periodic statistics (every 100 frames = ~5 seconds)
            debug_counter += 1
            if debug_counter % 100 == 0:
                if data_received:
                    print(f"\n--- Debug Stats (Frame {debug_counter}) ---")
                    for imu_id, imu_name in IMU_MAPPING.items():
                        buffer = imu_data_buffers[imu_id]
                        if len(buffer['accel_magnitude']) > 0:
                            accel_data = np.array(buffer['accel_magnitude'])
                            print(f"{imu_name}: Samples={len(accel_data)}, "
                                  f"Range=[{np.min(accel_data):.2f}, {np.max(accel_data):.2f}], "
                                  f"Mean={np.mean(accel_data):.2f}, Std={np.std(accel_data):.2f}")
                    print("---")
                else:
                    elapsed = time.time() - start_time
                    remaining = max(0, 30 - elapsed)
                    if remaining > 0:
                        print(f"⏳ Still waiting for UDP data... {remaining:.0f}s until auto-test mode")
            
            if not new_data_available:
                return []
            
            # Update plots for each IMU
            current_time = time.time()
            plot_artists = []
            
            status_lines = []
            
            for imu_id, imu_name in IMU_MAPPING.items():
                buffer = imu_data_buffers[imu_id]
                
                if len(buffer['timestamp']) < 10:
                    continue
                
                # Convert to numpy arrays
                timestamps = np.array(buffer['timestamp'])
                accel_mag = np.array(buffer['accel_magnitude'])
                
                # Filter data for plot window
                time_window = current_time - PLOT_WINDOW
                mask = timestamps >= time_window
                
                if np.any(mask):
                    plot_times = timestamps[mask] - current_time  # Relative time
                    plot_accel = accel_mag[mask]
                    
                    # Apply filtering
                    if len(plot_accel) > 10:
                        filtered_accel = self.butter_filter(plot_accel)
                        
                        # Detect steps
                        peaks, step_times = self.detect_steps_realtime(
                            plot_accel, timestamps[mask], imu_name)
                        
                        # Update step count
                        if len(peaks) > 0:
                            # Count new steps
                            latest_step_time = timestamps[mask][peaks[-1]]
                            if latest_step_time > buffer['last_step_time']:
                                new_steps = len([p for p in peaks 
                                               if timestamps[mask][p] > buffer['last_step_time']])
                                buffer['step_count'] += new_steps
                                buffer['last_step_time'] = latest_step_time
                                
                                # Calculate cadence (steps per minute)
                                if buffer['step_count'] > 1:
                                    time_span = latest_step_time - timestamps[0]
                                    if time_span > 0:
                                        buffer['cadence'] = (buffer['step_count'] / time_span) * 60
                        
                        # Update plot lines
                        if imu_name in self.plot_lines:
                            self.plot_lines[imu_name]['raw'].set_data(plot_times, plot_accel)
                            self.plot_lines[imu_name]['filtered'].set_data(plot_times, filtered_accel)
                            
                            # Update step markers
                            if len(peaks) > 0:
                                step_plot_times = plot_times[peaks]
                                step_plot_values = plot_accel[peaks]
                                self.plot_lines[imu_name]['steps'].set_data(step_plot_times, step_plot_values)
                            else:
                                self.plot_lines[imu_name]['steps'].set_data([], [])
                            
                            plot_artists.extend([
                                self.plot_lines[imu_name]['raw'],
                                self.plot_lines[imu_name]['filtered'],
                                self.plot_lines[imu_name]['steps']
                            ])
                
                # Add to status
                status_lines.append(f"{imu_name}: Steps={buffer['step_count']}, "
                                  f"Cadence={buffer['cadence']:.1f} steps/min")
            
            # Update status text
            status_text = "Real-time Step Detection\n" + "\n".join(status_lines)
            self.status_text.set_text(status_text)
            plot_artists.append(self.status_text)
            
            # Adjust plot limits
            for imu_name in IMU_MAPPING.values():
                if imu_name in self.plot_lines:
                    ax = self.plot_lines[imu_name]['raw'].axes
                    ax.set_xlim(-PLOT_WINDOW, 0)
                    
                    # Get current data for proper y-axis scaling
                    raw_line = self.plot_lines[imu_name]['raw']
                    filtered_line = self.plot_lines[imu_name]['filtered']
                    
                    # Get y-data from both lines
                    raw_ydata = raw_line.get_ydata()
                    filtered_ydata = filtered_line.get_ydata()
                    
                    if len(raw_ydata) > 0 and len(filtered_ydata) > 0:
                        # Combine all y-data to get proper range
                        all_ydata = np.concatenate([raw_ydata, filtered_ydata])
                        if len(all_ydata) > 0 and not np.all(np.isnan(all_ydata)):
                            y_min = np.nanmin(all_ydata)
                            y_max = np.nanmax(all_ydata)
                            
                            # Add some padding (10% margin)
                            y_range = y_max - y_min
                            if y_range > 0:
                                margin = y_range * 0.1
                                ax.set_ylim(max(0, y_min - margin), y_max + margin)
                            else:
                                # If no variation, center around the value
                                center = y_min
                                ax.set_ylim(max(0, center - 2), center + 2)
                    
                    ax.relim()
                    ax.autoscale_view(scalex=False, scaley=False)  # Don't auto-scale Y since we set it manually
            
            return plot_artists
            
        except Exception as e:
            print(f"Plot update error: {e}")
            return []

    def process_data_point(self, data_point):
        """Process a single data point from UDP stream"""
        try:
            imu_id = data_point['IMU']
            if imu_id not in IMU_MAPPING:
                return
            
            timestamp = time.time()  # Use current time for real-time processing
            
            # Extract accelerometer data
            accel_x = float(data_point['AccelX'])
            accel_y = float(data_point['AccelY'])
            accel_z = float(data_point['AccelZ'])
            
            # Calculate acceleration magnitude
            accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
            
            # Debug: Print acceleration values for first few samples of each IMU
            buffer = imu_data_buffers[imu_id]
            if len(buffer['accel_magnitude']) < 5:  # Print first 5 samples per IMU
                print(f"IMU {imu_id} ({IMU_MAPPING[imu_id]}): AccelMag={accel_magnitude:.2f} (x={accel_x:.2f}, y={accel_y:.2f}, z={accel_z:.2f})")
            
            # Store in buffer
            buffer['timestamp'].append(timestamp)
            buffer['accel_x'].append(accel_x)
            buffer['accel_y'].append(accel_y)
            buffer['accel_z'].append(accel_z)
            buffer['accel_magnitude'].append(accel_magnitude)
            
        except Exception as e:
            print(f"Data processing error: {e}")

def udp_receiver():
    """UDP receiver thread"""
    global running
    
    try:
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)
        sock.bind((UDP_IP, UDP_PORT))
        
        print(f"UDP receiver started on {UDP_IP}:{UDP_PORT}")
        
        while running:
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE)
                raw_data = data.decode('utf-8').strip()
                
                # Parse multiple lines (multiple IMU readings)
                lines = raw_data.split('\n')
                for line in lines:
                    if not line.strip():
                        continue
                    
                    fields = line.strip().split(',')
                    if len(fields) >= 15:
                        try:
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
                                "QuatZ": fields[15] if len(fields) > 15 else fields[14]
                            }
                            
                            # Add to processing queue
                            if not data_queue.full():
                                data_queue.put_nowait(data_dict)
                                global data_received
                                data_received = True  # Mark that real UDP data has been received
                            
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

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\nStopping gracefully...")
    running = False

def test_data_generator():
    """Generate simulated walking data for testing when no real UDP data is available"""
    global running
    import random
    
    print("Starting test data generator...")
    
    while running:
        try:
            for imu_id in IMU_MAPPING.keys():
                # Simulate walking with varying acceleration
                t = time.time()
                base_accel = 9.8  # Gravity
                
                # Simulate walking pattern with periodic motion
                walking_freq = 1.5  # Steps per second
                walking_amplitude = 8.0  # Acceleration variation
                
                # Generate realistic walking acceleration pattern
                accel_x = base_accel * 0.1 + walking_amplitude * np.sin(2 * np.pi * walking_freq * t) + random.gauss(0, 0.5)
                accel_y = base_accel * 0.1 + walking_amplitude * np.cos(2 * np.pi * walking_freq * t * 1.1) + random.gauss(0, 0.5)
                accel_z = base_accel + walking_amplitude * 0.3 * np.sin(2 * np.pi * walking_freq * t * 0.8) + random.gauss(0, 0.3)
                
                # Add some sensor-specific variation
                if 'Below' in IMU_MAPPING[imu_id]:
                    accel_x *= 1.2  # More movement in lower leg
                    accel_y *= 1.2
                
                data_dict = {
                    "Count": str(int(t * 100)),
                    "Timestamp": str(int(t * 1000)),
                    "IMU": imu_id,
                    "AccelX": f"{accel_x:.3f}",
                    "AccelY": f"{accel_y:.3f}",
                    "AccelZ": f"{accel_z:.3f}",
                    "GyroX": "0.0", "GyroY": "0.0", "GyroZ": "0.0",
                    "MagX": "0.0", "MagY": "0.0", "MagZ": "0.0",
                    "QuatW": "1.0", "QuatX": "0.0", "QuatY": "0.0", "QuatZ": "0.0"
                }
                
                if not data_queue.full():
                    data_queue.put_nowait(data_dict)
            
            time.sleep(0.01)  # 100 Hz simulation
            
        except Exception as e:
            print(f"Test data generator error: {e}")
            break

def main():
    """Main function to start real-time step length calculation"""
    global running, data_received, start_time
    
    print("Real-Time Step Length Calculator")
    print("=================================")
    print("IMU Mapping:")
    for imu_id, imu_name in IMU_MAPPING.items():
        print(f"  {imu_id}: {imu_name}")
    print(f"\n🔍 Checking for UDP data on {UDP_IP}:{UDP_PORT}...")
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start UDP receiver thread first
    udp_thread = threading.Thread(target=udp_receiver, daemon=True)
    udp_thread.start()
    
    # Wait briefly to detect incoming UDP data
    print("⏳ Detecting incoming data...")
    detection_timeout = 3  # Wait 3 seconds to detect real data
    detection_start = time.time()
    
    while time.time() - detection_start < detection_timeout and not data_received:
        time.sleep(0.1)  # Check every 100ms
        if data_received:
            break
    
    use_test_data = False
    
    if data_received:
        print("✅ Real UDP data detected! Using live IMU data.")
        print("📊 Starting real-time analysis...")
    else:
        print("❌ No UDP data detected in first 3 seconds.")
        # Ask user if they want to use test data
        use_test_data = input("\nUse simulated test data instead? (y/n): ").lower().strip() == 'y'
        
        if use_test_data:
            print("🚀 Test data generator will start - you should see dynamic acceleration patterns!")
        else:
            print("\n📡 Continuing to wait for real IMU data...")
            print("📋 Make sure your IMU sensors are:")
            print("   - Connected to the same network")
            print("   - Sending data to this computer's IP on port 12345")
            print("   - Using the correct IMU mapping (IMU1, IMU2, IMU3, IMU4, IMU7)")
            print("\n⏳ Plots will remain empty until data arrives...")
            print("💡 Auto-test mode will activate after 30 seconds if no data comes")
    
    print("\nPress Ctrl+C to stop")
    
    # Start test data generator if requested
    if use_test_data:
        test_thread = threading.Thread(target=test_data_generator, daemon=True)
        test_thread.start()
    
    # Reset start time for the 30-second auto-fallback timer
    start_time = time.time()
    
    # Create calculator and start animation
    calculator = RealTimeStepLengthCalculator()
    
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
        print("\nShutting down...")
    finally:
        running = False
        
    print("Real-time step length calculation stopped.")

if __name__ == "__main__":
    main() 