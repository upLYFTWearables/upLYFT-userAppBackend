import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import numpy as np
import time
from datetime import datetime, timedelta
import signal
import sys
import os
import glob
import socket
import threading
from collections import deque, defaultdict

# Configuration
WINDOW_SIZE = 60  # Show last 60 seconds of data
UPDATE_INTERVAL = 1000  # Update plot every 1000ms (reduced from 100ms for better performance)
MAX_POINTS = 300  # Reduced from 1000 for better performance
ANNOTATION_LIMIT = 5  # Only show annotations for last 5 points per IMU
DATA_DECIMATION = 3  # Show every 3rd point to reduce congestion

# UDP Configuration
UDP_PORT = 12345
BUFFER_SIZE = 1024

class RealTimeFrequencyMonitor:
    def __init__(self):
        self.running = True
        self.data_buffers = {}  # Store data for each IMU
        self.active_imus = set()
        self.last_update_time = time.time()
        
        # Real-time data structures
        self.raw_data_buffer = defaultdict(lambda: deque(maxlen=1000))  # Raw timestamps per IMU
        self.frequency_data = defaultdict(lambda: deque())  # Calculated frequencies per IMU (no maxlen to keep all data)
        self.time_data = defaultdict(lambda: deque())  # Time stamps for frequency data (no maxlen to keep all data)
        self.start_time = time.time()
        
        # Setup plot
        plt.style.use('default')
        self.fig = None
        self.axs = None
        self.lines = {}
        self.scatter_plots = {}
        self.annotations = {}
        
        # UDP socket setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        self.sock.bind(('0.0.0.0', UDP_PORT))
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\nStopping frequency monitor...")
        self.running = False
        
    def parse_udp_data(self, data_str):
        """Parse UDP data stream and extract IMU information"""
        try:
            lines = data_str.strip().split('\n')
            parsed_data = []
            
            for line in lines:
                if not line.strip():
                    continue
                    
                parts = line.split(',')
                if len(parts) >= 13:  # Ensure we have enough data fields
                    try:
                        imu_id = parts[2].strip()
                        
                        # Convert time format to timestamp
                        time_str = parts[1].strip()
                        try:
                            # Try parsing as microseconds first
                            timestamp = int(time_str) / 1e6  # Convert to seconds
                        except ValueError:
                            # If that fails, parse as time string
                            try:
                                # Parse time string and convert to seconds
                                h, m, s = time_str.split(':')
                                s, ms = s.split('.')
                                total_seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
                                timestamp = total_seconds
                            except Exception:
                                timestamp = time.time()  # Fallback to current time
                        
                        data_point = {
                            'imu_id': imu_id,
                            'timestamp': timestamp,
                            'current_time': time.time()
                        }
                        parsed_data.append(data_point)
                        
                    except (ValueError, IndexError) as e:
                        continue
            
            return parsed_data
            
        except Exception as e:
            print(f"Error parsing UDP data: {e}")
            return []

    def process_realtime_data(self, data_points):
        """Process real-time data and calculate frequencies"""
        if not data_points:
            return
            
        current_time = time.time()
        
        # Group data by IMU
        for data_point in data_points:
            imu_id = data_point['imu_id']
            timestamp = data_point['current_time']
            
            # Add timestamp to raw data buffer
            self.raw_data_buffer[imu_id].append(timestamp)
            
            # Calculate frequency every second
            if len(self.raw_data_buffer[imu_id]) > 0:
                # Count packets in the last second
                one_second_ago = current_time - 1.0
                recent_packets = [t for t in self.raw_data_buffer[imu_id] if t >= one_second_ago]
                frequency = len(recent_packets)
                
                # Update frequency data (only add new data points every second)
                if (not self.time_data[imu_id] or 
                    current_time - self.time_data[imu_id][-1] >= 0.8):
                    
                    relative_time = current_time - self.start_time
                    self.frequency_data[imu_id].append(frequency)
                    self.time_data[imu_id].append(relative_time)
                    
                    # Keep all data from start to end (no sliding window)
                    # Comment out the sliding window logic to show complete history
                    # while (self.time_data[imu_id] and 
                    #        relative_time - self.time_data[imu_id][0] > WINDOW_SIZE):
                    #     self.frequency_data[imu_id].popleft()
                    #     self.time_data[imu_id].popleft()
    
    def start_udp_data_collection(self):
        """Start collecting data from UDP socket"""
        print(f"Listening for UDP data on port {UDP_PORT}")
        
        while self.running:
            try:
                data, addr = self.sock.recvfrom(BUFFER_SIZE)
                data_str = data.decode('utf-8')
                
                # Parse UDP data
                parsed_data = self.parse_udp_data(data_str)
                
                if parsed_data:
                    # Process real-time data
                    self.process_realtime_data(parsed_data)
                    
            except Exception as e:
                if self.running:  # Only print error if we're still running
                    print(f"Error in UDP data collection: {e}")

    def find_latest_csv(self):
        """Find the most recent CSV file created by UDP.py"""
        files = glob.glob("received_data_*.csv")
        if not files:
            return None
        return max(files, key=os.path.getctime)

    def get_imu_color(self, imu_name):
        """Get consistent color for each IMU"""
        # Extract IMU number from name (e.g., "IMU1" -> 1)
        imu_num = int(imu_name.replace("IMU", ""))
        colors = cm.get_cmap('tab10', 10)
        return colors((imu_num - 1) % 10)

    def setup_plot(self, unique_imus):
        """Initialize the plot exactly like frqcount.py"""
        n_imus = len(unique_imus)
        
        # Clear the existing figure and set new size
        self.fig.clear()
        self.fig.set_size_inches(16, 3 * n_imus)
        
        # Create subplots
        self.axs = self.fig.subplots(n_imus, 1, sharex=True)
        
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
            
            # Style the subplot exactly like frqcount.py
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("Freq (Hz)", fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(80, 250)  # Adjusted range for better visibility
            
            # Set title
            ax.set_title(f"{imu} – Frequency Over Time", 
                        color=color, fontsize=12, fontweight='bold')
            
        # Set common x-axis label
        self.axs[-1].set_xlabel("Time (seconds)", fontsize=10)
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

    def update_plot(self, frame):
        """Update function for animation"""
        try:
            current_time = time.time()
            
            # Only update if enough time has passed or we have new data
            if (current_time - self.last_update_time) > 2.0 or self.frequency_data:
                # Process real-time data
                if self.frequency_data:
                    # Get unique IMUs (sorted for consistent ordering)
                    unique_imus = sorted(self.frequency_data.keys())
                    
                    # Create/recreate plot if needed
                    if self.fig is None or set(unique_imus) != self.active_imus:
                        self.setup_plot(unique_imus)
                        self.active_imus = set(unique_imus)
                        self.data_buffers = {}
                    
                    # Update each subplot
                    for imu in unique_imus:
                        # Get data for this IMU from real-time buffers
                        if imu in self.frequency_data and self.frequency_data[imu]:
                            timestamps = list(self.time_data[imu])
                            frequencies = list(self.frequency_data[imu])
                            
                            # Initialize buffer if needed
                            if imu not in self.data_buffers:
                                self.data_buffers[imu] = {'t': [], 'f': []}
                            
                            # Update data buffers with real-time data
                            self.data_buffers[imu]['t'] = timestamps
                            self.data_buffers[imu]['f'] = frequencies
                            
                            # Decimate data for better performance
                            if len(self.data_buffers[imu]['t']) > MAX_POINTS:
                                t_data, f_data = self.decimate_data(
                                    self.data_buffers[imu]['t'], 
                                    self.data_buffers[imu]['f']
                                )
                            else:
                                t_data = self.data_buffers[imu]['t']
                                f_data = self.data_buffers[imu]['f']
                            
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
                                
                                # Update title with statistics
                                avg_freq = np.mean(f_data)
                                current_freq = f_data[-1] if f_data else 0
                                ax_index = list(unique_imus).index(imu)
                                self.axs[ax_index].set_title(
                                    f"{imu} – Frequency Over Time (Current: {current_freq:.0f} Hz, Avg: {avg_freq:.1f} Hz)",
                                    color=self.get_imu_color(imu),
                                    fontsize=11, fontweight='bold'
                                )
                                
                                # Update x-axis limits to show all data from start to end
                                latest_time = max(t_data)
                                self.axs[ax_index].set_xlim(
                                    0, 
                                    latest_time + 5
                                )
                
                # Update tracking variables
                self.last_update_time = current_time
                
        except Exception as e:
            print(f"Error updating plot: {e}")
        
        # Return all artists that need to be redrawn
        artists = []
        for imu in self.lines:
            artists.append(self.lines[imu])
            artists.append(self.scatter_plots[imu])
            artists.extend(self.annotations[imu])
        return artists

    def run(self):
        """Main function to run the monitor"""
        try:
            print("Enhanced Real-time Frequency Monitor started")
            print("Features:")
            print("- Direct UDP data processing")
            print("- Reduced update rate for better performance")
            print("- Cleaner annotations (last 5 points only)")
            print("- Consistent IMU colors")
            print("- Complete data history from start to end")
            print(f"Listening for UDP data on port {UDP_PORT}")
            print("Press Ctrl+C to stop")
            
            # Start UDP data collection in a separate thread
            data_thread = threading.Thread(target=self.start_udp_data_collection)
            data_thread.daemon = True
            data_thread.start()
            
            # Create a placeholder figure to avoid multiple figures
            self.fig, self.axs = plt.subplots(1, 1, figsize=(16, 3))
            self.axs.text(0.5, 0.5, 'Waiting for UDP data...', ha='center', va='center', transform=self.axs.transAxes)
            self.axs.set_title('Real-time Frequency Monitor - Waiting for Data')
            
            # Start animation
            ani = animation.FuncAnimation(
                self.fig, self.update_plot,
                interval=UPDATE_INTERVAL,
                blit=False,  # Set to False to avoid issues with changing plot structure
                cache_frame_data=False
            )
            
            plt.show()
            
        except Exception as e:
            print(f"Error running monitor: {e}")
        finally:
            print("\nMonitor stopped.")
            self.running = False
            self.sock.close()

if __name__ == "__main__":
    monitor = RealTimeFrequencyMonitor()
    monitor.run() 