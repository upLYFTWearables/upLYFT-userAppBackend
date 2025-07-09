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

# Configuration
WINDOW_SIZE = 60  # Show last 60 seconds of data
UPDATE_INTERVAL = 1000  # Update plot every 1000ms (reduced from 100ms for better performance)
MAX_POINTS = 300  # Reduced from 1000 for better performance
ANNOTATION_LIMIT = 5  # Only show annotations for last 5 points per IMU
DATA_DECIMATION = 3  # Show every 3rd point to reduce congestion

class RealTimeFrequencyMonitor:
    def __init__(self):
        self.running = True
        self.last_read_size = 0
        self.data_buffers = {}  # Store data for each IMU
        self.active_imus = set()
        self.last_update_time = time.time()
        
        # Setup plot
        plt.style.use('default')
        self.fig = None
        self.axs = None
        self.lines = {}
        self.scatter_plots = {}
        self.annotations = {}
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\nStopping frequency monitor...")
        self.running = False

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
        
        # Create subplots
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
        csv_file = self.find_latest_csv()
        if not csv_file:
            return []
            
        try:
            # Get file size
            file_size = os.path.getsize(csv_file)
            current_time = time.time()
            
            # Only read if file has grown or enough time has passed
            if file_size > self.last_read_size or (current_time - self.last_update_time) > 2.0:
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Process data exactly like frqcount.py
                if not df.empty:
                    # Convert timestamp to datetime format
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S.%f')
                    
                    # Round down to the nearest second
                    df['Timestamp_sec'] = df['Timestamp'].dt.floor('S')
                    
                    # Calculate frequencies per IMU per second
                    freq_data = df.groupby(['IMU', 'Timestamp_sec']).size().reset_index(name='Frequency')
                    
                    # Get unique IMUs (sorted for consistent ordering)
                    unique_imus = sorted(freq_data['IMU'].unique())
                    
                    # Create/recreate plot if needed
                    if self.fig is None or set(unique_imus) != self.active_imus:
                        if self.fig is not None:
                            plt.close(self.fig)
                        self.setup_plot(unique_imus)
                        self.active_imus = set(unique_imus)
                        self.data_buffers = {}
                    
                    # Update each subplot
                    for imu in unique_imus:
                        # Get data for this IMU
                        imu_data = freq_data[freq_data['IMU'] == imu]
                        
                        if not imu_data.empty:
                            # Calculate relative timestamps (from start of data)
                            min_time = freq_data['Timestamp_sec'].min()
                            timestamps = (imu_data['Timestamp_sec'] - min_time).dt.total_seconds()
                            frequencies = imu_data['Frequency'].values
                            
                            # Initialize buffer if needed
                            if imu not in self.data_buffers:
                                self.data_buffers[imu] = {'t': [], 'f': []}
                            
                            # Replace data instead of appending (for real-time view)
                            self.data_buffers[imu]['t'] = timestamps.tolist()
                            self.data_buffers[imu]['f'] = frequencies.tolist()
                            
                            # Keep only recent data (last WINDOW_SIZE seconds)
                            if self.data_buffers[imu]['t']:
                                latest_time = max(self.data_buffers[imu]['t'])
                                cutoff_time = latest_time - WINDOW_SIZE
                                
                                # Filter data
                                filtered_data = [(t, f) for t, f in zip(self.data_buffers[imu]['t'], 
                                                                       self.data_buffers[imu]['f']) 
                                               if t >= cutoff_time]
                                
                                if filtered_data:
                                    self.data_buffers[imu]['t'], self.data_buffers[imu]['f'] = zip(*filtered_data)
                                    self.data_buffers[imu]['t'] = list(self.data_buffers[imu]['t'])
                                    self.data_buffers[imu]['f'] = list(self.data_buffers[imu]['f'])
                                else:
                                    self.data_buffers[imu]['t'] = []
                                    self.data_buffers[imu]['f'] = []
                            
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
                                
                                # Update x-axis limits
                                latest_time = max(t_data)
                                self.axs[ax_index].set_xlim(
                                    max(0, latest_time - WINDOW_SIZE), 
                                    latest_time + 5
                                )
                
                # Update tracking variables
                self.last_read_size = file_size
                self.last_update_time = current_time
                
        except Exception as e:
            if "No columns to parse from file" not in str(e):
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
            print("- Reduced update rate for better performance")
            print("- Cleaner annotations (last 5 points only)")
            print("- Consistent IMU colors")
            print("- 60-second sliding window")
            print("Waiting for UDP.py to create data file...")
            print("Press Ctrl+C to stop")
            
            # Start animation
            ani = animation.FuncAnimation(
                plt.gcf(), self.update_plot,
                interval=UPDATE_INTERVAL,
                blit=True,
                cache_frame_data=False
            )
            
            plt.show()
            
        except Exception as e:
            print(f"Error running monitor: {e}")
        finally:
            print("\nMonitor stopped.")

if __name__ == "__main__":
    monitor = RealTimeFrequencyMonitor()
    monitor.run() 