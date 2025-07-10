import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import os

class StepLengthProcessor:
    def __init__(self):
        self.window_size = 5  # 5 seconds window
        self.data_buffers = {
            'Left_Below_Knee': {'raw': [], 'filtered': [], 'steps': [], 'time': []},
            'Right_Below_Knee': {'raw': [], 'filtered': [], 'steps': [], 'time': []},
            'Left_Above_Knee': {'raw': [], 'filtered': [], 'steps': [], 'time': []},
            'Right_Above_Knee': {'raw': [], 'filtered': [], 'steps': [], 'time': []},
            'Pelvis': {'raw': [], 'filtered': [], 'steps': [], 'time': []}
        }
        self.last_read_size = 0
        
    def find_latest_csv(self):
        """Find the most recent CSV file"""
        files = glob.glob("received_data_*.csv")
        if not files:
            return None
        return max(files, key=os.path.getctime)
        
    def process_data(self):
        """Process the latest data and update plots"""
        csv_file = self.find_latest_csv()
        if not csv_file:
            return None
            
        try:
            file_size = os.path.getsize(csv_file)
            if file_size > self.last_read_size:
                df = pd.read_csv(csv_file)
                if not df.empty:
                    # Process data for each sensor
                    for sensor in self.data_buffers.keys():
                        sensor_data = df[df['IMU'] == sensor]
                        if not sensor_data.empty:
                            # Calculate time in seconds from the start
                            times = pd.to_datetime(sensor_data['Timestamp'])
                            start_time = times.min()
                            times = [(t - start_time).total_seconds() for t in times]
                            
                            # Calculate acceleration magnitude
                            accel_mag = np.sqrt(
                                sensor_data['AccX']**2 + 
                                sensor_data['AccY']**2 + 
                                sensor_data['AccZ']**2
                            )
                            
                            # Apply simple moving average filter
                            window = 5
                            filtered = pd.Series(accel_mag).rolling(window=window, center=True).mean()
                            
                            # Detect steps (simple threshold-based detection)
                            threshold = filtered.mean() + filtered.std()
                            steps = times[filtered > threshold]
                            
                            # Update buffers
                            self.data_buffers[sensor]['time'] = times[-self.window_size*100:]
                            self.data_buffers[sensor]['raw'] = accel_mag[-self.window_size*100:]
                            self.data_buffers[sensor]['filtered'] = filtered[-self.window_size*100:]
                            self.data_buffers[sensor]['steps'] = steps
                
                self.last_read_size = file_size
                return self.data_buffers
                
        except Exception as e:
            print(f"Error processing data: {e}")
            return None
            
    def get_metrics(self):
        """Calculate step detection metrics"""
        metrics = {}
        for sensor, data in self.data_buffers.items():
            if data['steps']:
                num_steps = len(data['steps'])
                if len(data['steps']) > 1:
                    # Calculate cadence (steps per minute)
                    time_diff = data['steps'][-1] - data['steps'][0]
                    if time_diff > 0:
                        cadence = (num_steps / time_diff) * 60
                    else:
                        cadence = 0
                else:
                    cadence = 0
                    
                metrics[sensor] = {
                    'steps': num_steps,
                    'cadence': cadence
                }
            else:
                metrics[sensor] = {
                    'steps': 0,
                    'cadence': 0
                }
        return metrics 