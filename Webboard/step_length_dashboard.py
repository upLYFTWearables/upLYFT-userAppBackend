import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import glob
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt, find_peaks

# Configuration
WINDOW_SIZE = 1000
PLOT_WINDOW = 5.0
FS_DEFAULT = 100
UPDATE_INTERVAL = 1  # Changed from 50ms to 10s

# IMU Mapping with anatomical names
IMU_MAPPING = {
    "IMU1": "Left Below Knee",
    "IMU2": "Right Below Knee", 
    "IMU3": "Left Above Knee",
    "IMU4": "Right Above Knee",
    "IMU7": "Pelvis"
}

# Step detection parameters
FILTER_CUTOFF = 5.0
HEIGHT_THRESHOLD = 1.5
DISTANCE_THRESHOLD = 50

# Page configuration
st.set_page_config(
    page_title="Step Length Analysis Dashboard",
    page_icon="👣",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for medical-style dashboard
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
        padding: 1rem;
    }
    
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Override Streamlit default text colors */
    .stMarkdown, .stText, p, div, span, h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    /* Streamlit specific overrides */
    .css-1d391kg, .css-1v0mbdj, .css-1avcm0n {
        color: #000000 !important;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
        color: #000000;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #007bff !important;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #333333 !important;
        margin-bottom: 0.5rem;
    }
    
    .header-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .header-section h1, .header-section h3 {
        color: white !important;
    }
    
    .status-active {
        background-color: #d4edda;
        color: #155724 !important;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .imu-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem;
        text-align: center;
        color: #000000;
    }
    
    .imu-value {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .imu-label {
        font-size: 0.8rem;
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

class StepLengthDashboard:
    def __init__(self):
        self.colors = {
            'IMU1': '#1f77b4',  # Blue
            'IMU2': '#ff7f0e',  # Orange
            'IMU3': '#2ca02c',  # Green
            'IMU4': '#d62728',  # Red
            'IMU7': '#9467bd',  # Purple
        }
        self.last_read_size = 0
        self.last_update_time = time.time()
        self.step_data = {imu_id: {'steps': 0, 'cadence': 0, 'last_step_time': 0} 
                         for imu_id in IMU_MAPPING}
        
    def butter_filter(self, data, cutoff=FILTER_CUTOFF, fs=FS_DEFAULT, order=4):
        if len(data) < 10:
            return np.array(data)
        nyquist = 0.5 * fs
        normalized_cutoff = min(cutoff / nyquist, 0.99)
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def detect_steps(self, accel_signal, timestamps):
        if len(accel_signal) < 50:
            return [], []
        filtered_signal = self.butter_filter(accel_signal)
        peaks, _ = find_peaks(filtered_signal, height=HEIGHT_THRESHOLD, distance=DISTANCE_THRESHOLD)
        if len(peaks) > 0:
            step_times = [timestamps[i] for i in peaks if i < len(timestamps)]
            return peaks, step_times
        return [], []

    def find_latest_csv(self):
        """Find the most recent CSV file created by UDP.py"""
        files = glob.glob("received_data_*.csv")
        if not files:
            return None
        return max(files, key=os.path.getctime)

    def process_data(self):
        """Process IMU data and create visualization"""
        csv_file = self.find_latest_csv()
        if not csv_file:
            return None, None
        
        try:
            # Get file size
            file_size = os.path.getsize(csv_file)
            current_time = time.time()
            
            # Only read if file has grown or enough time has passed (10 seconds)
            if file_size > self.last_read_size or (current_time - self.last_update_time) > UPDATE_INTERVAL:
                # Read the CSV file
                df = pd.read_csv(csv_file)
                if df.empty:
                    return None, None
                
                # Convert timestamp to datetime and get reference time
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S.%f')
                start_time = df['Timestamp'].min()
                
                # Process data for each IMU
                fig = make_subplots(
                    rows=len(IMU_MAPPING), cols=1,
                    subplot_titles=[f"{name} - Step Detection" for name in IMU_MAPPING.values()],
                    vertical_spacing=0.05,
                    shared_xaxes=True
                )
                
                metrics = {}
                
                for imu_id in IMU_MAPPING:
                    imu_data = df[df['IMU'] == imu_id].copy()
                    if not imu_data.empty:
                        # Calculate acceleration magnitude
                        accel_x = imu_data['AccelX'].values
                        accel_y = imu_data['AccelY'].values
                        accel_z = imu_data['AccelZ'].values
                        accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
                        
                        # Calculate relative timestamps in seconds
                        timestamps = (imu_data['Timestamp'] - start_time).dt.total_seconds().values
                        
                        # Filter and detect steps
                        filtered_signal = self.butter_filter(accel_magnitude)
                        peaks, step_times = self.detect_steps(accel_magnitude, timestamps)
                        
                        # Update step data
                        if len(peaks) > 0:
                            latest_step_time = timestamps[peaks[-1]]
                            if latest_step_time > self.step_data[imu_id]['last_step_time']:
                                new_steps = len([p for p in peaks 
                                               if timestamps[p] > self.step_data[imu_id]['last_step_time']])
                                self.step_data[imu_id]['steps'] += new_steps
                                self.step_data[imu_id]['last_step_time'] = latest_step_time
                                
                                if self.step_data[imu_id]['steps'] > 1:
                                    time_span = latest_step_time
                                    if time_span > 0:
                                        self.step_data[imu_id]['cadence'] = (self.step_data[imu_id]['steps'] / time_span) * 60
                        
                        # Add traces to plot
                        row_idx = list(IMU_MAPPING.keys()).index(imu_id) + 1
                        
                        # Raw signal
                        fig.add_trace(
                            go.Scatter(
                                x=timestamps,
                                y=accel_magnitude,
                                mode='lines',
                                name=f'{IMU_MAPPING[imu_id]} Raw',
                                line=dict(color=self.colors[imu_id], width=1, dash='dot'),
                                showlegend=False
                            ),
                            row=row_idx, col=1
                        )
                        
                        # Filtered signal
                        fig.add_trace(
                            go.Scatter(
                                x=timestamps,
                                y=filtered_signal,
                                mode='lines',
                                name=f'{IMU_MAPPING[imu_id]} Filtered',
                                line=dict(color=self.colors[imu_id], width=2),
                                showlegend=False
                            ),
                            row=row_idx, col=1
                        )
                        
                        # Step markers
                        if len(peaks) > 0:
                            fig.add_trace(
                                go.Scatter(
                                    x=timestamps[peaks],
                                    y=accel_magnitude[peaks],
                                    mode='markers',
                                    name=f'{IMU_MAPPING[imu_id]} Steps',
                                    marker=dict(
                                        symbol='x',
                                        size=10,
                                        color='red',
                                        line=dict(width=2)
                                    ),
                                    showlegend=False
                                ),
                                row=row_idx, col=1
                            )
                        
                        # Store metrics
                        metrics[imu_id] = {
                            'steps': self.step_data[imu_id]['steps'],
                            'cadence': self.step_data[imu_id]['cadence']
                        }
                
                # Update layout
                fig.update_layout(
                    height=250 * len(IMU_MAPPING),
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=50, r=50, t=100, b=50)
                )
                
                # Update axes
                fig.update_xaxes(
                    title_text="Time (seconds)",
                    gridcolor='lightgray',
                    showgrid=True
                )
                
                fig.update_yaxes(
                    title_text="Acceleration (m/s²)",
                    gridcolor='lightgray',
                    showgrid=True
                )
                
                # Update tracking variables
                self.last_read_size = file_size
                self.last_update_time = current_time
                
                return fig, metrics
                
        except Exception as e:
            st.error(f"Error processing data: {e}")
            return None, None

def main():
    # Initialize dashboard
    dashboard = StepLengthDashboard()
    
    # Header Section
    st.markdown("""
    <div class="header-section">
        <h1>👣 Step Length Analysis Dashboard</h1>
        <h3>Real-time Gait Analysis • Movement Monitoring</h3>
        <div style="font-size: 0.8rem; color: white; margin-top: 0.5rem;">Updates every 10 seconds</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for layout
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown('<h3 style="color: #000000;">📊 Real-time Step Analysis</h3>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div style="color: #28a745; font-weight: bold; text-align: center; padding: 0.5rem;">Auto-refresh Active (10s)</div>', unsafe_allow_html=True)
    
    with col3:
        auto_refresh = st.checkbox("Auto-refresh", value=True)
    
    # Process data
    fig, metrics = dashboard.process_data()
    
    if fig and metrics:
        # IMU Metrics Cards
        st.markdown('<h3 style="color: #000000;">🎯 Step Detection Metrics</h3>', unsafe_allow_html=True)
        
        # Create columns for IMU cards
        cols = st.columns(len(IMU_MAPPING))
        
        for i, (imu_id, imu_name) in enumerate(IMU_MAPPING.items()):
            with cols[i]:
                if imu_id in metrics:
                    m = metrics[imu_id]
                    color = dashboard.colors[imu_id]
                    
                    st.markdown(f"""
                    <div class="imu-card" style="border-left: 4px solid {color};">
                        <div class="imu-label">{imu_name}</div>
                        <div class="imu-value" style="color: {color};">{int(m['steps'])}</div>
                        <div class="imu-label">Total Steps</div>
                        <hr style="margin: 0.5rem 0;">
                        <div style="font-size: 0.8rem; color: #6c757d;">
                            Cadence: {m['cadence']:.1f} steps/min
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Main plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Clinical insights section
        st.markdown('<h3 style="color: #000000;">🔍 Clinical Insights</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; padding: 1rem; color: #0c5460;">
                <strong>Real-time Analysis Active</strong><br><br>
                • Monitoring step patterns<br>
                • Tracking cadence<br>
                • Analyzing gait symmetry
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Calculate average cadence across all IMUs
            avg_cadence = np.mean([m['cadence'] for m in metrics.values()])
            if avg_cadence > 100:
                st.markdown(f"""
                <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 1rem; color: #155724;">
                    <strong>✅ Normal Gait Pattern</strong><br>
                    Average cadence: {avg_cadence:.1f} steps/min
                </div>
                """, unsafe_allow_html=True)
            elif avg_cadence > 80:
                st.markdown(f"""
                <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 1rem; color: #856404;">
                    <strong>⚠️ Moderate Gait Speed</strong><br>
                    Average cadence: {avg_cadence:.1f} steps/min
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; padding: 1rem; color: #721c24;">
                    <strong>❌ Slow Gait Pattern</strong><br>
                    Average cadence: {avg_cadence:.1f} steps/min
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # No data available
        st.markdown("""
        <div class="metric-card" style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📡</div>
            <div style="font-size: 1.2rem; color: #6c757d; margin-bottom: 1rem;">
                Waiting for IMU Data
            </div>
            <div style="font-size: 0.9rem; color: #6c757d;">
                Please ensure UDP.py is running and collecting data
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(10)  # Changed from 0.1 to 10 seconds
        st.rerun()

if __name__ == "__main__":
    main() 