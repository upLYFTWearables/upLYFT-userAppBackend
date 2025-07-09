import streamlit as st
import socket
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our custom modules
from realtimestickman import create_kinematic_model, update_kinematic_model, plot_link_recursive
from realtimePelvicMetrics import RealtimePelvicMetrics
from realtime_powermetrics import RealTimePowerMetrics
from realtimesteplenghtcalculation import RealTimeStepLengthCalculator
from realtimedataFrequency import RealTimeFrequencyMonitor

# UDP Configuration
UDP_IP = "localhost"  # Changed to localhost since UDP.py sends to localhost
UDP_PORT = 12345

class UDPDataDistributor:
    def __init__(self, ip, port):
        """Initialize UDP receiver and data distribution"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        self.running = True
        self.data_handlers = []
        
    def add_handler(self, handler):
        """Add a data handler function"""
        self.data_handlers.append(handler)
        
    def start(self):
        """Start receiving and distributing data"""
        self.thread = threading.Thread(target=self._receive_and_distribute, daemon=True)
        self.thread.start()
        
    def _receive_and_distribute(self):
        """Receive UDP data and distribute to all handlers"""
        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
                data_str = data.decode('utf-8')
                
                # Parse the multi-IMU data packet
                imu_data = {}
                for line in data_str.split('\n'):
                    if not line.strip():
                        continue
                    
                    parts = line.split(',')
                    if len(parts) >= 9:  # Ensure we have all expected fields
                        imu_id = parts[2]
                        imu_data[imu_id] = {
                            'count': int(parts[0]),
                            'timestamp': int(parts[1]),
                            'accel': [float(parts[3]), float(parts[4]), float(parts[5])],
                            'gyro': [float(parts[6]), float(parts[7]), float(parts[8])]
                        }
                
                # Distribute to all handlers
                for handler in self.data_handlers:
                    try:
                        handler(imu_data)
                    except Exception as e:
                        print(f"Error in handler: {e}")
                        
            except Exception as e:
                if self.running:
                    print(f"UDP receive error: {e}")
                continue
                
    def stop(self):
        """Stop the distributor"""
        self.running = False
        self.sock.close()

# Set page config
st.set_page_config(
    page_title="upLYFT Real-Time Dashboard",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (unchanged)
st.markdown("""
    <style>
    .stApp {
        background-color: white;
        color: black;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 16px;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    
    # Initialize metrics processors
    st.session_state.metrics = {
        'pelvic': RealtimePelvicMetrics(),
        'power': RealTimePowerMetrics(),
        'step_length': RealTimeStepLengthCalculator(),
        'frequency': RealTimeFrequencyMonitor()
    }
    
    # Initialize data buffers
    st.session_state.data_buffers = {
        'timestamps': [],
        'pelvic_metrics': {'tilt': [], 'obliquity': [], 'rotation': []},
        'power_metrics': {'left': [], 'right': []},
        'step_length': {'left': [], 'right': []},
        'frequency': []
    }
    
    # Create UDP distributor
    st.session_state.distributor = UDPDataDistributor(UDP_IP, UDP_PORT)
    
    # Define data handlers for each metric
    def handle_pelvic_data(imu_data):
        if 'IMU7' in imu_data:  # Pelvis IMU
            result = st.session_state.metrics['pelvic'].process_quaternion(imu_data['IMU7'])
            if result:
                st.session_state.data_buffers['pelvic_metrics'].update(result)
    
    def handle_power_data(imu_data):
        if 'IMU5' in imu_data:  # Foot IMU
            result = st.session_state.metrics['power'].process_data(imu_data['IMU5'])
            if result:
                st.session_state.data_buffers['power_metrics'].update(result)
    
    def handle_step_length_data(imu_data):
        # Process relevant IMUs for step length
        result = st.session_state.metrics['step_length'].process_data(imu_data)
        if result:
            st.session_state.data_buffers['step_length'].update(result)
    
    def handle_frequency_data(imu_data):
        result = st.session_state.metrics['frequency'].process_data(imu_data)
        if result:
            st.session_state.data_buffers['frequency'].append(result)
    
    # Add handlers to distributor
    st.session_state.distributor.add_handler(handle_pelvic_data)
    st.session_state.distributor.add_handler(handle_power_data)
    st.session_state.distributor.add_handler(handle_step_length_data)
    st.session_state.distributor.add_handler(handle_frequency_data)
    
    # Start the distributor
    st.session_state.distributor.start()
    
    st.session_state.initialized = True

def main():
    st.title("upLYFT Real-Time Analysis Dashboard")
    
    # Create two columns - left for graphs, right for metrics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Stickman visualization
        st.subheader("Real-time Stickman Visualization")
        stickman_placeholder = st.empty()
        
        # Pelvic metrics plot
        st.subheader("Pelvic Metrics")
        pelvic_plot = st.empty()
        
        # Power metrics plot
        st.subheader("Power Metrics")
        power_plot = st.empty()
        
        # Step length plot
        st.subheader("Step Length")
        step_plot = st.empty()
        
        # Frequency plot
        st.subheader("Movement Frequency")
        freq_plot = st.empty()
    
    with col2:
        # Real-time metrics display
        st.subheader("Real-time Metrics")
        
        # Pelvic metrics
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### Pelvic Metrics")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Tilt", f"{st.session_state.data_buffers['pelvic_metrics']['tilt'][-1]:.1f}°" if st.session_state.data_buffers['pelvic_metrics']['tilt'] else "0.0°")
            with cols[1]:
                st.metric("Obliquity", f"{st.session_state.data_buffers['pelvic_metrics']['obliquity'][-1]:.1f}°" if st.session_state.data_buffers['pelvic_metrics']['obliquity'] else "0.0°")
            with cols[2]:
                st.metric("Rotation", f"{st.session_state.data_buffers['pelvic_metrics']['rotation'][-1]:.1f}°" if st.session_state.data_buffers['pelvic_metrics']['rotation'] else "0.0°")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Power metrics
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### Power Metrics")
            cols = st.columns(2)
            with cols[0]:
                st.metric("Left Power", f"{st.session_state.data_buffers['power_metrics']['left'][-1]:.1f}W" if st.session_state.data_buffers['power_metrics']['left'] else "0.0W")
            with cols[1]:
                st.metric("Right Power", f"{st.session_state.data_buffers['power_metrics']['right'][-1]:.1f}W" if st.session_state.data_buffers['power_metrics']['right'] else "0.0W")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Step length
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### Step Length")
            cols = st.columns(2)
            with cols[0]:
                st.metric("Left Step", f"{st.session_state.data_buffers['step_length']['left'][-1]:.2f}m" if st.session_state.data_buffers['step_length']['left'] else "0.00m")
            with cols[1]:
                st.metric("Right Step", f"{st.session_state.data_buffers['step_length']['right'][-1]:.2f}m" if st.session_state.data_buffers['step_length']['right'] else "0.00m")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Frequency
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### Movement Frequency")
            st.metric("Frequency", f"{st.session_state.data_buffers['frequency'][-1]:.1f}Hz" if st.session_state.data_buffers['frequency'] else "0.0Hz")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 