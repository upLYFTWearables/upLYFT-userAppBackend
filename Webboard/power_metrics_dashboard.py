import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import socket
import json
import threading
import queue
import time
from datetime import datetime
import traceback
from scipy.signal import find_peaks

# Page configuration
st.set_page_config(
    page_title="Power Metrics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
        padding: 1rem;
    }
    
    .stApp {
        background-color: #f8f9fa;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #007bff !important;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    
    .header-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .status-active {
        background-color: #d4edda;
        color: #155724 !important;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .status-inactive {
        background-color: #f8d7da;
        color: #721c24 !important;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

class PowerMetricsDashboard:
    def __init__(self):
        self.data_queue = queue.Queue()
        self.buffer_size = 500
        self.data_buffer = {
            'timestamp': [],
            'accel_x': [], 'accel_y': [], 'accel_z': [],
            'gyro_x': [], 'gyro_y': [], 'gyro_z': [],
            'trans_power': [], 'rot_power': [], 'total_power': [],
            'power_weight': [], 'rolling_avg': []
        }
        self.metrics = {
            'trans_power_peak': 0,
            'trans_power_mean': 0,
            'rot_power_peak': 0,
            'rot_power_mean': 0,
            'total_power_peak': 0,
            'total_power_mean': 0,
            'stride_time': 0,
            'stride_time_std': 0,
            'cadence': 0,
            'gct': 0,
            'gct_std': 0,
            'swing': 0,
            'swing_std': 0,
            'stride_length': 0,
            'stride_length_std': 0,
            'peak_load': 0
        }
        self.start_time = None
        self.udp_thread = None
        self.running = False
        self.last_update_time = None
        self.connection_status = "inactive"
        self.error_message = None
        self.port = 12345
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_reconnect_time = 0
        self.reconnect_delay = 5  # seconds

    def find_free_port(self):
        """Find a free UDP port to use"""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def start_udp_listener(self):
        """Start UDP listener in a separate thread"""
        if self.udp_thread is None or not self.udp_thread.is_alive():
            try:
                # Try to use default port first
                try:
                    test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    test_sock.bind(('0.0.0.0', self.port))
                    test_sock.close()
                except OSError:
                    # If default port is busy, find a free one
                    self.port = self.find_free_port()
                    print(f"Using alternative port: {self.port}")
                
                self.running = True
                self.udp_thread = threading.Thread(target=self._udp_listener)
                self.udp_thread.daemon = True
                self.udp_thread.start()
                self.connection_status = "active"
                self.error_message = None
                self.reconnect_attempts = 0
            except Exception as e:
                self.error_message = f"Error starting UDP listener: {e}"
                self.connection_status = "inactive"
                self.attempt_reconnect()
                traceback.print_exc()

    def attempt_reconnect(self):
        """Attempt to reconnect if connection is lost"""
        current_time = time.time()
        if (current_time - self.last_reconnect_time >= self.reconnect_delay and 
            self.reconnect_attempts < self.max_reconnect_attempts):
            
            self.reconnect_attempts += 1
            self.last_reconnect_time = current_time
            print(f"Attempting reconnection ({self.reconnect_attempts}/{self.max_reconnect_attempts})...")
            
            try:
                if self.udp_thread is not None:
                    self.running = False
                    self.udp_thread.join(timeout=1.0)
                self.start_udp_listener()
            except Exception as e:
                print(f"Reconnection attempt failed: {e}")

    def _udp_listener(self):
        """UDP listener thread function"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(('0.0.0.0', self.port))
            sock.settimeout(0.1)
            
            print(f"Listening for IMU data on port {self.port}...")
            last_data_time = time.time()
            
            while self.running:
                try:
                    data, addr = sock.recvfrom(1024)
                    data_str = data.decode('utf-8')
                    self.parse_imu_data(data_str)
                    self.last_update_time = datetime.now()
                    last_data_time = time.time()
                    if self.connection_status != "active":
                        self.connection_status = "active"
                        self.error_message = None
                except socket.timeout:
                    # Check if we haven't received data for too long
                    if time.time() - last_data_time > 5.0:  # 5 seconds timeout
                        self.connection_status = "inactive"
                        self.error_message = "No data received for 5 seconds"
                    continue
                except Exception as e:
                    print(f"Error in UDP listener: {e}")
                    traceback.print_exc()
                    time.sleep(1)
        except Exception as e:
            self.error_message = f"UDP listener failed: {e}"
            self.connection_status = "inactive"
            traceback.print_exc()
        finally:
            try:
                sock.close()
            except:
                pass
            
            # Attempt reconnection if connection was lost
            if self.running:
                self.attempt_reconnect()

    def parse_imu_data(self, data_str):
        """Parse IMU data and update buffer"""
        try:
            lines = data_str.strip().split('\n')
            for line in lines:
                if not line.strip():
                    continue
                    
                parts = line.split(',')
                if len(parts) >= 13 and parts[2].strip() == 'IMU5':
                    try:
                        # Parse timestamp
                        time_str = parts[1].strip()
                        try:
                            timestamp = int(time_str)
                        except ValueError:
                            h, m, s = time_str.split(':')
                            s, ms = s.split('.')
                            total_seconds = int(h) * 3600 + int(m) * 60 + int(s)
                            timestamp = total_seconds * 1000000 + int(ms) * 1000

                        # Create data point
                        data_point = {
                            'timestamp': timestamp,
                            'accel_x': float(parts[3]),
                            'accel_y': float(parts[4]),
                            'accel_z': float(parts[5]),
                            'gyro_x': float(parts[6]),
                            'gyro_y': float(parts[7]),
                            'gyro_z': float(parts[8])
                        }
                        
                        # Calculate powers
                        mass = 70  # kg
                        g = 9.81  # m/s²
                        
                        # Translational power
                        accel_mag = np.sqrt(data_point['accel_x']**2 + 
                                          data_point['accel_y']**2 + 
                                          data_point['accel_z']**2)
                        trans_power = mass * g * accel_mag
                        
                        # Rotational power
                        gyro_mag = np.sqrt(data_point['gyro_x']**2 + 
                                         data_point['gyro_y']**2 + 
                                         data_point['gyro_z']**2)
                        moment_inertia = 0.1  # kg·m²
                        rot_power = moment_inertia * gyro_mag
                        
                        # Total power and derivatives
                        total_power = trans_power + rot_power
                        power_weight = total_power / mass
                        
                        # Update data buffer
                        for key in self.data_buffer:
                            if key in data_point:
                                self.data_buffer[key].append(data_point[key])
                            elif key == 'trans_power':
                                self.data_buffer[key].append(trans_power)
                            elif key == 'rot_power':
                                self.data_buffer[key].append(rot_power)
                            elif key == 'total_power':
                                self.data_buffer[key].append(total_power)
                            elif key == 'power_weight':
                                self.data_buffer[key].append(power_weight)
                            
                        # Maintain buffer size
                        if len(self.data_buffer['timestamp']) > self.buffer_size:
                            for key in self.data_buffer:
                                self.data_buffer[key] = self.data_buffer[key][-self.buffer_size:]
                                
                        # Update metrics
                        self.update_metrics()
                        
                    except Exception as e:
                        print(f"Error processing IMU5 data: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error parsing data: {e}")

    def update_metrics(self):
        """Update real-time metrics"""
        try:
            if len(self.data_buffer['trans_power']) > 0:
                self.metrics.update({
                    'trans_power_peak': max(self.data_buffer['trans_power']),
                    'trans_power_mean': np.mean(self.data_buffer['trans_power']),
                    'rot_power_peak': max(self.data_buffer['rot_power']),
                    'rot_power_mean': np.mean(self.data_buffer['rot_power']),
                    'total_power_peak': max(self.data_buffer['total_power']),
                    'total_power_mean': np.mean(self.data_buffer['total_power']),
                    'peak_load': max(self.data_buffer['accel_z']) * 70 * 9.81  # F = ma
                })
                
                # Calculate gait metrics if enough data
                if len(self.data_buffer['accel_z']) > 50:
                    # Simple peak detection for stride analysis
                    peaks, _ = find_peaks(self.data_buffer['accel_z'], height=1.5)
                    if len(peaks) > 1:
                        stride_times = np.diff(peaks) / 100  # Assuming 100Hz sampling
                        self.metrics.update({
                            'stride_time': np.mean(stride_times),
                            'stride_time_std': np.std(stride_times),
                            'cadence': 60 / np.mean(stride_times),
                            'stride_length': 1.2  # Placeholder - needs actual calculation
                        })
        except Exception as e:
            print(f"Error updating metrics: {e}")

    def create_plots(self):
        """Create all plots using Plotly"""
        try:
            if not self.data_buffer['timestamp']:
                return None
                
            # Convert timestamps to relative time in seconds
            t = np.array(self.data_buffer['timestamp'])
            t = (t - t[0]) / 1e6  # Convert to seconds
            
            # Create subplots
            fig = make_subplots(
                rows=6, cols=1,
                subplot_titles=(
                    'Raw Acceleration',
                    'Translational Power',
                    'Rotational Power',
                    'Total Power',
                    'Power/Weight Ratio',
                    'Rolling Average Power'
                ),
                vertical_spacing=0.05,
                shared_xaxes=True
            )
            
            # Add acceleration traces
            for i, axis in enumerate(['x', 'y', 'z']):
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=self.data_buffer[f'accel_{axis}'],
                        name=f'Accel {axis.upper()}',
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
            
            # Add power traces
            fig.add_trace(
                go.Scatter(x=t, y=self.data_buffer['trans_power'],
                          name='Trans Power', line=dict(width=2)),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=t, y=self.data_buffer['rot_power'],
                          name='Rot Power', line=dict(width=2)),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=t, y=self.data_buffer['total_power'],
                          name='Total Power', line=dict(width=2)),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=t, y=self.data_buffer['power_weight'],
                          name='Power/Weight', line=dict(width=2)),
                row=5, col=1
            )
            
            # Calculate and add rolling average
            window = 20
            rolling_avg = np.convolve(
                self.data_buffer['total_power'],
                np.ones(window)/window,
                mode='valid'
            )
            t_rolling = t[window-1:]
            
            fig.add_trace(
                go.Scatter(x=t_rolling, y=rolling_avg,
                          name='Rolling Avg', line=dict(width=2)),
                row=6, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=1200,
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=50, r=50, t=50, b=50),
                font=dict(color='black')  # Set all text to black
            )
            
            # Update subplot title colors
            for annotation in fig.layout.annotations:
                annotation.update(font=dict(color='black', size=14))  # Make subplot titles black and slightly larger
            
            # Update axes
            fig.update_xaxes(title_text="Time (s)", row=6, col=1)
            fig.update_yaxes(title_text="m/s²", row=1, col=1)
            fig.update_yaxes(title_text="Watts", row=2, col=1)
            fig.update_yaxes(title_text="Watts", row=3, col=1)
            fig.update_yaxes(title_text="Watts", row=4, col=1)
            fig.update_yaxes(title_text="W/kg", row=5, col=1)
            fig.update_yaxes(title_text="Watts", row=6, col=1)
            
            return fig
            
        except Exception as e:
            print(f"Error creating plots: {e}")
            traceback.print_exc()
            return None

def main():
    st.markdown("""
    <div class='header-section'>
        <h1>⚡ Power Metrics Dashboard</h1>
        <p>Real-time power analysis from IMU sensors</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = PowerMetricsDashboard()
        st.session_state.dashboard.start_udp_listener()
    
    # Create columns for layout
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown('<h3>📊 Real-time Power Analysis</h3>', unsafe_allow_html=True)
    
    with col2:
        # Connection status
        status_class = "status-active" if st.session_state.dashboard.connection_status == "active" else "status-inactive"
        status_text = "● Connected" if st.session_state.dashboard.connection_status == "active" else "● Disconnected"
        st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
        
        # Show port information
        st.markdown(f"<div style='text-align: center; font-size: 0.8em;'>Port: {st.session_state.dashboard.port}</div>", unsafe_allow_html=True)
    
    with col3:
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        
        # Reconnect button
        if st.session_state.dashboard.connection_status == "inactive":
            if st.button("Reconnect"):
                st.session_state.dashboard.reconnect_attempts = 0
                st.session_state.dashboard.start_udp_listener()
    
    # Show error message if any
    if st.session_state.dashboard.error_message:
        st.error(st.session_state.dashboard.error_message)
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    # Display metrics in cards
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Translational Power</div>
            <div class='metric-value'>{:.2f} W</div>
            <div>Peak: {:.2f} W</div>
        </div>
        """.format(
            st.session_state.dashboard.metrics['trans_power_mean'],
            st.session_state.dashboard.metrics['trans_power_peak']
        ), unsafe_allow_html=True)
        
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Stride Time</div>
            <div class='metric-value'>{:.2f} s</div>
            <div>±{:.2f} s</div>
        </div>
        """.format(
            st.session_state.dashboard.metrics['stride_time'],
            st.session_state.dashboard.metrics['stride_time_std']
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Rotational Power</div>
            <div class='metric-value'>{:.2f} W</div>
            <div>Peak: {:.2f} W</div>
        </div>
        """.format(
            st.session_state.dashboard.metrics['rot_power_mean'],
            st.session_state.dashboard.metrics['rot_power_peak']
        ), unsafe_allow_html=True)
        
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Cadence</div>
            <div class='metric-value'>{:.1f}</div>
            <div>steps/min</div>
        </div>
        """.format(st.session_state.dashboard.metrics['cadence']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Total Power</div>
            <div class='metric-value'>{:.2f} W</div>
            <div>Peak: {:.2f} W</div>
        </div>
        """.format(
            st.session_state.dashboard.metrics['total_power_mean'],
            st.session_state.dashboard.metrics['total_power_peak']
        ), unsafe_allow_html=True)
        
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Peak Load</div>
            <div class='metric-value'>{:.1f}</div>
            <div>N</div>
        </div>
        """.format(st.session_state.dashboard.metrics['peak_load']), unsafe_allow_html=True)
    
    # Create and display plots
    fig = st.session_state.dashboard.create_plots()
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 8px;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📡</div>
            <div style="font-size: 1.2rem; color: #6c757d; margin-bottom: 1rem;">
                Waiting for IMU Data
            </div>
            <div style="font-size: 0.9rem; color: #6c757d;">
                Please ensure your IMU device is connected and sending data
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data insights
    if len(st.session_state.dashboard.data_buffer['timestamp']) > 0:
        with st.expander("📊 Performance Insights"):
            col1, col2 = st.columns(2)
            with col1:
                avg_power = st.session_state.dashboard.metrics['total_power_mean']
                if avg_power > 100:
                    performance = "Excellent"
                    color = "#28a745"
                elif avg_power > 50:
                    performance = "Good"
                    color = "#ffc107"
                else:
                    performance = "Low"
                    color = "#dc3545"
                
                st.markdown(f"""
                <div style="padding: 1rem; background-color: {color}20; border-radius: 8px;">
                    <h4 style="color: {color};">Power Output: {performance}</h4>
                    <p>Average Total Power: {avg_power:.1f} W</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                cadence = st.session_state.dashboard.metrics['cadence']
                if cadence > 0:
                    st.markdown(f"""
                    <div style="padding: 1rem; background-color: #17a2b820; border-radius: 8px;">
                        <h4 style="color: #17a2b8;">Gait Analysis</h4>
                        <p>Cadence: {cadence:.1f} steps/min</p>
                        <p>Stride Time: {st.session_state.dashboard.metrics['stride_time']:.2f} s</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main() 