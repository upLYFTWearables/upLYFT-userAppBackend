import streamlit as st
import numpy as np
import plotly.graph_objects as go
import socket
import threading
import queue
import time
from datetime import datetime
import sys
import os

# Add parent directory to path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RealTimeDataProcessing_MetricsCalculation.imumocap import Link

# Page configuration
st.set_page_config(
    page_title="Real-time Stickman Dashboard",
    page_icon="🏃",
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

class StickmanDashboard:
    def __init__(self):
        # Model dimensions
        self.PELVIS_LENGTH = 1.0
        self.PELVIS_WIDTH = 2.0
        self.UPPER_LEG_LENGTH = 4.0
        self.LOWER_LEG_LENGTH = 4.0
        self.FOOT_LENGTH = 1.0
        
        # UDP settings
        self.UDP_IP = "0.0.0.0"
        self.UDP_PORT = 12345
        
        # IMU mapping
        self.IMU_MAPPING = {
            'pelvis': 'IMU7',
            'left_upper_leg': 'IMU1',
            'left_lower_leg': 'IMU3',
            'right_upper_leg': 'IMU2',
            'right_lower_leg': 'IMU4'
        }
        
        # Initialize data structures
        self.imu_data = {
            'IMU1': {'quat': [1.0, 0.0, 0.0, 0.0], 'timestamp': 0},
            'IMU2': {'quat': [1.0, 0.0, 0.0, 0.0], 'timestamp': 0},
            'IMU3': {'quat': [1.0, 0.0, 0.0, 0.0], 'timestamp': 0},
            'IMU4': {'quat': [1.0, 0.0, 0.0, 0.0], 'timestamp': 0},
            'IMU7': {'quat': [1.0, 0.0, 0.0, 0.0], 'timestamp': 0}
        }
        
        # Initialize kinematic model
        self.pelvis = None
        self.left_upper_leg = None
        self.left_lower_leg = None
        self.right_upper_leg = None
        self.right_lower_leg = None
        
        # Status variables
        self.running = False
        self.connection_status = "inactive"
        self.error_message = None
        self.last_update_time = None
        self.data_queue = queue.Queue()
        
        # Create kinematic model
        self.create_kinematic_model()
        
    def create_kinematic_model(self):
        """Create the kinematic model"""
        # Left Leg Links
        left_foot = Link("left_foot", self.FOOT_LENGTH)
        self.left_lower_leg = Link("left_lower_leg", self.LOWER_LEG_LENGTH,
            [(left_foot, Link.matrix(pitch=-90, x=self.LOWER_LEG_LENGTH))])
        self.left_upper_leg = Link("left_upper_leg", self.UPPER_LEG_LENGTH,
            [(self.left_lower_leg, Link.matrix(x=self.UPPER_LEG_LENGTH))])

        # Right Leg Links  
        right_foot = Link("right_foot", self.FOOT_LENGTH)
        self.right_lower_leg = Link("right_lower_leg", self.LOWER_LEG_LENGTH,
            [(right_foot, Link.matrix(pitch=-90, x=self.LOWER_LEG_LENGTH))])
        self.right_upper_leg = Link("right_upper_leg", self.UPPER_LEG_LENGTH,
            [(self.right_lower_leg, Link.matrix(x=self.UPPER_LEG_LENGTH))])

        # Pelvis Link with Both Legs Attached
        self.pelvis = Link("pelvis", self.PELVIS_LENGTH,
            [
                (self.left_upper_leg, Link.matrix(y=self.PELVIS_WIDTH / 2, roll=180, yaw=180)),
                (self.right_upper_leg, Link.matrix(y=-self.PELVIS_WIDTH / 2, roll=180, yaw=180))
            ])

        # Set initial calibration - make pelvis vertical
        self.pelvis.joint = Link.matrix(pitch=-90)
        
        # Set initial IMU orientations
        self.update_imu_orientations()
    
    def update_imu_orientations(self):
        """Update IMU orientations in the kinematic model"""
        self.pelvis.set_imu_global(Link.matrix(quaternion=self.imu_data['IMU7']['quat']))
        self.left_upper_leg.set_imu_global(Link.matrix(quaternion=self.imu_data['IMU1']['quat']))
        self.left_lower_leg.set_imu_global(Link.matrix(quaternion=self.imu_data['IMU3']['quat']))
        self.right_upper_leg.set_imu_global(Link.matrix(quaternion=self.imu_data['IMU2']['quat']))
        self.right_lower_leg.set_imu_global(Link.matrix(quaternion=self.imu_data['IMU4']['quat']))
    
    def start_udp_listener(self):
        """Start UDP listener in a separate thread"""
        if not self.running:
            self.running = True
            self.udp_thread = threading.Thread(target=self._udp_listener)
            self.udp_thread.daemon = True
            self.udp_thread.start()
    
    def _udp_listener(self):
        """UDP listener thread function"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(0.1)
            sock.bind((self.UDP_IP, self.UDP_PORT))
            
            print(f"Listening for IMU data on {self.UDP_IP}:{self.UDP_PORT}")
            
            while self.running:
                try:
                    data, addr = sock.recvfrom(4096)
                    data_str = data.decode('utf-8').strip()
                    
                    # Process each line in the packet
                    for line in data_str.split('\n'):
                        if not line.strip():
                            continue
                            
                        fields = line.strip().split(',')
                        if len(fields) >= 15:
                            imu_id = fields[2]
                            
                            # Only process IMUs we're interested in
                            if imu_id in self.imu_data:
                                try:
                                    quat = [
                                        float(fields[12]),  # w
                                        float(fields[13]),  # x
                                        float(fields[14]),  # y
                                        float(fields[15]) if len(fields) > 15 else float(fields[14])  # z
                                    ]
                                    
                                    # Update IMU data
                                    self.imu_data[imu_id]['quat'] = quat
                                    self.imu_data[imu_id]['timestamp'] = time.time()
                                    
                                    # Update connection status
                                    self.connection_status = "active"
                                    self.last_update_time = datetime.now()
                                    self.error_message = None
                                    
                                except (ValueError, IndexError) as e:
                                    print(f"Error parsing IMU data: {e}")
                    
                    # Update kinematic model
                    self.update_imu_orientations()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"UDP receive error: {e}")
                    if self.running:
                        self.error_message = str(e)
                        self.connection_status = "inactive"
                    continue
                    
        except Exception as e:
            print(f"UDP listener failed: {e}")
            self.error_message = str(e)
            self.connection_status = "inactive"
        finally:
            sock.close()
    
    def plot_link_recursive(self, link, parent_transform=None):
        """Convert link data to plotly traces"""
        traces = []
        
        if parent_transform is None:
            parent_transform = np.eye(4)
            
        # Get current link's global transform
        global_transform = parent_transform @ link.joint
        
        # Extract positions
        start = parent_transform[0:3, 3]
        end = global_transform[0:3, 3]
        
        # Color mapping for different body parts
        color_map = {
            'pelvis': '#FF4B4B',  # Red
            'left_upper_leg': '#4B9FFF',  # Blue
            'left_lower_leg': '#4BC0FF',  # Light Blue
            'right_upper_leg': '#FF9F4B',  # Orange
            'right_lower_leg': '#FFB04B',  # Light Orange
            'left_foot': '#4BFFB0',  # Cyan
            'right_foot': '#FFE74B'  # Yellow
        }
        
        link_color = color_map.get(link.name, 'gray')
        
        # Add link line with custom color
        traces.append(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode='lines',
                line=dict(color=link_color, width=8),
                name=link.name,
                showlegend=True if link.name in color_map else False
            )
        )
        
        # Add joint point
        traces.append(
            go.Scatter3d(
                x=[end[0]],
                y=[end[1]],
                z=[end[2]],
                mode='markers',
                marker=dict(size=10, color=link_color),
                name=f"{link.name}_joint",
                showlegend=False
            )
        )
        
        # Add coordinate axes at joint
        AXIS_LENGTH = 0.4
        colors = ['red', 'green', 'blue']
        for i, (color, axis) in enumerate(zip(colors, ['X', 'Y', 'Z'])):
            direction = global_transform[0:3, i] * AXIS_LENGTH
            traces.append(
                go.Scatter3d(
                    x=[end[0], end[0] + direction[0]],
                    y=[end[1], end[1] + direction[1]],
                    z=[end[2], end[2] + direction[2]],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f"{link.name}_{axis}",
                    showlegend=False
                )
            )
        
        # Plot connections to children
        for child, child_offset in link.children:
            child_transform = global_transform @ child_offset
            child_start = global_transform[0:3, 3]
            child_end = child_transform[0:3, 3]
            
            child_color = color_map.get(child.name, 'gray')
            
            # Add connection line
            traces.append(
                go.Scatter3d(
                    x=[child_start[0], child_end[0]],
                    y=[child_start[1], child_end[1]],
                    z=[child_start[2], child_end[2]],
                    mode='lines',
                    line=dict(color=child_color, width=8),
                    name=f"{link.name}_to_{child.name}",
                    showlegend=False
                )
            )
            
            # Recursively plot child
            traces.extend(self.plot_link_recursive(child, child_transform))
        
        return traces
    
    def create_stickman_plot(self):
        """Create the stickman visualization using Plotly"""
        traces = self.plot_link_recursive(self.pelvis)
        
        fig = go.Figure(data=traces)
        
        # Update layout with white background and improved styling
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    range=[-10, 10],
                    showgrid=True,
                    zeroline=True,
                    showbackground=True,
                    backgroundcolor='white',
                    gridcolor='lightgray',
                    zerolinecolor='lightgray'
                ),
                yaxis=dict(
                    range=[-10, 10],
                    showgrid=True,
                    zeroline=True,
                    showbackground=True,
                    backgroundcolor='white',
                    gridcolor='lightgray',
                    zerolinecolor='lightgray'
                ),
                zaxis=dict(
                    range=[-10, 10],
                    showgrid=True,
                    zeroline=True,
                    showbackground=True,
                    backgroundcolor='white',
                    gridcolor='lightgray',
                    zerolinecolor='lightgray'
                ),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            height=700
        )
        
        return fig

def main():
    st.markdown("""
    <div class='header-section'>
        <h1>🏃 Real-time Stickman Visualization</h1>
        <p>3D motion capture visualization using IMU sensors</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize dashboard in session state
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = StickmanDashboard()
        st.session_state.dashboard.start_udp_listener()
    
    # Create columns for layout
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown('<h3>📊 Real-time Motion Capture</h3>', unsafe_allow_html=True)
    
    with col2:
        # Connection status
        status_class = "status-active" if st.session_state.dashboard.connection_status == "active" else "status-inactive"
        status_text = "● Connected" if st.session_state.dashboard.connection_status == "active" else "● Disconnected"
        st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
    
    with col3:
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        
        # Reconnect button
        if st.session_state.dashboard.connection_status == "inactive":
            if st.button("Reconnect"):
                st.session_state.dashboard.running = False
                time.sleep(0.5)
                st.session_state.dashboard = StickmanDashboard()
                st.session_state.dashboard.start_udp_listener()
    
    # Show error message if any
    if st.session_state.dashboard.error_message:
        st.error(st.session_state.dashboard.error_message)
    
    # Create columns for IMU status cards
    cols = st.columns(5)
    
    # Display IMU status cards
    for i, (imu_id, data) in enumerate(st.session_state.dashboard.imu_data.items()):
        with cols[i]:
            last_update = datetime.fromtimestamp(data['timestamp']) if data['timestamp'] > 0 else None
            status = "Active" if (last_update and (datetime.now() - last_update).total_seconds() < 5) else "Inactive"
            color = "#28a745" if status == "Active" else "#dc3545"
            
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{imu_id}</div>
                <div style='color: {color}; font-weight: bold;'>{status}</div>
                <div style='font-size: 0.8em;'>
                    {last_update.strftime('%H:%M:%S') if last_update else 'No data'}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Create and display stickman visualization
    fig = st.session_state.dashboard.create_stickman_plot()
    st.plotly_chart(fig, use_container_width=True)
    
    # Display quaternion data
    with st.expander("📊 Raw Quaternion Data"):
        for imu_id, data in st.session_state.dashboard.imu_data.items():
            st.markdown(f"""
            <div style='padding: 0.5rem; background-color: #f8f9fa; border-radius: 5px; margin: 0.2rem 0;'>
                <strong>{imu_id}:</strong> w={data['quat'][0]:.3f}, x={data['quat'][1]:.3f}, y={data['quat'][2]:.3f}, z={data['quat'][3]:.3f}
            </div>
            """, unsafe_allow_html=True)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(0.1)  # Shorter refresh time for smoother animation
        st.rerun()

if __name__ == "__main__":
    main() 