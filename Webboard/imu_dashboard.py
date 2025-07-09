import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time
import os
import glob
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Medical Rehabilitation Platform",
    page_icon="🏥",
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
    
    .status-paused {
        background-color: #fff3cd;
        color: #856404 !important;
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
    
    /* Ensure all text in containers is black */
    .element-container, .stMarkdown > div {
        color: #000000 !important;
    }
    
    /* Button text */
    .stButton > button {
        color: #000000 !important;
    }
    
    /* Checkbox text */
    .stCheckbox > label {
        color: #000000 !important;
    }
    
    /* Expander text */
    .streamlit-expanderHeader {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

class IMUDashboard:
    def __init__(self):
        self.colors = {
            'IMU1': '#1f77b4',  # Blue
            'IMU2': '#ff7f0e',  # Orange
            'IMU3': '#2ca02c',  # Green
            'IMU4': '#d62728',  # Red
            'IMU5': '#9467bd',  # Purple
            'IMU6': '#8c564b',  # Brown
            'IMU7': '#e377c2',  # Pink
        }
        
    def find_latest_csv(self):
        """Find the most recent CSV file created by UDP.py"""
        files = glob.glob("received_data_*.csv")
        if not files:
            return None
        return max(files, key=os.path.getctime)
    
    def load_and_process_data(self):
        """Load and process IMU data from CSV file"""
        csv_file = self.find_latest_csv()
        if not csv_file:
            return None, None, None
        
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                return None, None, None
            
            # Convert timestamp to datetime format
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S.%f')
            
            # Round down to the nearest second
            df['Timestamp_sec'] = df['Timestamp'].dt.floor('S')
            
            # Calculate frequencies per IMU per second
            freq_data = df.groupby(['IMU', 'Timestamp_sec']).size().reset_index(name='Frequency')
            
            # Get unique IMUs
            unique_imus = sorted(freq_data['IMU'].unique())
            
            # Calculate relative timestamps
            if not freq_data.empty:
                min_time = freq_data['Timestamp_sec'].min()
                freq_data['Time_Relative'] = (freq_data['Timestamp_sec'] - min_time).dt.total_seconds()
            
            return freq_data, unique_imus, csv_file
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None, None
    
    def create_frequency_plot(self, freq_data, unique_imus):
        """Create interactive frequency plot using Plotly"""
        if freq_data is None or freq_data.empty:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=len(unique_imus), cols=1,
            subplot_titles=[f"{imu} - Frequency Over Time" for imu in unique_imus],
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        for i, imu in enumerate(unique_imus):
            imu_data = freq_data[freq_data['IMU'] == imu]
            
            if not imu_data.empty:
                # Add line trace
                fig.add_trace(
                    go.Scatter(
                        x=imu_data['Time_Relative'],
                        y=imu_data['Frequency'],
                        mode='lines+markers',
                        name=f"{imu}",
                        line=dict(color=self.colors.get(imu, '#1f77b4'), width=2),
                        marker=dict(size=6),
                        showlegend=False
                    ),
                    row=i+1, col=1
                )
                
                # Add annotations for recent points
                recent_points = imu_data.tail(3)
                for _, point in recent_points.iterrows():
                    fig.add_annotation(
                        x=point['Time_Relative'],
                        y=point['Frequency'],
                        text=str(int(point['Frequency'])),
                        showarrow=False,
                        yshift=10,
                        font=dict(size=10, color=self.colors.get(imu, '#1f77b4')),
                        row=i+1, col=1
                    )
        
        # Update layout
        fig.update_layout(
            height=200 * len(unique_imus),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Update axes
        fig.update_xaxes(
            title_text="Time (seconds)",
            gridcolor='lightgray',
            row=len(unique_imus), col=1
        )
        
        for i in range(len(unique_imus)):
            fig.update_yaxes(
                title_text="Freq (Hz)",
                gridcolor='lightgray',
                range=[80, 250],
                row=i+1, col=1
            )
        
        return fig
    
    def calculate_metrics(self, freq_data, unique_imus):
        """Calculate key metrics for dashboard"""
        if freq_data is None or freq_data.empty:
            return {}
        
        metrics = {}
        for imu in unique_imus:
            imu_data = freq_data[freq_data['IMU'] == imu]
            if not imu_data.empty:
                metrics[imu] = {
                    'current_freq': imu_data['Frequency'].iloc[-1],
                    'avg_freq': imu_data['Frequency'].mean(),
                    'max_freq': imu_data['Frequency'].max(),
                    'min_freq': imu_data['Frequency'].min(),
                    'sample_count': len(imu_data)
                }
        
        return metrics

def main():
    # Initialize dashboard
    dashboard = IMUDashboard()
    
    # Header Section
    st.markdown("""
    <div class="header-section">
        <h1>🏥 Medical Rehabilitation Platform</h1>
        <h3>IMU Movement Therapy • Real-time Monitoring</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for layout
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown('<h3 style="color: #000000;">📊 Real-time IMU Frequency Analysis</h3>', unsafe_allow_html=True)
    
    with col2:
        # Session status - auto-refresh enabled
        st.markdown('<div style="color: #28a745; font-weight: bold; text-align: center; padding: 0.5rem;">Auto-refresh Active</div>', unsafe_allow_html=True)
    
    with col3:
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh", value=True)
    
    # Load and process data
    freq_data, unique_imus, csv_file = dashboard.load_and_process_data()
    
    if freq_data is not None and not freq_data.empty:
        # Calculate metrics
        metrics = dashboard.calculate_metrics(freq_data, unique_imus)
        
        # Session info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Session Status</div>
                <div class="status-active">● Active Session</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_samples = sum([m['sample_count'] for m in metrics.values()])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Samples</div>
                <div class="metric-value">{total_samples}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Active IMUs</div>
                <div class="metric-value">{len(unique_imus)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if csv_file:
                file_time = datetime.fromtimestamp(os.path.getctime(csv_file))
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Last Update</div>
                    <div style="font-size: 0.9rem; color: #28a745; font-weight: bold;">
                        {file_time.strftime('%H:%M:%S')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # IMU Metrics Cards
        st.markdown('<h3 style="color: #000000;">🎯 IMU Performance Metrics</h3>', unsafe_allow_html=True)
        
        # Create columns for IMU cards
        cols = st.columns(min(len(unique_imus), 4))
        
        for i, imu in enumerate(unique_imus):
            with cols[i % 4]:
                if imu in metrics:
                    m = metrics[imu]
                    color = dashboard.colors.get(imu, '#1f77b4')
                    
                    st.markdown(f"""
                    <div class="imu-card" style="border-left: 4px solid {color};">
                        <div class="imu-label">{imu}</div>
                        <div class="imu-value" style="color: {color};">{m['current_freq']:.0f} Hz</div>
                        <div class="imu-label">Current Frequency</div>
                        <hr style="margin: 0.5rem 0;">
                        <div style="font-size: 0.8rem; color: #6c757d;">
                            Avg: {m['avg_freq']:.1f} Hz<br>
                            Range: {m['min_freq']:.0f}-{m['max_freq']:.0f} Hz
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Main frequency plot
        st.markdown('<h3 style="color: #000000;">📈 Real-time Frequency Monitoring</h3>', unsafe_allow_html=True)
        
        fig = dashboard.create_frequency_plot(freq_data, unique_imus)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table (collapsible)
        with st.expander("📋 Raw Data View"):
            st.dataframe(
                freq_data.tail(50),
                use_container_width=True,
                hide_index=True
            )
        
        # Clinical insights section
        st.markdown('<h3 style="color: #000000;">🔍 Clinical Insights</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; padding: 1rem; color: #0c5460;">
                <strong>Real-time Analysis Active</strong><br><br>
                • Monitoring IMU sensor frequencies<br>
                • Detecting movement patterns<br>
                • Tracking rehabilitation progress
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if metrics:
                avg_all = np.mean([m['avg_freq'] for m in metrics.values()])
                if avg_all > 200:
                    st.markdown(f"""
                    <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 1rem; color: #155724;">
                        <strong>✅ Excellent Performance</strong><br>
                        Average frequency: {avg_all:.1f} Hz
                    </div>
                    """, unsafe_allow_html=True)
                elif avg_all > 150:
                    st.markdown(f"""
                    <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 1rem; color: #856404;">
                        <strong>⚠️ Good Performance</strong><br>
                        Average frequency: {avg_all:.1f} Hz
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; padding: 1rem; color: #721c24;">
                        <strong>❌ Check Sensors</strong><br>
                        Average frequency: {avg_all:.1f} Hz
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
        
        st.markdown("""
        <div style="background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; padding: 1rem; color: #0c5460;">
            <strong>Instructions:</strong><br>
            1. Start UDP.py to collect IMU data<br>
            2. Ensure IMU sensors are connected<br>
            3. Data will appear automatically once available
        </div>
        """, unsafe_allow_html=True)
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main() 