import socket
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import threading
import queue
import copy

# Import our custom imumocap library
import imumocap

# UDP configuration
UDP_IP = "0.0.0.0"
UDP_PORT = 12345

# Data queue and control
data_queue = queue.Queue(maxsize=1000)
running = True

# IMU mapping for body parts
IMU_MAPPING = {
    'pelvis': 'IMU7',
    'left_upper_leg': 'IMU1', 
    'left_lower_leg': 'IMU3',
    'right_upper_leg': 'IMU2',
    'right_lower_leg': 'IMU4'
}

# Store latest IMU data for each sensor
imu_data = {
    'IMU1': {'quat': [1.0, 0.0, 0.0, 0.0], 'timestamp': 0},
    'IMU2': {'quat': [1.0, 0.0, 0.0, 0.0], 'timestamp': 0},
    'IMU3': {'quat': [1.0, 0.0, 0.0, 0.0], 'timestamp': 0},
    'IMU4': {'quat': [1.0, 0.0, 0.0, 0.0], 'timestamp': 0},
    'IMU7': {'quat': [1.0, 0.0, 0.0, 0.0], 'timestamp': 0}
}

# Kinematic model dimensions (same as original script)
PELVIS_LENGTH = 1.0
PELVIS_WIDTH = 2.0
UPPER_LEG_LENGTH = 4.0
LOWER_LEG_LENGTH = 4.0
FOOT_LENGTH = 1.0

# Global kinematic model
pelvis = None
left_upper_leg = None
left_lower_leg = None
right_upper_leg = None
right_lower_leg = None

def create_kinematic_model():
    """Create the kinematic model using the exact same structure as Stickman_correlation_filtered.py"""
    global pelvis, left_upper_leg, left_lower_leg, right_upper_leg, right_lower_leg
    
    # Left Leg Links
    left_foot = imumocap.Link("left_foot", FOOT_LENGTH)  # Placeholder
    left_lower_leg = imumocap.Link("left_lower_leg", LOWER_LEG_LENGTH,
        [(left_foot, imumocap.Link.matrix(pitch=-90, x=LOWER_LEG_LENGTH))])
    left_upper_leg = imumocap.Link("left_upper_leg", UPPER_LEG_LENGTH,
        [(left_lower_leg, imumocap.Link.matrix(x=UPPER_LEG_LENGTH))])

    # Right Leg Links  
    right_foot = imumocap.Link("right_foot", FOOT_LENGTH)  # Placeholder
    right_lower_leg = imumocap.Link("right_lower_leg", LOWER_LEG_LENGTH,
        [(right_foot, imumocap.Link.matrix(pitch=-90, x=LOWER_LEG_LENGTH))])
    right_upper_leg = imumocap.Link("right_upper_leg", UPPER_LEG_LENGTH,
        [(right_lower_leg, imumocap.Link.matrix(x=UPPER_LEG_LENGTH))])

    # Pelvis Link with Both Legs Attached
    pelvis = imumocap.Link("pelvis", PELVIS_LENGTH,
        [
            (left_upper_leg,  imumocap.Link.matrix(y=PELVIS_WIDTH / 2, roll=180, yaw=180)),
            (right_upper_leg, imumocap.Link.matrix(y=-PELVIS_WIDTH / 2, roll=180, yaw=180))
        ])

    # Set initial calibration - make pelvis vertical
    pelvis.joint = imumocap.Link.matrix(pitch=-90)

    # Set initial IMU global orientations
    pelvis.set_imu_global(imumocap.Link.matrix(quaternion=imu_data['IMU7']['quat']))
    left_upper_leg.set_imu_global(imumocap.Link.matrix(quaternion=imu_data['IMU1']['quat']))
    left_lower_leg.set_imu_global(imumocap.Link.matrix(quaternion=imu_data['IMU3']['quat']))
    right_upper_leg.set_imu_global(imumocap.Link.matrix(quaternion=imu_data['IMU2']['quat']))
    right_lower_leg.set_imu_global(imumocap.Link.matrix(quaternion=imu_data['IMU4']['quat']))

    return pelvis

def udp_receiver():
    """Receive UDP packets and update IMU data"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(0.1)
    sock.bind((UDP_IP, UDP_PORT))
    
    print(f"Real-time Stickman UDP Receiver started")
    print(f"Listening on {UDP_IP}:{UDP_PORT}")
    print("IMU Mapping:")
    for body_part, imu_id in IMU_MAPPING.items():
        print(f"  {body_part}: {imu_id}")
    print("Ready to receive IMU data...")
    
    packet_count = 0
    last_stats_time = time.time()
    
    while running:
        try:
            data, addr = sock.recvfrom(4096)
            raw_data = data.decode('utf-8').strip()
            
            # Debug: Print first few packets
            packet_count += 1
            if packet_count <= 3:
                print(f"Packet {packet_count} from {addr[0]}:{addr[1]}")
            
            # Process each line in the packet
            for line in raw_data.split('\n'):
                if not line.strip():
                    continue
                    
                fields = line.strip().split(',')
                if len(fields) >= 15:
                    imu_id = fields[2]
                    
                    # Only process IMUs we're interested in
                    if imu_id in imu_data:
                        try:
                            quat = [
                                float(fields[12]),  # w
                                float(fields[13]),  # x
                                float(fields[14]),  # y
                                float(fields[15]) if len(fields) > 15 else float(fields[14])  # z
                            ]
                            
                            # Update IMU data
                            imu_data[imu_id]['quat'] = quat
                            imu_data[imu_id]['timestamp'] = time.time()
                            
                            # Signal that new data is available
                            try:
                                data_queue.put_nowait({'imu_id': imu_id, 'quat': quat})
                            except queue.Full:
                                pass  # Drop data if queue is full
                                
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing IMU data: {e}")
            
            # Performance stats
            if packet_count % 100 == 0:
                current_time = time.time()
                elapsed = current_time - last_stats_time
                packet_rate = 100 / elapsed if elapsed > 0 else 0
                print(f"Packet rate: {packet_rate:.1f}/sec | Total packets: {packet_count}")
                last_stats_time = current_time
                        
        except socket.timeout:
            continue
        except Exception as e:
            if running:
                print(f"UDP receive error: {e}")
            break
    
    sock.close()

def update_kinematic_model():
    """Update the kinematic model with latest IMU data using the exact same method as correlation script"""
    global pelvis, left_upper_leg, left_lower_leg, right_upper_leg, right_lower_leg
    
    # Update each link with corresponding IMU data (same as correlation script)
    pelvis.set_joint_from_imu_global(imumocap.Link.matrix(quaternion=imu_data['IMU7']['quat']))
    left_upper_leg.set_joint_from_imu_global(imumocap.Link.matrix(quaternion=imu_data['IMU1']['quat']))
    left_lower_leg.set_joint_from_imu_global(imumocap.Link.matrix(quaternion=imu_data['IMU3']['quat']))
    right_upper_leg.set_joint_from_imu_global(imumocap.Link.matrix(quaternion=imu_data['IMU2']['quat']))
    right_lower_leg.set_joint_from_imu_global(imumocap.Link.matrix(quaternion=imu_data['IMU4']['quat']))

def plot_link_recursive(link, ax, parent_transform=None):
    """Plot a single link and its children recursively with proper styling"""
    if parent_transform is None:
        parent_transform = np.eye(4)
        
    # Get current link's global transform
    global_transform = parent_transform @ link.joint
    
    # Extract positions
    start = parent_transform[0:3, 3]
    end = global_transform[0:3, 3]
    
    # Plot the link as a line
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
            'gray', linewidth=2, label='Link' if link.name == 'pelvis' else None)
            
    # Plot joint as a black dot
    ax.scatter(end[0], end[1], end[2], c='k', s=50)
    
    # Plot coordinate axes at joint (smaller than before)
    AXIS_LENGTH = 0.4  # Reduced from 0.5
    colors = ['r', 'g', 'b']  # x=red, y=green, z=blue
    for i in range(3):
        direction = global_transform[0:3, i] * AXIS_LENGTH
        ax.quiver(end[0], end[1], end[2],
                 direction[0], direction[1], direction[2],
                 color=colors[i], length=AXIS_LENGTH, normalize=True)
    
    # Add text label for the link
    if link.name:
        ax.text(end[0], end[1], end[2], link.name, fontsize=8)
    
    # Plot connections to children with proper transforms
    for child, child_offset in link.children:
        child_transform = global_transform @ child_offset
        # Plot connection line
        child_start = global_transform[0:3, 3]
        child_end = child_transform[0:3, 3]
        ax.plot([child_start[0], child_end[0]], 
                [child_start[1], child_end[1]], 
                [child_start[2], child_end[2]], 
                'gray', linewidth=2)
        # Recursively plot child
        plot_link_recursive(child, ax, child_transform)

def update_plot(frame):
    """Update function for matplotlib animation"""
    if data_queue.empty():
        return
    
    # Process all available data in queue
    while not data_queue.empty():
        try:
            data_queue.get_nowait()
        except queue.Empty:
            break
    
    # Update kinematic model with latest IMU data
    update_kinematic_model()
    
    # Clear the current plot
    ax.clear()
    
    # Plot the kinematic chain with proper styling
    plot_link_recursive(pelvis, ax)
    
    # Set consistent view and labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    # Set view angle to match reference implementation
    ax.view_init(elev=10, azim=45)
    
    # Add grid
    ax.grid(True)
    
    # Add legend with proper placement
    ax.legend(["Link", "Joint", "X", "Y", "Z"], loc="upper left", frameon=False)
    
    # Add data timestamp info
    latest_time = max([data['timestamp'] for data in imu_data.values()])
    if latest_time > 0:
        time_str = datetime.fromtimestamp(latest_time).strftime('%H:%M:%S.%f')[:-3]
        ax.text2D(0.02, 0.98, f"Latest data: {time_str}", transform=ax.transAxes, 
                 fontsize=8, verticalalignment='top')
    
    # Adjust subplot to maximize figure space
    plt.subplots_adjust(top=0.95, bottom=0, left=0, right=1)

if __name__ == "__main__":
    try:
        print("Starting Real-time Dynamic Stickman Visualization...")
        
        # Create kinematic model first
        pelvis_root = create_kinematic_model()
        
        # Create 3D plot with same settings as correlation script
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Start UDP receiver thread
        receiver_thread = threading.Thread(target=udp_receiver, daemon=True)
        receiver_thread.start()
        
        print("Waiting for IMU data...")
        
        # Show static plot first to verify stickman structure
        plot_link_recursive(pelvis, ax)
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Real-time IMU Motion Capture - Static Initial Pose')
        ax.view_init(elev=10, azim=45)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize=8)
        plt.draw()
        plt.pause(0.1)
        
        # Create animation
        ani = FuncAnimation(
            fig, 
            update_plot, 
            frames=None,
            interval=100,  # 10 FPS
            cache_frame_data=False,
            blit=False
        )
        
        plt.show()
        
    except KeyboardInterrupt:
        print("\nStopping gracefully...")
        running = False
        
    finally:
        running = False
        print("Real-time stickman visualization stopped.")