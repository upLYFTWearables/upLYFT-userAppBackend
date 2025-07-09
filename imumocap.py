import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.transform import Rotation
import copy

class Link:
    def __init__(self, name, length, children=None):
        self.name = name
        self.length = length
        self.children = children or []
        self.joint = np.eye(4)  # 4x4 transformation matrix
        self.imu_global = np.eye(4)  # IMU global orientation
        self.imu_local = np.eye(4)   # IMU local orientation
        self.position = np.zeros(3)
        
    @staticmethod
    def matrix(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, quaternion=None):
        """Create a 4x4 transformation matrix from position and rotation parameters"""
        T = np.eye(4)
        T[0:3, 3] = [x, y, z]
        
        if quaternion is not None:
            # Convert quaternion to rotation matrix
            w, qx, qy, qz = quaternion
            R = Rotation.from_quat([qx, qy, qz, w]).as_matrix()
            T[0:3, 0:3] = R
        else:
            # Convert Euler angles to rotation matrix
            roll_rad = np.radians(roll)
            pitch_rad = np.radians(pitch)
            yaw_rad = np.radians(yaw)
            
            R = Rotation.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad]).as_matrix()
            T[0:3, 0:3] = R
            
        return T
    
    def set_imu_global(self, matrix):
        """Set the global IMU orientation"""
        self.imu_global = matrix.copy()
        self.imu_local = np.linalg.inv(self.joint) @ self.imu_global
    
    def set_joint_from_imu_global(self, imu_matrix):
        """Update joint orientation based on current IMU reading"""
        self.joint = imu_matrix @ np.linalg.inv(self.imu_local)
    
    def get_global_transform(self, parent_transform=None):
        """Get the global transformation matrix for this link"""
        if parent_transform is None:
            parent_transform = np.eye(4)
        
        global_transform = parent_transform @ self.joint
        
        # Update position
        self.position = global_transform[0:3, 3]
        
        return global_transform
    
    def get_endpoints(self, parent_transform=None):
        """Get start and end points of this link"""
        global_transform = self.get_global_transform(parent_transform)
        
        start = global_transform[0:3, 3]
        
        # End point is at length distance along local x-axis
        local_end = np.array([self.length, 0, 0, 1])
        end = (global_transform @ local_end)[0:3]
        
        return start, end
    
    @staticmethod
    def flatten(root_link):
        """Flatten the link hierarchy into a list"""
        result = [root_link]
        for child, _ in root_link.children:
            result.extend(Link.flatten(child))
        return result
    
    def copy(self):
        """Create a deep copy of the transformation matrix"""
        return self.joint.copy()

def plot(root_link, frames=None, fps=30, elev=10, azim=45, file_name=None, figsize=(10, 6), dpi=100):
    """Plot the kinematic chain, optionally with animation"""
    
    def plot_link_recursive(link, parent_transform=None, ax=None):
        """Recursively plot links"""
        if parent_transform is None:
            parent_transform = np.eye(4)
            
        # Get current link's global transform
        global_transform = parent_transform @ link.joint
        
        # Plot the link as a line from parent to current position
        start = parent_transform[0:3, 3]
        end = global_transform[0:3, 3]
        
        if ax is not None:
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                   'b-', linewidth=2, label=link.name if link.name else None)
            ax.scatter(end[0], end[1], end[2], c='r', s=50)
        
        # Recursively plot children
        for child, child_offset in link.children:
            child_transform = global_transform @ child_offset
            plot_link_recursive(child, child_transform, ax)
    
    def update_frame(frame_idx):
        """Update function for animation"""
        if frames is None:
            return
            
        ax.clear()
        
        # Apply frame data to links
        flat_links = Link.flatten(root_link)
        for i, link in enumerate(flat_links):
            if i < len(frames[frame_idx]):
                link.joint = frames[frame_idx][i]
        
        # Plot the updated structure
        plot_link_recursive(root_link, ax=ax)
        
        # Set consistent axis limits
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Motion Capture Animation - Frame {frame_idx}')
        ax.view_init(elev=elev, azim=azim)
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    if frames is None:
        # Static plot
        plot_link_recursive(root_link, ax=ax)
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Static Pose')
        ax.view_init(elev=elev, azim=azim)
        return fig
    else:
        # Animation
        ani = FuncAnimation(fig, update_frame, frames=len(frames), 
                          interval=1000/fps, repeat=True)
        
        if file_name:
            # Save as GIF
            writer = PillowWriter(fps=fps)
            ani.save(file_name, writer=writer)
            print(f"Animation saved as {file_name}")
        
        return ani 