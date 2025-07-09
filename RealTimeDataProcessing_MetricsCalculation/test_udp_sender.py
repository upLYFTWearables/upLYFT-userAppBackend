import socket
import time
import math
import random

# UDP configuration
UDP_IP = "127.0.0.1"  # localhost
UDP_PORT = 12345

def create_sample_data(count, timestamp):
    """Create sample IMU data in the exact format shown by user"""
    
    # Base quaternion values for different IMUs
    base_quats = {
        "IMU1": [0.01, 0.01, 0.87, 0.49],
        "IMU2": [0.02, 0.01, 0.91, 0.41],
        "IMU3": [-0.0, 0.0, 0.65, 0.76],
        "IMU4": [0.01, 0.0, 0.91, 0.43],
        "IMU5": [-0.01, 0.01, 0.45, 0.89],
        "IMU6": [0.0, 0.0, 0.86, 0.52],
        "IMU7": [0.01, 0.0, 0.69, 0.72]  # This is our target IMU
    }
    
    # Create dynamic quaternion for IMU7 (simulate movement)
    t = time.time()
    
    # Simulate realistic pelvic motion
    tilt_angle = 0.1 * math.sin(0.5 * t)  # Slow tilt motion
    obliquity_angle = 0.05 * math.sin(0.3 * t)  # Slow obliquity motion
    rotation_angle = 0.08 * math.sin(0.7 * t)  # Slow rotation motion
    
    # Convert to quaternion (simplified)
    qw = math.cos(tilt_angle/2) * math.cos(obliquity_angle/2) * math.cos(rotation_angle/2)
    qx = math.sin(tilt_angle/2) * math.cos(obliquity_angle/2) * math.cos(rotation_angle/2)
    qy = math.cos(tilt_angle/2) * math.sin(obliquity_angle/2) * math.cos(rotation_angle/2)
    qz = math.cos(tilt_angle/2) * math.cos(obliquity_angle/2) * math.sin(rotation_angle/2)
    
    # Update IMU7 quaternion with dynamic values
    base_quats["IMU7"] = [qw, qx, qy, qz]
    
    lines = []
    
    # Create data for multiple IMUs (like in user's example)
    for imu_id in ["IMU1", "IMU2", "IMU3", "IMU4", "IMU5", "IMU6", "IMU7"]:
        # Base accelerometer values (around gravity)
        accel_x = random.uniform(-0.2, 0.3)
        accel_y = random.uniform(-0.1, 0.3)
        accel_z = random.uniform(9.7, 9.9)
        
        # Add some motion for IMU7
        if imu_id == "IMU7":
            accel_x += 0.1 * math.sin(t)
            accel_y += 0.1 * math.cos(t * 0.8)
            accel_z += 0.05 * math.sin(t * 0.6)
        
        # Gyroscope values (mostly zero for stationary)
        gyro_x = random.uniform(-0.01, 0.01)
        gyro_y = random.uniform(-0.01, 0.01)
        gyro_z = random.uniform(-0.01, 0.01)
        
        # Magnetometer values
        mag_x = random.uniform(10, 30)
        mag_y = random.uniform(-25, 10)
        mag_z = random.uniform(-40, -10)
        
        # Get quaternion
        quat = base_quats[imu_id]
        
        # Format: Count,Timestamp,IMU,AccelX,AccelY,AccelZ,GyroX,GyroY,GyroZ,MagX,MagY,MagZ,QuatW,QuatX,QuatY,QuatZ
        line = f"{count},{timestamp},{imu_id},{accel_x:.2f},{accel_y:.2f},{accel_z:.2f},{gyro_x:.1f},{gyro_y:.1f},{gyro_z:.1f},{mag_x:.2f},{mag_y:.2f},{mag_z:.2f},{quat[0]:.2f},{quat[1]:.2f},{quat[2]:.2f},{quat[3]:.2f}"
        lines.append(line)
    
    return lines

def main():
    """Send test UDP data"""
    print("=== UDP Test Data Sender ===")
    print(f"Sending IMU data to {UDP_IP}:{UDP_PORT}")
    print("This will send data for IMU1-IMU7, with IMU7 having dynamic motion")
    print("Press Ctrl+C to stop")
    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    count = 2494089  # Starting count like in user's example
    
    try:
        while True:
            # Generate timestamp
            timestamp = time.strftime("%H:%M:%S.%f")[:-3]  # Format like "03:34:28.248"
            
            # Create sample data
            data_lines = create_sample_data(count, timestamp)
            
            # Send data
            message = "\n".join(data_lines)
            sock.sendto(message.encode('utf-8'), (UDP_IP, UDP_PORT))
            
            # Print status
            if count % 100 == 0:
                print(f"Sent packet {count} with {len(data_lines)} IMU readings")
            
            count += 1
            time.sleep(0.01)  # 100Hz simulation
            
    except KeyboardInterrupt:
        print("\nStopping UDP sender...")
    finally:
        sock.close()

if __name__ == "__main__":
    main() 