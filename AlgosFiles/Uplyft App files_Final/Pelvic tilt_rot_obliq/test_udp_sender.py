import socket
import time
import math
import numpy as np

def generate_test_data(t):
    """
    Generate test IMU data that simulates a slow oscillating motion.
    Returns data in the format:
    [AccelX, AccelY, AccelZ, GyroX, GyroY, GyroZ, MagX, MagY, MagZ, QuatW, QuatX, QuatY, QuatZ]
    """
    # Generate a slow oscillating rotation around Y axis
    freq = 0.5  # Hz
    angle = math.radians(30 * math.sin(2 * math.pi * freq * t))  # 30 degree max rotation
    
    # Convert to quaternion (w, x, y, z)
    w = math.cos(angle/2)
    x = 0
    y = math.sin(angle/2)
    z = 0
    
    # Normalize quaternion
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Simulate accelerometer data (including gravity)
    accel_x = -0.5 * math.sin(2 * math.pi * freq * t)  # Small oscillation
    accel_y = 8.5  # Mostly gravity
    accel_z = 0.5 * math.cos(2 * math.pi * freq * t)  # Small oscillation
    
    # Simulate gyroscope data (angular velocities)
    gyro_x = 0.1 * math.cos(2 * math.pi * freq * t)
    gyro_y = 0.15 * math.sin(2 * math.pi * freq * t)
    gyro_z = 0.05 * math.sin(2 * math.pi * freq * t)
    
    # Create the data string in the same format as the file
    data_str = f"0,{accel_x:.2f},{accel_y:.2f},{accel_z:.2f}," + \
               f"{gyro_x:.2f},{gyro_y:.2f},{gyro_z:.2f}," + \
               f"0,0,0," + \
               f"{w:.2f},{x:.2f},{y:.2f},{z:.2f}"
    
    return data_str

def main():
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Target address (where realtimePelvicMetrics.py is listening)
    server_address = ('localhost', 12345)
    
    print("Starting to send test IMU data...")
    t = 0
    try:
        while True:
            # Generate test data
            data_str = generate_test_data(t)
            
            # Send the data
            sock.sendto(data_str.encode(), server_address)
            
            # Wait for a short time (simulate 100Hz data rate)
            time.sleep(0.01)  # 10ms = 100Hz
            t += 0.01
            
    except KeyboardInterrupt:
        print("\nStopping sender...")
    finally:
        sock.close()

if __name__ == "__main__":
    main() 