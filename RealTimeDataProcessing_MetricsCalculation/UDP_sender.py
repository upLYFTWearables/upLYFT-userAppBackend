import socket
import time
import math
import random

def generate_imu_data(count, imu_id):
    """Generate simulated IMU data for a specific IMU"""
    t = time.time()
    # Simulate different motion patterns for different IMUs
    freq = 2.0  # Hz
    omega = 2 * math.pi * freq
    t_sim = count / 100.0  # 100 Hz sampling
    
    # Base amplitudes for different IMUs
    if imu_id == 'IMU5':  # Foot IMU
        amp_accel = [2.0, 1.5, 3.0]
        amp_gyro = [50.0, 40.0, 30.0]
        phase = [0, math.pi/3, math.pi/4]
    elif imu_id == 'IMU1':  # Upper leg IMU
        amp_accel = [1.5, 1.0, 2.0]
        amp_gyro = [30.0, 25.0, 20.0]
        phase = [math.pi/6, math.pi/4, math.pi/3]
    elif imu_id == 'IMU2':  # Lower leg IMU
        amp_accel = [1.8, 1.2, 2.5]
        amp_gyro = [40.0, 30.0, 25.0]
        phase = [math.pi/4, math.pi/3, math.pi/2]
    else:  # Other IMUs
        amp_accel = [1.0, 0.8, 1.5]
        amp_gyro = [20.0, 15.0, 10.0]
        phase = [0, math.pi/6, math.pi/3]
    
    # Generate accelerometer data
    accel_x = amp_accel[0] * math.sin(omega * t_sim + phase[0]) + random.gauss(0, 0.1)
    accel_y = amp_accel[1] * math.cos(omega * t_sim + phase[1]) + random.gauss(0, 0.1)
    accel_z = amp_accel[2] * math.sin(omega * t_sim + phase[2]) + random.gauss(0, 0.1)
    
    # Generate gyroscope data
    gyro_x = amp_gyro[0] * math.cos(omega * t_sim) + random.gauss(0, 1)
    gyro_y = amp_gyro[1] * math.sin(omega * t_sim) + random.gauss(0, 1)
    gyro_z = amp_gyro[2] * math.sin(omega * t_sim + phase[2]) + random.gauss(0, 1)
    
    # Format data string (CSV format)
    data = f"{count},{int(t*1e6)},{imu_id},{accel_x:.6f},{accel_y:.6f},{accel_z:.6f},{gyro_x:.6f},{gyro_y:.6f},{gyro_z:.6f}"
    return data

def main():
    # UDP setup
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('localhost', 12345)
    
    # List of IMUs to simulate
    imus = ['IMU1', 'IMU2', 'IMU3', 'IMU4', 'IMU5', 'IMU6', 'IMU7']
    
    print(f"Starting multi-IMU data simulation...")
    print(f"Simulating IMUs: {', '.join(imus)}")
    print(f"Sending to {server_address}")
    
    count = 0
    try:
        while True:
            # Generate and send data for each IMU
            data_packet = []
            for imu_id in imus:
                data = generate_imu_data(count, imu_id)
                data_packet.append(data)
            
            # Send all IMU data in one packet
            packet = '\n'.join(data_packet)
            sock.sendto(packet.encode(), server_address)
            
            # Sleep to maintain ~100Hz sampling rate
            time.sleep(0.01)  # 10ms delay
            count += 1
            
            # Print status every second
            if count % 100 == 0:
                print(f"Sent {count} samples for {len(imus)} IMUs", end='\r')
                
    except KeyboardInterrupt:
        print("\nStopping data simulation...")
    finally:
        sock.close()

if __name__ == "__main__":
    main() 