import socket
import time
from datetime import datetime
import threading
import queue
import signal
import csv
import os
from collections import defaultdict

# Import the UDP configuration and constants from UDP.py
from UDP import UDP_IP, UDP_PORT, COLUMNS

# IMU separation configuration
IMU_WRITE_BATCH_SIZE = 50   # Write every 50 samples per IMU
IMU_QUEUE_SIZE = 1000       # Queue size per IMU
CSV_BUFFER_SIZE = 32768     # 32KB CSV write buffer

# Data queues for each IMU and control
imu_queues = defaultdict(lambda: queue.Queue(maxsize=IMU_QUEUE_SIZE))
running = True
stats = {'received': 0, 'written': defaultdict(int), 'packets': 0, 'start_time': time.time()}

# File handles and threads for each IMU
imu_files = {}
imu_writers = {}
writer_threads = {}

def signal_handler(sig, frame):
    global running
    print("\nStopping IMU separator gracefully...")
    running = False

def get_imu_filename(imu_name):
    """Generate filename for specific IMU"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"IMU_{imu_name}_data_{timestamp}.csv"

def write_worker(imu_name):
    """Ultra-fast CSV writer thread for specific IMU"""
    csv_file = None
    csv_writer = None
    filename = get_imu_filename(imu_name)
    
    try:
        # Open CSV file for writing with large buffer
        csv_file = open(filename, 'w', newline='', buffering=CSV_BUFFER_SIZE)
        csv_writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
        csv_writer.writeheader()
        csv_file.flush()
        
        # Store file handles globally
        imu_files[imu_name] = csv_file
        imu_writers[imu_name] = csv_writer
        
        print(f"Created separated file for {imu_name}: {filename}")
        
        batch = []
        
        while running or not imu_queues[imu_name].empty():
            try:
                # Collect batch as quickly as possible
                batch_collected = 0
                while batch_collected < IMU_WRITE_BATCH_SIZE:
                    try:
                        item = imu_queues[imu_name].get_nowait()
                        batch.append(item)
                        batch_collected += 1
                    except queue.Empty:
                        break
                
                # Write batch to CSV immediately
                if batch:
                    csv_writer.writerows(batch)
                    csv_file.flush()  # Force write to disk
                    stats['written'][imu_name] += len(batch)
                    batch = []
                
                # Minimal sleep only if queue is empty
                if imu_queues[imu_name].empty() and running:
                    time.sleep(0.001)  # 1ms sleep
                    
            except Exception as e:
                print(f"Write error for {imu_name}: {e}")
                break
        
        # Write any remaining data
        if batch:
            csv_writer.writerows(batch)
            csv_file.flush()
            stats['written'][imu_name] += len(batch)
            
    except Exception as e:
        print(f"CSV file error for {imu_name}: {e}")
    finally:
        if csv_file:
            try:
                csv_file.close()
                print(f"Closed separated file for {imu_name}: {filename}")
            except:
                pass

def start_imu_writer(imu_name):
    """Start a writer thread for a specific IMU"""
    if imu_name not in writer_threads:  # Only start if not already running
        writer_thread = threading.Thread(target=write_worker, args=(imu_name,), daemon=True)
        writer_threads[imu_name] = writer_thread
        writer_thread.start()
        print(f"Started writer thread for {imu_name}")
        return writer_thread
    return writer_threads[imu_name]

def process_udp_data(data_dict):
    """Process individual data point from UDP.py and separate by IMU"""
    global stats
    
    try:
        imu_name = data_dict["IMU"]
        
        # Start writer thread for new IMU if needed
        if imu_name not in writer_threads:
            start_imu_writer(imu_name)
            print(f"Started tracking {imu_name} in separator")
        
        # Add to appropriate IMU queue
        try:
            imu_queues[imu_name].put_nowait(data_dict)
            stats['received'] += 1
            
        except queue.Full:
            print(f"QUEUE OVERFLOW for {imu_name} in separator! Dropping sample. Total: {stats['received']}")
            
    except Exception as e:
        print(f"Error processing data in separator: {e}")

def create_udp_socket():
    """Create UDP socket using same configuration as UDP.py"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)  # 1MB receive buffer
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)     # Enable broadcast reception
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)     # Allow address reuse
    sock.settimeout(0.1)  # 100ms timeout
    
    # Use a different port to avoid conflict with UDP.py
    separator_port = UDP_PORT + 1
    sock.bind((UDP_IP, separator_port))
    return sock, separator_port

def parse_udp_packet(raw_data):
    """Parse UDP packet data using same logic as UDP.py"""
    lines = raw_data.split('\n')
    data_points = []
    
    for line in lines:
        if not line.strip():  # Skip empty lines
            continue
            
        fields = line.strip().split(',')
        if len(fields) >= 15:
            try:
                # Create data dict with correct field mapping (same as UDP.py)
                data_dict = {
                    "Count": fields[0],
                    "Timestamp": fields[1], 
                    "IMU": fields[2],
                    "AccelX": float(fields[3]),
                    "AccelY": float(fields[4]), 
                    "AccelZ": float(fields[5]),
                    "GyroX": float(fields[6]),
                    "GyroY": float(fields[7]),
                    "GyroZ": float(fields[8]),
                    "MagX": float(fields[9]),
                    "MagY": float(fields[10]),
                    "MagZ": float(fields[11]),
                    "QuatX": float(fields[12]),
                    "QuatY": float(fields[13]),
                    "QuatZ": float(fields[14]),
                    "QuatW": float(fields[15]) if len(fields) > 15 else float(fields[14])
                }
                data_points.append(data_dict)
                
            except ValueError as e:
                print(f"Error parsing numbers in line: {line[:50]}... Error: {e}")
        else:
            print(f"Invalid field count: {len(fields)} in line: {line[:50]}...")
    
    return data_points

def run_imu_separator():
    """Main function to run the IMU separator"""
    global running, stats
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create UDP socket on different port
    sock, separator_port = create_udp_socket()
    
    print(f"IMU Data Separator started")
    print(f"Listening on {UDP_IP}:{separator_port} (broadcast enabled)")
    print(f"Will create separate CSV files for each IMU")
    print(f"Ready to receive and separate data from any ESP32 on network")
    print("Press Ctrl+C to stop")
    
    # Main receiving loop
    packet_count = 0
    sample_count = 0
    last_stats_time = time.time()
    active_imus = set()
    
    try:
        while running:
            try:
                data, addr = sock.recvfrom(4096)  # Larger receive buffer
                
                try:
                    # Parse received data using same logic as UDP.py
                    raw_data = data.decode('utf-8').strip()
                    stats['packets'] += 1
                    
                    # Debug: Print first few packets to see format and sender
                    if stats['packets'] <= 5:
                        print(f"Separator Packet {stats['packets']} from {addr[0]}:{addr[1]} - {raw_data[:100]}...")
                    
                    # Parse data points using same logic as UDP.py
                    data_points = parse_udp_packet(raw_data)
                    samples_in_packet = 0
                    
                    # Process each data point and separate by IMU
                    for data_dict in data_points:
                        imu_name = data_dict["IMU"]
                        
                        # Track active IMUs
                        if imu_name not in active_imus:
                            active_imus.add(imu_name)
                        
                        # Process the data point
                        process_udp_data(data_dict)
                        samples_in_packet += 1
                    
                    # Count this packet
                    packet_count += 1
                    sample_count += samples_in_packet
                    
                    # Performance stats every 50 packets
                    if packet_count >= 50:
                        current_time = time.time()
                        elapsed = current_time - last_stats_time
                        sample_rate = sample_count / elapsed if elapsed > 0 else 0
                        packet_rate = packet_count / elapsed if elapsed > 0 else 0
                        
                        # Show queue status for each active IMU
                        queue_status = []
                        for imu in active_imus:
                            queue_size = imu_queues[imu].qsize()
                            queue_percent = (queue_size / IMU_QUEUE_SIZE) * 100
                            queue_status.append(f"{imu}:{queue_size}({queue_percent:.1f}%)")
                        
                        print(f"Separator - Samples: {sample_rate:.0f}/sec | Packets: {packet_rate:.0f}/sec | Queues: {', '.join(queue_status)} | Total: {stats['received']}")
                        
                        packet_count = 0
                        sample_count = 0
                        last_stats_time = current_time
                            
                except (ValueError, UnicodeDecodeError) as e:
                    print(f"Error parsing packet in separator: {e}")
                    continue
                    
            except socket.timeout:
                # Timeout is normal, just continue
                continue
            except Exception as e:
                if running:
                    print(f"Receive error in separator: {e}")
                break
    
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down IMU separator...")
        running = False
        
        # Wait for all writer threads to finish
        print("Waiting for IMU writers to finish...")
        for imu_name in active_imus:
            if imu_name in writer_threads:
                writer_threads[imu_name].join(timeout=5)
            if imu_name in imu_files:
                try:
                    imu_files[imu_name].close()
                except:
                    pass
        
        sock.close()
        
        # Final stats
        elapsed = time.time() - stats['start_time']
        print(f"\n=== IMU Separator Final Statistics ===")
        print(f"Total packets received: {stats['packets']}")
        print(f"Total samples received: {stats['received']}")
        print(f"Average sample rate: {stats['received']/elapsed:.1f} samples/sec")
        print(f"Average packet rate: {stats['packets']/elapsed:.1f} packets/sec")
        print(f"Data collection time: {elapsed:.1f} seconds")
        
        print(f"\n=== IMU Separation Statistics ===")
        total_written = 0
        for imu_name in active_imus:
            written = stats['written'].get(imu_name, 0)
            total_written += written
            print(f"{imu_name}: {written} samples written to separated file")
        
        if stats['received'] != total_written:
            lost_samples = stats['received'] - total_written
            print(f"WARNING: {lost_samples} samples lost ({lost_samples/stats['received']*100:.2f}%)")
        else:
            print("✓ All samples successfully separated!")
        
        print("IMU Separator shutdown complete.")

if __name__ == "__main__":
    run_imu_separator() 