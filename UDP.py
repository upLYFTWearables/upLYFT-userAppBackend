import socket
import time
from datetime import datetime
import threading
import queue
import signal
import csv
import os

# UDP configuration
UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 12345
OUTPUT_FILE = f"received_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Column headers for the CSV file
COLUMNS = [
    "Count", "Timestamp", "IMU",
    "AccelX", "AccelY", "AccelZ",
    "GyroX", "GyroY", "GyroZ",
    "MagX", "MagY", "MagZ",
    "QuatW", "QuatX", "QuatY", "QuatZ"
]

# Optimized configuration for multi-IMU packets (600 samples/sec total)
WRITE_BATCH_SIZE = 50   # Write every 50 samples 
MAX_QUEUE_SIZE = 6000   # 10 seconds buffer (600 * 10)
CSV_BUFFER_SIZE = 32768 # 32KB CSV write buffer for better performance

# Data queue and control
data_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
running = True
stats = {'received': 0, 'written': 0, 'packets': 0, 'start_time': time.time()}

def signal_handler(sig, frame):
    global running
    print("\nStopping gracefully...")
    running = False

def write_worker():
    """Ultra-fast CSV writer thread"""
    csv_file = None
    csv_writer = None
    
    try:
        # Open CSV file for writing with large buffer
        csv_file = open(OUTPUT_FILE, 'w', newline='', buffering=CSV_BUFFER_SIZE)
        csv_writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
        csv_writer.writeheader()
        csv_file.flush()
        
        batch = []
        
        while running or not data_queue.empty():
            try:
                # Collect batch as quickly as possible
                batch_collected = 0
                while batch_collected < WRITE_BATCH_SIZE:
                    try:
                        item = data_queue.get_nowait()
                        batch.append(item)
                        batch_collected += 1
                    except queue.Empty:
                        break
                
                # Write batch to CSV immediately
                if batch:
                    csv_writer.writerows(batch)
                    csv_file.flush()  # Force write to disk
                    stats['written'] += len(batch)
                    batch = []
                
                # Minimal sleep only if queue is empty
                if data_queue.empty() and running:
                    time.sleep(0.001)  # 1ms sleep
                    
            except Exception as e:
                print(f"Write error: {e}")
                break
        
        # Write any remaining data
        if batch:
            csv_writer.writerows(batch)
            csv_file.flush()
            stats['written'] += len(batch)
            
    except Exception as e:
        print(f"CSV file error: {e}")
    finally:
        if csv_file:
            csv_file.close()
            print(f"CSV file saved: {OUTPUT_FILE}")

# Setup signal handler
signal.signal(signal.SIGINT, signal_handler)

# Start writer thread
writer_thread = threading.Thread(target=write_worker, daemon=False)
writer_thread.start()

# Create UDP socket with broadcast support
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)  # 1MB receive buffer
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)     # Enable broadcast reception
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)     # Allow address reuse
sock.settimeout(0.1)  # 100ms timeout (more reasonable)
sock.bind((UDP_IP, UDP_PORT))

print(f"Multi-IMU UDP Broadcast Receiver started")
print(f"Listening on {UDP_IP}:{UDP_PORT} (broadcast enabled)")
print(f"Output CSV file: {OUTPUT_FILE}")
print(f"Optimized for multi-IMU broadcast packets")
print(f"Ready to receive from any ESP32 on network 192.168.18.x")
print("Press Ctrl+C to stop")

# Main receiving loop
packet_count = 0
sample_count = 0
last_stats_time = time.time()

try:
    while running:
        try:
            data, addr = sock.recvfrom(4096)  # Larger receive buffer
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            try:
                # Parse received data - handle multi-line packets for multiple IMUs
                raw_data = data.decode('utf-8').strip()
                stats['packets'] += 1
                
                # Debug: Print first few packets to see format and sender
                if stats['packets'] <= 5:
                    print(f"Packet {stats['packets']} from {addr[0]}:{addr[1]} - {raw_data[:100]}...")
                
                # Split by newlines to handle multiple IMU readings in one packet
                lines = raw_data.split('\n')
                samples_in_packet = 0
                
                for line in lines:
                    if not line.strip():  # Skip empty lines
                        continue
                        
                    fields = line.strip().split(',')
                    if len(fields) >= 15:
                        try:
                            # Create data dict with correct field mapping
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
                                "QuatW": float(fields[12]),
                                "QuatX": float(fields[13]),
                                "QuatY": float(fields[14]),
                                "QuatZ": float(fields[15]) if len(fields) > 15 else float(fields[14])
                            }
                            
                            # Add to queue
                            try:
                                data_queue.put_nowait(data_dict)
                                stats['received'] += 1
                                samples_in_packet += 1
                                
                            except queue.Full:
                                print(f"QUEUE OVERFLOW! Dropping sample. Total: {stats['received']}")
                                
                        except ValueError as e:
                            print(f"Error parsing numbers in line: {line[:50]}... Error: {e}")
                    else:
                        print(f"Invalid field count: {len(fields)} in line: {line[:50]}...")
                
                # Count this packet
                packet_count += 1
                sample_count += samples_in_packet
                
                # Performance stats every 50 packets (more frequent for debugging)
                if packet_count >= 50:
                    current_time = time.time()
                    elapsed = current_time - last_stats_time
                    sample_rate = sample_count / elapsed if elapsed > 0 else 0
                    packet_rate = packet_count / elapsed if elapsed > 0 else 0
                    queue_size = data_queue.qsize()
                    queue_percent = (queue_size / MAX_QUEUE_SIZE) * 100
                    
                    print(f"Samples: {sample_rate:.0f}/sec | Packets: {packet_rate:.0f}/sec | Queue: {queue_size} ({queue_percent:.1f}%) | Total: {stats['received']}")
                    
                    packet_count = 0
                    sample_count = 0
                    last_stats_time = current_time
                        
            except (ValueError, UnicodeDecodeError) as e:
                print(f"Error parsing packet: {e}")
                continue
                
        except socket.timeout:
            # Timeout is normal, just continue
            continue
        except Exception as e:
            if running:
                print(f"Receive error: {e}")
            break

except KeyboardInterrupt:
    pass
finally:
    print("\nShutting down...")
    running = False
    
    # Wait for writer to finish
    print("Waiting for writer to finish...")
    writer_thread.join(timeout=30)
    sock.close()
    
    # Final stats
    elapsed = time.time() - stats['start_time']
    print(f"\n=== Final Statistics ===")
    print(f"Total packets received: {stats['packets']}")
    print(f"Total samples received: {stats['received']}")
    print(f"Total samples written: {stats['written']}")
    print(f"Average sample rate: {stats['received']/elapsed:.1f} samples/sec")
    print(f"Average packet rate: {stats['packets']/elapsed:.1f} packets/sec")
    print(f"Average samples per packet: {stats['received']/stats['packets']:.1f}" if stats['packets'] > 0 else "")
    print(f"Data collection time: {elapsed:.1f} seconds")
    print(f"CSV file: {OUTPUT_FILE}")
    
    if stats['received'] != stats['written']:
        lost_samples = stats['received'] - stats['written']
        print(f"WARNING: {lost_samples} samples lost ({lost_samples/stats['received']*100:.2f}%)")
    else:
        print("✓ All samples successfully saved!")
    
    print("Shutdown complete.")