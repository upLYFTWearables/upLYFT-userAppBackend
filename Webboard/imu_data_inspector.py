import pandas as pd
import glob
import os
from datetime import datetime

def inspect_imu_data():
    """Inspect all CSV files to understand IMU data structure"""
    
    print("=== IMU Data Inspector ===")
    print("Analyzing all CSV files in current directory...\n")
    
    # Find all CSV files
    csv_files = glob.glob("received_data_*.csv")
    
    if not csv_files:
        print("No CSV files found matching pattern 'received_data_*.csv'")
        return
    
    # Sort files by creation time
    csv_files.sort(key=os.path.getctime)
    
    print(f"Found {len(csv_files)} CSV files:")
    
    all_imus = set()
    file_summary = []
    
    for csv_file in csv_files:
        try:
            # Get file info
            file_size = os.path.getsize(csv_file)
            file_time = datetime.fromtimestamp(os.path.getctime(csv_file))
            
            # Read CSV
            df = pd.read_csv(csv_file)
            
            if df.empty:
                print(f"  {csv_file}: EMPTY FILE")
                continue
            
            # Get unique IMUs
            unique_imus = sorted(df['IMU'].unique())
            all_imus.update(unique_imus)
            
            # Get data range
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S.%f')
            start_time = df['Timestamp'].min()
            end_time = df['Timestamp'].max()
            duration = (end_time - start_time).total_seconds()
            
            # Calculate frequencies
            df['Timestamp_sec'] = df['Timestamp'].dt.floor('S')
            freq_data = df.groupby(['IMU', 'Timestamp_sec']).size().reset_index(name='Frequency')
            
            summary = {
                'file': csv_file,
                'size_kb': file_size / 1024,
                'created': file_time,
                'imus': unique_imus,
                'total_samples': len(df),
                'duration_sec': duration,
                'avg_freq_per_imu': {}
            }
            
            # Calculate average frequency per IMU
            for imu in unique_imus:
                imu_freq = freq_data[freq_data['IMU'] == imu]['Frequency']
                summary['avg_freq_per_imu'][imu] = imu_freq.mean()
            
            file_summary.append(summary)
            
            print(f"  {csv_file}:")
            print(f"    Size: {file_size/1024:.1f} KB")
            print(f"    Created: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    IMUs: {unique_imus}")
            print(f"    Samples: {len(df)}")
            print(f"    Duration: {duration:.1f} seconds")
            
            for imu in unique_imus:
                avg_freq = summary['avg_freq_per_imu'][imu]
                print(f"    {imu}: {avg_freq:.1f} Hz average")
            print()
            
        except Exception as e:
            print(f"  {csv_file}: ERROR - {e}")
    
    print(f"\n=== Summary ===")
    print(f"Total unique IMUs found: {sorted(all_imus)}")
    
    if file_summary:
        latest_file = file_summary[-1]
        print(f"Latest file: {latest_file['file']}")
        print(f"Latest file IMUs: {latest_file['imus']}")
        print(f"Latest file created: {latest_file['created'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check for missing IMUs
        expected_imus = [f"IMU{i}" for i in range(1, 8)]  # IMU1-IMU7
        missing_imus = [imu for imu in expected_imus if imu not in all_imus]
        
        if missing_imus:
            print(f"\nMissing IMUs (never seen in data): {missing_imus}")
        
        # Check which IMUs are in latest vs older files
        if len(file_summary) > 1:
            older_imus = set()
            for summary in file_summary[:-1]:
                older_imus.update(summary['imus'])
            
            latest_imus = set(latest_file['imus'])
            
            only_in_older = older_imus - latest_imus
            only_in_latest = latest_imus - older_imus
            
            if only_in_older:
                print(f"IMUs only in older files: {sorted(only_in_older)}")
            if only_in_latest:
                print(f"IMUs only in latest file: {sorted(only_in_latest)}")
    
    print(f"\n=== Recommendations ===")
    if len(all_imus) < 7:
        print("- Not all IMUs (1-7) are present in your data")
        print("- Check if all IMU devices are connected and transmitting")
    
    if file_summary:
        latest_size = file_summary[-1]['size_kb']
        if latest_size > 1000:  # > 1MB
            print("- Large CSV files detected - consider data cleanup for better performance")
        
        latest_duration = file_summary[-1]['duration_sec']
        if latest_duration > 300:  # > 5 minutes
            print("- Long data sessions detected - real-time view shows last 60 seconds only")

if __name__ == "__main__":
    inspect_imu_data() 