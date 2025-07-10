import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the CSV file
df = pd.read_csv("received_data_20250704_113436.csv")  # Update path as needed
# Convert timestamp to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%H:%M:%S.%f")

# Round down to the nearest second
df['Timestamp_sec'] = df['Timestamp'].dt.floor('S')


# Group by IMU and timestamp to count frequency per second
frequency_over_time = df.groupby(['IMU', 'Timestamp_sec']).size().reset_index(name='Frequency')

# Get unique IMUs and assign unique colors
unique_imus = sorted(frequency_over_time['IMU'].unique())
n = len(unique_imus)
colors = cm.get_cmap('tab10', n)

# Create subplots
fig, axs = plt.subplots(n, 1, figsize=(16, 3 * n), sharex=True)

for i, imu in enumerate(unique_imus):
    imu_data = frequency_over_time[frequency_over_time['IMU'] == imu]
    color = colors(i)

    # Calculate average frequency
    avg_freq = imu_data["Frequency"].mean()
    total_samples = df[df["IMU"] == imu].shape[0]

    axs[i].plot(imu_data['Timestamp_sec'], imu_data['Frequency'], marker='o', color=color)
    
    # Annotate each frequency point
    for x, y in zip(imu_data["Timestamp_sec"], imu_data["Frequency"]):
        axs[i].text(x, y + 1, str(y), ha='center', va='bottom', fontsize=7, color=color)

    axs[i].set_title(
        f"{imu} – Frequency Over Time (Avg: {avg_freq:.1f} Hz",
        color=color,
        fontsize=12,
    )
    axs[i].set_ylabel("Freq (Hz)")
    axs[i].grid(True)
    axs[i].tick_params(axis='x', rotation=45)

axs[-1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.show()
