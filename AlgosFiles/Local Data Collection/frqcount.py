import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Optional: Seaborn theme for a modern look
sns.set_theme(style="whitegrid")

# Load and clean the CSV
file_path = "AlgosFiles/Test_20250530124006.csv"
df = pd.read_csv(file_path, on_bad_lines='skip')

# Filter rows with IMU values
df = df[df['IMU'].astype(str).str.contains('IMU', na=False)]

# Parse timestamps
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S', errors='coerce')
df = df.dropna(subset=['Timestamp'])

# Group per second
df['Time_Second'] = df['Timestamp'].dt.strftime('%H:%M:%S')
frequency = df.groupby(['Time_Second', 'IMU']).size().unstack(fill_value=0)

# Plot setup
imu_list = frequency.columns
num_imus = len(imu_list)
fig, axs = plt.subplots(num_imus, 1, figsize=(16, 4.5 * num_imus), sharex=True)

if num_imus == 1:
    axs = [axs]

colors = sns.color_palette("Set2", n_colors=num_imus)

for i, imu in enumerate(imu_list):
    imu_data = frequency[imu]
    avg_freq = imu_data.mean()

    axs[i].plot(
        imu_data.index, imu_data.values,
        marker='o', linestyle='-',
        color=colors[i],
        linewidth=2, markersize=7,
        label=f'{imu}  |  Avg: {avg_freq:.2f}'
    )

    # Annotate points
    for x, y in enumerate(imu_data.values):
        if y > 0:
            axs[i].text(x, y + 0.2, str(y), ha='center', fontsize=9, color=colors[i], fontweight='bold')

    axs[i].set_title(f'{imu} Frequency Over Time', fontsize=14, fontweight='bold', pad=10)
    axs[i].set_ylabel('Count', fontsize=12)
    axs[i].legend(frameon=True, loc='upper left')
    axs[i].grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)

# X-axis shared label
plt.xlabel('Time (HH:MM:SS)', fontsize=13)
plt.xticks(rotation=45, fontsize=10)
plt.suptitle('IMU Frequency Analysis Over Time', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
