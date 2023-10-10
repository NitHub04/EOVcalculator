"""
EOV calculator:
Determine if a HF patient has Exercise Oscillatory Ventilation based on rules outlined in https://doi.org/10.1378/chest.07-2146
Use the data export file from Cosmed CPEX machines for existing code
"""

conda install pandas numpy scipy
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

# Load the data from the Cosmed Excel export
filename = 'your_file_path_here.xlsx'  # <replace this with the actual file path
df = pd.read_excel(filename)

# Calculate VE_baseline from 'REST' phase
rest_df = df[df['Phase'] == 'REST']
VE_baseline = rest_df['VE'].mean()
threshold = 0.15 * VE_baseline  # 15% of VE_baseline

# Extract the relevant columns for 'WARMUP' and 'EXERCISE' phases
exercise_df = df[df['Phase'].isin(['WARMUP', 'EXERCISE'])]

# Convert time to seconds
def time_to_seconds(dt):
    total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
    return total_seconds

exercise_df['t'] = exercise_df['t'].apply(time_to_seconds)
time = exercise_df['t'].reset_index(drop=True)
ve = exercise_df['VE'].reset_index(drop=True)
total_duration_seconds = exercise_df['t'].max() - exercise_df['t'].min()

# Find local minima and maxima
local_minima = argrelextrema(np.array(ve), np.less)[0]
local_maxima = argrelextrema(np.array(ve), np.greater)[0]

# Initialize lists to store values
max_values = []
nadir_values = []
mean_values = []
amplitude = []
cycle_lengths = []
eov_cycles = []

# Iterate over the pairs of nadirs
for i in range(len(local_minima) - 1):
    nadir1 = local_minima[i]
    nadir2 = local_minima[i + 1]

    # Get the peaks that are within this cycle
    peaks_in_cycle = [peak for peak in local_maxima if nadir1 < peak < nadir2]
    
    # If there are no peaks in this cycle, append None to the lists and continue to the next iteration
    if not peaks_in_cycle:
        max_values.append(None)
        nadir_values.append(None)
        mean_values.append(None)
        amplitude.append(None)
        cycle_lengths.append(None)
        eov_cycles.append(None)
        continue
    
    # Find the peak with the highest VE value in this cycle
    max_ = max(peaks_in_cycle, key=lambda peak: ve[peak])
    peak = ve[max_]
    
    # Calculate the mean VE of the nadirs in this cycle
    mean_nadir = np.mean([ve[nadir1], ve[nadir2]])
    
    # Calculate the mean VE over the entire cycle
    mean_ve = np.mean(ve[nadir1:nadir2+1])
    
    # Calculate the amplitude of the cycle in raw units
    amp_raw = peak - mean_nadir if mean_nadir != 0 else 0  # prevent division by zero

    # Determine whether this is an EOV cycle based on the threshold calculated from VE_baseline
    eov_cycle = amp_raw > threshold
    
    # Store the values
    max_values.append(peak)
    nadir_values.append(mean_nadir)
    mean_values.append(mean_ve)
    amplitude.append(amp_raw)
    cycle_lengths.append(exercise_df['t'].iloc[nadir2] - exercise_df['t'].iloc[nadir1])  # Use actual time for cycle length
    eov_cycles.append(eov_cycle)

# Calculate the total duration of the EOV cycles
total_duration_EOV_cycles = sum(cycle_length for cycle_length, eov in zip(cycle_lengths, eov_cycles) if eov and cycle_length is not None)

# Calculate EOV fraction
eov_fraction = total_duration_EOV_cycles / total_duration_seconds

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(time, ve, label='VE')

# Plot nadirs and peaks
plt.scatter(exercise_df['t'].iloc[local_minima], ve[local_minima], color='red')  # Nadirs are in red
for t, y in zip(exercise_df['t'].iloc[local_minima], ve[local_minima]):
    plt.text(t, y, 'N', color='red', fontsize=12, ha='center', va='bottom')

plt.scatter(exercise_df['t'].iloc[local_maxima], ve[local_maxima], color='green')  # Peaks are in green
for t, y in zip(exercise_df['t'].iloc[local_maxima], ve[local_maxima]):
    plt.text(t, y, 'P', color='green', fontsize=12, ha='center', va='bottom')

plt.xlabel('Time (s)')
plt.ylabel('VE (L/min)')

# Determine the title based on EOV fraction
if eov_fraction >= 0.6:
    plt.title(f"{filename} - EOV Detected")
else:
    plt.title(f"{filename} - EOV Not Detected")

plt.legend()
plt.show()
