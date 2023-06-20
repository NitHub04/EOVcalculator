"""
Created on Sun Jun 18 21:28:27 2023

@author: nikhil88
"""

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


# Load the data from Excel
filename ='[Filename].xlsx'
df = pd.read_excel(filename)

def time_to_seconds(dt):
    total_seconds = dt.hour*3600 + dt.minute*60 + dt.second
    return total_seconds

# Extract the relevant columns
df = df[df['Phase'].isin(['WARMUP', 'EXERCISE'])]
df['t'] = df['t'].apply(time_to_seconds)
time = df['t'].reset_index(drop=True)
ve = df['VE'].reset_index(drop=True)
total_duration_seconds = df['t'].max() - df['t'].min()

# Find local minima and maxima
local_minima = argrelextrema(np.array(ve), np.less)[0]
local_maxima = argrelextrema(np.array(ve), np.greater)[0]

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
    
    # If there are no peaks in this cycle, append None or np.nan to the lists and continue to the next iteration
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

    # Calculate the amplitude of the cycle in percentage for EOV cycle determination
    amp_percent = (peak - mean_nadir) / mean_nadir * 100 if mean_nadir != 0 else 0  # prevent division by zero
    
    # Determine whether this is an EOV cycle
    eov_cycle = amp_percent > 25
    
    # Store the values
    max_values.append(peak)
    nadir_values.append(mean_nadir)
    mean_values.append(mean_ve)
    amplitude.append(amp_raw)
    cycle_lengths.append(df['t'].iloc[nadir2] - df['t'].iloc[nadir1])  # Use actual time for cycle length
    eov_cycles.append(eov_cycle)

# Calculate the total duration of the EOV cycles
total_duration_EOV_cycles = sum(cycle_length for cycle_length, eov in zip(cycle_lengths, eov_cycles) if eov and cycle_length is not None)

# Calculate EOV fraction
eov_fraction = total_duration_EOV_cycles / total_duration_seconds
print('EOV Fraction:', eov_fraction)

# Create DataFrame
results = pd.DataFrame({
    'Cycle': range(1, len(max_values)+1),
    'Max VE': max_values,
    'Mean Nadir VE': nadir_values,
    'Mean VE': mean_values,
    'Amplitude': amplitude,
    'EOV Cycle': eov_cycles,
    'Cycle Length': cycle_lengths
})

print(results)

if eov_fraction >= 0.6:
    # Calculate the average and maximum oscillatory amplitude and cycle length
    eov_amplitude = [amp_raw for amp_raw, cycle in zip(amplitude, eov_cycles) if cycle]
    eov_cycle_length = [cycle for cycle, eov in zip(cycle_lengths, eov_cycles) if eov]

    avg_amplitude = np.mean(eov_amplitude)
    max_amplitude = np.max(eov_amplitude)
    avg_cycle_length = np.mean(eov_cycle_length)
    max_cycle_length = np.max(eov_cycle_length)

    print("Average Oscillatory Amplitude: ", avg_amplitude)
    print("Maximum Oscillatory Amplitude: ", max_amplitude)
    print("Average Oscillatory Cycle Length: ", avg_cycle_length)
    print("Maximum Oscillatory Cycle Length: ", max_cycle_length)
    print("EOV detected: oscillatory ventilation persists for >= 60% of exercise duration")
else:
    print("EOV not detected")

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(time, ve, label='VE')

# Plot nadirs and peaks
plt.scatter(df['t'].iloc[local_minima], ve[local_minima], color='red')  # Nadirs are in red
for t, y in zip(df['t'].iloc[local_minima], ve[local_minima]):
    plt.text(t, y, 'N', color='red', fontsize=12, ha='center', va='bottom')

plt.scatter(df['t'].iloc[local_maxima], ve[local_maxima], color='green')  # Peaks are in green
for t, y in zip(df['t'].iloc[local_maxima], ve[local_maxima]):
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

