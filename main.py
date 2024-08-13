"""
EOV calculator:
Determine if a HF patient has Exercise Oscillatory Ventilation based on rules outlined in https://doi.org/10.1378/chest.07-2146
Use the data export file from Cosmed CPEX machines as a CSV input.
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

def time_to_seconds(dt):
    """
    Convert time strings to total seconds.
    """
    if isinstance(dt, str):
        dt = dt.strip()
        try:
            # Parse as MM:SS or HH:MM:SS
            if len(dt.split(':')) == 2:
                dt_parsed = pd.to_datetime(dt, format='%M:%S')
                return dt_parsed.minute * 60 + dt_parsed.second
            elif len(dt.split(':')) == 3:
                dt_parsed = pd.to_datetime(dt, format='%H:%M:%S')
                return dt_parsed.hour * 3600 + dt_parsed.minute * 60 + dt_parsed.second
        except ValueError:
            return np.nan
    return np.nan

def process_data(file_path):
    """
    Process the CSV file to determine EOV status and generate a VE plot.
    """
    try:
        print(f"Processing file: {file_path}")
        df = pd.read_csv(file_path)
        
        # Calculate VE_baseline from 'REST' phase
        rest_df = df[df['Phase'] == 'REST']
        VE_baseline = rest_df['VE'].mean()
        threshold = 0.15 * VE_baseline  # 15% of VE_baseline

        # Extract data from 'WARMUP' and 'EXERCISE' phases
        exercise_df = df[df['Phase'].isin(['WARMUP', 'EXERCISE'])]
        exercise_df['t'] = exercise_df['t'].apply(time_to_seconds)
        time = exercise_df['t'].reset_index(drop=True)
        ve = exercise_df['VE'].reset_index(drop=True)
        total_duration_seconds = exercise_df['t'].max() - exercise_df['t'].min()

        # Find local minima and maxima
        local_minima = argrelextrema(np.array(ve), np.less)[0]
        local_maxima = argrelextrema(np.array(ve), np.greater)[0]

        # Initialize lists to store values
        amplitude = []
        cycle_lengths = []
        eov_cycles = []

        for i in range(len(local_minima) - 1):
            nadir1 = local_minima[i]
            nadir2 = local_minima[i + 1]

            peaks_in_cycle = [peak for peak in local_maxima if nadir1 < peak < nadir2]

            if not peaks_in_cycle:
                continue

            max_ = max(peaks_in_cycle, key=lambda peak: ve[peak])
            peak = ve[max_]
            mean_nadir = np.mean([ve[nadir1], ve[nadir2]])

            amp_raw = peak - mean_nadir if mean_nadir != 0 else 0
            eov_cycle = amp_raw > threshold

            amplitude.append(amp_raw)
            cycle_lengths.append(exercise_df['t'].iloc[nadir2] - exercise_df['t'].iloc[nadir1])
            eov_cycles.append(eov_cycle)

        total_duration_EOV_cycles = sum(cycle_length for cycle_length, eov in zip(cycle_lengths, eov_cycles) if eov and cycle_length is not None)
        eov_fraction = total_duration_EOV_cycles / total_duration_seconds if total_duration_seconds > 0 else 0
        avg_amplitude_all = np.mean(amplitude) if amplitude else 0
        avg_cycle_length_all = np.mean(cycle_lengths) if cycle_lengths else 0

        # Determine EOV detection
        eov_detected = eov_fraction >= 0.6

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(time, ve, label='VE')
        plt.xlabel('Time (s)')
        plt.ylabel('VE (L/min)')
        plt.legend(['Baseline' if 'Baseline' in file_path else 'Follow-up'])

        # Save the plot
        output_file = f"{file_path.split('.')[0]}_VE_plot.png"
        plt.savefig(output_file)
        plt.close()

        # Print the summary
        print(f"""
        Patient details:
            Filename: {file_path}
            Date of Test: {df['Date'].iloc[0] if 'Date' in df.columns else 'Unknown'}

        EOV status:
            EOV Detected: {eov_detected}
            EOV Fraction: {eov_fraction}
            Avg Amplitude All: {avg_amplitude_all}
            Avg Cycle Length All: {avg_cycle_length_all}

        For reference:
            VE_rest: {VE_baseline}
            VE_rest15: {threshold}
        """)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def main():
    # Example CSV path (replace with the path to the CSV file)
    csv_file_path = 'your_csv_file.csv'  # Replace this with your actual CSV file path
    
    # Process the data
    process_data(csv_file_path)

if __name__ == "__main__":
    main()
