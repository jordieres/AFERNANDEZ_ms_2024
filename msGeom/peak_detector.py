import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.signal import find_peaks


class DetectPeaks:
    def __init__(self):
        pass

    def detect_triplet_peaks(self, df: pd.DataFrame, column: str, distance: int = 10, prominence: float = 0.5) -> pd.DataFrame:

        """Detects triplets of peaks (entry-secondary, main, exit-secondary) in gyroscope data.
            Args:

                df (pd.DataFrame): Input DataFrame with a 'datetime' column in milliseconds and a gyroscope data column.
                column (str): Name of the column containing gyroscope values.
                distance (int): Minimum horizontal distance (in samples) between peaks.
                prominence (float): Minimum prominence of a peak to be considered significant.
        
        Returns:
            pd.DataFrame: A DataFrame containing the time and values of the identified peak triplets with labels.

        """

        values = df[column].values
        peaks, properties = find_peaks(values, distance=distance, prominence=prominence)
        peak_df = df.iloc[peaks].copy()
        peak_df["peak_type"] = "unlabeled"
    
        # Heuristics: look for triplets where a main peak is preceded and followed by smaller peaks

        triplet_peaks = []
        for i in range(1, len(peaks) - 1):
            prev_idx, curr_idx, next_idx = peaks[i - 1], peaks[i], peaks[i + 1]
            prev_val, curr_val, next_val = values[prev_idx], values[curr_idx], values[next_idx]
            if curr_val > prev_val and curr_val > next_val:
                triplet_peaks.extend([
                    {"timestamp": df.iloc[prev_idx]["time"], "value": prev_val, "peak_type": "entry"},
                    {"timestamp": df.iloc[curr_idx]["time"], "value": curr_val, "peak_type": "main"},
                    {"timestamp": df.iloc[next_idx]["time"], "value": next_val, "peak_type": "exit"}

                ])
    
        return pd.DataFrame(triplet_peaks)
    
    
    def plot_peaks(self, df: pd.DataFrame, signal_column: str, peak_df: pd.DataFrame, signal_name: str = None) -> None:
        """
        Plots the gyroscope or accelerometer signal and overlays detected peak triplets.

        Args:
            df (pd.DataFrame): Original DataFrame with time series data.
            signal_column (str): Name of the column with signal values.
            peak_df (pd.DataFrame): DataFrame with labeled peaks.
            signal_name (str, optional): Name to display in the title (e.g., 'modG', 'modA'). Defaults to signal_column.
        """
        if signal_name is None:
            signal_name = signal_column

        plt.figure(figsize=(15, 4))
        plt.plot(df["time"], df[signal_column], label=f"{signal_name} signal", color="orange")

        for label, color in zip(["entry", "main", "exit"], ["blue", "red", "green"]):
            points = peak_df[peak_df["peak_type"] == label]
            plt.scatter(points["timestamp"], points["value"], label=label, color=color)

        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel(f"{signal_name} value")
        plt.title(f"Detected Peak Triplets - {signal_name}")
        plt.tight_layout()


    def analyze_step_robustness(self,triplets: pd.DataFrame, signal_name: str, total_time: float, window_size: float = 10.0):
        print(f"\n Validación de pasos detectados en ventanas de {window_size:.0f}s para {signal_name}:")
        triplets = triplets.copy()
        triplets["timestamp"] = pd.to_numeric(triplets["timestamp"], errors="coerce")
        n_windows = int(np.ceil(total_time / window_size))
        for i in range(n_windows):
            start_t = i * window_size
            end_t = (i + 1) * window_size
            in_window = triplets[
                (triplets['peak_type'] == 'main') &
                (triplets['timestamp'] >= start_t) &
                (triplets['timestamp'] < end_t)
            ]
            print(f" Ventana {i+1}: {len(in_window)} pasos detectados entre {start_t:.1f}s y {end_t:.1f}s")


