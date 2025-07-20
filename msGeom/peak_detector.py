import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.signal import find_peaks

class DetectPeaks:
    """
    Class for detecting step-related peaks in IMU signals and computing stride statistics.
    """
    def __init__(self):
        pass

    def detect_triplet_peaks(self, df: pd.DataFrame, column: str, distance: int = 10, prominence: float = 0.5) -> pd.DataFrame:

        """
        Detects triplets of peaks (entry-secondary, main, exit-secondary) in gyroscope data.

        :param df: Input DataFrame with a 'time' column and a gyroscope signal column.
        :type df: pd.DataFrame
        :param column: Name of the column containing gyroscope values.
        :type column: str
        :param distance: Minimum horizontal distance (in samples) between peaks.
        :type distance: int
        :param prominence: Minimum prominence of a peak to be considered significant.
        :type prominence: float
        :return: DataFrame containing time, values, and labels ('entry', 'main', 'exit') of detected peaks.
        :rtype: pd.DataFrame
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
                {"timestamp": df.iloc[prev_idx]["time"], "value": prev_val, "peak_type": "entry", "orig_index": prev_idx},
                {"timestamp": df.iloc[curr_idx]["time"], "value": curr_val, "peak_type": "main", "orig_index": curr_idx},
                {"timestamp": df.iloc[next_idx]["time"], "value": next_val, "peak_type": "exit", "orig_index": next_idx},
                ])

    
        return pd.DataFrame(triplet_peaks)
    
    
    def plot_peaks(self, df, signal_column, peak_df, signal_name = None) -> None:
        """
        Plot a time series signal with overlaid labeled peak triplets (entry, main, exit).

        This function visualizes a gyroscope or accelerometer signal along with the detected 
        peaks classified as entry, main, or exit, using different colors for each label.

        :param df: Original DataFrame containing the time series signal.
        :type df: pd.DataFrame
        :param signal_column: Name of the column containing the signal values to be plotted.
        :type signal_column: str
        :param peak_df: DataFrame containing labeled peaks with columns 'timestamp', 'value', and 'peak_type'.
        :type peak_df: pd.DataFrame
        :param signal_name: Optional custom name for the signal to display in the plot title. If None, uses `signal_column`.
        :type signal_name: str or None
        :return: None
        :rtype: None
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


    def analyze_step_robustness(self,triplets, signal_name , total_time, window_size: float = 10.0):
        """
        Evaluate the consistency of detected steps over time using a sliding window approach.

        The function divides the total duration into time windows and counts the number 
        of main peaks (representing steps) in each window, printing the result.

        :param triplets: DataFrame containing detected peak triplets with a 'peak_type' and 'timestamp' column.
        :type triplets: pd.DataFrame
        :param signal_name: Name of the signal used (for display in output messages).
        :type signal_name: str
        :param total_time: Total duration of the signal in seconds.
        :type total_time: float
        :param window_size: Duration of each window in seconds to count steps. Default is 10 seconds.
        :type window_size: float
        :return: None
        :rtype: None
        """


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
            

    def compute_stride_stats_per_minute(self, df_steps, pos_kalman, step_peaks, output_dir=None):
        """
        Compute per-minute stride statistics based on detected step peaks and position data.

        Calculates stride lengths between consecutive step peaks and aggregates step counts, 
        mean stride length, standard deviation, and total distance covered per minute.

        :param df_steps: DataFrame with time and datetime columns for each detected step.
        :type df_steps: pd.DataFrame
        :param pos_kalman: Array of Kalman-filtered positions with shape (N, 2) or (N, 3).
        :type pos_kalman: np.ndarray
        :param step_peaks: List of indices in df_steps indicating the main step peaks.
        :type step_peaks: list[int]
        :param output_dir: Optional output directory for saving results (currently unused).
        :type output_dir: str or None
        :return: 
            - df_stats: Aggregated stride statistics per minute.
            - df_stride: Raw stride information per individual step.
        :rtype: tuple[pd.DataFrame, pd.DataFrame]
        """
        
        df_steps = df_steps.reset_index(drop=True)

        stride_data = []
        for i in range(1, len(step_peaks)):
            idx_start = step_peaks[i - 1]
            idx_end = step_peaks[i]

            if idx_start >= len(df_steps) or idx_end >= len(df_steps):
                continue

            timetride = df_steps.loc[idx_end, 'time']
            stride_length = np.linalg.norm(pos_kalman[idx_end, :2] - pos_kalman[idx_start, :2])

            stride_data.append({
                'time': timetride,
                'stride_length_m': stride_length,
                'datetime': df_steps.loc[idx_end, 'datetime']
            })

        df_stride = pd.DataFrame(stride_data)
        df_stride['minute'] = df_stride['time'].astype(int) // 60

        summary = []
        for minute, group in df_stride.groupby('minute'):
            start_time = group['datetime'].min().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            end_time = group['datetime'].max().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            summary.append({
                'minute': minute,
                'steps': len(group),
                'mean_stride_length': group['stride_length_m'].mean(),
                'std_stride_length': group['stride_length_m'].std(),
                'distance_m': group['stride_length_m'].sum(),
                'start_time': start_time,
                'end_time': end_time
            })

        df_stats = pd.DataFrame(summary)
        return df_stats, df_stride