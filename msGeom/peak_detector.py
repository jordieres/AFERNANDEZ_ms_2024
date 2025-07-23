import numpy as np
import pandas as pd
from geopy.distance import geodesic
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from tabulate import tabulate

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
    


    def plot_trajectory_with_strides(self,pos, gps_pos, time_pos, df_stride_raw_clean, title="Trajectory with Valid Strides", traj_label="Kalman"):
        """
        Plot the estimated trajectory versus GPS and mark valid strides as green points.

        This function displays the macroscopic comparison between an estimated trajectory (e.g., from a Kalman filter)
        and the GPS reference path. Additionally, it highlights the positions of valid strides by interpolating their
        timestamps onto the trajectory.

        :param pos: Estimated trajectory with shape (N, 3) or (N, 2). If 3D, only X and Y are used.
        :type pos: np.ndarray
        :param gps_pos: GPS reference trajectory with shape (N, 2).
        :type gps_pos: np.ndarray
        :param time_pos: Timestamps associated with the estimated trajectory.
        :type time_pos: np.ndarray
        :param df_stride_raw_clean: DataFrame containing valid strides. Must include a "time" column.
        :type df_stride_raw_clean: pd.DataFrame
        :param title: Title of the plot. Default is "Trajectory with Valid Strides".
        :type title: str
        :param traj_label: Label for the estimated trajectory line. Default is "Kalman".
        :type traj_label: str
        :return: None. The function produces a matplotlib plot of the trajectory and valid stride points.
        :rtype: None
        """

        # Ensure that `pos` has only X, Y if it comes with Z
        if pos.shape[1] == 3:
            pos = pos[:, :2]

  
        stride_times = df_stride_raw_clean["time"].values
        stride_x = np.interp(stride_times, time_pos, pos[:, 0])
        stride_y = np.interp(stride_times, time_pos, pos[:, 1])
        stride_positions = np.vstack((stride_x, stride_y)).T

        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(pos[:, 0], pos[:, 1], label=f'{traj_label} Trajectory')
        plt.plot(gps_pos[:, 0], gps_pos[:, 1], 'k--', label='GPS Reference')
        plt.plot(gps_pos[-1, 0], gps_pos[-1, 1], 'ko', label='Final GPS')
        plt.scatter(stride_positions[:, 0], stride_positions[:, 1], color='green', label='Valid Strides', zorder=5)
        plt.title(title)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.grid()
        plt.legend()
        plt.tight_layout()



    # def generate_stride_and_gps_jump_table1(self, df_stride_raw_clean, gps_pos, gps_time, gps_jump_threshold=10.0, start_datetime=None):
    #     """
    #     Generate a table showing valid stride times and significant GPS jumps.

    #     This function combines:
    #     - The timestamps of valid strides (typically shown as green points).
    #     - The timestamps and locations where GPS shows a sudden jump (e.g., >10 meters).
    #     - Optionally includes a datetime column if start_datetime is provided.

    #     :param df_stride_raw_clean: DataFrame with valid strides. Must include 'time'.
    #     :type df_stride_raw_clean: pd.DataFrame
    #     :param gps_pos: Numpy array of GPS positions (shape: Nx2).
    #     :type gps_pos: np.ndarray
    #     :param gps_time: Numpy array of timestamps corresponding to gps_pos (shape: N,).
    #     :type gps_time: np.ndarray
    #     :param gps_jump_threshold: Minimum distance (in meters) to consider a GPS jump.
    #     :type gps_jump_threshold: float
    #     :param start_datetime: Optional starting datetime (as pandas.Timestamp or datetime). If given, adds a datetime column.
    #     :type start_datetime: pd.Timestamp or datetime.datetime or None
    #     :return: Combined DataFrame with event type, time, coordinates, and optionally datetime.
    #     :rtype: pd.DataFrame
    #     """
    #     # Detect GPS jumps
    #     gps_diff = np.linalg.norm(np.diff(gps_pos, axis=0), axis=1)
    #     jump_indices = np.where(gps_diff > gps_jump_threshold)[0] + 1  # +1 to get point *after* jump

    #     gps_jump_times = gps_time[jump_indices]
    #     gps_jump_coords = gps_pos[jump_indices]

    #     df_jumps = pd.DataFrame({
    #         "type": "GPS Jump",
    #         "time": gps_jump_times,
    #         "x": gps_jump_coords[:, 0],
    #         "y": gps_jump_coords[:, 1],
    #         "delta_dist": gps_diff[jump_indices - 1]
    #     })

    #     # Valid strides
    #     stride_times = df_stride_raw_clean["time"].values
    #     stride_x = np.interp(stride_times, gps_time, gps_pos[:, 0])
    #     stride_y = np.interp(stride_times, gps_time, gps_pos[:, 1])

    #     df_strides = pd.DataFrame({
    #         "type": "Valid Stride",
    #         "time": stride_times,
    #         "x": stride_x,
    #         "y": stride_y,
    #         "delta_dist": np.nan
    #     })

    #     # Combine
    #     df_combined = pd.concat([df_strides, df_jumps], ignore_index=True)
    #     df_combined.sort_values(by="time", inplace=True)
    #     df_combined.reset_index(drop=True, inplace=True)

    #     # Add datetime if reference provided
    #     if start_datetime is not None:
    #         df_combined["datetime"] = pd.to_datetime(start_datetime) + pd.to_timedelta(df_combined["time"], unit="s")

    #     return df_combined


    def generate_stride_and_gps_jump_table1(self,
                df_stride_raw_clean,
                gps_pos,
                gps_time,
                gps_jump_threshold=10.0,
                start_datetime=None,
                gps_lat=None,
                gps_lon=None
            ):
   
        # --- 1. Detect GPS jumps ---
        gps_diff = np.linalg.norm(np.diff(gps_pos, axis=0), axis=1)
        jump_indices = np.where(gps_diff > gps_jump_threshold)[0] + 1

        df_jumps = pd.DataFrame({
            "type": "GPS Jump",
            "time": gps_time[jump_indices],
            "x": gps_pos[jump_indices, 0],
            "y": gps_pos[jump_indices, 1],
            "delta_dist": gps_diff[jump_indices - 1],
            "source": "event"
        })

        # --- 2. Add Valid Strides ---
        if df_stride_raw_clean is not None and not df_stride_raw_clean.empty:
            stride_times = df_stride_raw_clean["time"].values
            stride_x = np.interp(stride_times, gps_time, gps_pos[:, 0])
            stride_y = np.interp(stride_times, gps_time, gps_pos[:, 1])
            df_strides = pd.DataFrame({
                "type": "Valid Stride",
                "time": stride_times,
                "x": stride_x,
                "y": stride_y,
                "delta_dist": np.nan,
                "source": "event"
            })
        else:
            df_strides = pd.DataFrame(columns=["type", "time", "x", "y", "delta_dist", "source"])

        # --- 3. GPS Samples ---
        delta_d = np.insert(np.linalg.norm(np.diff(gps_pos, axis=0), axis=1), 0, np.nan)
        df_samples = pd.DataFrame({
            "type": "GPS Sample",
            "time": gps_time,
            "x": gps_pos[:, 0],
            "y": gps_pos[:, 1],
            "delta_dist": delta_d,
            "source": "gps_sample"
        })

        if gps_lat is not None and gps_lon is not None:
            df_samples["lat"] = gps_lat
            df_samples["lon"] = gps_lon

        # --- 4. Merge ---
        df_combined = pd.concat([df_samples, df_strides, df_jumps], ignore_index=True)
        df_combined.sort_values("time", inplace=True)
        df_combined.reset_index(drop=True, inplace=True)

        # --- 5. Datetime ---
        if start_datetime is not None:
            gps_datetime = pd.to_datetime(start_datetime) + pd.to_timedelta(gps_time, unit="s")
            df_combined["datetime"] = np.interp(
                df_combined["time"], gps_time, gps_datetime.astype(np.int64)
            ).astype("datetime64[ns]")

        # --- 6. Lat/lon interpolation ---
        if gps_lat is not None and gps_lon is not None:
            df_combined["lat"] = np.interp(df_combined["time"], gps_time, gps_lat)
            df_combined["lon"] = np.interp(df_combined["time"], gps_time, gps_lon)

        return df_combined