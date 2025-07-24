import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from geopy.distance import geodesic


class StrideProcessor:
    """
    Processor class for filtering, validating, and analyzing stride-based gait data.
    It includes tools for quality checks, spatial alignment, and segment-level diagnostics.
    """

    def __init__(self, min_stride=0.2, max_stride=2.5, window_sec=1.5):
        """
        Initialize the stride processor with stride length constraints.

        :param min_stride: Minimum valid stride length (in meters).
        :type min_stride: float
        :param max_stride: Maximum valid stride length (in meters).
        :type max_stride: float
        :param window_sec: Time window (in seconds) around each stride for data extraction.
        :type window_sec: float
        """
        self.min_stride = min_stride
        self.max_stride = max_stride
        self.window = window_sec


    def clean_stride_data(self, df_stride_raw):
        """
        Filter out strides whose length falls outside the valid range.

        :param df_stride_raw: DataFrame containing raw stride data.
        :type df_stride_raw: pd.DataFrame
        :return: Filtered DataFrame with valid strides.
        :rtype: pd.DataFrame
        """

        before = len(df_stride_raw)
        cleaned = df_stride_raw[
            (df_stride_raw["stride_length_m"] >= self.min_stride) &
            (df_stride_raw["stride_length_m"] <= self.max_stride)
        ].copy()
        after = len(cleaned)
        return cleaned
    

    def recompute_stats_per_minute(self, df_stride_clean):
        """
        Recompute per-minute stride statistics after filtering.

        :param df_stride_clean: DataFrame of valid strides.
        :type df_stride_clean: pd.DataFrame
        :return: Summary DataFrame grouped by minute.
        :rtype: pd.DataFrame
        """
        df_stats_clean = df_stride_clean.groupby('minute').agg(
            steps=('stride_length_m', 'count'),
            mean_stride_length=('stride_length_m', 'mean'),
            std_stride_length=('stride_length_m', 'std'),
            distance_m=('stride_length_m', 'sum')
        ).reset_index()
        return df_stats_clean
    
    
    def check_distance_similarity(self, df_stats, gps_distance, tolerance = 0.15) :
        """
        Check whether estimated distance is consistent with GPS total distance.

        :param df_stats: DataFrame with stride statistics including 'distance_m'.
        :type df_stats: pd.DataFrame
        :param gps_distance: Total GPS-measured distance (in meters).
        :type gps_distance: float
        :param tolerance: Acceptable deviation ratio (e.g., 0.15 = ±15%).
        :type tolerance: float
        :return: Updated DataFrame with a new column 'gps_consistent'.
        :rtype: pd.DataFrame
        """
        total_stride_distance = df_stats['distance_m'].sum()
        ratio = total_stride_distance / gps_distance if gps_distance > 0 else 0
        df_stats['gps_consistent'] = np.abs(ratio - 1) <= tolerance
        return df_stats
    

    def check_stride_length_range(self, df_stats, min_valid=0.2, max_valid=1.5) :
        """
        Check whether average stride lengths fall within an expected range.

        :param df_stats: DataFrame with 'mean_stride_length' column.
        :type df_stats: pd.DataFrame
        :param min_valid: Minimum acceptable mean stride length (meters).
        :type min_valid: float
        :param max_valid: Maximum acceptable mean stride length (meters).
        :type max_valid: float
        :return: Updated DataFrame with 'stride_length_valid' column.
        :rtype: pd.DataFrame
        """
        df_stats['stride_length_valid'] = (
            (df_stats['mean_stride_length'] >= min_valid) &
            (df_stats['mean_stride_length'] <= max_valid)
        )
        return df_stats
    

    def check_trajectory_smoothness(self, df_steps, velocity_threshold = 3.0) :
        """
        Detect abrupt velocity spikes in trajectory as potential noise indicators.

        :param df_steps: DataFrame with 'velocity_m_s' column.
        :type df_steps: pd.DataFrame
        :param velocity_threshold: Maximum allowed velocity change (in m/s).
        :type velocity_threshold: float
        :return: DataFrame with a 'velocity_spike' column indicating anomalies.
        :rtype: pd.DataFrame
        """

        df_steps = df_steps.copy()
        df_steps['velocity_spike'] = df_steps['velocity_m_s'].diff().abs() > velocity_threshold
        num_spikes = df_steps['velocity_spike'].sum()
        return df_steps
    

    def check_spatial_alignment(self, pos_est: np.ndarray, pos_gps: np.ndarray, threshold: float = 10.0) -> np.ndarray:
        """
        Evaluate spatial error between estimated and GPS positions.

        :param pos_est: Estimated positions (N, 2).
        :type pos_est: np.ndarray
        :param pos_gps: GPS positions (N, 2), matching timestamps.
        :type pos_gps: np.ndarray
        :param threshold: Maximum acceptable spatial error (in meters).
        :type threshold: float
        :return: Tuple of (alignment mask, percentage of aligned samples).
        :rtype: tuple[np.ndarray, float]
        :raises ValueError: If input shapes do not match.
        """

        if pos_est.shape != pos_gps.shape:
            raise ValueError("Estimated and GPS positions must have the same shape.")
        spatial_error = np.linalg.norm(pos_est - pos_gps, axis=1)
        alignment = spatial_error <= threshold
        percent_ok = 100 * np.mean(alignment)
        return alignment,percent_ok


    def evaluate_quality_segments(self, df_stride_stats, df_steps ,gps_pos, imu_pos, velocity_threshold = 3.0, gps_distance = None, error_tolerance_m = 10.0, min_alignment_ratio = 0.95 ):
        """
        Assess per-minute gait quality by combining multiple criteria:
        distance consistency, stride length, velocity smoothness, and GPS alignment.

        :param df_stride_stats: Per-minute stride statistics (filtered).
        :type df_stride_stats: pd.DataFrame
        :param df_steps: Step-level DataFrame with 'velocity_m_s' and 'minute' columns.
        :type df_steps: pd.DataFrame
        :param gps_pos: GPS positions (N, 2).
        :type gps_pos: np.ndarray
        :param imu_pos: Estimated IMU positions (N, 2).
        :type imu_pos: np.ndarray
        :param velocity_threshold: Maximum velocity allowed before being marked as a spike.
        :type velocity_threshold: float
        :param gps_distance: Total GPS distance (used to validate distance estimates).
        :type gps_distance: float
        :param error_tolerance_m: Maximum allowed spatial error (in meters).
        :type error_tolerance_m: float
        :param min_alignment_ratio: Minimum proportion of aligned points per segment (0-1).
        :type min_alignment_ratio: float
        :return: DataFrame with an 'all_criteria_ok' column summarizing segment validity.
        :rtype: pd.DataFrame
        """

        stats = df_stride_stats.copy()
        stats["gps_consistent"] = False
        stats["velocity_ok"] = False
        stats["spatially_aligned"] = False

        if gps_distance is not None:
            stats["gps_consistent"] = stats["distance_m"].between(gps_distance * 0.85, gps_distance * 1.15)
        
        for i, row in stats.iterrows():
            minute = row["minute"]
            vel_segment = df_steps[df_steps["minute"] == minute]["velocity_m_s"]
            stats.at[i, "velocity_ok"] = (vel_segment <= velocity_threshold).all()

            imu_seg = imu_pos[df_steps["minute"] == minute]
            gps_seg = gps_pos[df_steps["minute"] == minute]

            if len(imu_seg) > 0 and len(gps_seg) == len(imu_seg):
                dists = np.linalg.norm(imu_seg - gps_seg, axis=1)
                aligned_ratio = np.mean(dists < error_tolerance_m)
                stats.at[i, "spatially_aligned"] = aligned_ratio >= min_alignment_ratio

        stats["all_criteria_ok"] = (
            stats["gps_consistent"] &
            stats["stride_length_valid"] &
            stats["velocity_ok"] &
            stats["spatially_aligned"]
        )

        return stats

    def plot_stride_filtering(self,df_stride_raw, df_stride_raw_clean,min_stride=0.2, max_stride=2.5, y_max=3.0, title="Strides before and after filtering"):
        """
        Visualize stride lengths before and after length-based filtering.

        :param df_stride_raw: Original DataFrame containing all stride lengths.
        :type df_stride_raw: pd.DataFrame
        :param df_stride_raw_clean: Filtered DataFrame with valid strides.
        :type df_stride_raw_clean: pd.DataFrame
        :param min_stride: Minimum valid stride length (meters).
        :type min_stride: float
        :param max_stride: Maximum valid stride length (meters).
        :type max_stride: float
        :param y_max: Maximum y-axis limit (not used explicitly).
        :type y_max: float
        :param title: Plot title.
        :type title: str
        """
        plt.figure(figsize=(12, 4))
        plt.plot(df_stride_raw["time"], df_stride_raw["stride_length_m"], label="Raw strides")
        plt.scatter(df_stride_raw_clean["time"], df_stride_raw_clean["stride_length_m"], label="Valid strides", color="green")
        plt.axhline(max_stride, color="red", linestyle="--", label="Max. threshold")
        plt.axhline(min_stride, color="red", linestyle="--", label="Min. threshold")
        plt.xlabel("Time (s)")
        plt.ylabel("Stride length (m)")
        plt.legend()
        plt.title(title)
        plt.tight_layout()

    def extract_region(self, df_imu, df_gps, stride_time):
        """
        Extracts a time window around a given stride from IMU and GPS signals.

        :param df_imu: DataFrame with IMU columns ['time', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'].
        :type df_imu: pd.DataFrame
        :param df_gps: DataFrame with GPS columns ['time', 'lat', 'lng'].
        :type df_gps: pd.DataFrame
        :param stride_time: Center time of the stride (in seconds).
        :type stride_time: float
        :return: Tuple of IMU/GPS windows, geodesic distance, start and end times.
        :rtype: tuple[pd.DataFrame, pd.DataFrame, float, float, float]
        """

        start = stride_time - self.window
        end = stride_time + self.window

        imu_window = df_imu[(df_imu["time"] >= start) & (df_imu["time"] <= end)].copy()
        gps_window = df_gps[(df_gps["time"] >= start) & (df_gps["time"] <= end)].copy()

        if len(gps_window) < 2:
            distance = None
        else:
            coord_start = (gps_window.iloc[0]['lat'], gps_window.iloc[0]['lng'])
            coord_end = (gps_window.iloc[-1]['lat'], gps_window.iloc[-1]['lng'])
            distance = geodesic(coord_start, coord_end).meters

        return imu_window, gps_window, distance, start, end

    def analyze_strides(self, df_imu, df_gps, df_strides, stride_type="invalid"):
        """
        Analyze and extract data windows around multiple strides.

        :param df_imu: IMU signal DataFrame.
        :type df_imu: pd.DataFrame
        :param df_gps: GPS data DataFrame.
        :type df_gps: pd.DataFrame
        :param df_strides: DataFrame with a 'time' column marking each stride.
        :type df_strides: pd.DataFrame
        :param stride_type: Label assigned to the analyzed strides (e.g., 'valid', 'invalid').
        :type stride_type: str
        :return: List of dictionaries, each containing stride-level analysis data.
        :rtype: list[dict]
        """
        results = []
        for i, row in df_strides.iterrows():
            t = row["time"]
            imu_data, gps_data, gps_dist, start, end = self.extract_region(df_imu, df_gps, t)

            results.append({
                "stride_index": i,
                "stride_time": t,
                "stride_type": stride_type,
                "gps_distance_m": gps_dist,
                "time_window_start": start,
                "time_window_end": end,
                "imu_window": imu_data,
                "gps_window": gps_data
            })

        return results

    def compute_kalman_gps_error_and_jumps(self, time, pos_kalman, gps_pos, threshold=7.0, min_separation=0.5):
        """
        Compute Kalman-GPS error and detect GPS jumps.

        :param time: Time vector.
        :type time: np.ndarray
        :param pos_kalman: Kalman-filtered positions (Nx2 or Nx3).
        :type pos_kalman: np.ndarray
        :param gps_pos: GPS positions (Nx2).
        :type gps_pos: np.ndarray
        :param threshold: Threshold for jump detection.
        :type threshold: float
        :param min_separation: Minimum time separation between jumps.
        :type min_separation: float
        :return: Tuple (error array, list of jump indices).
        :rtype: Tuple[np.ndarray, List[int]]
        """
        pos_kalman = np.asarray(pos_kalman)
        gps_pos = np.asarray(gps_pos)
        time = np.asarray(time)

        if pos_kalman.ndim == 1 or pos_kalman.shape[1] < 2:
            raise ValueError("pos_kalman must have at least 2 columns (X, Y).")
        if gps_pos.ndim == 1 or gps_pos.shape[1] != 2:
            raise ValueError("gps_pos must be a (N x 2) matrix.")
        if len(pos_kalman) != len(gps_pos) or len(pos_kalman) != len(time):
            raise ValueError("time, pos_kalman, and gps_pos must have the same length.")

        kalman_gps_error = np.linalg.norm(pos_kalman[:, :2] - gps_pos, axis=1)
        jump_indices = np.where(kalman_gps_error > threshold)[0]

        filtered_jump_indices = []
        for idx in jump_indices:
            t = time[idx]
            if not filtered_jump_indices or (t - time[filtered_jump_indices[-1]]) > min_separation:
                filtered_jump_indices.append(idx)

        return kalman_gps_error, filtered_jump_indices

    def plot_gps_jumps(self, time, kalman_gps_error, jump_indices, threshold=7.0):
        """
        Plot GPS jumps based on Kalman-GPS error.

        :param time: Time vector.
        :type time: np.ndarray
        :param kalman_gps_error: Error vector.
        :type kalman_gps_error: np.ndarray
        :param jump_indices: Indices of GPS jumps.
        :type jump_indices: List[int]
        :param threshold: Threshold line to plot.
        :type threshold: float
        """
        plt.figure(figsize=(10, 5))
        plt.plot(time, kalman_gps_error, label="Kalman-GPS Error", color='blue')
        plt.axhline(threshold, color='red', linestyle='--', label=f"Threshold ({threshold} m)")
        plt.scatter(time[jump_indices], kalman_gps_error[jump_indices],
                    color='red', edgecolor='black', label="Filtered Jumps")
        plt.xlabel("Time (s)")
        plt.ylabel("Kalman vs GPS Error (m)")
        plt.title("GPS Jump Detection based on Kalman-GPS Error")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    def plot_jumps_and_strides(self, time, kalman_gps_error, jump_times, stride_times, valid_strides, threshold = 7.0):
        """
        Plot Kalman-GPS error with marked GPS jumps and stride events.

        :param time: Time vector.
        :type time: np.ndarray
        :param kalman_gps_error: Kalman-GPS error values.
        :type kalman_gps_error: np.ndarray
        :param jump_times: List of GPS jump timestamps.
        :type jump_times: list[float]
        :param stride_times: Dict mapping stride numbers to their times.
                            e.g., {30: 39.10, 31: 40.08, ...}
        :type stride_times: dict[int, float]
        :param valid_strides: List of stride numbers considered valid.
        :type valid_strides: list[int]
        :param threshold: Threshold value for GPS error.
        :type threshold: float
        """
        plt.figure(figsize=(12, 6))
        plt.plot(time, kalman_gps_error, label='Kalman-GPS Error', color='blue')

        # Plot GPS jumps
        for jt in jump_times:
            y = kalman_gps_error[np.argmin(np.abs(time - jt))]
            plt.plot(jt, y, 'ro', label=f"GPS Jump at {jt:.2f}s")
            plt.text(jt + 0.2, y + 0.5, f"Jump\n{jt:.2f}s", fontsize=9, color='red')

        # Plot stride lines
        for stride_num, t in stride_times.items():
            if stride_num in valid_strides:
                color = 'green'
                label = f"Valid Stride #{stride_num} ({t:.2f}s)"
            else:
                color = 'red'
                label = f"Invalid Stride #{stride_num} ({t:.2f}s)"
            plt.axvline(t, color=color, linestyle='--', label=label)

        # Plot threshold
        plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold} m)')

        plt.title("Kalman-GPS Error and Stride Events")
        plt.xlabel("Time (s)")
        plt.ylabel("Error (m)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
