import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy import signal
import yaml
from typing import Tuple
from ahrs.filters import Madgwick, Mahony
from ahrs.common.orientation import q_conj, q_rot, axang2quat
from matplotlib import pyplot as plt
from pyproj import Proj, Transformer
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import plotly.graph_objects as go
import folium
from folium.plugins import PolyLineTextPath
from folium import FeatureGroup, LayerControl
import os
from scipy.signal import find_peaks


class DataPreprocessor:
    """
    Class for loading configuration and Excel data, resampling, and preprocessing sensor data.
    """

    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """
        Load YAML configuration file from the specified path.

        :param config_path: Path to the YAML configuration file.
        :type config_path: str
        :return: Dictionary containing configuration data.
        :rtype: dict
        """
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def load_data(self, file_path):
        """
        Load Excel data from the specified path.

        :param file_path: Path to the Excel file.
        :type file_path: str
        :return: DataFrame containing raw data.
        :rtype: pd.DataFrame
        """
        df = pd.read_excel(file_path)
        return df

    def resample_to_40hz(self, df, time_col = '_time', freq_hz = 40, gap_threshold_ms = 200):
        """
        Resample data to the target frequency handling session gaps.

        :param df: Raw DataFrame.
        :type df: pd.DataFrame
        :param time_col: Name of the time column.
        :type time_col: str
        :param freq_hz: Target frequency in Hz.
        :type freq_hz: int
        :param gap_threshold_ms: Gap threshold to split sessions.
        :type gap_threshold_ms: int
        :return: Interpolated DataFrame.
        :rtype: pd.DataFrame
        """
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)
        df['delta'] = df[time_col].diff().dt.total_seconds() * 1000
        df['session'] = (df['delta'] > gap_threshold_ms).cumsum()

        interpolated = []
        for session_id, group in df.groupby('session'):
            group = group.set_index(time_col)
            group = group.sort_index()
            new_index = pd.date_range(start=group.index[0], end=group.index[-1], freq=f'{int(1000/freq_hz)}ms')
            df_interp = pd.DataFrame(index=new_index)

            for col in group.columns.difference(['delta', 'session']):
                clean = group[col].dropna()
                if len(clean) >= 4:
                    t = (clean.index - clean.index[0]).total_seconds().to_numpy()
                    y = clean.to_numpy()
                    cs = CubicSpline(t, y)
                    t_new = (new_index - clean.index[0]).total_seconds().to_numpy()
                    df_interp[col] = cs(t_new)
                else:
                    df_interp[col] = np.nan

            df_interp.reset_index(inplace=True)
            df_interp.rename(columns={'index': time_col}, inplace=True)
            df_interp['session'] = session_id
            interpolated.append(df_interp)

        result = pd.concat(interpolated, ignore_index=True)
        result.dropna(inplace=True)
        return result

    def preprocess_data(self, df):
        """
        Process DataFrame to compute time, sample rate, and sensor arrays.

        :param df: Preprocessed DataFrame.
        :type df: pd.DataFrame
        :return: Tuple of time array, sample rate, gyroscope, accelerometer, magnetometer arrays, sample period and filtered GPS DataFrame..
        :rtype: tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float, pd.DataFrame]
        """
        ...
        df['time'] = (df['_time'] - df['_time'].iloc[0]).dt.total_seconds()
        time = df['time'].to_numpy()
        sample_period = np.mean(np.diff(time))
        sample_rate = 1.0 / sample_period

        gyr = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180
        acc = df[['Ax', 'Ay', 'Az']].to_numpy() 
        mag = df[['Mx', 'My', 'Mz']].to_numpy() * 100
        
        

        return time, sample_rate, gyr, acc, mag
    
    def compute_positions(self, df, config):
        """
        Convert GPS coordinates to local Cartesian positions using projection configuration.

        :param df: DataFrame with 'lat' and 'lng' columns.
        :type df: pd.DataFrame
        :param config: Configuration dictionary containing 'Location' section with projection params.
        :type config: dict
        :return: Tuple of GPS position array and final GPS position.
        :rtype: tuple[np.ndarray, np.ndarray]
        :raises KeyError: If required projection parameters are missing in config['Location'].
        """
        location_cfg = config["Location"]
        proj = Proj(
            proj=location_cfg["proj"],
            zone=location_cfg["zone"],
            ellps=location_cfg["ellps"],
            south=location_cfg["south"]
        )

        df_gps = df[['lat', 'lng', 'time']].dropna().reset_index(drop=True)

        lat = df_gps['lat'].to_numpy()
        lng = df_gps['lng'].to_numpy()
        
        x, y = proj(lng, lat)
        gps_pos = np.stack((x - x[0], y - y[0]), axis=1)
        return df_gps,gps_pos, gps_pos[-1]



class IMUProcessor:
    """
    Class for processing IMU data: gravity estimation, motion detection, and position estimation.
    """

    def __init__(self):
        pass

    def estimate_gravity_vector(self, acc, alpha = 0.9):
        """
        Estimate normalized gravity vector from accelerometer data using exponential moving average.

        :param acc: Acceleration data array with shape (N, 3).
        :type acc: np.ndarray
        :param alpha: Smoothing coefficient (0 < alpha < 1).
        :type alpha: float
        :return: Normalized gravity vectors.
        :rtype: np.ndarray
        :raises ValueError: If input shape is invalid or alpha is out of bounds.
        """
        if acc.ndim != 2 or acc.shape[1] != 3:
            raise ValueError("Input array must have shape (N, 3) with columns [Ax, Ay, Az].")
        if not (0 < alpha < 1):
            raise ValueError("Parameter alpha must be in the range (0, 1).")

        gravity = np.zeros_like(acc)
        gravity[0] = acc[0]

        for i in range(1, len(acc)):
            gravity[i] = alpha * gravity[i - 1] + (1 - alpha) * acc[i]

        # Normalización segura
        norm = np.linalg.norm(gravity, axis=1, keepdims=True)
        norm[norm == 0] = 1.0  # Evita división por cero
        gravity_normalized = gravity / norm

        return gravity_normalized

    def detect_stationary(self, acc, sample_rate):
        """
        Detect stationary periods from accelerometer data using filtering and thresholding.

        :param acc: Acceleration data array with shape (N, 3).
        :type acc: np.ndarray
        :param sample_rate: Sampling rate of the accelerometer signal in Hz.
        :type sample_rate: float
        :return: A tuple containing stationary: Boolean array indicating stationary (True) or moving (False) states.
        :rtype: tuple[np.ndarray]
        :raises ValueError: If input shape is invalid or sample_rate is non-positive.
        """
        acc_mag = np.linalg.norm(acc, axis=1)
        acc_mag_clipped = np.clip(acc_mag, 0, 20)

        b, a = signal.butter(1, 0.01 / (sample_rate / 2), 'highpass')
        acc_hp = signal.filtfilt(b, a, acc_mag_clipped)

        b, a = signal.butter(1, 5.0 / (sample_rate / 2), 'lowpass')
        acc_lp = signal.filtfilt(b, a, np.abs(acc_hp))

        threshold = np.percentile(acc_lp, 15)
        stationary = acc_lp < threshold

        return stationary

    # Agregación de ZUPH
    def detect_no_rotation(self, gyr, threshold = 0.05, duration_samples = 5):
        """
        Detect periods where there is negligible angular velocity on Z-axis (i.e., no yaw rotation).

        :param gyr: Gyroscope data array of shape (N, 3), in rad/s.
        :type gyr: np.ndarray
        :param threshold: Threshold for Z-axis angular velocity (in rad/s) to define no rotation.
        :type threshold: float
        :param duration_samples: Minimum number of consecutive samples below the threshold to validate no rotation.
        :type duration_samples: int
        :return: Boolean array of shape (N,), where True indicates no rotation around the Z-axis.
        :rtype: np.ndarray
        """

        gz = np.abs(gyr[:, 2])  # Only Z axis
        mask = gz < threshold
        stable = np.copy(mask)
        for i in range(len(mask)):
            if not mask[i]:
                continue
            if i + duration_samples <= len(mask) and np.all(mask[i:i + duration_samples]):
                stable[i:i + duration_samples] = True
        return stable

    def estimate_position_generic(self, method, use_mag, gyr, acc, mag, time, sample_rate, stationary):
        """
        Estimate the 2D position from IMU sensor data using either the Madgwick or Mahony filter.

        This method computes orientation quaternions using an AHRS filter, rotates the 
        acceleration to the earth frame, removes gravity, integrates to velocity and 
        position, and compensates for drift during stationary periods.

        :param method: Algorithm to use ('madgwick' or 'mahony').
        :type method: str
        :param use_mag: Whether to use magnetometer data (True for MARG, False for IMU-only).
        :type use_mag: bool
        :param gyr: Gyroscope data array of shape (N, 3), in rad/s.
        :type gyr: np.ndarray
        :param acc: Accelerometer data array of shape (N, 3), in m/s^2.
        :type acc: np.ndarray
        :param mag: Magnetometer data array of shape (N, 3), in arbitrary units.
        :type mag: np.ndarray
        :param time: Time vector of shape (N,), in seconds.
        :type time: np.ndarray
        :param sample_rate: Sampling frequency in Hz.
        :type sample_rate: float
        :param stationary: Boolean array of shape (N,) indicating stationary periods.
        :type stationary: np.ndarray
        :return: Estimated position array of shape (N, 3).
        :rtype: np.ndarray
        :raises ValueError: If the method is not recognized.
        """

        if method == "madgwick":
            base_gain = 0.005
            filter_ = Madgwick(frequency=sample_rate, gain=base_gain)
            q = axang2quat([0, 0, 1], np.deg2rad(45))
        elif method == "mahony":
            base_kp = 1.5
            filter_ = Mahony(Kp=base_kp, Ki=0.01, frequency=sample_rate)
            q = np.array([1.0, 0.0, 0.0, 0.0])
            acc_init = np.median(acc[time <= time[0] + 2], axis=0)
            for _ in range(2000):
                q = filter_.updateIMU(q, gyr=np.zeros(3), acc=acc_init)
        else:
            raise ValueError("Unknown method. Use 'madgwick' or 'mahony'.")

        quats = np.zeros((len(time), 4))
        no_rotation = self.detect_no_rotation(gyr)
        no_motion = stationary & no_rotation

        for t in range(len(time)):
            if method == "madgwick":
                filter_.gain = 0.001 if no_motion[t] else base_gain
                q_new = filter_.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t]) if use_mag else filter_.updateIMU(q, gyr=gyr[t], acc=acc[t])
            else:
                filter_.Kp = base_kp * 0.1 if no_motion[t] else (base_kp if stationary[t] else 0.0)
                q_new = filter_.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t]) if use_mag else filter_.updateIMU(q, gyr=gyr[t], acc=acc[t])
            if q_new is not None:
                q = q_new
            quats[t] = q

        acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)])
        acc_earth -= self.estimate_gravity_vector(acc, 0.95)
        acc_earth *= 9.81

        vel = np.zeros_like(acc_earth)
        for t in range(1, len(vel)):
            vel[t] = vel[t - 1] + acc_earth[t] * (1 / sample_rate)
            if stationary[t]:
                vel[t] = 0

        drift = np.zeros_like(vel)
        starts = np.where(np.diff(stationary.astype(int)) == -1)[0] + 1
        ends = np.where(np.diff(stationary.astype(int)) == 1)[0] + 1
        for s, e in zip(starts, ends):
            drift_rate = vel[e - 1] / (e - s)
            drift[s:e] = np.outer(np.arange(e - s), drift_rate)
        vel -= drift

        pos = np.zeros_like(vel)
        for t in range(1, len(pos)):
            pos[t] = pos[t - 1] + vel[t] * (1 / sample_rate)

        return pos


class KalmanProcessor:
    """
    Class for applying Kalman filters to fuse IMU and GPS position data.
    """

    def apply_filter1(self, pos_imu, gps_pos, sample_rate):
        """
        Apply a Kalman filter using filterpy to fuse IMU and GPS data.

        :param pos_imu: IMU-estimated positions array of shape (N, 2).
        :type pos_imu: np.ndarray
        :param gps_pos: GPS positions array of shape (N, 2).
        :type gps_pos: np.ndarray
        :param sample_rate: Sampling rate in Hz.
        :type sample_rate: float
        :return: Kalman-filtered fused positions of shape (N, 2).
        :rtype: np.ndarray
        """

        n = pos_imu.shape[0]
        dt = 1.0 / sample_rate

        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1,  0],
                            [0, 0, 0,  1]])
        kf.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        kf.R = np.eye(2) * 20.0
        kf.P = np.eye(4) * 10.0
        kf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=2)

        # Estado inicial: posición y velocidad estimada por IMU
        vel_imu = np.gradient(pos_imu, dt, axis=0)
        kf.x = np.zeros((4, 1))
        kf.x[:2] = pos_imu[0, :2].reshape(2, 1)
        kf.x[2:] = vel_imu[0, :2].reshape(2, 1)

        fused = np.zeros((n, 2))
        for i in range(n):
            # Predicción basada en modelo (usa estado anterior)
            kf.predict()

            # Corrección basada en GPS si disponible
            if i < gps_pos.shape[0]:
                z = gps_pos[i].reshape(2, 1)
                kf.update(z)

            fused[i] = kf.x[:2, 0]

        return fused


    def apply_filter2(self, pos_imu, gps_pos, time):
        """
        Apply a 2D Kalman filter to adjust IMU trajectory using GPS corrections.

        :param pos_imu: IMU-estimated positions array of shape (N, 3).
        :type pos_imu: np.ndarray
        :param gps_pos: GPS projected positions array of shape (N, 2).
        :type gps_pos: np.ndarray
        :param time: Time vector array of shape (N,).
        :type time: np.ndarray
        :return: Kalman-corrected position array of shape (N, 3).
        :rtype: np.ndarray
        """
        dt = np.mean(np.diff(time))
        n = len(time)

        # Estado: [x, y, vx, vy]
        X = np.zeros((4, n))
        X[:2, 0] = gps_pos[0]  # posición inicial GPS
        X[2:, 0] = 0  # velocidad inicial 0

        # Matriz de transición del estado
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0 ],
            [0, 0, 0, 1 ]
        ])

        # Matriz de observación (solo medimos x, y del GPS)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Matrices de covarianza
        Q = np.eye(4) * 0.05   # Ruido del proceso (más alto si confías poco en la IMU)
        R = np.eye(2) * 5.0    # Ruido de la medición GPS
        P = np.eye(4) * 1.0    # Inicialización de incertidumbre

        estimated = np.zeros((n, 4))
        for k in range(1, n):
            # Predicción
            X[:, k] = F @ X[:, k - 1]
            P = F @ P @ F.T + Q

            # Solo corregimos si hay dato GPS
            if k < len(gps_pos):
                Z = gps_pos[k]  # medición GPS
                y = Z - H @ X[:, k]  # error de innovación
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)  # ganancia de Kalman
                X[:, k] += K @ y
                P = (np.eye(4) - K @ H) @ P

            estimated[k] = X[:, k]

        pos_kalman = np.copy(pos_imu)
        pos_kalman[:, 0] = estimated[:, 0]
        pos_kalman[:, 1] = estimated[:, 1]
        return pos_kalman
    



class PlotProcessor:
    """
    Class for handling various types of trajectory visualizations.
    """

    def plot_trajectories_interactive(self, resultados, errores, gps_pos, gps_final, title="Trajectory Comparison", save_path=None):
        """
        Create an interactive Plotly plot comparing estimated and GPS trajectories.

        :param resultados: Dictionary of estimated positions.
        :type resultados: dict[str, np.ndarray]
        :param errores: Dictionary of position errors.
        :type errores: dict[str, float]
        :param gps_pos: GPS positions array.
        :type gps_pos: np.ndarray
        :param gps_final: Final GPS position.
        :type gps_final: np.ndarray
        :param title: Plot title.
        :type title: str
        :param save_path: Optional path to save the HTML plot.
        :type save_path: str or None
        """
        fig = go.Figure()
        for name, pos in resultados.items():
            fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode='lines', name=f"{name} ({errores[name]:.2f} m)"))

        fig.add_trace(go.Scatter(x=gps_pos[:, 0], y=gps_pos[:, 1], mode='lines', name="GPS (reference)", line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=[gps_final[0]], y=[gps_final[1]], mode='markers', name="GPS final", marker=dict(color='black', size=10)))

        fig.update_layout(title=title, xaxis_title='X (m)', yaxis_title='Y (m)', width=900, height=700)
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive plot saved to: {save_path}")

    def plot_trajectories_all(self, all_results, gps_pos, gps_final, title="Trajectory Comparison"):
        """
        Plot all estimated trajectories together using Matplotlib.

        :param all_results: Dictionary with keys as method names and values as (positions, error).
        :type all_results: dict[str, tuple[np.ndarray, float]]
        :param gps_pos: GPS positions array.
        :type gps_pos: np.ndarray
        :param gps_final: Final GPS position.
        :type gps_final: np.ndarray
        :param title: Plot title.
        :type title: str
        """
        plt.figure(figsize=(10, 8))
        for label, (pos, err) in all_results.items():
            linestyle = '--' if 'Kalman' in label else '-'
            alpha = 0.6 if 'Kalman' not in label else 1.0
            plt.plot(pos[:, 0], pos[:, 1], linestyle, alpha=alpha, label=f"{label} ({err:.2f} m)")
        plt.plot(gps_pos[:, 0], gps_pos[:, 1], 'k--', label="GPS (reference)")
        plt.plot(gps_final[0], gps_final[1], 'ko', label="GPS final")
        plt.title(title)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.legend()
        plt.grid()

    def plot_trajectories_split(self, resultados, errores, gps_pos, gps_final, title, save_path=None):
        """
        Plot each estimated trajectory separately using Matplotlib.

        :param resultados: Dictionary of estimated positions.
        :type resultados: dict[str, np.ndarray]
        :param errores: Dictionary of errors per method.
        :type errores: dict[str, float]
        :param gps_pos: GPS positions array.
        :type gps_pos: np.ndarray
        :param gps_final: Final GPS point.
        :type gps_final: np.ndarray
        :param title: Plot title.
        :type title: str
        :param save_path: Optional path to save plot.
        :type save_path: str or None
        """
        plt.figure(figsize=(10, 8))
        for name, pos in resultados.items():
            plt.plot(pos[:, 0], pos[:, 1], label=f"{name} ({errores[name]:.2f} m)")
        plt.plot(gps_pos[:, 0], gps_pos[:, 1], 'k--', label="GPS (reference)")
        plt.plot(gps_final[0], gps_final[1], 'ko', label="GPS final")
        plt.title(title)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.legend()
        plt.grid()
        if save_path:
            plt.savefig(save_path)

    def generate_map_with_estimates(self, df_gps, resultados, output_html_path, config):
        """
        Generates an interactive map with GPS and estimated IMU/Kalman trajectories using Folium.

        :param df_gps: DataFrame with 'lat' and 'lng' columns.
        :type df_gps: pd.DataFrame
        :param resultados: Dictionary of trajectory name to estimated XY positions.
        :type resultados: dict[str, np.ndarray]
        :param output_html_path: Path to save HTML map.
        :type output_html_path: str
        :param config: YAML configuration with 'Location' projection info.
        :type config: dict
        """
        location_cfg = config["Location"]
        proj = Proj(proj=location_cfg["proj"], zone=location_cfg["zone"], ellps=location_cfg["ellps"], south=location_cfg["south"])
        ref_code = location_cfg["code"]
        transformer = Transformer.from_proj(proj, ref_code, always_xy=True)

        lat0, lon0 = df_gps.loc[0, 'lat'], df_gps.loc[0, 'lng']
        fmap = folium.Map(location=[lat0, lon0], zoom_start=18)

        gps_coords = df_gps[['lat', 'lng']].values.tolist()
        gps_group = folium.FeatureGroup(name='GPS (reference)')
        folium.PolyLine(gps_coords, color='grey', weight=4).add_to(gps_group)
        folium.Marker(location=gps_coords[0], popup="Start", icon=folium.Icon(color='green')).add_to(gps_group)
        folium.Marker(location=gps_coords[-1], popup="End", icon=folium.Icon(color='red')).add_to(gps_group)
        gps_group.add_to(fmap)

        color_list = ['cadetblue','#F04BF0','#FAA43A', "#056641",'#F17CB0','#DECF3F','#F15854','#5DA5DA']
        for i, (name, traj) in enumerate(resultados.items()):
            x_coords = traj[:, 0]
            y_coords = traj[:, 1]
            lon_est, lat_est = transformer.transform(
                x_coords + proj(lon0, lat0)[0],
                y_coords + proj(lon0, lat0)[1]
            )
            path = list(zip(lat_est, lon_est))
            color = color_list[i % len(color_list)]
            group = folium.FeatureGroup(name=name)
            folium.PolyLine(path, color=color, weight=3).add_to(group)
            group.add_to(fmap)

        folium.LayerControl(collapsed=False).add_to(fmap)
        fmap.save(output_html_path)
        print(f"Interactive map saved to: {output_html_path}")



    def plot_trajectories(self, resultados, errores, gps_pos, gps_final, title="Trajectory Comparison", save_path=None):
        """
        Plot estimated and GPS trajectories.

        :param resultados: Dictionary of estimated positions.
        :type resultados: dict[str, np.ndarray]
        :param errores: Dictionary of position errors.
        :type errores: dict[str, float]
        :param gps_pos: GPS positions array.
        :type gps_pos: np.ndarray
        :param gps_final: Final GPS position.
        :type gps_final: np.ndarray
        :param title: Title for the plot.
        :type title: str
        :param save_path: If given, path to save the plot image.
        :type save_path: str or None
        """
        plt.figure(figsize=(10, 8))
        for name, pos in resultados.items():
            plt.plot(pos[:, 0], pos[:, 1], label=f"{name} ({errores[name]:.2f} m)")
        plt.plot(gps_pos[:, 0], gps_pos[:, 1], 'k--', label="GPS (reference)")
        plt.plot(gps_final[0], gps_final[1], 'ko', label="GPS final")
        plt.title(title)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.legend()
        plt.grid()
        if save_path:
            plt.savefig(save_path)

        
    def plot_results_madwick(self,time, pos, vel, gps_pos=None, output_dir=None, title="Trajectory Comparison", base_name="trajectory", verbose=3, traj_label="IMU" ):
        """
        Plot diagnostic and trajectory figures for IMU data, and optionally save them.

        This function generates several plots based on filtered acceleration, position,
        velocity, 2D and 3D trajectory. If GPS data is provided, it includes a comparative plot. 
        Depending on the verbosity level and output directory, plots can also be saved.

        :param time: Array of timestamps in seconds.
        :type time: np.ndarray
        :param acc_lp: Low-pass filtered acceleration magnitude.
        :type acc_lp: np.ndarray
        :param threshold: Threshold value used to detect stationary periods.
        :type threshold: float
        :param pos: Estimated position array of shape (N, 3).
        :type pos: np.ndarray
        :param vel: Estimated velocity array of shape (N, 3).
        :type vel: np.ndarray
        :param gps_pos: Optional GPS position array of shape (N, 2). Defaults to None.
        :type gps_pos: np.ndarray or None
        :param output_dir: Directory to save plots. If None, figures are displayed but not saved.
        :type output_dir: str or None
        :param title: Title used in trajectory comparison plots.
        :type title: str
        :param base_name: Base filename for saving output plots (without extension).
        :type base_name: str
        :param verbose: Verbosity level:
                        - 1: No plots
                        - 2: Show/save trajectory plots
                        - 3: Show/save all plots (acceleration, velocity, etc.)
        :type verbose: int
        """
        def save_figure(title):
            if output_dir:
                filename = f"{base_name}_{title.lower().replace(' ', '_')}.png"
                path = os.path.join(output_dir, filename)
                plt.savefig(path, bbox_inches='tight')

        # Filtered acceleration
        if verbose == 3:
            # Position
            plt.figure(figsize=(15, 5))
            plt.plot(time, pos[:, 0], 'r', label='x')
            plt.plot(time, pos[:, 1], 'g', label='y')
            plt.plot(time, pos[:, 2], 'b', label='z')
            plt.title("Position")
            plt.xlabel("Time (s)")
            plt.ylabel("Position (m)")
            plt.legend()
            plt.grid()
            save_figure("Position")

            # Velocity
            plt.figure(figsize=(15, 5))
            plt.plot(time, vel[:, 0], 'r', label='x')
            plt.plot(time, vel[:, 1], 'g', label='y')
            plt.plot(time, vel[:, 2], 'b', label='z')
            plt.title("Velocity")
            plt.xlabel("Time (s)")
            plt.ylabel("Velocity (m/s)")
            plt.legend()
            plt.grid()
            save_figure("Velocity")

            # XY Trajectory
            plt.figure()
            plt.plot(pos[:, 0], pos[:, 1])
            plt.axis('equal')
            plt.title("XY Trajectory")
            plt.grid()
            save_figure("XY Trajectory")

            # 3D Trajectory
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])
            ax.set_title("3D Trajectory")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            if output_dir:
                filename = f"{base_name}_3d_trajectory.png"
                path = os.path.join(output_dir, filename)
                plt.savefig(path, bbox_inches='tight')

        # IMU vs GPS comparison (always in verbose 2 or 3)
        if gps_pos is not None and verbose >= 2:
            plt.figure(figsize=(10, 8))
            plt.plot(pos[:, 0], pos[:, 1], label=f'{traj_label} Trajectory')
            plt.plot(gps_pos[:, 0], gps_pos[:, 1], 'k--', label='GPS Reference')
            plt.plot(gps_pos[-1, 0], gps_pos[-1, 1], 'ko', label='Final GPS')
            plt.title(title)
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.axis("equal")
            plt.grid()
            plt.legend()
            save_figure(f"Trajectory Comparison ({traj_label} vs GPS)")




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







class PositionVelocityEstimator:
    """
    Estimates orientation, acceleration, velocity, and position from IMU sensors (gyroscope, accelerometer, magnetometer).
    Uses the Madgwick filter with adaptive gain for ZUPH and velocity corrections using ZUPT.
    """
    def __init__(self, sample_rate, sample_period):
        self.sample_rate = sample_rate
        self.sample_period = sample_period
        self.base_gain = 0.041
        self.low_gain = 0.001

    def estimate_orientation_and_position(self, time, gyr, acc, mag, stationary):
        """
        Estimate orientation, linear acceleration, velocity, and position from sensor data using a Madgwick filter
        and zero-velocity updates.

        Applies orientation estimation with adaptive gain based on motion state (ZUPH),
        and velocity correction during stationary periods (ZUPT).

        :param time: Array of time stamps.
        :type time: np.ndarray
        :param gyr: Gyroscope data array with shape (N, 3), in rad/s.
        :type gyr: np.ndarray
        :param acc: Accelerometer data array with shape (N, 3), in m/s².
        :type acc: np.ndarray
        :param mag: Magnetometer data array with shape (N, 3), in µT.
        :type mag: np.ndarray
        :param sample_period: Time between samples, in seconds.
        :type sample_period: float
        :param sample_rate: Sampling rate in Hz.
        :type sample_rate: float
        :param stationary: Boolean array indicating stationary states (True for stationary).
        :type stationary: np.ndarray
        :return: Tuple (quats, acc_earth, vel, pos) where:
                - quats: Quaternion orientation estimates.
                - acc_earth: Linear acceleration in earth frame (gravity removed).
                - vel: Estimated velocity with ZUPT correction.
                - pos: Estimated position.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        :raises ValueError: If input shapes are inconsistent.
        """
        madgwick = Madgwick(frequency=self.sample_rate, gain=self.base_gain)
        q = np.array([1.0, 0.0, 0.0, 0.0])
        quats = np.zeros((len(time), 4))
        quats[0] = q

        gyro_norm = np.linalg.norm(gyr, axis=1)
        no_rotation = gyro_norm < 0.1
        no_motion = stationary & no_rotation

        for t in range(1, len(time)):
            madgwick.gain = self.low_gain if no_motion[t] else self.base_gain
            q = madgwick.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t])
            quats[t] = q

        imu_proc = IMUProcessor()
        gravity = imu_proc.estimate_gravity_vector(acc, 0.95)

        acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)])
        acc_earth -= gravity
        acc_earth *= 9.81
        
        vel = np.zeros_like(acc_earth)
        for t in range(1, len(vel)):
            vel[t] = vel[t - 1] + acc_earth[t] * self.sample_period
            if stationary[t] and not stationary[t - 1]:
                vel[t] = 0  

        vel_drift = np.zeros_like(vel)
        starts = np.where(np.diff(stationary.astype(int)) == -1)[0] + 1
        ends = np.where(np.diff(stationary.astype(int)) == 1)[0] + 1
        for s, e in zip(starts, ends):
            if e > s:
                drift_rate = vel[e - 1] / (e - s)
                vel_drift[s:e] = np.outer(np.arange(e - s), drift_rate)
        vel -= vel_drift

        pos = np.zeros_like(vel)
        for t in range(1, len(pos)):
            pos[t] = pos[t - 1] + vel[t] * self.sample_period

        return quats, acc_earth, vel, pos



class SensorFusionFilters:
    """
    Class containing filtering and correction methods for sensor fusion,
    including an Extended Kalman Filter (EKF), complementary filter, and linear drift correction.
    """

    def __init__(self, alpha: float = 0.98):
        """
        Initialize the filter with a default weighting factor for complementary fusion.

        :param alpha: Weighting factor for the complementary filter (0 < alpha < 1).
        """
        self.alpha = alpha



    

    def ekf_fusion_2d(self,gps_pos, imu_acc, time):
        """
        Extended Kalman Filter (EKF) for 2D position estimation using GPS and IMU acceleration.

        :param gps_pos: GPS positions, shape (N, 2).
        :type gps_pos: np.ndarray
        :param imu_acc: IMU accelerations, shape (N, 2).
        :type imu_acc: np.ndarray
        :param time: Time stamps, shape (N,).
        :type time: np.ndarray
        :return: Estimated 2D positions, shape (N, 2).
        :rtype: np.ndarray
        """
        N = len(time)
        dt = np.mean(np.diff(time))

        # Initial state: [x, y, vx, vy, ax, ay]
        x = np.array([gps_pos[0, 0], gps_pos[0, 1], 0.0, 0.0, imu_acc[0, 0], imu_acc[0, 1]])
        P = np.eye(6) * 10.0

        Q = np.diag([0.05, 0.05, 0.05, 0.05, 0.1, 0.1])  # Process noise
        R = np.eye(2) * 3.0                             # Measurement noise

        estimates = np.zeros((N, 6))

        for k in range(1, N):
            u = imu_acc[k]

            # Prediction step
            x_pred = np.zeros(6)
            x_pred[0] = x[0] + x[2]*dt + 0.5*u[0]*dt**2
            x_pred[1] = x[1] + x[3]*dt + 0.5*u[1]*dt**2
            x_pred[2] = x[2] + u[0]*dt
            x_pred[3] = x[3] + u[1]*dt
            x_pred[4] = 0.9 * x[4] + 0.1 * u[0]  # Smooth update of acceleration
            x_pred[5] = 0.9 * x[5] + 0.1 * u[1]

            # Jacobian of motion model
            F = np.array([
                [1, 0, dt, 0, 0.5*dt**2, 0],
                [0, 1, 0, dt, 0, 0.5*dt**2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])

            P = F @ P @ F.T + Q

            # Correction step
            H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0]
            ])
            z = gps_pos[k]
            y = z - H @ x_pred
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x_pred + K @ y
            P = (np.eye(6) - K @ H) @ P

            # Optional: clip extreme values
            x = np.clip(x, -1e6, 1e6)

            estimates[k] = x

        return estimates[:, :2]

    def complementary_filter(self,pos_imu, gps_pos):
        """
        Fuse IMU and GPS positions using a complementary filter.

        :param pos_imu: IMU-estimated positions, shape (N, 3).
        :type pos_imu: np.ndarray
        :param gps_pos: GPS positions, shape (N, 2).
        :type gps_pos: np.ndarray
        :param alpha: Weighting factor for IMU data (0 < alpha < 1). Default is 0.98.
        :type alpha: float
        :return: Fused position estimate, shape (N, 3).
        :rtype: np.ndarray
        """
        fused = np.zeros_like(pos_imu)
        fused[:, :2] = self.alpha * pos_imu[:, :2] + (1 - self.alpha) * gps_pos
        fused[:, 2] = pos_imu[:, 2]  # Retain IMU Z-axis
        return fused


        
    def linear_drift_correction(self,pos_imu, gps_start, gps_end):
        """
        Apply linear correction to IMU position to reduce drift between known GPS start and end points.

        The correction is applied progressively across the time series to match GPS anchors.

        :param pos_imu: IMU-estimated positions (N, 3).
        :type pos_imu: np.ndarray
        :param gps_start: 2D GPS start position (2,).
        :type gps_start: np.ndarray
        :param gps_end: 2D GPS end position (2,).
        :type gps_end: np.ndarray
        :return: Drift-corrected IMU position array (N, 3).
        :rtype: np.ndarray
        """
        N = len(pos_imu)
        imu_start = pos_imu[0, :2]
        imu_end = pos_imu[-1, :2]
        drift = (gps_end - gps_start) - (imu_end - imu_start)
        correction = np.linspace(0, 1, N).reshape(-1, 1) * drift
        corrected = pos_imu.copy()
        corrected[:, :2] += correction
        return corrected



import numpy as np

class KalmanFilter2D:
    def __init__(self, dt, q, r):
        # dt: tiempo entre muestras
        # q: covarianza del ruido del proceso (modelo de movimiento del IMU)
        # r: covarianza del ruido de la medición (GPS)

        # Estado inicial (ej. [x, y, vx, vy])
        self.state = np.zeros(4)
        # Covarianza del estado
        self.P = np.eye(4) * 1000 # Gran incertidumbre inicial

        # Matriz de Transición de Estado (F) - Modelo de movimiento (ej. Posición = Posición anterior + Velocidad*dt)
        # Asumiendo un modelo de velocidad constante o aceleración constante
        self.F = np.array([
            [1, 0, dt, 0],  # x = x + vx*dt
            [0, 1, 0, dt],  # y = y + vy*dt
            [0, 0, 1, 0],   # vx = vx
            [0, 0, 0, 1]    # vy = vy
        ])

        # Matriz de Control (B) - Si aplicas entradas de control (aceleraciones del IMU)
        # Si usas aceleraciones del IMU para predecir, B sería:
        self.B = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]
        ])
        # Y necesitarías pasar 'acc' a la predicción. 

        # Matriz de Ruido del Proceso (Q) - Incertidumbre en nuestro modelo de movimiento
        self.Q = np.eye(4) * q

        # Matriz de Observación (H) - Cómo las mediciones se relacionan con el estado (GPS mide [x, y])
        self.H = np.array([
            [1, 0, 0, 0],  # Medición de x
            [0, 1, 0, 0]   # Medición de y
        ])

        # Matriz de Ruido de la Medición (R) - Incertidumbre del sensor GPS
        self.R = np.eye(2) * r

    def initialize(self, initial_gps_pos):
        # Inicializa el estado con la primera posición del GPS. Velocidad inicial cero.
        self.state = np.array([initial_gps_pos[0], initial_gps_pos[1], 0.0, 0.0])
        # Resetea la incertidumbre para la inicialización
        self.P = np.eye(4) * 1.0 # Menor incertidumbre si estamos seguros del punto de partida

    def predict(self, acc_imu=None):
        # Predicción del estado: x_k = F * x_{k-1} (+ B * u)
        # Si usaras las aceleraciones del IMU directamente en el modelo de predicción:
        if acc_imu is not None:
            u = np.array([acc_imu[0], acc_imu[1]]) # Asumiendo acc_earth ya en 2D
            self.state = self.F @ self.state + self.B @ u  

        else:
          self.state = self.F @ self.state

        # Predicción de la covarianza: P_k = F * P_{k-1} * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement_gps):
        # Innovación/Error de Medición: y = z - H * x_k
        y = measurement_gps - (self.H @ self.state)

        # Covarianza de Innovación: S = H * P_k * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Ganancia de Kalman: K = P_k * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Actualización del estado: x_k = x_k + K * y
        self.state = self.state + (K @ y)

        # Actualización de la covarianza: P_k = (I - K * H) * P_k
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

        return self.state[:2] # Retorna la posición (x, y) estimada
