import numpy as np
import pandas as pd
import yaml
import plotly.graph_objects as go
import folium
import os

from ahrs.filters import Madgwick, Mahony
from ahrs.common.orientation import q_conj, q_rot, axang2quat
from matplotlib import pyplot as plt
from pyproj import Proj, Transformer
from filterpy.kalman import KalmanFilter
from scipy.interpolate import CubicSpline
from scipy import signal
from scipy.signal import find_peaks
from geopy.distance import geodesic

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

    def __init__(self, sample_rate, sample_period):
        self.sample_period = sample_period
        self.sample_rate = sample_rate
        

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
            base_gain = 0.02
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

        gravity = self.estimate_gravity_vector(acc, 0.95)
        acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)])
        acc_earth -= gravity
        acc_earth *= 9.81


        vel = np.zeros_like(acc_earth)
        for t in range(1, len(vel)):
            vel[t] = vel[t - 1] + acc_earth[t] * (self.sample_period)
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
            pos[t] = pos[t - 1] + vel[t] * self.sample_period

        return pos
    

        
    def estimate_position_madwick(self, time, gyr, acc, mag, stationary):
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
        madgwick = Madgwick(frequency=self.sample_rate, gain=0.02)
        low_gain = 0.001
        q = np.array([1.0, 0.0, 0.0, 0.0])
        quats = np.zeros((len(time), 4))
        quats[0] = q

        no_rotation = self.detect_no_rotation(gyr)
        no_motion = stationary & no_rotation

        for t in range(1, len(time)):
            madgwick.gain = low_gain if no_motion[t] else 0.02
            q = madgwick.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t])
            quats[t] = q

        gravity = self.estimate_gravity_vector(acc, 0.95)
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



class PositionVelocityEstimator:
    """
    Estimates orientation, acceleration, velocity, and position from IMU sensors (gyroscope, accelerometer, magnetometer).
    Uses the Madgwick filter with adaptive gain for ZUPH and velocity corrections using ZUPT.
    """
    def __init__(self, sample_rate, sample_period):
        self.sample_rate = sample_rate
        self.sample_period = sample_period
        # self.base_gain = 0.041
        self.base_gain = 0.02
        # self.low_gain = 0.001
        self.low_gain = 0.001
        self.imu_proc = IMUProcessor(sample_rate, sample_period)

    def estimate_position_madwick(self, time, gyr, acc, mag, stationary):
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

        no_rotation = self.imu_proc.detect_no_rotation(gyr)
        no_motion = stationary & no_rotation

        for t in range(1, len(time)):
            madgwick.gain = self.low_gain if no_motion[t] else self.base_gain
            q = madgwick.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t])
            quats[t] = q

        
        gravity = self.imu_proc.estimate_gravity_vector(acc, 0.95)
        acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)])
        acc_earth -= gravity
        acc_earth *= 9.81

        # gravity_est = np.array([q_rot(q_conj(qt), [0, 0, 1]) for qt in quats]) * 9.81
        # acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)]) * 9.81
        # acc_earth -= gravity_est

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



class KalmanProcessor:
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
    

    
    def initialize_kalman(self, gps_pos, sample_rate):
        dt = 1.0 / sample_rate
        kf = KalmanFilter(dt=dt, q=0.1, r=0.5)
        kf.initialize(gps_pos[0])
        return kf


    def run_filter_with_acc_and_gps(self, acc_earth, gps_pos):
        fused = []
        for i in range(len(acc_earth)):
            self.predict(acc_imu=acc_earth[i, :2])
            self.update(gps_pos[i])
            fused.append(self.state[:2])
        return np.array(fused)





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

        
    def plot_macroscopic_comparision(self, pos, gps_pos=None, output_dir=None, title="Trajectory Comparison", base_name="trajectory", traj_label= None ):
        """
        Plot diagnostic and trajectory figures for IMU data, and optionally save them.

        This function generates several plots based on filtered acceleration, position,
        velocity, 2D and 3D trajectory. If GPS data is provided, it includes a comparative plot. 
        Depending on the verbosity level and output directory, plots can also be saved.


        :param pos: Estimated position array of shape (N, 3).
        :type pos: np.ndarray
        :param gps_pos: Optional GPS position array of shape (N, 2). Defaults to None.
        :type gps_pos: np.ndarray or None
        :param title: Title used in trajectory comparison plots.
        :type title: str

        """
        if gps_pos is not None:
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

    def plot_resume(self, time, pos, vel, output_dir=None, base_name="summary"):
        """
        Plot position, velocity, and 3D trajectory of the estimated movement.

        This function summarizes motion-related variables: 1D time series of position and velocity
        (in x, y, z), and a 3D trajectory plot in space.

        :param time: Time vector in seconds.
        :type time: np.ndarray
        :param pos: Estimated position array of shape (N, 3).
        :type pos: np.ndarray
        :param vel: Estimated velocity array of shape (N, 3).
        :type vel: np.ndarray
        :param output_dir: (Not used anymore; figures are not saved).
        :type output_dir: str or None
        :param base_name: (Unused). Name that would be used for saving if enabled.
        :type base_name: str
        """

        # Position over time
        plt.figure(figsize=(15, 5))
        plt.plot(time, pos[:, 0], 'r', label='x')
        plt.plot(time, pos[:, 1], 'g', label='y')
        plt.plot(time, pos[:, 2], 'b', label='z')
        plt.title("Position over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.legend()
        plt.grid()

        # Velocity over time
        plt.figure(figsize=(15, 5))
        plt.plot(time, vel[:, 0], 'r', label='vx')
        plt.plot(time, vel[:, 1], 'g', label='vy')
        plt.plot(time, vel[:, 2], 'b', label='vz')
        plt.title("Velocity over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.legend()
        plt.grid()

        # 3D Trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])
        ax.set_title("3D Trajectory")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")





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
                {"timestamp": df.iloc[prev_idx]["time"], "value": prev_val, "peak_type": "entry", "orig_index": prev_idx},
                {"timestamp": df.iloc[curr_idx]["time"], "value": curr_val, "peak_type": "main", "orig_index": curr_idx},
                {"timestamp": df.iloc[next_idx]["time"], "value": next_val, "peak_type": "exit", "orig_index": next_idx},
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

    def compute_stride_stats_per_minute(self, df_steps, pos_kalman, step_peaks, output_dir=None):
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






class ResultsProcessor:
    """
    Class for preparing and exporting final results such as step-wise metrics, positions, and velocities.
    """

    def __init__(self):
        pass

    def prepare_step_dataframe(self, time, gps_lat, gps_lng, pos_kalman, vel, df_inter):
        """
        Generate a DataFrame with step-wise position, velocity, and GPS data.

        :param time: Time vector in seconds.
        :type time: np.ndarray
        :param gps_lat: GPS latitude array.
        :type gps_lat: np.ndarray
        :param gps_lng: GPS longitude array.
        :type gps_lng: np.ndarray
        :param pos_kalman: Kalman-filtered position array (N, 2+).
        :type pos_kalman: np.ndarray
        :param vel: Velocity array (N, 3).
        :type vel: np.ndarray
        :return: DataFrame with time, position, velocity, and step distance.
        :rtype: pd.DataFrame
        """
        step_distance = np.zeros(len(time))
        step_distance[1:] = np.linalg.norm(pos_kalman[1:, :2] - pos_kalman[:-1, :2], axis=1)

        df_steps = pd.DataFrame({
            'time': time,
            'lat': gps_lat,
            'lng': gps_lng,
            'pos_x_m': pos_kalman[:, 0],
            'pos_y_m': pos_kalman[:, 1],
            'velocity_m_s': np.linalg.norm(vel, axis=1),
            'step_distance_m': step_distance,
            'datetime': df_inter['_time'].values
        })

        return df_steps
    
    def print_metrics(self, name, traj, gps_final):
        final_err = np.linalg.norm(traj[-1, :2] - gps_final)
        total_dist = np.sum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1))
        print(f"- {name}   -> Final error: {final_err:.2f} m | Distance: {total_dist:.2f} m")

    def export_to_excel(self, df: pd.DataFrame, file_path: str):
        try:
            df.to_excel(file_path, index=False)
            print(f"\n Excel saved: {file_path}")
        except Exception as e:
            print(f" Error saving Excel {file_path}: {e}")
    
    def get_output_path(self, filename: str, args) -> str:
        return os.path.join(args.output_dir or ".", filename)
    




class StrideProcessor:
    def __init__(self, min_stride=0.2, max_stride=2.5, window_sec=6.0):
        """
        Filtro para eliminar zancadas fuera de rango razonable.

        :param min_stride: Mínima longitud válida de zancada (m)
        :param max_stride: Máxima longitud válida de zancada (m)
        :param window_sec: Time before and after each stride to extract.
        """
        self.min_stride = min_stride
        self.max_stride = max_stride
        self.window = window_sec

    def clean_stride_data(self, df_stride_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina zancadas con longitud fuera del rango especificado.

        :param df_stride_raw: DataFrame de zancadas individuales
        :return: DataFrame filtrado
        """
        before = len(df_stride_raw)
        cleaned = df_stride_raw[
            (df_stride_raw["stride_length_m"] >= self.min_stride) &
            (df_stride_raw["stride_length_m"] <= self.max_stride)
        ].copy()
        after = len(cleaned)
        return cleaned

    def recompute_stats_per_minute(self, df_stride_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Recalcula estadísticas por minuto tras filtrar.

        :param df_stride_clean: DataFrame de zancadas válidas
        :return: DataFrame de estadísticas por minuto
        """
        df_stats_clean = df_stride_clean.groupby('minute').agg(
            steps=('stride_length_m', 'count'),
            mean_stride_length=('stride_length_m', 'mean'),
            std_stride_length=('stride_length_m', 'std'),
            distance_m=('stride_length_m', 'sum')
        ).reset_index()
        return df_stats_clean
    
    def check_distance_similarity(self, df_stats: pd.DataFrame, gps_distance: float, tolerance: float = 0.15) -> pd.DataFrame:
        """
        Comprueba si la distancia estimada por minuto es razonablemente cercana al GPS (±tolerancia).

        :param df_stats: DataFrame con columnas 'minute' y 'distance_m'
        :param gps_distance: Distancia total medida por GPS (m)
        :param tolerance: Porcentaje permitido de desviación (ej. 0.15 = ±15%)
        :return: DataFrame con columna adicional 'gps_consistent': True/False por minuto
        """
        total_stride_distance = df_stats['distance_m'].sum()
        ratio = total_stride_distance / gps_distance if gps_distance > 0 else 0
        df_stats['gps_consistent'] = np.abs(ratio - 1) <= tolerance
        return df_stats

    def check_stride_length_range(self, df_stats: pd.DataFrame, min_valid=0.2, max_valid=1.5) -> pd.DataFrame:
        """
        Verifica si la longitud media de zancada está dentro de un rango razonable.

        :param df_stats: DataFrame con columna 'mean_stride_length'
        :param min_valid: Longitud mínima aceptable (m)
        :param max_valid: Longitud máxima aceptable (m)
        :return: DataFrame con columna adicional 'stride_length_valid': True/False
        """
        df_stats['stride_length_valid'] = (
            (df_stats['mean_stride_length'] >= min_valid) &
            (df_stats['mean_stride_length'] <= max_valid)
        )
        return df_stats

    def check_trajectory_smoothness(self, df_steps: pd.DataFrame, velocity_threshold: float = 3.0) -> pd.DataFrame:
        """
        Detecta saltos/picos en la velocidad (ruido) como indicio de errores de trayectoria.

        :param df_steps: DataFrame con columna 'velocity_m_s'
        :param velocity_threshold: Límite superior para velocidad razonable (m/s)
        :return: DataFrame con columna 'velocity_spike': True si hay un pico brusco
        """
        df_steps = df_steps.copy()
        df_steps['velocity_spike'] = df_steps['velocity_m_s'].diff().abs() > velocity_threshold
        num_spikes = df_steps['velocity_spike'].sum()
        return df_steps

    def check_spatial_alignment(self, pos_est: np.ndarray, pos_gps: np.ndarray, threshold: float = 10.0) -> np.ndarray:
        """
        Evalúa el error espacial entre posición estimada y GPS (por paso).

        :param pos_est: Array (N, 2) de posiciones estimadas
        :param pos_gps: Array (N, 2) de posiciones GPS (mismos timestamps)
        :param threshold: Umbral máximo de error permitido (en metros)
        :return: Array booleano (N,) donde True indica alineación GPS aceptable
        """
        if pos_est.shape != pos_gps.shape:
            raise ValueError("Las posiciones estimadas y GPS deben tener la misma forma.")
        spatial_error = np.linalg.norm(pos_est - pos_gps, axis=1)
        alignment = spatial_error <= threshold
        percent_ok = 100 * np.mean(alignment)
        return alignment,percent_ok


    def evaluate_quality_segments(self,df_stride_stats: pd.DataFrame,df_steps: pd.DataFrame,gps_pos: np.ndarray, imu_pos: np.ndarray, 
                                  velocity_threshold: float = 3.0, gps_distance: float = None, error_tolerance_m: float = 10.0, min_alignment_ratio: float = 0.95 ) -> pd.DataFrame:
        """
        Evalúa por minuto si todos los criterios de calidad se cumplen.

        :param df_stride_stats: DataFrame de estadísticas de zancada por minuto (filtrado)
        :param df_steps: DataFrame paso a paso con columnas ['time', 'velocity_m_s']
        :param gps_pos: Posiciones GPS (Nx2)
        :param imu_pos: Posiciones estimadas IMU (Nx2)
        :param velocity_threshold: Umbral de velocidad para considerar un pico
        :param gps_distance: Distancia total medida por GPS (para ratio de distancia)
        :param error_tolerance_m: Tolerancia en metros para error espacial
        :param min_alignment_ratio: Mínimo % de puntos bien alineados (0-1)
        :return: DataFrame de calidad por minuto con columna 'all_criteria_ok'
        """
        stats = df_stride_stats.copy()
        stats["gps_consistent"] = False
        stats["velocity_ok"] = False
        stats["spatially_aligned"] = False

        if gps_distance is not None:
            # Comprobar coherencia de distancia
            stats["gps_consistent"] = stats["distance_m"].between(gps_distance * 0.85, gps_distance * 1.15)

        # Evaluar por minuto si hay picos de velocidad
        for i, row in stats.iterrows():
            minute = row["minute"]
            vel_segment = df_steps[df_steps["minute"] == minute]["velocity_m_s"]
            stats.at[i, "velocity_ok"] = (vel_segment <= velocity_threshold).all()

            # Error espacial con GPS
            imu_seg = imu_pos[df_steps["minute"] == minute]
            gps_seg = gps_pos[df_steps["minute"] == minute]

            if len(imu_seg) > 0 and len(gps_seg) == len(imu_seg):
                dists = np.linalg.norm(imu_seg - gps_seg, axis=1)
                aligned_ratio = np.mean(dists < error_tolerance_m)
                stats.at[i, "spatially_aligned"] = aligned_ratio >= min_alignment_ratio

        # Combinar todo
        stats["all_criteria_ok"] = (
            stats["gps_consistent"] &
            stats["stride_length_valid"] &
            stats["velocity_ok"] &
            stats["spatially_aligned"]
        )

        return stats

    def plot_stride_filtering(self,df_stride_raw, df_stride_raw_clean,min_stride=0.2, max_stride=2.5, y_max=3.0, title="Zancadas antes y después del filtrado"):
        """
        Visualiza la longitud de zancadas antes y después del filtrado.

        Parameters:
            df_stride_raw (pd.DataFrame): DataFrame original con todas las zancadas.
            df_stride_raw_clean (pd.DataFrame): DataFrame tras aplicar el filtrado por longitud.
            min_stride (float): Umbral mínimo de longitud válida (m).
            max_stride (float): Umbral máximo de longitud válida (m).
            title (str): Título del gráfico.
        """
        plt.figure(figsize=(12, 4))
        plt.plot(df_stride_raw["time"], df_stride_raw["stride_length_m"], label="Zancadas brutas")
        plt.scatter(df_stride_raw_clean["time"], df_stride_raw_clean["stride_length_m"], label="Zancadas válidas", color="green")
        plt.axhline(max_stride, color="red", linestyle="--", label="Umbral máx")
        plt.axhline(min_stride, color="red", linestyle="--", label="Umbral mín")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Longitud de zancada (m)")
        plt.legend()
        plt.title(title)
        plt.tight_layout()

    def extract_region(self, df_imu, df_gps, stride_time):
        """
        Extracts IMU and GPS data in a window around a given stride.

        :param df_imu: DataFrame with columns ['time', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
        :param df_gps: DataFrame with columns ['time', 'lat', 'lng']
        :param stride_time: Central time of the stride (in seconds)
        :return: imu_window, gps_window, gps_distance_m, start, end
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
        Analyzes multiple strides without saving files, returns key data for visualization and comparison.

        :param df_imu: IMU signal DataFrame.
        :param df_gps: GPS position DataFrame.
        :param df_strides: DataFrame with a 'time' column indicating stride times.
        :param stride_type: 'valid' or 'invalid' (for classification).
        :return: List of dictionaries with analysis per stride.
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

    