import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt
from ahrs.filters import Madgwick, Mahony
from ahrs.common.orientation import q_conj, q_rot, axang2quat
from pyproj import Proj
import argparse
import os
import yaml
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import plotly.graph_objects as go



def load_data(file_path):
    """
    Load Excel data from the specified path.

    :param file_path: Path to the Excel file.
    :type file_path: str
    :return: DataFrame containing raw data.
    :rtype: pd.DataFrame
    """
    df = pd.read_excel(file_path)
    return df


def load_config(config_path):
    """
    Load YAML configuration file from the specified path.

    :param config_path: Path to the YAML configuration file.
    :type config_path: str
    :return: Dictionary containing configuration data.
    :rtype: dict
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



def parse_args():
    """
    Parse command-line arguments for the IMU processing pipeline.

    :return: Parsed command-line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="IMU data processing pipeline")
    parser.add_argument("-f", "--file_paths", type=str, nargs="+", required=True, help="Paths to one or more Excel files")
    parser.add_argument('-v', '--verbose', type=int, default=3, help='Verbosity level')
    parser.add_argument('-c', '--config', type=str, default='.config.yaml', help='Path to the configuration file')
    parser.add_argument('--output_mode', choices=["screen", "save", "both"], default="screen", help="How to handle output plots: 'screen', 'save', or 'both'")
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='Directory to save output plots')
    parser.add_argument('-m','--methods', nargs='+', choices=['madgwick_imu', 'madgwick_marg', 'mahony_imu', 'mahony_marg'], required=True, help="Algoritmos a ejecutar (elige uno o varios)")
    parser.add_argument('-g','--plot_mode', choices=['split', 'all', 'interactive'], default='split', help="How to plot trajectories: 'split' (default), 'all', or 'interactive'")
    return parser.parse_args()

def resample_to_40hz(df, time_col='_time', freq_hz=40, gap_threshold_ms=200):
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


def preprocess_data(df):
    """
    Process DataFrame to compute time, sample rate, and sensor arrays.

    :param df: Preprocessed DataFrame.
    :type df: pd.DataFrame
    :return: Tuple of time array, sample rate, gyroscope, accelerometer, magnetometer arrays, and sample period.
    :rtype: tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float]
    """
    ...
    df['time'] = (df['_time'] - df['_time'].iloc[0]).dt.total_seconds()
    time = df['time'].to_numpy()
    sample_period = np.mean(np.diff(time))
    sample_rate = 1.0 / sample_period

    gyr = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180
    acc = df[['Ax', 'Ay', 'Az']].to_numpy() 
    mag = df[['Mx', 'My', 'Mz']].to_numpy() * 0.1
    

    return time, sample_rate, gyr, acc, mag, sample_period


def estimate_gravity_vector(acc: np.ndarray, alpha: float = 0.9) -> np.ndarray:
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

def detect_stationary(acc, sample_rate):
    """
    Detect stationary periods from accelerometer data using filtering and thresholding.

    :param acc: Acceleration data array with shape (N, 3).
    :type acc: np.ndarray
    :param sample_rate: Sampling rate of the accelerometer signal in Hz.
    :type sample_rate: float
    :return: A tuple containing:
             - stationary: Boolean array indicating stationary (True) or moving (False) states.
             - acc_lp: Low-pass filtered absolute high-pass signal used for thresholding.
             - threshold: Threshold value used to determine stationary periods.
    :rtype: tuple[np.ndarray, np.ndarray, float]
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
def detect_no_rotation(gyr: np.ndarray, threshold: float = 0.05, duration_samples: int = 5) -> np.ndarray:
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

def compute_gps_positions(df, config):
    """
    Convert GPS coordinates to local Cartesian positions using projection configuration.

    :param df: DataFrame with 'lat' and 'lng' columns.
    :type df: pd.DataFrame
    :param config: Configuration dictionary containing a 'Location' section with projection parameters:
                   - proj (str): Projection type (e.g., 'utm')
                   - zone (int): UTM zone number
                   - ellps (str): Ellipsoid model (e.g., 'WGS84')
                   - south (bool): True if in southern hemisphere
    :type config: dict
    :return: Tuple of GPS position array and final GPS position.
    :rtype: tuple[np.ndarray, np.ndarray]
    :raises KeyError: If required projection parameters are missing in config['Location'].
    """


    # Get projection configuration
    location_cfg = config["Location"]  # Raises KeyError if 'Location' section is missing

    proj = Proj(
        proj=location_cfg["proj"],
        zone=location_cfg["zone"],
        ellps=location_cfg["ellps"],
        south=location_cfg["south"]
    )

    # Extract latitude and longitude values
    lat = df['lat'].to_numpy()
    lng = df['lng'].to_numpy()

    # Project to Cartesian coordinates
    x, y = proj(lng, lat)
    gps_pos = np.stack((x - x[0], y - y[0]), axis=1)

    return gps_pos, gps_pos[-1]

def estimate_position_generic(method, use_mag, gyr, acc, mag, time, sample_rate, stationary):
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
        raise ValueError("Método no reconocido. Usa 'madgwick' o 'mahony'.")

    quats = np.zeros((len(time), 4))
    no_rotation = detect_no_rotation(gyr)
    no_motion = stationary & no_rotation

    for t in range(len(time)):
        if method == "madgwick":
            filter_.gain = 0.001 if no_motion[t] else base_gain
            if use_mag:
                q_new = filter_.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t])
            else:
                q_new = filter_.updateIMU(q, gyr=gyr[t], acc=acc[t])
        else:  # mahony
            if use_mag:
                filter_.Kp = base_kp * 0.1 if no_motion[t] else (base_kp if stationary[t] else 0.0)
                q_new = filter_.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t])
            else:
                filter_.Kp = base_kp if stationary[t] else 0.0
                q_new = filter_.updateIMU(q, gyr=gyr[t], acc=acc[t])
        
        if q_new is not None:
            q = q_new
        quats[t] = q

    acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)])
    acc_earth -= estimate_gravity_vector(acc, 0.95)
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



def plot_trajectories(resultados, errores, gps_pos, gps_final, title="Trajectory Comparison", save_path=None):

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

      

def apply_kalman_filter1(pos_imu: np.ndarray, gps_pos: np.ndarray, sample_rate: float) -> np.ndarray:


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


def apply_kalman_filter2(pos_imu, gps_pos, time):
    """
    Ajusta la trayectoria estimada por la IMU usando un filtro de Kalman 2D con GPS como corrección.

    :param pos_imu: Posiciones estimadas por la IMU (N, 3).
    :param gps_pos: Posiciones GPS proyectadas en 2D (N, 2).
    :param time: Vector de tiempo.
    :return: Posiciones corregidas (N, 3)
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





def plot_trajectories_interactive(resultados, errores, gps_pos, gps_final, title="Trajectory Comparison", save_path=None):
    fig = go.Figure()

    # Añadir trayectorias estimadas
    for name, pos in resultados.items():
        fig.add_trace(go.Scatter(
            x=pos[:, 0], y=pos[:, 1],
            mode='lines',
            name=f"{name} ({errores[name]:.2f} m)"
        ))

    # Añadir trayectorias GPS
    fig.add_trace(go.Scatter(
        x=gps_pos[:, 0], y=gps_pos[:, 1],
        mode='lines',
        name="GPS (reference)",
        line=dict(dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=[gps_final[0]], y=[gps_final[1]],
        mode='markers',
        name="GPS final",
        marker=dict(color='black', size=10)
    ))

    fig.update_layout(
        title=title,
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        width=900,
        height=700
    )

    # Si se especifica, guardar como HTML
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive plot saved to: {save_path}")

def plot_trajectories_all(all_results, gps_pos, gps_final, title="Trajectory Comparison"):
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


def plot_trajectories_split(resultados, errores, gps_pos, gps_final, title, save_path=None):
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
