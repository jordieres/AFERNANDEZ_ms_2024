"""

Madwick con Magnetometro aplicando ZUPT,ZUPH y lo de la rotación.
Además se ven distintas formas de ajustar la deriva aunque la mejor es la de kalman basico.

"""



import numpy as np
import pandas as pd
import argparse
import os
import yaml
from scipy import signal
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ahrs.filters import Madgwick
from ahrs.common.orientation import q_conj, q_rot
from pyproj import Proj


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

    df['time'] = (df['_time'] - df['_time'].iloc[0]).dt.total_seconds()
    time = df['time'].to_numpy()
    sample_period = np.mean(np.diff(time))
    sample_rate = 1.0 / sample_period

    gyr = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180
    acc = df[['Ax', 'Ay', 'Az']].to_numpy()
    mag = df[['Mx', 'My', 'Mz']].to_numpy() *0.1

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

    
    norm = np.linalg.norm(gravity, axis=1, keepdims=True)
    norm[norm == 0] = 1.0  # evita division x cero
    gravity_normalized = gravity / norm

    return gravity_normalized

def detect_stationary_zones(acc, sample_rate):
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

    return stationary, acc_lp, threshold


def estimate_orientation_and_position(time, gyr, acc, mag, sample_period, sample_rate, stationary):
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
    base_gain = 0.041  # Ganancia base de Madgwick
    low_gain = 0.001   # Ganancia reducida para ZUPH
    madgwick = Madgwick(frequency=sample_rate, gain=base_gain)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    quats = np.zeros((len(time), 4))
    quats[0] = q

    # Detecta zonas sin rotación (giro bajo)
    gyro_norm = np.linalg.norm(gyr, axis=1)
    no_rotation = gyro_norm < 0.1  # umbral en rad/s
    no_motion = stationary & no_rotation  # ZUPH: sin movimiento ni rotación

    for t in range(1, len(time)):
        # Aplica ZUPH: reduce el gain si estamos en zona estacionaria sin rotación
        madgwick.gain = low_gain if no_motion[t] else base_gain
        q = madgwick.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t])
        quats[t] = q

    # Convertir aceleraciones al marco terrestre
    acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)])
    acc_earth -= estimate_gravity_vector(acc, 0.95)
    acc_earth *= 9.81

    # Integración de velocidad con ZUPT
    vel = np.zeros_like(acc_earth)
    for t in range(1, len(vel)):
        vel[t] = vel[t - 1] + acc_earth[t] * sample_period
        if stationary[t]:
            vel[t] = 0  # ZUPT aplicado: velocidad puesta a cero si está estacionario

    # Corrección de deriva por interpolación entre zonas ZUPT
    vel_drift = np.zeros_like(vel)
    starts = np.where(np.diff(stationary.astype(int)) == -1)[0] + 1
    ends = np.where(np.diff(stationary.astype(int)) == 1)[0] + 1
    for s, e in zip(starts, ends):
        if e > s:
            drift_rate = vel[e - 1] / (e - s)
            vel_drift[s:e] = np.outer(np.arange(e - s), drift_rate)
    vel -= vel_drift

    # Integración para obtener posición
    pos = np.zeros_like(vel)
    for t in range(1, len(pos)):
        pos[t] = pos[t - 1] + vel[t] * sample_period

    return quats, acc_earth, vel, pos



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

def plot_results(time, acc_lp, threshold, pos, vel, gps_pos=None, output_dir=None, title="Trajectory Comparison", base_name="trajectory", verbose=3, traj_label="IMU" ):
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
        plt.figure(figsize=(10, 4))
        plt.plot(time, acc_lp, label='acc_lp')
        plt.axhline(threshold, color='r', linestyle='--')
        plt.title("Filtered Acceleration")
        plt.xlabel("Time (s)")
        plt.grid()
        plt.legend()
        save_figure("Filtered Acceleration")

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

def apply_kalman_filter(pos_imu, gps_pos, time):
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



def ekf_fusion_2d(gps_pos: np.ndarray, imu_acc: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    Extended Kalman Filter (EKF) for 2D position estimation using GPS and IMU acceleration.

    :param gps_pos: Nx2 array of GPS positions [x, y].
    :param imu_acc: Nx2 array of IMU accelerations [ax, ay].
    :param time: Array of timestamps (N,).
    :return: Nx2 array of estimated positions.
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

def complementary_filter(pos_imu: np.ndarray, gps_pos: np.ndarray, alpha: float = 0.98) -> np.ndarray:
    """
    Fuse IMU and GPS positions using a complementary filter.

    :param pos_imu: IMU-estimated positions, shape (N, 3).
    :param gps_pos: GPS positions, shape (N, 2).
    :param alpha: Weighting factor for IMU data (0 < alpha < 1).
    :return: Fused position estimate, shape (N, 3).
    """
    fused = np.zeros_like(pos_imu)
    fused[:, :2] = alpha * pos_imu[:, :2] + (1 - alpha) * gps_pos
    fused[:, 2] = pos_imu[:, 2]  # Retain IMU Z-axis
    return fused


def linear_drift_correction(pos_imu: np.ndarray, gps_start: np.ndarray, gps_end: np.ndarray) -> np.ndarray:
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


def parse_args():
    """
    Parse command-line arguments for the IMU processing pipeline.

    :return: Parsed command-line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="IMU data processing pipeline")
    parser.add_argument("-f", "--file_paths", type=str, nargs="+", required=True, help="Paths to one or more Excel files")
    parser.add_argument("--threshold", type=float, default=0.1, help="Stationary detection threshold")
    parser.add_argument('-v', '--verbose', type=int, default=3, help='Verbosity level')
    parser.add_argument('-c', '--config', type=str, default='.config.yaml', help='Path to the configuration file')
    parser.add_argument('--output_mode', choices=["screen", "save", "both"], default="screen", help="How to handle output plots: 'screen', 'save', or 'both'")
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='Directory to save output plots')
    return parser.parse_args()

prev_gps_latlng = None
prev_gps_pos = None

def main():
    """
    Main function to process IMU Excel files and plot estimated trajectories vs GPS.
    """
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    prev_gps_latlng = None
    prev_gps_pos = None

    for file_path in args.file_paths:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        foot_label = "Left Foot" if "left" in base_name.lower() else "Right Foot" if "right" in base_name.lower() else base_name

        if args.verbose >= 2:
            print(f"\n{'*'*33}  {foot_label}  {'*'*33}\n")
            print(f"{'-'*80}")
            print(f"Processing file: {base_name}...")
            print(f"{'-'*80}\n")

        try:
            df = load_data(file_path)

            if 'lat' in df.columns and 'lng' in df.columns:
                current_gps_latlng = df[['lat', 'lng']].to_numpy()
                use_prev_gps = False

                if prev_gps_latlng is None:
                    print("Este es el primer archivo con GPS cargado.")
                elif current_gps_latlng.shape == prev_gps_latlng.shape:
                    same_gps = np.allclose(current_gps_latlng, prev_gps_latlng, atol=1e-6)
                    print(f"¿GPS idéntico a archivo anterior?: {same_gps}")
                    if same_gps:
                        use_prev_gps = True
                else:
                    print("GPS no comparable: diferente número de muestras.")

                prev_gps_latlng = current_gps_latlng

            df = resample_to_40hz(df)
            time, sample_rate, gyr, acc, mag, sample_period = preprocess_data(df)
            stationary, acc_lp, threshold = detect_stationary_zones(acc, sample_rate)
            quats, acc_earth, vel, pos = estimate_orientation_and_position(
                time, gyr, acc, mag, sample_period, sample_rate, stationary
            )

            if use_prev_gps and prev_gps_pos is not None:
                gps_pos = prev_gps_pos
                gps_final = gps_pos[-1]
                print("Se reutilizó el GPS del primer archivo.")
            else:
                gps_pos, gps_final = compute_gps_positions(df, config)
                prev_gps_pos = gps_pos

            # Apply filters
            pos_kalman = apply_kalman_filter(pos, gps_pos, time)
            pos_ekf_2d = ekf_fusion_2d(gps_pos=gps_pos, imu_acc=acc_earth[:, :2], time=time)
            pos_complementary = complementary_filter(pos, gps_pos, alpha=0.98)
            pos_drift_corrected = linear_drift_correction(pos, gps_start=gps_pos[0], gps_end=gps_pos[-1])

            dist_gps = np.sum(np.linalg.norm(np.diff(gps_pos, axis=0), axis=1)) if gps_pos is not None else 0

            if args.verbose >= 2:
                def print_metrics(name, traj):
                    final_err = np.linalg.norm(traj[-1, :2] - gps_final)
                    total_dist = np.sum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1))
                    print(f"- {name}    -> Final error: {final_err:.2f} m | Distance: {total_dist:.2f} m")

                print("\n Quantitative Comparison: ")
                print(f"- Total GPS distance: {dist_gps:.2f} m")
                print_metrics("IMU", pos)
                print_metrics("Kalman", pos_kalman)
                print_metrics("EKF 2D", np.hstack((pos_ekf_2d, pos[:, 2:3])))
                print_metrics("Complementary", pos_complementary)
                print_metrics("Linear Drift", pos_drift_corrected)
                print(f"{'-'*80}")

            save_path = None
            if output_dir and args.output_mode in ("save", "both"):
                save_path = os.path.join(output_dir, f"{base_name}_trajectory.png")

            # Plots
            plot_results(time, acc_lp, threshold, pos, vel, gps_pos=gps_pos,
                         output_dir=output_dir if args.output_mode in ("save", "both") else None,
                         title=f"Trajectory Comparison - {foot_label} (IMU vs GPS)",
                         base_name=base_name + "_pre_kalman",
                         verbose=args.verbose, traj_label="IMU")

            plot_results(time, acc_lp, threshold, pos_kalman, vel, gps_pos=gps_pos,
                         output_dir=output_dir if args.output_mode in ("save", "both") else None,
                         title=f"Trajectory Comparison - {foot_label} (Kalman vs GPS)",
                         base_name=base_name + "_post_kalman",
                         verbose=args.verbose, traj_label="Kalman")

            plot_results(time, acc_lp, threshold, np.hstack((pos_ekf_2d, pos[:, 2:3])), vel, gps_pos=gps_pos,
                         output_dir=output_dir if args.output_mode in ("save", "both") else None,
                         title=f"Trajectory Comparison - {foot_label} (EKF 2D vs GPS)",
                         base_name=base_name + "_post_ekf2d",
                         verbose=args.verbose, traj_label="EKF 2D")

            plot_results(time, acc_lp, threshold, pos_complementary, vel, gps_pos=gps_pos,
                         output_dir=output_dir if args.output_mode in ("save", "both") else None,
                         title=f"Trajectory Comparison - {foot_label} (Complementary Filter vs GPS)",
                         base_name=base_name + "_post_complementary",
                         verbose=args.verbose, traj_label="Complementary")


            plot_results(time, acc_lp, threshold, pos_drift_corrected, vel, gps_pos=gps_pos,
                         output_dir=output_dir if args.output_mode in ("save", "both") else None,
                         title=f"Trajectory Comparison - {foot_label} (Linear Drift Correction vs GPS)",
                         base_name=base_name + "_post_drift",
                         verbose=args.verbose, traj_label="LinearDrift")

            if save_path:
                plt.savefig(save_path)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if output_dir and args.output_mode in ("save", "both"):
        print(f"{'-'*80}")
        print(f"\nTrajectory plots successfully saved to:\n{output_dir}\n")

    if args.output_mode in ("screen", "both"):
        plt.show()


if __name__ == "__main__":
    main()