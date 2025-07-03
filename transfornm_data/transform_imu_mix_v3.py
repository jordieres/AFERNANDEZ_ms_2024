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
    ...
    df['time'] = (df['_time'] - df['_time'].iloc[0]).dt.total_seconds()
    time = df['time'].to_numpy()
    sample_period = np.mean(np.diff(time))
    sample_rate = 1.0 / sample_period

    gyr = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180
    acc = df[['Ax', 'Ay', 'Az']].to_numpy() #* 9.81
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
 
def detect_stationary(acc: np.ndarray, sample_rate: float):
    """
    Detect stationary periods from acceleration magnitude.

    :param acc: Acceleration data.
    :type acc: np.ndarray
    :param sample_rate: Sampling frequency in Hz.
    :type sample_rate: float
    :return: Boolean array indicating stationary samples.
    :rtype: np.ndarray
    """
    acc_mag = np.linalg.norm(acc, axis=1)
    acc_mag_clipped = np.clip(acc_mag, 0, 20)
    b, a = signal.butter(1, 0.01 / (sample_rate / 2), 'highpass')
    acc_hp = signal.filtfilt(b, a, acc_mag_clipped)
    b, a = signal.butter(1, 5.0 / (sample_rate / 2), 'lowpass')
    acc_lp = signal.filtfilt(b, a, np.abs(acc_hp))
    threshold = np.percentile(acc_lp, 15)
    return acc_lp < threshold

def update_quaternion_madgwick(q, t, filter_, gyr, acc, mag, use_mag):
    """
    Update Madgwick filter quaternion with or without magnetometer.

    This function applies the Madgwick update step either in MARG mode (gyro, accel, mag)
    or IMU-only mode (gyro, accel) depending on the `use_mag` flag.

    :param q: Current quaternion estimate (4,).
    :type q: np.ndarray
    :param t: Current time index.
    :type t: int
    :param filter_: Instance of the Madgwick filter.
    :type filter_: Madgwick
    :param gyr: Gyroscope data array of shape (N, 3), in rad/s.
    :type gyr: np.ndarray
    :param acc: Accelerometer data array of shape (N, 3), in m/s².
    :type acc: np.ndarray
    :param mag: Magnetometer data array of shape (N, 3), in µT.
    :type mag: np.ndarray
    :param use_mag: Whether to include magnetometer data in the update.
    :type use_mag: bool
    :return: Updated quaternion estimate (4,).
    :rtype: np.ndarray
    """
    if use_mag:
        return filter_.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t])
    else:
        return filter_.updateIMU(q, gyr=gyr[t], acc=acc[t])

def update_quaternion_mahony(q, t, filter_, gyr, acc, mag, use_mag):
    """
    Update Mahony filter quaternion with or without magnetometer.

    This function applies the Mahony update step in either MARG mode (gyro, accel, mag)
    or IMU-only mode (gyro, accel) depending on the `use_mag` flag.

    :param q: Current quaternion estimate (4,).
    :type q: np.ndarray
    :param t: Current time index.
    :type t: int
    :param filter_: Instance of the Mahony filter.
    :type filter_: Mahony
    :param gyr: Gyroscope data array of shape (N, 3), in rad/s.
    :type gyr: np.ndarray
    :param acc: Accelerometer data array of shape (N, 3), in m/s².
    :type acc: np.ndarray
    :param mag: Magnetometer data array of shape (N, 3), in µT.
    :type mag: np.ndarray
    :param use_mag: Whether to include magnetometer data in the update.
    :type use_mag: bool
    :return: Updated quaternion estimate (4,).
    :rtype: np.ndarray
    """
    if use_mag:
        return filter_.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t])
    else:
        return filter_.updateIMU(q, gyr=gyr[t], acc=acc[t])
    
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

def estimate_position(name: str, gyr, acc, mag, time, sample_rate, stationary, use_madgwick=True, use_mag=True):
    """
    Estimate position from IMU data using Madgwick or Mahony filter.

    :param name: Label for identifying the current dataset or subject.
    :type name: str
    :param gyr: Gyroscope data array of shape (N, 3), in rad/s.
    :type gyr: np.ndarray
    :param acc: Accelerometer data array of shape (N, 3), in m/s².
    :type acc: np.ndarray
    :param mag: Magnetometer data array of shape (N, 3), in µT.
    :type mag: np.ndarray
    :param time: Time array of shape (N,), in seconds.
    :type time: np.ndarray
    :param sample_rate: Sampling rate of the data in Hz.
    :type sample_rate: float
    :param stationary: Boolean array indicating stationary periods (ZUPT).
    :type stationary: np.ndarray
    :param use_madgwick: Whether to use Madgwick filter (True) or Mahony filter (False).
    :type use_madgwick: bool
    :param use_mag: Whether to use magnetometer data in the orientation filter.
    :type use_mag: bool
    :return: Estimated position array of shape (N, 3), in meters.
    :rtype: np.ndarray

    """
    if use_madgwick:
        base_gain = 0.005 if use_mag else 0.01
        filter_ = Madgwick(frequency=sample_rate, gain=base_gain)
        q = axang2quat([0, 0, 1], np.deg2rad(45))
        update_function = update_quaternion_madgwick
    else:
        base_kp = 1.5
        filter_ = Mahony(Kp=base_kp, Ki=0.01, frequency=sample_rate)
        q = np.array([1.0, 0.0, 0.0, 0.0])
        acc_init = np.median(acc[time <= time[0] + 2], axis=0)
        for _ in range(2000):
            q = filter_.updateIMU(q, gyr=np.zeros(3), acc=acc_init)
        update_function = update_quaternion_mahony

    quats = np.zeros((len(time), 4))
    no_rotation = detect_no_rotation(gyr)
    no_motion = stationary & no_rotation

    for t in range(len(time)):
        if not use_madgwick:
            filter_.Kp = base_kp if stationary[t] else 0.0

        # Apply ZUPH
        if use_mag and no_motion[t]:
            if use_madgwick:
                filter_.gain = 0.001
            else:
                filter_.Kp = base_kp * 0.1
        elif use_madgwick:
            filter_.gain = base_gain

        q_new = update_function(q, t, filter_, gyr, acc, mag, use_mag)
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

def apply_kalman_filter(pos_imu: np.ndarray, gps_pos: np.ndarray, sample_rate: float) -> np.ndarray:


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



from scipy.interpolate import interp1d

def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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
            df = resample_to_40hz(df)
            time, sample_rate, gyr, acc, mag, sample_period = preprocess_data(df)
            stationary = detect_stationary(acc, sample_rate)

            resultados = {
                "Madgwick with mag": estimate_position("Madgwick with mag", gyr, acc, mag, time, sample_rate, stationary, True, True),
                "Madgwick without mag": estimate_position("Madgwick without mag", gyr, acc, mag, time, sample_rate, stationary, True, False),
                "Mahony with mag": estimate_position("Mahony with mag", gyr, acc, mag, time, sample_rate, stationary, False, True),
                "Mahony without mag": estimate_position("Mahony without mag", gyr, acc, mag, time, sample_rate, stationary, False, False),
            }

            gps_pos, gps_final = compute_gps_positions(df, config)

            gps_time = df.loc[~df['lat'].isna(), 'time'].to_numpy()
            imu_time = df['time'].to_numpy()


            kalman_results = {}
            errors = {}

            for name, pos in resultados.items():
                imu_len = len(pos)
                time_cut = imu_time[:imu_len]

                # Crear función de interpolación GPS para esta trayectoria
                try:
                    gps_interp_fn = interp1d(gps_time, gps_pos, axis=0, bounds_error=False, fill_value="extrapolate")
                    gps_interp = gps_interp_fn(time_cut)
                except Exception as e:
                    print(f"Warning: Failed to interpolate GPS for {name} – {e}")
                    continue

                print(f"Kalman → {name}")
                print(f"  IMU start pos: {pos[0]}, end: {pos[-1]}")
                print(f"  GPS interp start: {gps_interp[0]}, end: {gps_interp[-1]}")

                kalman_fused = apply_kalman_filter(pos, gps_interp, sample_rate)
                kalman_name = f"{name} + Kalman"
                kalman_results[kalman_name] = kalman_fused

                kalman_error = np.mean(np.linalg.norm(kalman_fused[:, :2] - gps_interp[:imu_len], axis=1))
                errors[kalman_name] = kalman_error

                base_error = np.mean(np.linalg.norm(pos[:imu_len, :2] - gps_interp[:imu_len], axis=1))
                errors[name] = base_error

            resultados.update(kalman_results)

            if args.verbose >= 2:
                print("Estimation results vs GPS (with Kalman):\n")
                for name, pos in resultados.items():
                    error = errors.get(name, 0.0)
                    distance = np.max(np.linalg.norm(pos[:, :2], axis=1))
                    print(f"- {name:<30} → Error: {error:7.2f} m | Max distance: {distance:7.2f} m")
                print(f"{'-'*80}")

            output_file = None
            if output_dir and args.output_mode in ("save", "both"):
                output_file = os.path.join(output_dir, f"{base_name}_trajectory.png")

            plot_trajectories(resultados, errors, gps_pos, gps_final,
                              title=f"Trajectory Comparison - {foot_label}",
                              save_path=output_file)

            if args.verbose >= 3:
                print(f"\nDiagnostics:\n")
                print(f"- Total interpolated samples    : {len(df):,}")
                print(f"- Estimated frequency (Hz)      : {sample_rate:.2f}")
                print(f"- Estimated period (s)          : {sample_period:.5f}")
                print(f"- Stationary samples            : {np.sum(stationary):,}")
                print(f"- Moving samples                : {np.sum(~stationary):,}")
                print(f"- Stationary percentage         : {np.mean(stationary) * 100:.1f} %")

                acc_init = np.median(acc[time <= time[0] + 2], axis=0)
                mag_init = np.median(mag[time <= time[0] + 2], axis=0)
                print(f"- Norm of acc_init (expected ~1): {np.linalg.norm(acc_init):.4f}")
                print(f"- Norm of mag_init              : {np.linalg.norm(mag_init):.4f}\n")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if output_dir and args.output_mode in ("save", "both"):
        print(f"{'-'*80}")
        print(f"\nTrajectory plots successfully saved to:\n{output_dir}\n")

    if args.output_mode in ("screen", "both"):
        plt.show()


if __name__ == "__main__":
    main()
