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
    mag = df[['Mx', 'My', 'Mz']].to_numpy() / 1000

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

def detect_stationary_zones(acc, sample_rate):
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
    madgwick = Madgwick(frequency=sample_rate, gain=0.041)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    quats = np.zeros((len(time), 4))
    quats[0] = q

    for t in range(1, len(time)):
        q = madgwick.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t])
        quats[t] = q

    acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)])
    acc_earth -= estimate_gravity_vector(acc, 0.95)
    acc_earth *= 9.81

    vel = np.zeros_like(acc_earth)
    for t in range(1, len(vel)):
        vel[t] = vel[t - 1] + acc_earth[t] * sample_period
        if stationary[t]:
            vel[t] = 0

    vel_drift = np.zeros_like(vel)
    starts = np.where(np.diff(stationary.astype(int)) == -1)[0] + 1
    ends = np.where(np.diff(stationary.astype(int)) == 1)[0] + 1
    for s, e in zip(starts, ends):
        drift_rate = vel[e - 1] / (e - s)
        vel_drift[s:e] = np.outer(np.arange(e - s), drift_rate)
    vel -= vel_drift

    pos = np.zeros_like(vel)
    for t in range(1, len(pos)):
        pos[t] = pos[t - 1] + vel[t] * sample_period

    return quats, acc_earth, vel, pos


def compute_gps_positions(df, config):
    """
    Convert GPS coordinates to local Cartesian positions using projection configuration.

    :param df: DataFrame with 'lat' and 'lng' columns.
    :type df: pd.DataFrame
    :param config: Full configuration dictionary (already loaded).
    :type config: dict
    :return: Tuple of GPS position array and final GPS position.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    location_cfg = config.get("Location", {}) 
    lat = df['lat'].to_numpy()
    lng = df['lng'].to_numpy()

    proj = Proj(
        proj=location_cfg.get('proj', 'utm'),
        zone=location_cfg.get('zone', 30),
        ellps=location_cfg.get('ellps', 'WGS84'),
        south=location_cfg.get('south', False)
    )

    x, y = proj(lng, lat)
    gps_pos = np.stack((x - x[0], y - y[0]), axis=1)


    return gps_pos, gps_pos[-1]


def plot_results(time, acc_lp, threshold, pos, vel, gps_pos=None, output_dir=None, base_name="trajectory", verbose=3):
    """
    Plot diagnostic and trajectory figures for IMU data, and optionally save them.

    This function generates several plots based on filtered acceleration, position,
    velocity, 2D and 3D trajectory, and a comparison with GPS data if available. 
    Depending on the verbosity level and output directory, plots can also be saved.

    :param time: Array of timestamps in seconds.
    :type time: np.ndarray
    :param acc_lp: Low-pass filtered acceleration magnitude.
    :type acc_lp: np.ndarray
    :param threshold: Threshold used for detecting stationary periods.
    :type threshold: float
    :param pos: Estimated position array of shape (N, 3).
    :type pos: np.ndarray
    :param vel: Estimated velocity array of shape (N, 3).
    :type vel: np.ndarray
    :param gps_pos: Optional GPS position array of shape (N, 2). Defaults to None.
    :type gps_pos: np.ndarray or None
    :param output_dir: Directory to save output plots. If None, plots are not saved.
    :type output_dir: str or None
    :param base_name: Base filename used when saving plots.
    :type base_name: str
    :param verbose: Verbosity level (2 shows trajectory comparison, 3 shows all plots).
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
        plt.plot(pos[:, 0], pos[:, 1], label='IMU Trajectory')
        plt.plot(gps_pos[:, 0], gps_pos[:, 1], 'k--', label='GPS Reference')
        plt.plot(gps_pos[-1, 0], gps_pos[-1, 1], 'ko', label='Final GPS')
        plt.title("Trajectory Comparison (IMU vs GPS)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.grid()
        plt.legend()
        save_figure("Trajectory Comparison (IMU vs GPS)")



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



def main():
    """
    Main function to process IMU Excel files and plot estimated trajectories vs GPS.
    """
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
            stationary, acc_lp, threshold = detect_stationary_zones(acc, sample_rate)
            quats, acc_earth, vel, pos = estimate_orientation_and_position(
                time, gyr, acc, mag, sample_period, sample_rate, stationary
            )

            gps_pos, gps_final = compute_gps_positions(df, config)
            error = np.linalg.norm(pos[-1, :2] - gps_final) if gps_pos is not None else None

            if args.verbose >=2:
                print("Estimation results vs GPS:\n")
                print(f"- Error with GPS: {error:7.2f} m")
                distance = np.max(np.linalg.norm(pos, axis=1))
                print(f"- Max distance: {distance:7.2f} m")
                print(f"{'-'*80}")

            elif args.verbose == 3:
                print("Diagnostics:")
                print(f"- Samples: {len(df)}")
                print(f"- Frequency (Hz): {sample_rate:.2f}")
                print(f"- Stationary samples: {np.sum(stationary)}")
                print(f"- Final position: {pos[-1]}")
                print(f"- Final velocity: {vel[-1]}")

            # Save or show plots
            save_path = None
            if output_dir and args.output_mode in ("save", "both"):
                save_path = os.path.join(output_dir, f"{base_name}_trajectory.png")

            plot_results(time, acc_lp, threshold, pos, vel, gps_pos=gps_pos, output_dir=output_dir if args.output_mode in ("save", "both") else None, base_name=base_name, verbose=args.verbose)

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
