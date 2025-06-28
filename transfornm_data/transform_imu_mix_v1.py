import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt
from ahrs.filters import Madgwick, Mahony
from ahrs.common.orientation import q_conj, q_rot, axang2quat
from pyproj import Proj
import argparse
import yaml
import os


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

    :param df: Raw DataFrame
    :param time_col: Name of the time column
    :param freq_hz: Target frequency in Hz
    :param gap_threshold_ms: Gap threshold to split sessions
    :return: Interpolated DataFrame
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
    :return: time array, sample_rate, sample_period, gyr, acc, mag arrays
    """
    df['time'] = (df['_time'] - df['_time'].iloc[0]).dt.total_seconds()
    time = df['time'].to_numpy()
    sample_period = np.mean(np.diff(time))
    sample_rate = 1.0 / sample_period

    gyr = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180
    acc = df[['Ax', 'Ay', 'Az']].to_numpy() #* 9.81
    mag = df[['Mx', 'My', 'Mz']].to_numpy() * 0.1

    return time, sample_rate, gyr, acc, mag, sample_period


def detect_stationary(acc: np.ndarray, sample_rate: float):
    """
    Detect stationary periods from acceleration magnitude.

    :param acc: Acceleration data.
    :param sample_rate: Sampling frequency (Hz).
    :return: Boolean array indicating stationary samples.
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
    if use_mag:
        return filter_.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t])
    else:
        return filter_.updateIMU(q, gyr=gyr[t], acc=acc[t])

def update_quaternion_mahony(q, t, filter_, gyr, acc, mag, use_mag):
    if use_mag:
        return filter_.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t])
    else:
        return filter_.updateIMU(q, gyr=gyr[t], acc=acc[t])

def estimate_position(name: str, gyr, acc, mag, time, sample_rate, stationary, use_madgwick=True, use_mag=True):
    """
    Estimate position from IMU data using Madgwick or Mahony filter.

    :param name: Filter name.
    :param gyr: Gyroscope data.
    :param acc: Accelerometer data.
    :param mag: Magnetometer data.
    :param time: Time vector.
    :param sample_rate: Sampling rate.
    :param stationary: Stationary mask.
    :param use_madgwick: Whether to use Madgwick filter.
    :param use_mag: Whether to include magnetometer.
    :return: Estimated position.
    """
    if use_madgwick:
        filter_ = Madgwick(frequency=sample_rate, gain=0.005 if use_mag else 0.01)
        q = axang2quat([0, 0, 1], np.deg2rad(45))
        update_function = update_quaternion_madgwick
    else:
        filter_ = Mahony(Kp=1.5, Ki=0.01, frequency=sample_rate)
        q = np.array([1.0, 0.0, 0.0, 0.0])
        acc_init = np.median(acc[time <= time[0] + 2], axis=0)
        for _ in range(2000):
            q = filter_.updateIMU(q, gyr=np.zeros(3), acc=acc_init)
        update_function = update_quaternion_mahony

    quats = np.zeros((len(time), 4))
    for t in range(len(time)):
        if not use_madgwick:
            filter_.Kp = 0.5 if stationary[t] else 0.0
        q_new = update_function(q, t, filter_, gyr, acc, mag, use_mag)
        if q_new is not None:
            q = q_new
        quats[t] = q

    acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)])
    acc_earth -= np.array([0, 0, 1])
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
            print(f"\n")
            print(f"{'*'*33}  {foot_label}  {'*'*33}\n ")
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
            
            gps_pos, gps_final = compute_gps_positions(df,config)

            errors = {}
            if args.verbose >= 2:
                print("Estimation results vs GPS:\n")
                
            for name, pos in resultados.items():
                error = np.linalg.norm(pos[-1, :2] - gps_final)
                errors[name] = error
                print(f"- {name:<25} → Error: {error:7.2f} m", end="")
                distance = np.max(np.linalg.norm(pos, axis=1))
                print(f" | Max distance: {distance:7.2f} m")
            print(f"{'-'*80}")


            output_file = None
            if output_dir and args.output_mode in ("save", "both"): output_file = os.path.join(output_dir, f"{base_name}_trajectory.png")

            plot_trajectories(resultados, errors, gps_pos, gps_final, title=f"Trajectory Comparison - {foot_label}", save_path=output_file)

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
