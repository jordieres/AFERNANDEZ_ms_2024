import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt
from ahrs.filters import Madgwick, Mahony
from ahrs.common.orientation import q_conj, q_rot, axang2quat
from pyproj import Proj
import argparse



def load_data(file_path):
    """
    Load Excel data from the specified path.

    :param file_path: Path to the Excel file
    :return: DataFrame containing raw data
    """
    df = pd.read_excel(file_path)
    return df

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


def compute_gps_positions(df):
    """
    Convert GPS coordinates to local Cartesian positions.

    :param df: DataFrame with 'lat' and 'lng' columns.
    :return: GPS position array and final GPS position.
    """
    lat = df['lat'].to_numpy()
    lng = df['lng'].to_numpy()
    proj = Proj(proj='utm', zone=30, ellps='WGS84', south=False)
    x, y = proj(lng, lat)
    gps_pos = np.stack((x - x[0], y - y[0]), axis=1)
    return gps_pos, gps_pos[-1]

def plot_trajectories(resultados, errores, gps_pos, gps_final):
    """
    Plot estimated and GPS trajectories.

    :param resultados: Dictionary of positions.
    :param errores: Dictionary of errors.
    :param gps_pos: GPS positions.
    :param gps_final: Final GPS position.
    """
    plt.figure(figsize=(10, 8))
    for name, pos in resultados.items():
        plt.plot(pos[:, 0], pos[:, 1], label=f"{name} ({errores[name]:.2f} m)")
    plt.plot(gps_pos[:, 0], gps_pos[:, 1], 'k--', label="GPS (reference)")
    plt.plot(gps_final[0], gps_final[1], 'ko', label="GPS final")
    plt.title("Trajectory Comparison (Estimated vs GPS)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    
    parser = argparse.ArgumentParser(description="IMU data processing pipeline")
    parser.add_argument("-f","--file_path", dest="file_path", type=str, required=True, help="Path to Excel file with IMU data")
    parser.add_argument("--threshold", type=float, default=0.1, help="Stationary detection threshold")
    args = parser.parse_args()

    df = load_data(args.file_path)
    df = resample_to_40hz(df)
    time, sample_rate, gyr, acc, mag, sample_period = preprocess_data(df)
    stationary = detect_stationary(acc, sample_rate)


    resultados = {
        "Madgwick with mag": estimate_position("Madgwick with mag", gyr, acc, mag, time, sample_rate, stationary, True, True),
        "Madgwick without mag": estimate_position("Madgwick without mag", gyr, acc, mag, time, sample_rate, stationary, True, False),
        "Mahony with mag": estimate_position("Mahony with mag", gyr, acc, mag, time, sample_rate, stationary, False, True),
        "Mahony without mag": estimate_position("Mahony without mag", gyr, acc, mag, time, sample_rate, stationary, False, False),
    }

    gps_pos, gps_final = compute_gps_positions(df)


    errores = {}
    for name, pos in resultados.items():
        error = np.linalg.norm(pos[-1, :2] - gps_final)
        errores[name] = error
        print(f"{name} → Error vs GPS: {error:.2f} m")
        print("Máxima distancia recorrida:", np.max(np.linalg.norm(pos, axis=1)))

    plot_trajectories(resultados, errores, gps_pos, gps_final)

if __name__ == "__main__":
    main()
