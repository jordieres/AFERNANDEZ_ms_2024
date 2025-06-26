
import numpy as np
import pandas as pd
import argparse
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
    acc = df[['Ax', 'Ay', 'Az']].to_numpy()
    mag = df[['Mx', 'My', 'Mz']].to_numpy() / 1000

    return time, sample_rate, gyr, acc, mag, sample_period


def detect_stationary_zones(acc, sample_rate):
    """
    Detect stationary zones using filtered acceleration.

    :param acc: Accelerometer data
    :param sample_rate: Sampling rate
    :return: Tuple of stationary flags, low-pass acceleration and threshold
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
    Compute orientation using Madgwick and integrate acceleration to get position.

    :return: Tuple of quaternions, Earth-frame acceleration, velocity, and position
    """
    madgwick = Madgwick(frequency=sample_rate, gain=0.041)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    quats = np.zeros((len(time), 4))
    quats[0] = q

    for t in range(1, len(time)):
        q = madgwick.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t])
        quats[t] = q

    acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)])
    acc_earth -= np.array([0, 0, 1])
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
        drift = np.outer(np.arange(e - s), drift_rate)
        vel_drift[s:e] = drift
    vel -= vel_drift

    pos = np.zeros_like(vel)
    for t in range(1, len(pos)):
        pos[t] = pos[t - 1] + vel[t] * sample_period

    return quats, acc_earth, vel, pos

def compare_with_gps(df, pos):
    """
    Compare final estimated position with GPS coordinates if available.

    :return: Tuple of GPS XY coordinates and final error
    """
    if 'lat' in df.columns and 'lng' in df.columns:
        lat = df['lat'].to_numpy()
        lng = df['lng'].to_numpy()
        proj = Proj(proj='utm', zone=30, ellps='WGS84', south=False)
        x_gps, y_gps = proj(lng, lat)
        gps_pos = np.stack((x_gps - x_gps[0], y_gps - y_gps[0]), axis=1)
        error = np.linalg.norm(pos[-1, :2] - gps_pos[-1])
        return gps_pos, error
    return None, None

def plot_results(time, acc_lp, threshold, pos, vel, gps_pos=None):
    """
    Generate plots for diagnostics, trajectory, and comparison with GPS.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(time, acc_lp, label='acc_lp')
    plt.axhline(threshold, color='r', linestyle='--')
    plt.title("Filtered Acceleration")
    plt.xlabel("Time (s)")
    plt.grid()
    plt.legend()

    plt.figure(figsize=(15, 5))
    plt.plot(time, pos[:, 0], 'r', label='x')
    plt.plot(time, pos[:, 1], 'g', label='y')
    plt.plot(time, pos[:, 2], 'b', label='z')
    plt.title("Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()

    plt.figure(figsize=(15, 5))
    plt.plot(time, vel[:, 0], 'r', label='x')
    plt.plot(time, vel[:, 1], 'g', label='y')
    plt.plot(time, vel[:, 2], 'b', label='z')
    plt.title("Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(pos[:, 0], pos[:, 1])
    plt.axis('equal')
    plt.title("XY Trajectory")
    plt.grid()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])
    ax.set_title("3D Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if gps_pos is not None:
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

    plt.show()

def main():

    parser = argparse.ArgumentParser(description="IMU data processing pipeline with Madwick")
    parser.add_argument("-f","--file_path", dest="file_path", type=str, required=True, help="Path to Excel file with IMU data")
    parser.add_argument("--threshold", type=float, default=0.1, help="Stationary detection threshold")
    args = parser.parse_args()


    df = load_data(args.file_path)
    df = resample_to_40hz(df)

    time, sample_rate, gyr, acc, mag, sample_period = preprocess_data(df)
    
    stationary, acc_lp, threshold = detect_stationary_zones(acc, sample_rate)
    quats, acc_earth, vel, pos = estimate_orientation_and_position(time, gyr, acc, mag, sample_period, sample_rate, stationary)

    gps_pos, error = compare_with_gps(df, pos)


    # === VERIFICACIONES ===
    acc_init = np.median(acc[time <= time[0] + 2], axis=0)
    mag_init = np.median(mag[time <= time[0] + 2], axis=0)

    print("====== VERIFICACIONES ======")
    print(f"Número total de muestras interpoladas: {len(df)}")
    print(f"Frecuencia estimada (Hz): {sample_rate:.2f}")
    print(f"Periodo estimado (s): {sample_period:.5f}")
    print(f"Muestras estacionarias: {np.sum(stationary)}")
    print(f"Muestras en movimiento: {np.sum(~stationary)}")
    print(f"Porcentaje estacionario: {np.mean(stationary) * 100:.1f}%")
    print("Norma acc_init (esperado ~1):", np.linalg.norm(acc_init))
    print("Norma mag_init:", np.linalg.norm(mag_init))
    print("Promedio de acc_earth (debería estar cerca de 0):", np.mean(acc_earth, axis=0))
    print("Velocidad final:", vel[-1])
    print("Norma velocidad final:", np.linalg.norm(vel[-1]))
    print("Posición final:", pos[-1])
    print("Máxima distancia recorrida:", np.max(np.linalg.norm(pos, axis=1)))
    print("Quaterníon promedio:", np.mean(quats, axis=0))

    if gps_pos is not None:
        print(f"GPS distance Error: {error:.2f} m")
    else:
        print("No 'lat' and 'lng' columns found in the data. Cannot compare with GPS.")

    plot_results(time, acc_lp, threshold, pos, vel, gps_pos)

if __name__ == "__main__":
    main()
