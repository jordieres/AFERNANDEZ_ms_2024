from scipy.interpolate import CubicSpline
import numpy as np
import pandas as pd
import argparse
from scipy import signal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ahrs.filters import Mahony
from ahrs.common.orientation import q_conj, q_rot


from scipy.interpolate import CubicSpline

def reinterpolar_a_40hz(df, time_col='_time', freq_hz=40, gap_threshold_ms=200):
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    df['delta'] = df[time_col].diff().dt.total_seconds() * 1000  # en ms
    df['session'] = (df['delta'] > gap_threshold_ms).cumsum()

    interpolados = []

    # Paso 3: Procesar cada segmento individualmente
    for session_id, grupo in df.groupby('session'):
        grupo = grupo.set_index(time_col)
        grupo = grupo.sort_index()

        start = grupo.index[0]
        end = grupo.index[-1]
        new_index = pd.date_range(start=start, end=end, freq=f'{int(1000/freq_hz)}ms')
        df_interp = pd.DataFrame(index=new_index)

        for col in grupo.columns.difference(['delta', 'session']):
            clean = grupo[col].dropna()
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
        interpolados.append(df_interp)

    resultado = pd.concat(interpolados, ignore_index=True)
    return resultado
 


def compute_time_parameters(file_path: str) -> dict:
    """
    Read an Excel file and compute basic timing parameters.

    Args:
        file_path (str): Path to the Excel file containing a '_time' column.

    Returns:
        dict: Dictionary containing:
            - start_time (float): Time of first sample (default 0).
            - stop_time (float): Time of last sample in seconds.
            - sample_period (float): Mean sampling interval in seconds.
            - frequency (float): Estimated sampling frequency in Hz.
            - dataframe (pd.DataFrame): Sorted DataFrame with '_time' as datetime.
    """
    df = pd.read_excel(file_path)
    if '_time' not in df.columns:
        raise ValueError("'_time' column not found.")

    df['_time'] = pd.to_datetime(df['_time'])
    df = df.sort_values('_time').reset_index(drop=True)
    dt = df['_time'].diff().dt.total_seconds().dropna()
    sample_period = dt.mean()
    frequency = 1 / sample_period
    stop_time = (len(df) - 1) * sample_period
    return {
        "start_time": 0,
        "stop_time": stop_time,
        "sample_period": sample_period,
        "frequency": frequency,
        "dataframe": df
    }

def filter_and_detect_stationary(acc: np.ndarray, sample_rate: float, threshold: float = 0.1) -> tuple:
    """
    Filter the acceleration signal and detect stationary periods.

    Args:
        acc (np.ndarray): Raw acceleration data of shape (N, 3).
        sample_rate (float): Sampling frequency in Hz.
        threshold (float): Magnitude threshold for detecting stationary state.

    Returns:
        tuple:
            - stationary (np.ndarray): Boolean array where True indicates stationary.
            - acc_lp (np.ndarray): Filtered magnitude of acceleration.
    """
    acc_mag = np.linalg.norm(acc, axis=1)
    b, a = signal.butter(1, 0.01 / (sample_rate / 2), 'highpass')
    acc_hp = signal.filtfilt(b, a, acc_mag)
    b, a = signal.butter(1, 5.0 / (sample_rate / 2), 'lowpass')
    acc_lp = signal.filtfilt(b, a, np.abs(acc_hp))
    stationary = acc_lp < threshold
    return stationary, acc_lp

def estimate_orientation(acc: np.ndarray, gyr: np.ndarray, stationary: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    Estimate orientation using Mahony filter from IMU data.

    Args:
        acc (np.ndarray): Accelerometer data of shape (N, 3).
        gyr (np.ndarray): Gyroscope data of shape (N, 3), in rad/s.
        stationary (np.ndarray): Boolean mask for stationary points.
        sample_rate (float): Sampling frequency in Hz.

    Returns:
        np.ndarray: Array of quaternions of shape (N, 4) representing orientation.
    """
    mahony = Mahony(Kp=1.5, Ki=0.01, frequency=sample_rate)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    acc_init = np.median(acc[:int(2 * sample_rate)], axis=0)
    for _ in range(2000):
        q = mahony.updateIMU(q, gyr=np.zeros(3), acc=acc_init)
    quats = np.zeros((len(acc), 4))
    for t in range(len(acc)):
        mahony.Kp = 0.5 if stationary[t] else 0.0
        q = mahony.updateIMU(q, gyr[t], acc[t])
        quats[t] = q
    return quats

def rotate_to_earth_frame(acc: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """
    Rotate body-frame acceleration vectors into the earth frame.

    Args:
        acc (np.ndarray): Raw acceleration data (N, 3).
        quats (np.ndarray): Orientation quaternions (N, 4).

    Returns:
        np.ndarray: Acceleration in earth frame (N, 3).
    """
    acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)])
    acc_earth -= np.array([0, 0, 1])
    acc_earth *= 9.81
    return acc_earth

def compute_velocity(acc_earth: np.ndarray, stationary: np.ndarray, sample_period: float) -> np.ndarray:
    """
    Compute velocity by integrating acceleration and correcting drift.

    Args:
        acc_earth (np.ndarray): Earth-frame acceleration (N, 3).
        stationary (np.ndarray): Boolean array indicating stationary periods.
        sample_period (float): Sampling period in seconds.

    Returns:
        np.ndarray: Velocity array of shape (N, 3).
    """
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
    return vel

def compute_position(vel: np.ndarray, sample_period: float) -> np.ndarray:
    """
    Compute position by integrating velocity.

    Args:
        vel (np.ndarray): Velocity array (N, 3).
        sample_period (float): Sampling period in seconds.

    Returns:
        np.ndarray: Position array (N, 3).
    """
    pos = np.zeros_like(vel)
    for t in range(1, len(pos)):
        pos[t] = pos[t - 1] + vel[t] * sample_period
    return pos

def plot_magnetometer(self):
    mag_norm = np.linalg.norm(self.mag, axis=1)
    print("Magnetometer norm (expected ~0.05 mT):", np.mean(mag_norm))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(self.mag[:, 0], self.mag[:, 1], self.mag[:, 2], s=1)
    ax.set_title("Magnetometer 3D Cloud")
    ax.set_xlabel("Mx (mT)")
    ax.set_ylabel("My (mT)")
    ax.set_zlabel("Mz (mT)")
    plt.show()

def visualize_data(time: np.ndarray, acc_lp: np.ndarray, threshold: float, gyr: np.ndarray, acc: np.ndarray, stationary: np.ndarray, pos: np.ndarray, vel: np.ndarray):
    """
    Plot filtered acceleration, gyroscope, accelerometer, position, velocity and 3D trajectory.

    Args:
        time (np.ndarray): Time vector.
        acc_lp (np.ndarray): Filtered acceleration magnitude.
        threshold (float): Threshold for stationary detection.
        gyr (np.ndarray): Gyroscope data.
        acc (np.ndarray): Accelerometer data.
        stationary (np.ndarray): Boolean stationary array.
        pos (np.ndarray): Position data.
        vel (np.ndarray): Velocity data.
    """
    plt.figure()
    plt.plot(time, acc_lp)
    plt.axhline(threshold, color='red', linestyle='--')
    plt.title("Filtered Acceleration Magnitude")

    plt.figure()
    plt.plot(time, gyr)
    plt.title("Gyroscope")

    plt.figure()
    plt.plot(time, acc)
    plt.plot(time, acc_lp, 'k:')
    plt.plot(time, stationary.astype(float), 'k')
    plt.title("Accelerometer")

    plt.figure()
    plt.plot(time, pos)
    plt.title("Position")

    plt.figure()
    plt.plot(time, vel)
    plt.title("Velocity")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])
    ax.set_title("3D Trajectory")

    plt.show()

def main():
    """
    Main function to run IMU signal processing pipeline using CLI arguments.
    """
    parser = argparse.ArgumentParser(description="IMU data processing pipeline")
    parser.add_argument("-f","--file_path", dest="file_path", type=str, required=True, help="Path to Excel file with IMU data")
    parser.add_argument("--threshold", type=float, default=0.1, help="Stationary detection threshold")
    args = parser.parse_args()

    results = compute_time_parameters(args.file_path)
    df = results["dataframe"]
    sample_period = results["sample_period"]
    sample_rate = results["frequency"]
    time = np.arange(len(df)) * sample_period

    acc = df.iloc[:, 2:5].values
    gyr = df.iloc[:, 5:8].values * np.pi / 180

    stationary, acc_lp = filter_and_detect_stationary(acc, sample_rate, threshold=args.threshold)
    quats = estimate_orientation(acc, gyr, stationary, sample_rate)
    acc_earth = rotate_to_earth_frame(acc, quats)
    vel = compute_velocity(acc_earth, stationary, sample_period)
    pos = compute_position(vel, sample_period)

    visualize_data(time, acc_lp, args.threshold, gyr, acc, stationary, pos, vel)

if __name__ == "__main__":
    main()
 

