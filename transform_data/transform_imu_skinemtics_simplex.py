import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend
from scipy.interpolate import CubicSpline
from scipy.ndimage import label
from skinematics.sensors.manual import MyOwnSensor
from skinematics import vector


def load__data(file_path):
        return pd.read_excel(file_path)

def preprocess_imu_signals(df):
    acc = df[['Ax', 'Ay', 'Az']].to_numpy() * 9.81
    gyr = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180
    mag = df[['Mx', 'My', 'Mz']].to_numpy() * 0.1
    return acc, gyr, mag

def reinterpolar_a_40hz(df, time_col, target_freq_hz, gap_threshold_ms=200):
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
        new_index = pd.date_range(start=start, end=end, freq=f'{int(1000/target_freq_hz)}ms')
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
 

def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, data, axis=0)


def compute_quaternions(acc, gyr, mag, freq):
    in_data = {'rate': freq, 'acc': acc, 'omega': gyr, 'mag': mag}
    imu = MyOwnSensor(in_file='manual', in_data=in_data)
    imu.set_qtype('madgwick')
    return np.array(imu.quat, dtype=float)


def compute_velocity_position(df, acc, gyr, mag, freq, time_col):
    quaternions = compute_quaternions(acc, gyr, mag, freq)
    gravity_local = np.tile([0, 0, 9.81], (len(quaternions), 1))
    gravity_global = vector.rotate_vector(gravity_local, quaternions)
    acc_global = vector.rotate_vector(acc, quaternions) - gravity_global
    acc_global = butter_lowpass_filter(acc_global, cutoff=5, fs=freq)
    acc_global = detrend(acc_global, axis=0, type='constant')

    acc_norm = np.linalg.norm(acc_global + gravity_global, axis=1)
    acc_norm_smooth = butter_lowpass_filter(acc_norm.reshape(-1, 1), cutoff=1.5, fs=freq).flatten()
    reposo = np.abs(acc_norm_smooth - 9.81) < 0.2

    if reposo.sum() < 10:
        reposo[:int(2 * freq)] = True

    acc_offset = acc_global[reposo].mean(axis=0)
    acc_inertial = acc_global - acc_offset
    acc_inertial = np.clip(acc_inertial, -30, 30)

    sample_period = 1.0 / freq
    velocity = np.zeros_like(acc_inertial)
    for i in range(1, len(acc_inertial)):
        velocity[i] = velocity[i-1] + acc_inertial[i] * sample_period

    labels, num = label(reposo)
    for i in range(1, num + 1):
        idx = np.where(labels == i)[0]
        if len(idx) > 10:
            velocity[idx] = 0

    velocity = butter_lowpass_filter(velocity, cutoff=0.1, fs=freq)
    position = np.cumsum(velocity * sample_period, axis=0)
    position -= position[0]

    df[['Vel_X', 'Vel_Y', 'Vel_Z']] = velocity
    df[['Pos_X', 'Pos_Y', 'Pos_Z']] = position
    return df


def plot_results(df, time_col):
    plt.figure()
    plt.plot(df[time_col], df['Vel_X'], label='Vel_X')
    plt.plot(df[time_col], df['Vel_Y'], label='Vel_Y')
    plt.plot(df[time_col], df['Vel_Z'], label='Vel_Z')
    plt.title('Velocidad'); plt.xlabel('Tiempo'); plt.ylabel('m/s'); plt.grid(True); plt.legend()

    plt.figure()
    plt.plot(df[time_col], df['Pos_X'], label='Pos_X')
    plt.plot(df[time_col], df['Pos_Y'], label='Pos_Y')
    plt.plot(df[time_col], df['Pos_Z'], label='Pos_Z')
    plt.title('Posición'); plt.xlabel('Tiempo'); plt.ylabel('m'); plt.grid(True); plt.legend()

    plt.figure()
    plt.plot(df['Pos_X'], df['Pos_Y'])
    plt.title('Trayectoria XY'); plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.axis('equal'); plt.grid(True)
    plt.show()


def main():
    file_path = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_tabuenca_left.xlsx"
    
    time_col = '_time'
    target_freq_hz = 40
    df = load__data(file_path)
    df_interp = reinterpolar_a_40hz(df, time_col=time_col, target_freq_hz=target_freq_hz)
    df_interp.dropna(inplace=True)

    acc, gyr, mag = preprocess_imu_signals(df_interp)
    df_final = compute_velocity_position(df_interp, acc, gyr, mag, target_freq_hz, time_col)

    
    plot_results(df_final, time_col)

if __name__ == '__main__':
    main()
