
import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt
from ahrs.filters import Madgwick
import matplotlib.pyplot as plt

# === CARGA Y PREPROCESAMIENTO ===
ruta_archivo = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_prueba6.xlsx"
df = pd.read_excel(ruta_archivo)

# Cargar datos con separador ';' y convertir coma a punto

df = df.astype({col: float for col in df.columns if col != '_time'})
df['_time'] = pd.to_datetime(df['_time'])
df['dt'] = df['_time'].diff().dt.total_seconds().fillna(0.02)

# === FILTRADO DE LA ACELERACIÓN ===
def butter_lowpass_filter(data, cutoff=2.5, fs=50, order=2):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)

for axis in ['Ax', 'Ay', 'Az']:
    df[axis] = butter_lowpass_filter(df[axis], fs=1/df['dt'].mean())

# === INTEGRACIÓN PARA VELOCIDAD Y POSICIÓN ===
df['Vx'] = cumulative_trapezoid(df['Ax'], dx=df['dt'], initial=0)
df['Vy'] = cumulative_trapezoid(df['Ay'], dx=df['dt'], initial=0)
df['Vz'] = cumulative_trapezoid(df['Az'], dx=df['dt'], initial=0)

df['Px'] = cumulative_trapezoid(df['Vx'], dx=df['dt'], initial=0)
df['Py'] = cumulative_trapezoid(df['Vy'], dx=df['dt'], initial=0)
df['Pz'] = cumulative_trapezoid(df['Vz'], dx=df['dt'], initial=0)

# === FILTRO MADGWICK PARA ORIENTACIÓN ===
madgwick = Madgwick()
quaternions = []
q = np.array([1.0, 0.0, 0.0, 0.0])

for i in range(len(df)):
    gyr = df.loc[i, ['Gx', 'Gy', 'Gz']].values
    acc = df.loc[i, ['Ax', 'Ay', 'Az']].values
    q = madgwick.updateIMU(q, gyr=gyr, acc=acc)
    quaternions.append(q)

quaternions = np.array(quaternions)
df['Qw'], df['Qx'], df['Qy'], df['Qz'] = quaternions.T

# === VISUALIZACIÓN BÁSICA ===
plt.figure(figsize=(10, 6))
plt.plot(df['_time'], df['Px'], label='Posición X')
plt.plot(df['_time'], df['Py'], label='Posición Y')
plt.plot(df['_time'], df['Pz'], label='Posición Z')
plt.legend()
plt.title('Estimación de Posición por Doble Integración')
plt.xlabel('Tiempo')
plt.ylabel('Posición (m)')
plt.grid()
plt.tight_layout()
plt.show()



df.to_excel(r'C:/Users/Gliglo/OneDrive - Universidad Politécnica de Madrid/Documentos/UPM/TFG/Proyecto_TFG/AFERNANDEZ_ms_2024/transfornm_data/imu_resultado_prueba.xlsx')