import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import signal
from scipy.signal import butter, filtfilt

from skinematics.sensors.manual import MyOwnSensor
from skinematics import vector
from skinematics.quat import Quaternion 

# Definicion de path
input_file = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_tabuenca_left.xlsx"


# Carga de datos
df = pd.read_excel(input_file)
df = df.sort_values(by="_time").reset_index(drop=True)
df['_time'] = pd.to_datetime(df['_time'])

# Ajuste de df
cols_sensor = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz']
for col in cols_sensor:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# Frecuencia de muestreo (nº de veces que se registra una medicion)
df['delta_time'] = df['_time'].diff().dt.total_seconds()
df['delta_time'].iloc[0] = df['delta_time'].iloc[1]
freq = 1 / df['delta_time'].mean()
print(f"Frecuencia de muestreo estimada: {freq:.2f} Hz")

# Matrices de cada sensor
acc = df[['Ax', 'Ay', 'Az']].values
gyr = df[['Gx', 'Gy', 'Gz']].values
mag = df[['Mx', 'My', 'Mz']].values
in_data = {'rate': freq, 'acc': acc, 'omega': gyr, 'mag': mag}

# Orientacion --> Trabajo con cuaterniones
imu = MyOwnSensor(in_file='manual', in_data=in_data)
imu.set_qtype('madgwick')
quaternions = imu.quat

# Aceleracion en marco inercial (sist de coordenadas --> utilizo gravedad real)
gravity = vector.rotate_vector(np.tile([0, 0, 9.81], (len(quaternions), 1)), quaternions)
acc_inertial = vector.rotate_vector(acc, quaternions) - gravity

# Suaviza aceleracion cuando hay variaciones altas
def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

acc_inertial = butter_lowpass_filter(acc_inertial, cutoff=5, fs=freq)

# Deteccion de reposo --> si hay reposo la aceleracion= gravedad
acc_norm = np.linalg.norm(acc, axis=1)
reposo = np.abs(acc_norm - 9.81) < 0.1

# Correccion de offset de acceleracion cuando hay reposo
acc_offset = acc_inertial[reposo].mean(axis=0) if np.any(reposo) else np.zeros(3)
acc_inertial -= acc_offset

# Integracion de velocidad (para datos mas reales)
velocity = np.cumsum(acc_inertial / freq, axis=0)

# Corrreccion de velocidad en reposo
for i in range(1, len(reposo)):
    if reposo[i] and reposo[i - 1]:
        velocity[i] = [0, 0, 0]

# === FILTRADO SUAVE DE VELOCIDAD ===
velocity = butter_lowpass_filter(velocity, cutoff=0.3, fs=freq)

# Integración de posición
position = np.cumsum(velocity / freq, axis=0)

#Guardar resultados en nuevo df
df[['Vel_X', 'Vel_Y', 'Vel_Z']] = velocity
df[['Pos_X', 'Pos_Y', 'Pos_Z']] = position



# Aceleraciones
fig = plt.figure()
plt.plot(df['_time'], df['Ax'], label='Ax')
plt.plot(df['_time'], df['Ay'], label='Ay')
plt.plot(df['_time'], df['Az'], label='Az')
plt.title('Aceleraciones'); plt.xlabel('Tiempo'); plt.ylabel('m/s²')
plt.legend(); plt.grid(True)


# Giroscopio
fig = plt.figure()
plt.plot(df['_time'], df['Gx'], label='Gx')
plt.plot(df['_time'], df['Gy'], label='Gy')
plt.plot(df['_time'], df['Gz'], label='Gz')
plt.title('Velocidades angulares (GIROSCOPIO)'); plt.xlabel('Tiempo'); plt.ylabel('rad/s')
plt.legend(); plt.grid(True)


# Velocidad
fig = plt.figure()
plt.plot(df['_time'], df['Vel_X'], label='Vel_X')
plt.plot(df['_time'], df['Vel_Y'], label='Vel_Y')
plt.plot(df['_time'], df['Vel_Z'], label='Vel_Z')
plt.title('Velocidad'); plt.xlabel('Tiempo'); plt.ylabel('m/s')
plt.legend(); plt.grid(True)


# Posición
fig = plt.figure()
plt.plot(df['_time'], df['Pos_X'], label='Pos_X')
plt.plot(df['_time'], df['Pos_Y'], label='Pos_Y')
plt.plot(df['_time'], df['Pos_Z'], label='Pos_Z')
plt.title('Posición'); plt.xlabel('Tiempo'); plt.ylabel('m')
plt.legend(); plt.grid(True)


# Trayectoria XY
fig = plt.figure()
plt.plot(df['Pos_X'], df['Pos_Y'])
plt.title('Trayectoria XY'); plt.xlabel('X (m)'); plt.ylabel('Y (m)')
plt.axis('equal'); plt.grid(True)


plt.show()

