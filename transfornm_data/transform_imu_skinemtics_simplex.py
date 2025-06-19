import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import signal
from scipy.signal import butter, filtfilt

# === CONFIGURACIÓN DE RUTAS ===
SKINEMATICS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "external_repos", "scikit_kinematics")
)
sys.path.insert(0, SKINEMATICS_PATH)

from skinematics.sensors.manual import MyOwnSensor
from skinematics import vector
from skinematics.quat import Quaternion 

# Definicion de path
input_file = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_tabuenca_left.xlsx"
output_dir = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\transfornm_data\out_transfornm_data"
img_dir = os.path.join(output_dir, "img")
output_file = os.path.join(output_dir, "dat_2024_tabuenca_left_transform_1.xlsx")

# Carga de datos
df = pd.read_excel(input_file)
df = df.sort_values(by="_time").reset_index(drop=True)
df['_time'] = pd.to_datetime(df['_time'])

# Ajuste de df
cols_sensor = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz']
for col in cols_sensor:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# Frecuencia de muestreo
df['delta_time'] = df['_time'].diff().dt.total_seconds()
df.loc[0, 'delta_time'] = df.loc[1, 'delta_time']
freq = 1 / df['delta_time'].mean()
print(f"Frecuencia de muestreo estimada: {freq:.2f} Hz")

# Datos para IMU
acc = df[['Ax', 'Ay', 'Az']].values
gyr = df[['Gx', 'Gy', 'Gz']].values
mag = df[['Mx', 'My', 'Mz']].values
in_data = {'rate': freq, 'acc': acc, 'omega': gyr, 'mag': mag}

# Procesamiento con scikit-kinematics
imu = MyOwnSensor(in_file='manual', in_data=in_data)
imu.set_qtype('madgwick')
imu.calc_position()  # <-- llamada a la función

# Extraemos posición y velocidad
position = imu.pos  # (N, 3)
velocity = imu.vel  # (N, 3)

# Guardar en el DataFrame
df[['Vel_X', 'Vel_Y', 'Vel_Z']] = velocity
df[['Pos_X', 'Pos_Y', 'Pos_Z']] = position

# Guardar a Excel
os.makedirs(output_dir, exist_ok=True)
df.to_excel(output_file, index=False)
print(f"Archivo exportado: {output_file}")

# Función para guardar gráficas
def guardar_grafica(nombre, fig):
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, f"{nombre}.png")
    fig.savefig(path)
    print(f"Guardado: {path}")
    plt.close(fig)

# Graficar
# Aceleraciones
fig = plt.figure()
plt.plot(df['_time'], df['Ax'], label='Ax')
plt.plot(df['_time'], df['Ay'], label='Ay')
plt.plot(df['_time'], df['Az'], label='Az')
plt.title('Aceleraciones'); plt.xlabel('Tiempo'); plt.ylabel('m/s²')
plt.legend(); plt.grid(True)
guardar_grafica("aceleraciones_10", fig)

# Giroscopio
fig = plt.figure()
plt.plot(df['_time'], df['Gx'], label='Gx')
plt.plot(df['_time'], df['Gy'], label='Gy')
plt.plot(df['_time'], df['Gz'], label='Gz')
plt.title('Velocidades angulares (GIROSCOPIO)'); plt.xlabel('Tiempo'); plt.ylabel('rad/s')
plt.legend(); plt.grid(True)
guardar_grafica("giroscopio_10", fig)

# Velocidad
fig = plt.figure()
plt.plot(df['_time'], df['Vel_X'], label='Vel_X')
plt.plot(df['_time'], df['Vel_Y'], label='Vel_Y')
plt.plot(df['_time'], df['Vel_Z'], label='Vel_Z')
plt.title('Velocidad'); plt.xlabel('Tiempo'); plt.ylabel('m/s')
plt.legend(); plt.grid(True)
guardar_grafica("velocidad_10", fig)

# Posición
fig = plt.figure()
plt.plot(df['_time'], df['Pos_X'], label='Pos_X')
plt.plot(df['_time'], df['Pos_Y'], label='Pos_Y')
plt.plot(df['_time'], df['Pos_Z'], label='Pos_Z')
plt.title('Posición'); plt.xlabel('Tiempo'); plt.ylabel('m')
plt.legend(); plt.grid(True)
guardar_grafica("posicion_10", fig)

# Trayectoria XY
fig = plt.figure()
plt.plot(df['Pos_X'], df['Pos_Y'])
plt.title('Trayectoria XY'); plt.xlabel('X (m)'); plt.ylabel('Y (m)')
plt.axis('equal'); plt.grid(True)
guardar_grafica("trayectoria_XY_10", fig)
