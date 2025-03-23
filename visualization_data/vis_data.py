import numpy as np
import pandas as pd
from ahrs.filters import Madgwick
from ahrs.common.orientation import acc2q, q2euler

#  Cargar los datos desde el archivo Excel
ruta_archivo = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_prueba6.xlsx"
df = pd.read_excel(ruta_archivo)

#2. Convertir datos a arrays numpy

gyro = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180  # Convertir a rad/s
accel = df[['Ax', 'Ay', 'Az']].to_numpy()  # Acelerómetro en g
mag = df[['Mx', 'My', 'Mz']].to_numpy()  # Magnetómetro en microteslas

# 3. Aplicar filtro de Madgwick
madgwick = Madgwick()
quaternions = np.zeros((len(df), 4))  # Inicializar matriz de cuaterniones
q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # Asegurar tipo float64

for i in range(len(df)):
    q = madgwick.updateIMU(q, gyr=gyro[i], acc=accel[i])  # Actualizar cuaternión
    quaternions[i] = q

# 4. Convertir cuaterniones a ángulos de Euler (yaw, pitch, roll)
euler_angles = np.apply_along_axis(q2euler, 1, quaternions)

# 5. Agregar resultados al DataFrame
df[['q_w', 'q_x', 'q_y', 'q_z']] = quaternions
df[['yaw', 'pitch', 'roll']] = euler_angles


import matplotlib.pyplot as plt
df['_time'] = pd.to_datetime(df['_time'], errors='coerce')
plt.figure(figsize=(10, 6))
plt.plot(df['_time'], df['yaw'], label='Yaw')
plt.plot(df['_time'], df['pitch'], label='Pitch')
plt.plot(df['_time'], df['roll'], label='Roll')
plt.legend()
plt.xlabel("Tiempo")
plt.ylabel("Ángulos (radianes)")
plt.title("Orientación del pie a lo largo del tiempo")
plt.xticks(rotation=45)
plt.show()