
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skinematics.sensors import ImuData
from skinematics.imus import IMU

# ======= 1. Cargar los datos del archivo Excel ========
ruta_archivo = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_prueba6.xlsx"
df = pd.read_excel(ruta_archivo)

# ======= 2. Simular columna de tiempo si no existe ========
# Frecuencia de muestreo (Hz). Usa el valor real si lo conoces.
frecuencia = 50  # 50 Hz -> dt = 0.02 s
dt = 1 / frecuencia
df['_time'] = pd.date_range(start='2023-01-01', periods=len(df), freq=f'{int(dt * 1000)}L')

# ======= 3. Preparar los datos para Skinematics ========
# Convertir giroscopio a radianes por segundo
gyr = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180  # deg/s -> rad/s
acc = df[['Ax', 'Ay', 'Az']].to_numpy()               # en m/s² directamente

# ======= 4. Crear objeto IMUData y calcular con Skinematics ========
imu_data = ImuData(acc=acc, gyr=gyr, rate=frecuencia)
imu = IMU(data=imu_data)

# ======= 5. Obtener velocidad y posición estimadas ========
vel = imu.vel   # (N x 3) matriz
pos = imu.pos   # (N x 3) matriz

# ======= 6. Añadir al DataFrame ========
df['Vx'] = vel[:, 0]
df['Vy'] = vel[:, 1]
df['Vz'] = vel[:, 2]

df['Px'] = pos[:, 0]
df['Py'] = pos[:, 1]
df['Pz'] = pos[:, 2]

# ======= 7. Graficar posición estimada ========
plt.figure(figsize=(10, 6))
plt.plot(df['Px'], df['Py'], label='Trayectoria estimada (IMU)', color='orange')
plt.xlabel('Posición X (m)')
plt.ylabel('Posición Y (m)')
plt.title('Trayectoria IMU con Skinematics')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ======= 8. Guardar el resultado (opcional) ========
guardar = True
if guardar:
    df.to_excel('imu_resultado_con_velocidad_posicion.xlsx', index=False)