import sys
import os

# Añadir la ruta del módulo skinematics a sys.path
sys.path.append(os.path.abspath("C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\external_repos\scikit_kinematics"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skinematics import imus

# Leer el archivo CSV
excel_file = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_prueba10.xlsx"
df = pd.read_csv(excel_file, delimiter='\t')

# Reemplazar comas por puntos y convertir a float
for col in ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# Asegurarse de que _time es tipo datetime
df['_time'] = pd.to_datetime(df['_time'])

# Calcular frecuencia de muestreo promedio
df['delta_time'] = df['_time'].diff().dt.total_seconds()
freq = 1 / df['delta_time'].mean()

print(f"Frecuencia de muestreo estimada: {freq:.2f} Hz")

# Extraer aceleraciones y velocidades angulares como arrays Nx3
acc = df[['Ax', 'Ay', 'Az']].values
gyr = df[['Gx', 'Gy', 'Gz']].values

# Crear el objeto IMU de scikit-kinematics
imu = imus.IMU(data=acc, rate=gyr, freq=freq)

# Calcular orientación, velocidad y posición
imu.calc_orientation()  # Estima orientación
imu.calc_position()     # Estima velocidad y posición

# Extraer resultados
velocity = imu.vel  # Velocidad en m/s (Nx3)
position = imu.pos  # Posición en metros (Nx3)

# Mostrar la posición final
print("Última posición estimada (x, y, z):", position[-1])

# Visualización de la trayectoria (solo X e Y)
plt.plot(position[:, 0], position[:, 1])
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Trayectoria estimada')
plt.grid(True)
plt.axis('equal')
plt.show()
