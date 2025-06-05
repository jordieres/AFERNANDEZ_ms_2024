# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from ahrs.filters import Madgwick
 
# # Cargar los datos del archivo Excel y realizar preprocesado
# ruta_archivo = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_prueba6.xlsx"
# df = pd.read_excel(ruta_archivo)
 
# # Simulación de columna de tiempo si no existe
# frecuencia = 50  # 50 Hz -> dt = 0.02 s
# dt = 1 / frecuencia
# df['_time'] = pd.date_range(start='2023-01-01', periods=len(df), freq=f'{int(dt * 1000)}ms')
 
# # Extraer y preparar los datos de aceleración y giroscopio
# acc = df[['Ax', 'Ay', 'Az']].to_numpy()  # valores en m/s²
# gyr = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180  # convertir de deg/s a rad/s
# # Datos de magnetómetro
# mag = df[['Mx', 'My', 'Mz']].to_numpy()
 
# # Crear objeto Madgwick con tasa de muestreo adecuada
# beta = 0.1  # para ajuste según necesites
# madgwick = Madgwick(beta=beta, freq=frecuencia)
 
# # Inicializar lista de resultados y primer cuaternión
# orientations = []
# q = np.array([1.0, 0.0, 0.0, 0.0])  # Cuaternión inicial

# # Procesar datos
# for i in range(len(df)):
#     q = madgwick.updateIMU(q, gyr[i], acc[i])
#     if mag is not None:
#         q = madgwick.updateMARG(q, gyr[i], acc[i], mag[i])
#     orientations.append(q)
 
# # Convertir quaternions a euler angles (roll, pitch, yaw) para orientación
# orientations = np.array(orientations)
# roll = np.arctan2(2*(orientations[:,0]*orientations[:,1] + orientations[:,2]*orientations[:,3]), 1 - 2*(orientations[:,1]**2 + orientations[:,2]**2))
# pitch = np.arcsin(2*(orientations[:,0]*orientations[:,2] - orientations[:,3]*orientations[:,1]))
# yaw = np.arctan2(2*(orientations[:,0]*orientations[:,3] + orientations[:,1]*orientations[:,2]), 1 - 2*(orientations[:,2]**2 + orientations[:,3]**2))
 
# # Añadir los resultados de orientación al DataFrame
# df['Roll'] = roll
# df['Pitch'] = pitch
# df['Yaw'] = yaw
 
# # Grafica de la orientación estimada
# plt.figure(figsize=(10, 6))
# plt.plot(df['_time'], df['Roll'], label='Roll (IMU)', color='blue')
# plt.plot(df['_time'], df['Pitch'], label='Pitch (IMU)', color='green')
# plt.plot(df['_time'], df['Yaw'], label='Yaw (IMU)', color='red')
# plt.xlabel('Tiempo')
# plt.ylabel('Ángulo (radianes)')
# plt.title('Orientación Estimada con Madgwick')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
 
# # Guardar el resultado
# df.to_excel(r'C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\transfornm_data\imu_resultado_modificado_madgwick.xlsx')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick
from scipy.spatial.transform import rotation as R

# Cargar datos
ruta_archivo = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_prueba6.xlsx"
df = pd.read_excel(ruta_archivo)

# Simular columna de tiempo
frecuencia = 50
dt = 1 / frecuencia
df['_time'] = pd.date_range(start='2023-01-01', periods=len(df), freq=f'{int(dt * 1000)}ms')

# Sensores
acc = df[['Ax', 'Ay', 'Az']].to_numpy()
gyr = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180
mag = df[['Mx', 'My', 'Mz']].to_numpy()

# Filtro Madgwick
madgwick = Madgwick(beta=0.1, freq=frecuencia)
q = np.array([1.0, 0.0, 0.0, 0.0])
quaternions = []

# Estimar orientación
for i in range(len(df)):
    q = madgwick.updateMARG(q, gyr[i], acc[i], mag[i])
    quaternions.append(q)

quaternions = np.array(quaternions)

# Convertir a ángulos de Euler (para visualización)
px = np.arctan2(2*(quaternions[:,0]*quaternions[:,1] + quaternions[:,2]*quaternions[:,3]),
                  1 - 2*(quaternions[:,1]**2 + quaternions[:,2]**2))
py = np.arcsin(2*(quaternions[:,0]*quaternions[:,2] - quaternions[:,3]*quaternions[:,1]))
pz = np.arctan2(2*(quaternions[:,0]*quaternions[:,3] + quaternions[:,1]*quaternions[:,2]),
                 1 - 2*(quaternions[:,2]**2 + quaternions[:,3]**2))

# Añadir orientación como "posición"
df['Px'] = px
df['Py'] = py
df['Pz'] = pz

# Inicializar velocidad
vel = np.zeros((len(df), 3))
gravity = np.array([0.0, 0.0, 9.81])  # m/s²

# Estimar velocidad integrando aceleración transformada al marco global
for i in range(1, len(df)):
    rot = R.from_quat([quaternions[i][1], quaternions[i][2], quaternions[i][3], quaternions[i][0]])  # AHRS: w,x,y,z → scipy: x,y,z,w
    acc_global = rot.apply(acc[i])  # transformar aceleración al marco global
    acc_global -= gravity  # quitar gravedad
    vel[i] = vel[i-1] + acc_global * dt  # integración simple

df['Vx'] = vel[:, 0]
df['Vy'] = vel[:, 1]
df['Vz'] = vel[:, 2]

# Graficar orientación
plt.figure(figsize=(10, 6))
plt.plot(df['_time'], df['Px'], label='Px (Roll)', color='blue')
plt.plot(df['_time'], df['Py'], label='Py (Pitch)', color='green')
plt.plot(df['_time'], df['Pz'], label='Pz (Yaw)', color='red')
plt.xlabel('Tiempo')
plt.ylabel('Ángulo (radianes)')
plt.title('Orientación Estimada con Madgwick')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Graficar velocidad
plt.figure(figsize=(10, 6))
plt.plot(df['_time'], df['Vx'], label='Vx', color='purple')
plt.plot(df['_time'], df['Vy'], label='Vy', color='orange')
plt.plot(df['_time'], df['Vz'], label='Vz', color='gray')
plt.xlabel('Tiempo')
plt.ylabel('Velocidad (m/s)')
plt.title('Velocidad Estimada por Integración')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Guardar resultados
df.to_excel(r'C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\transfornm_data\imu_resultado_madgwick.xlsx')
