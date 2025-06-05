# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from filterpy.kalman import ExtendedKalmanFilter
 
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
# mag = df[['Mx', 'My', 'Mz']].to_numpy()
 
# # Inicializar el filtro de Kalman extendido (EKF)
# ekf = ExtendedKalmanFilter(dim_x=6, dim_z=3)
# ekf.x = np.array([1, 0, 0, 0, 0, 0])    # Estado inicial
# ekf.P *= 0.01                           # Covarianza inicial
# ekf.R = np.eye(3) * 0.01                # Covarianza de medición
# ekf.Q = np.eye(6) * 0.01                # Covarianza de proceso
 
# # Definir función de transición
# def f(x, dt):
#     F = np.eye(6)
#     F[0:3, 3:6] = np.eye(3) * dt
#     return F @ x
 
# # Definir función de medición
# def h(x):
#     return x[0:3]
 
# # Inicializar listas para orientación y posición
# quaternions = []
# posiciones = []
# velocidades = []
 
# # Procesar datos
# for i in range(len(df)):
#     ekf.predict()
#     ekf.update(acc[i])
#     quaternions.append(ekf.x[:3])
#     posiciones.append(ekf.x[:3])
#     velocidades.append(ekf.x[3:])
 
# # Convertir listas a arrays
# quaternions = np.array(quaternions)
# posiciones = np.array(posiciones)
# velocidades = np.array(velocidades)
 
# # Añadir los resultados al DataFrame
# df['Px'] = posiciones[:, 0]
# df['Py'] = posiciones[:, 1]
# df['Pz'] = posiciones[:, 2]
# df['Vx'] = velocidades[:, 0]
# df['Vy'] = velocidades[:, 1]
# df['Vz'] = velocidades[:, 2]
 
# # Grafica de la trayectoria estimada
# plt.figure(figsize=(10, 6))
# plt.plot(df['Px'], df['Py'], label='Trayectoria Estimada (IMU)', color='orange')
# plt.xlabel('Posición X (m)')
# plt.ylabel('Posición Y (m)')
# plt.title('Trayectoria Estimada con FilterPy - Datos IMU')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
 
# # Guardar el resultado
# df.to_excel(r'C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\transfornm_data\imu_resultado_modificado_filterpy.xlsx')
from scipy.spatial.transform import Rotation as R

# Crear una rotación de 90 grados sobre el eje X
rot = R.from_euler('x', 90, degrees=True)
print(rot.as_quat())  # Debería imprimir un quaternion