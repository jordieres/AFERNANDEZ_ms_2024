# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from ahrs.filters import Madgwick
# # from ahrs.common.orientation import q2euler
 
# # # Cargar los datos del archivo Excel y realizar preprocesado
# # ruta_archivo = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_prueba6.xlsx"
# # df = pd.read_excel(ruta_archivo)
 
# # # Simulación de columna de tiempo si no existe
# # frecuencia = 50  # 50 Hz -> dt = 0.02 s
# # dt = 1 / frecuencia
# # df['_time'] = pd.date_range(start='2023-01-01', periods=len(df), freq=f'{int(dt * 1000)}ms')
 
# # # Extraer y preparar los datos de aceleración y giroscopio
# # acc = df[['Ax', 'Ay', 'Az']].to_numpy()  # valores en m/s²
# # gyr = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180  # convertir de deg/s a rad/s
# # mag = df[['Mx', 'My', 'Mz']].to_numpy()
 
# # # Crear objeto Madgwick utilizando AHRS
# # filtro_madgwick = Madgwick(frequency=frecuencia)
 
# # # Inicializar listas para orientación y posición
# # quaternions = np.zeros((len(df), 4))
# # quaternions[0] = [1, 0, 0, 0]  # Quaternión inicial
 
# # # Procesar datos
# # for i in range(1, len(df)):
# #     quaternions[i] = filtro_madgwick.updateIMU(quaternions[i-1], gyr[i], acc[i])
 
# # # Convertir quaternions a ángulos de Euler (roll, pitch, yaw) para orientación
# # orientaciones = np.array([q2euler(q) for q in quaternions])

# # # Añadir los resultados de orientación al DataFrame
# # df['Roll'] = orientaciones[:, 0]
# # df['Pitch'] = orientaciones[:, 1]
# # df['Yaw'] = orientaciones[:, 2]
 
# # # Graficar la orientación estimada
# # plt.figure(figsize=(10, 6))
# # plt.plot(df['_time'], df['Roll'], label='Roll (IMU)', color='blue')
# # plt.plot(df['_time'], df['Pitch'], label='Pitch (IMU)', color='green')
# # plt.plot(df['_time'], df['Yaw'], label='Yaw (IMU)', color='red')
# # plt.xlabel('Tiempo')
# # plt.ylabel('Ángulo (radianes)')
# # plt.title('Orientación Estimada con AHRS Madgwick')
# # plt.legend()
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()
 
# # # Guardar el resultado en Excel
# # df.to_excel(r'C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\transfornm_data\imu_resultado_ahrs.xlsx')


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from ahrs.filters import Madgwick
# from ahrs.common.orientation import q2euler
# from scipy.spatial.transform import Rotation as R

# # Cargar archivo
# ruta_archivo = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_prueba6.xlsx"
# df = pd.read_excel(ruta_archivo)

# # Crear columna de tiempo
# frecuencia = 50  # Hz
# dt = 1 / frecuencia
# df['_time'] = pd.date_range(start='2024-11-02', periods=len(df), freq=f'{int(dt * 1000)}ms')

# # Sensores
# acc = df[['Ax', 'Ay', 'Az']].to_numpy()
# gyr = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180
# mag = df[['Mx', 'My', 'Mz']].to_numpy()

# # Filtro Madgwick
# filtro = Madgwick(frequency=frecuencia)
# quats = np.zeros((len(df), 4))
# quats[0] = np.array([1.0, 0.0, 0.0, 0.0])

# # Posición y velocidad
# vel = np.zeros((len(df), 3))
# pos = np.zeros((len(df), 3))

# # Bucle
# for i in range(1, len(df)):
#     # Estimar orientación
#     q = filtro.updateIMU(quats[i-1], gyr[i], acc[i])
#     quats[i] = q

#     # Transformar aceleración a global
#     rot = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy usa formato [x, y, z, w]
#     acc_global = rot.apply(acc[i])

#     # Quitar gravedad
#     acc_global[2] -= 9.81

#     # Integración
#     vel[i] = vel[i-1] + acc_global * dt
#     pos[i] = pos[i-1] + vel[i] * dt

# # Calcular ejes Euler
# euler = np.array([q2euler(q) for q in quats])
# df['Roll'] = euler[:, 0]
# df['Pitch'] = euler[:, 1]
# df['Yaw'] = euler[:, 2]

# # Guardar Px,Py,Pz y Vx,Vy,Vz
# df['Px'] = pos[:, 0]
# df['Py'] = pos[:, 1]
# df['Pz'] = pos[:, 2]
# df['Vx'] = vel[:, 0]
# df['Vy'] = vel[:, 1]
# df['Vz'] = vel[:, 2]

# # Guardar a Excel
# df.to_excel(r'C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\transfornm_data\imu_resultado_ahrs.xlsx', index=False)

# # Gráfica comparativa
# plt.figure(figsize=(10, 6))
# plt.plot(df['Px'], df['Py'], label='Trayectoria Madgwick', color='blue')
# plt.xlabel('Px (m)')
# plt.ylabel('Py (m)')
# plt.title('Trayectoria estimada con AHRS-Madgwick')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick
from ahrs.common.orientation import q2euler
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt

# === 1. Cargar archivo ===
ruta_archivo = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_prueba6.xlsx"
df = pd.read_excel(ruta_archivo)

frecuencia = 50  # Hz
dt = 1 / frecuencia
df['_time'] = pd.date_range(start='2024-11-02', periods=len(df), freq=f'{int(dt * 1000)}ms')

# === 2. Filtro pasa bajos a la aceleración ===
def butter_filter(data, cutoff=5, fs=50, order=2):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data, axis=0)

acc_raw = df[['Ax', 'Ay', 'Az']].to_numpy()
acc = butter_filter(acc_raw, cutoff=5, fs=frecuencia)

# === 3. Sensores ===
gyr = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180
mag = df[['Mx', 'My', 'Mz']].to_numpy()

# === 4. Filtro Madgwick ===
filtro = Madgwick(frequency=frecuencia)
quats = np.zeros((len(df), 4))
quats[0] = np.array([1.0, 0.0, 0.0, 0.0])

vel = np.zeros((len(df), 3))
pos = np.zeros((len(df), 3))

# === 5. Bucle principal ===
for i in range(1, len(df)):
    q = filtro.updateIMU(quats[i-1], gyr[i], acc[i])
    quats[i] = q

    rot = R.from_quat([q[1], q[2], q[3], q[0]])
    acc_global = rot.apply(acc[i])
    acc_global[2] -= 9.81

    vel[i] = vel[i-1] + acc_global * dt
    pos[i] = pos[i-1] + vel[i] * dt

    # ZUPT: cuando hay presión en el pie (S0, S1, S2 altos), asumir que no hay movimiento
    if df.loc[i, 'S0'] > 700 and df.loc[i, 'S1'] > 700 and df.loc[i, 'S2'] > 700:
        vel[i] = np.zeros(3)  # Reinicio de velocidad para evitar deriva

# === 6. Calcular orientación en Euler ===
euler = np.array([q2euler(q) for q in quats])
df['Roll'] = euler[:, 0]
df['Pitch'] = euler[:, 1]
df['Yaw'] = euler[:, 2]

# === 7. Guardar resultados ===
df['Px'] = pos[:, 0]
df['Py'] = pos[:, 1]
df['Pz'] = pos[:, 2]
df['Vx'] = vel[:, 0]
df['Vy'] = vel[:, 1]
df['Vz'] = vel[:, 2]

salida = r'C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\transfornm_data\imu_resultado_ahrs.xlsx'
df.to_excel(salida, index=False)

# === 8. Gráfico de trayectoria ===
plt.figure(figsize=(10, 6))
plt.plot(df['Px'], df['Py'], label='Trayectoria Madgwick Filtrada', color='blue')
plt.xlabel('Px (m)')
plt.ylabel('Py (m)')
plt.title('Trayectoria estimada (con filtrado y ZUPT)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
