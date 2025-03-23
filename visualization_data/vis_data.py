# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import folium
# from scipy.spatial.transform import Rotation as R
# from scipy.signal import butter, filtfilt


# #  Cargar los datos desde el archivo Excel
# ruta_archivo = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_prueba6.xlsx"
# df = pd.read_excel(ruta_archivo)

# # Reemplazar comas por puntos en valores numéricos si es necesario
# df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)

# # Convertir las columnas numéricas a tipo float
# for col in df.columns[1:]:  # Asumiendo que la primera columna es '_time'
#     df[col] = pd.to_numeric(df[col], errors="coerce")

# df["_time"] = pd.to_datetime(df["_time"])  # Convertir columna de tiempo

# # Asegurar que los datos sean numéricos
# for col in df.columns[1:]:
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# # Filtrar ruido con filtro Butterworth
# def butter_lowpass_filter(data, cutoff=0.1, fs=50, order=2):
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     return filtfilt(b, a, data)

# df["Ax"] = butter_lowpass_filter(df["Ax"])
# df["Ay"] = butter_lowpass_filter(df["Ay"])
# df["Az"] = butter_lowpass_filter(df["Az"])

# # Integrar Aceleración para obtener Velocidad y Posición
# dt = np.diff(df["_time"]).mean().total_seconds()  # Intervalo de tiempo medio
# velocity_x = np.cumsum(df["Ax"] * dt)  # Integración simple
# velocity_y = np.cumsum(df["Ay"] * dt)
# position_x = np.cumsum(velocity_x * dt)
# position_y = np.cumsum(velocity_y * dt)

# # Visualización de la trayectoria
# plt.figure(figsize=(8, 6))
# plt.plot(position_x, position_y, marker="o", linestyle="-", alpha=0.7)
# plt.xlabel("Posición X (m)")
# plt.ylabel("Posición Y (m)")
# plt.title("Trayectoria Estimada del Paciente")
# plt.grid()
# plt.show()
import numpy as np
import pandas as pd
from ahrs.filters import Madgwick
from ahrs.common.orientation import acc2q, q2euler

#  Cargar los datos desde el archivo Excel
ruta_archivo = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_prueba6.xlsx"
df = pd.read_excel(ruta_archivo)

# 📌 2. Convertir datos a arrays numpy

gyro = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180  # Convertir a rad/s
accel = df[['Ax', 'Ay', 'Az']].to_numpy()  # Acelerómetro en g
mag = df[['Mx', 'My', 'Mz']].to_numpy()  # Magnetómetro en microteslas

# 📌 3. Aplicar filtro de Madgwick
madgwick = Madgwick()
quaternions = np.zeros((len(df), 4))  # Inicializar matriz de cuaterniones
q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # Asegurar tipo float64

for i in range(len(df)):
    q = madgwick.updateIMU(q, gyr=gyro[i], acc=accel[i])  # Actualizar cuaternión
    quaternions[i] = q

# 📌 4. Convertir cuaterniones a ángulos de Euler (yaw, pitch, roll)
euler_angles = np.apply_along_axis(q2euler, 1, quaternions)

# 📌 5. Agregar resultados al DataFrame
df[['q_w', 'q_x', 'q_y', 'q_z']] = quaternions
df[['yaw', 'pitch', 'roll']] = euler_angles

# # 📌 6. Guardar el nuevo dataset con orientación
# df.to_csv("datos_con_orientacion.", index=False)


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