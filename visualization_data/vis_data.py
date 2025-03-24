import numpy as np
import pandas as pd
from ahrs.filters import Madgwick
from ahrs.common.orientation import acc2q, q2euler


import matplotlib.pyplot as plt
from clase_madgwick import MadgwickAHRS
from clase_quaternion import Quaternion

#  Cargar los datos desde el archivo Excel
ruta_archivo = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_prueba6.xlsx"
df = pd.read_excel(ruta_archivo)

# #2. Convertir datos a arrays numpy

# gyro = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180  # Convertir a rad/s
# accel = df[['Ax', 'Ay', 'Az']].to_numpy()  # Acelerómetro en g
# mag = df[['Mx', 'My', 'Mz']].to_numpy()  # Magnetómetro en microteslas

# # 3. Aplicar filtro de Madgwick
# madgwick = Madgwick()
# quaternions = np.zeros((len(df), 4))  # Inicializar matriz de cuaterniones
# q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # Asegurar tipo float64

# for i in range(len(df)):
#     q = madgwick.updateIMU(q, gyr=gyro[i], acc=accel[i])  # Actualizar cuaternión
#     quaternions[i] = q

# # 4. Convertir cuaterniones a ángulos de Euler (yaw, pitch, roll)
# euler_angles = np.apply_along_axis(q2euler, 1, quaternions)

# # 5. Agregar resultados al DataFrame
# df[['q_w', 'q_x', 'q_y', 'q_z']] = quaternions
# df[['yaw', 'pitch', 'roll']] = euler_angles


# import matplotlib.pyplot as plt
# df['_time'] = pd.to_datetime(df['_time'], errors='coerce')
# plt.figure(figsize=(10, 6))
# plt.plot(df['_time'], df['yaw'], label='Yaw')
# plt.plot(df['_time'], df['pitch'], label='Pitch')
# plt.plot(df['_time'], df['roll'], label='Roll')
# plt.legend()
# plt.xlabel("Tiempo")
# plt.ylabel("Ángulos (radianes)")
# plt.title("Orientación del pie a lo largo del tiempo")
# plt.xticks(rotation=45)
# plt.show()




# Cargar los datos
df = pd.read_csv("datos_pie_derecho.csv", delimiter="\t")  # Ajusta según el formato de tu dataset

# Convertir los datos correctamente
df["Gx"] = df["Gx"].str.replace(",", ".").astype(float) * np.pi / 180  # Convertir a rad/s
df["Gy"] = df["Gy"].str.replace(",", ".").astype(float) * np.pi / 180
df["Gz"] = df["Gz"].str.replace(",", ".").astype(float) * np.pi / 180
df["Ax"] = df["Ax"].str.replace(",", ".").astype(float)
df["Ay"] = df["Ay"].str.replace(",", ".").astype(float)
df["Az"] = df["Az"].str.replace(",", ".").astype(float)
df["Mx"] = df["Mx"].str.replace(",", ".").astype(float)
df["My"] = df["My"].str.replace(",", ".").astype(float)
df["Mz"] = df["Mz"].str.replace(",", ".").astype(float)

# Inicializar el filtro de Madgwick
madgwick = MadgwickAHRS(sampleperiod=1/256, beta=0.1)

# Almacenar los resultados
orientations = []

# Procesar cada fila
for _, row in df.iterrows():
    gyro = [row["Gx"], row["Gy"], row["Gz"]]
    accel = [row["Ax"], row["Ay"], row["Az"]]
    mag = [row["Mx"], row["My"], row["Mz"]]
    
    # Actualizar filtro de fusión de sensores
    madgwick.update(gyroscope=gyro, accelerometer=accel, magnetometer=mag)
    
    # Obtener cuaternión y convertir a ángulos de Euler
    roll, pitch, yaw = madgwick.quaternion.to_euler_angles()
    orientations.append([roll, pitch, yaw])

# Convertir resultados en DataFrame
orientations_df = pd.DataFrame(orientations, columns=["Roll", "Pitch", "Yaw"])
df = pd.concat([df, orientations_df], axis=1)

# Graficar la orientación en el tiempo
plt.figure(figsize=(10, 5))
plt.plot(df["_time"], df["Yaw"], label="Yaw")
plt.plot(df["_time"], df["Pitch"], label="Pitch")
plt.plot(df["_time"], df["Roll"], label="Roll")
plt.legend()
plt.xlabel("Tiempo")
plt.ylabel("Ángulos (rad)")
plt.title("Evolución de la orientación del pie derecho")
plt.show()
