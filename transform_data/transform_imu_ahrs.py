
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ahrs
import pandas as pd
from ahrs.common.orientation import q_conj, q_rot
from set_var import compute_time_parameters

# === CONFIGURACIÓN ===
file_path = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_tabuenca_left.xlsx"

# === CARGA Y CÁLCULO DE PARÁMETROS TEMPORALES ===
results = compute_time_parameters(file_path)
if not results:
    raise RuntimeError("Time parameters could not be retrieved.")

df = results["dataframe"]
start_time = results["start_time"]
stop_time = results["stop_time"]
sample_period = results["sample_period"]
sample_rate = results["frequency"]

# === PREPROCESADO DATA ===
time = np.arange(0, len(df)) * sample_period
df.drop(columns=['_time'], inplace=True)
print(" Dimensiones originales: ",df.shape)

df.drop_duplicates(subset=['Ax','Ay','Az','Gx','Gy','Gz','Mx','My','Mz'], keep='first', inplace=True)
print("Dimensiones tras eliminación de duplicados: ",df.shape)

time = np.arange(0, len(df)) * sample_period

gyr = df.iloc[:, 5:8].values * np.pi / 180  # convertir a rad/s
acc = df.iloc[:, 2:5].values

# cortar por tiempo para trabajar con la parte util de los datos
idx = (time >= start_time) & (time <= stop_time)
time = time[idx]
gyr = gyr[idx]
acc = acc[idx]


# Magnitud de la aceleración
acc_mag = np.linalg.norm(acc, axis=1)

# Frecuencia de corte = 0.01 Hz (más permisiva)
cutoff_hp = 0.01  # Hz
b, a = signal.butter(1, cutoff_hp / (sample_rate / 2), 'highpass')
acc_hp = signal.filtfilt(b, a, acc_mag, padtype='odd', padlen=3*(max(len(a), len(b)) - 1))

# Filtro pasa-baja
cutoff_lp = 5.0  # Hz
b, a = signal.butter(1, cutoff_lp / (sample_rate / 2), 'lowpass')
acc_lp = signal.filtfilt(b, a, np.abs(acc_hp), padtype='odd', padlen=3*(max(len(a), len(b)) - 1))

# percentil dinamico bajo del movimiento
threshold = np.percentile(acc_lp, 90) * 0.5
threshold = 0.1 #---->  Se tiene que definir segun el grafico de los datos para marcar el umbral para tener el max numero de muestras

stationary = acc_lp < threshold

# Visualización
plt.figure(figsize=(10, 4))
plt.plot(time, acc_lp, label='acc_lp')
plt.axhline(threshold, color='red', linestyle='--', label=f'Umbral {threshold:.3f}')
plt.title("Magnitud de aceleración filtrada")
plt.xlabel("Tiempo (s)")
plt.ylabel("|acc_hp| filtrada")
plt.legend()
plt.grid()
plt.tight_layout()


# Reporte
print("Threshold usado:", threshold)
print("Stationary únicos:", np.unique(stationary))
print("Muestras estacionarias:", np.sum(stationary))
print("Muestras en movimiento:", np.sum(~stationary))


# Mostrar datos

# GIROSCOPIO
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(time, gyr, linewidth=0.5)
plt.title("Giroscopio")
plt.legend(["X", "Y", "Z"])

#ACELEROMETRO
plt.subplot(2, 1, 2)
plt.plot(time, acc, linewidth=0.5)
plt.plot(time, acc_lp, 'k:', label='Filtered Magnitude')
plt.plot(time, stationary.astype(float), 'k', label='Stationary')
plt.title("Acelerómetro")
plt.legend()



# Estimación de orientación
#mahony = ahrs.filters.Mahony(Kp=1.0, Ki=0.0, frequency=sample_rate)
mahony = ahrs.filters.Mahony(Kp=1.5, Ki=0.01, frequency=sample_rate) # cambio valores de Ki y Kp para mejorar deriva
q = np.array([1.0, 0.0, 0.0, 0.0])
init_idx = time <= time[0] + 2
#acc_init = acc[init_idx].mean(axis=0)
acc_init = np.median(acc[init_idx], axis=0)


for _ in range(2000):
    q = mahony.updateIMU(q, gyr=np.zeros(3), acc=acc_init)

quats = np.zeros((len(time), 4))
for t in range(len(time)):
    mahony.Kp = 0.5 if stationary[t] else 0.0
    q = mahony.updateIMU(q, gyr[t] * np.pi / 180, acc[t])
    quats[t] = q

# Aceleración en marco terrestre
acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)])
acc_earth -= np.array([0, 0, 1])
acc_earth *= 9.81
#acc_earth -= np.array([0, 0, 9.81])          # Quitar gravedad del marco terrestre ( ya la he restado antes)
print("Acc_earth promedio post-corrección:", np.mean(acc_earth, axis=0))
print("Acc init:", acc_init)
print("Norm acc init:", np.linalg.norm(acc_init))
# Integración: velocidad
vel = np.zeros_like(acc_earth)
for t in range(1, len(vel)):
    vel[t] = vel[t - 1] + acc_earth[t] * sample_period
    if stationary[t]:
        vel[t] = 0

# Comprobacion de parametros
print("Acc. promedio:", np.mean(acc_earth, axis=0))
print("Primer acc rotado:", acc_earth[0])
print("Acc_earth promedio:", np.mean(acc_earth, axis=0))
print("Velocidad final:", vel[-1])
print("Stationary únicos:", np.unique(stationary))
print("Quaterniones promedio:", np.mean(quats, axis=0))



# Corrección por deriva
vel_drift = np.zeros_like(vel)
starts = np.where(np.diff(stationary.astype(int)) == -1)[0] + 1
ends = np.where(np.diff(stationary.astype(int)) == 1)[0] + 1
for s, e in zip(starts, ends):
    drift_rate = vel[e - 1] / (e - s)
    drift = np.outer(np.arange(e - s), drift_rate)
    vel_drift[s:e] = drift
vel -= vel_drift

# Integración: posición
pos = np.zeros_like(vel)
for t in range(1, len(pos)):
    pos[t] = pos[t - 1] + vel[t] * sample_period



ime = np.arange(len(pos)) * sample_period

# GRAFICA DE POSICION
plt.figure(figsize=(15, 5))
plt.plot(time, pos[:, 0], 'r', label='x')
plt.plot(time, pos[:, 1], 'g', label='y')
plt.plot(time, pos[:, 2], 'b', label='z')
plt.xlabel("time (s)")
plt.ylabel("position (m)")
plt.title("position")
plt.legend()
plt.grid()


# GRAFICA DE VELOCIDAD
plt.figure(figsize=(15, 5))
plt.plot(time, vel[:, 0], color='red', label='x')
plt.plot(time, vel[:, 1], color='green', label='y')
plt.plot(time, vel[:, 2], color='blue', label='z')
plt.xlabel("time (s)")
plt.ylabel("velocity (m/s)")
plt.title("velocity")
plt.legend()
plt.grid()



# GRAFICAS TRAYECTORIA

# Gráfico XY
plt.figure()
plt.plot(pos[:, 0], pos[:, 1])
plt.title("Trayectoria (X vs Y)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.axis('equal')
plt.grid()


# Gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])
ax.set_title("Trayectoria 3D")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


# Mostrar todas las gracicas
plt.show()
