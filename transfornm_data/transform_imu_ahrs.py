import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from ahrs.filters import Mahony
from ahrs.common.orientation import q_rot, q_conj
from mpl_toolkits.mplot3d import Axes3D

# ------------------ CONFIGURACIÓN ------------------
excel_file = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_prueba6.xlsx"
sample_rate = 256  # Hz
sample_period = 1 / sample_rate
init_period = 2  # segundos
# ----------------------------------------------------

# Leer datos (soporta coma como decimal)
df = pd.read_excel(excel_file)
df = df.applymap(lambda x: float(str(x).replace(',', '.')) if isinstance(x, str) else x)

# Extraer sensores
acc = df[['Ax', 'Ay', 'Az']].values
gyr = df[['Gx', 'Gy', 'Gz']].values * np.pi / 180  # convertir a rad/s
mag = df[['Mx', 'My', 'Mz']].values
time = np.arange(len(acc)) * sample_period

# Magnitud de acelerómetro y filtrado
acc_mag = np.linalg.norm(acc, axis=1)
b, a = signal.butter(1, 0.001/(0.5*sample_rate), 'highpass')
acc_mag_filt = signal.filtfilt(b, a, acc_mag)
acc_mag_filt = np.abs(acc_mag_filt)
b, a = signal.butter(1, 5/(0.5*sample_rate), 'lowpass')
acc_mag_filt = signal.filtfilt(b, a, acc_mag_filt)
stationary = acc_mag_filt < 0.05

# Inicialización filtro Mahony
mahony = Mahony(Kp=1, Ki=0, KpInit=1.0, frequency=sample_rate)
q = np.array([1.0, 0.0, 0.0, 0.0])
init_idx = time <= init_period
acc_mean = acc[init_idx].mean(axis=0)
for _ in range(2000):
    q = mahony.updateIMU(q, gyr=np.zeros(3), acc=acc_mean)

# Estimar orientación para todo el tiempo
quats = np.zeros((len(time), 4))
for i in range(len(time)):
    mahony.Kp = 0.5 if stationary[i] else 0
    q = mahony.updateIMU(q, gyr=gyr[i], acc=acc[i])
    quats[i] = q

# Rotar aceleraciones al marco terrestre
acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)])
acc_earth -= np.array([0, 0, 1])  # quitar gravedad
acc_earth *= 9.81  # convertir g a m/s^2

# Integrar para obtener velocidad
vel = np.zeros_like(acc_earth)
for i in range(1, len(vel)):
    vel[i] = vel[i-1] + acc_earth[i] * sample_period
    if stationary[i]:
        vel[i] = np.zeros(3)

# Corregir deriva por integración
vel_drift = np.zeros_like(vel)
start = np.where(np.diff(stationary.astype(int)) == -1)[0] + 1
end = np.where(np.diff(stationary.astype(int)) == 1)[0] + 1
for s, e in zip(start, end):
    drift_rate = vel[e-1] / (e - s)
    drift = np.outer(np.arange(e - s), drift_rate)
    vel_drift[s:e] = drift
vel -= vel_drift

# Integrar velocidad para obtener posición
pos = np.zeros_like(vel)
for i in range(1, len(pos)):
    pos[i] = pos[i-1] + vel[i] * sample_period

# ------------------ GRAFICAR ------------------
plt.figure()
plt.plot(time, vel)
plt.title("Velocidad")
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad (m/s)")
plt.legend(['X', 'Y', 'Z'])
plt.grid()

plt.figure()
plt.plot(time, pos)
plt.title("Posición")
plt.xlabel("Tiempo (s)")
plt.ylabel("Posición (m)")
plt.legend(['X', 'Y', 'Z'])
plt.grid()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])
ax.set_title("Trayectoria 3D")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
plt.tight_layout()
plt.show()
