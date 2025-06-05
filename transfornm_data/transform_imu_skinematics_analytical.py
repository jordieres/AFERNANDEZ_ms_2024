import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skinematics.imus import analytical
 

ruta_archivo = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_prueba6.xlsx"
df = pd.read_excel(ruta_archivo)
 
frecuencia = 50         
dt = 1 / frecuencia
df['_time'] = pd.date_range(start='2023-01-01', periods=len(df), freq=f'{int(dt * 1000)}L')
 


acc = df[['Ax', 'Ay', 'Az']].to_numpy()            # m/s² o g según se haya registrado
gyr = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180  # convertir de grados/s a rad/s
 

# Definir orientación inicial (matriz identidad) y posición inicial (vector cero)
R_initialOrientation = np.eye(3)
initialPosition = np.zeros(3)
 
# Llamada a la función analytical, que devuelve:
#   q   -> orientación en forma de quaternion (N x 4)
#   pos -> posición en el espacio (N x 3)
#   vel -> velocidad (N x 3)
q, pos, vel = analytical(R_initialOrientation=R_initialOrientation,
                          omega=gyr,
                          initialPosition=initialPosition,
                          accMeasured=acc,
                          rate=frecuencia)
 

df['Px'] = pos[:, 0]
df['Py'] = pos[:, 1]
df['Pz'] = pos[:, 2]
df['Vx'] = vel[:, 0]
df['Vy'] = vel[:, 1]
df['Vz'] = vel[:, 2]
 
# Grafica de la trayectoria estimada
plt.figure(figsize=(10, 6))
plt.plot(df['Px'], df['Py'], label='Trayectoria Estimada (Analytical)', color='orange')
plt.xlabel('Posición X (m)')
plt.ylabel('Posición Y (m)')
plt.title('Trayectoria Estimada con Skinematics - Función Analytical')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Guardar el resultado

df.to_excel(r'C:/Users/Gliglo/OneDrive - Universidad Politécnica de Madrid/Documentos/UPM/TFG/Proyecto_TFG/AFERNANDEZ_ms_2024/transfornm_data/imu_resultado_skinematics_funcAnalitical.xlsx')