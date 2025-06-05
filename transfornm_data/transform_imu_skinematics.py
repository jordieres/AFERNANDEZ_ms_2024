# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from skinematics.sensors.manual import MyOwnSensor
# # from filterpy.kalman import ExtendedKalmanFilter

# # #  Cargar los datos del archivo Excel y realizar preprocesado

# # ruta_archivo = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_reducido.xlsx"
# # df = pd.read_excel(ruta_archivo)
 
# # # Simulación de columna de tiempo si no existe
# # frecuencia = 50         # 50 Hz -> dt = 0.02 s
# # dt = 1 / frecuencia
# # df['_time'] = pd.to_datetime(df['_time'])


# # # Extraer y preparar los datos de aceleración y giroscopio
# # acc = df[['Ax', 'Ay', 'Az']].to_numpy()  # valores en m/s² (o en g, según corresponda)
# # gyr = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180  # convertir de deg/s a rad/s
 
# # # Datos de magnetómetro ( si no para probar se puede hacer: mag = None )
# # mag = df[['Mx','My','Mz']].to_numpy()
 

# # # Crear el objeto sensor usando la clase manual MyOwnSensor

# # in_data = {
# #     'rate': frecuencia,
# #     'acc': acc,
# #     'omega': gyr,
# #     'mag': mag
# # }
 
# # # La cadena 'in_file' puede ser cualquier identificador; en este caso indicamos que son datos manuales.
# # sensor = MyOwnSensor(in_file="Datos manuales", in_data=in_data)
 
# # # Calculo de la posición (y la orientación si procede) usando el método calc_position.
# # sensor.calc_position()
 

# # # Extraer los resultados y añadirlos al DataFrame

# # # Los atributos calculados son:
# # #   sensor.pos -> matriz de posición (N x 3)
# # #   sensor.vel -> matriz de velocidad (N x 3)
# # df['Px'] = sensor.pos[:, 0]
# # df['Py'] = sensor.pos[:, 1]
# # df['Pz'] = sensor.pos[:, 2]
# # df['Vx'] = sensor.vel[:, 0]
# # df['Vy'] = sensor.vel[:, 1]
# # df['Vz'] = sensor.vel[:, 2]
 

# # # Grafica de la trayectoria estimada
# # plt.figure(figsize=(10, 6))
# # plt.plot(df['Px'], df['Py'], label='Trayectoria Estimada (IMU)', color='orange')
# # plt.xlabel('Posición X (m)')
# # plt.ylabel('Posición Y (m)')
# # plt.title('Trayectoria Estimada con Skinematics - Datos Manuales')
# # plt.legend()
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()
 

# # # Guardar el resultado

# # df.to_excel(r'C:/Users/Gliglo/OneDrive - Universidad Politécnica de Madrid/Documentos/UPM/TFG/Proyecto_TFG/AFERNANDEZ_ms_2024/transfornm_data/imu_resultado_skinematics.xlsx')

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from skinematics.sensors.manual import MyOwnSensor
# import time

# # --- CONFIGURACIÓN ---
# ruta_archivo = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_reducido.xlsx"
# frecuencia = 50  # Hz
# usar_magnetometro = False  # Cambia a True si realmente necesitas orientación absoluta
# limite_filas = 5000        # Para pruebas. Usa None para procesar todo

# # --- CARGA Y PREPROCESADO DE DATOS ---
# df = pd.read_excel(ruta_archivo)
# if limite_filas:
#     df = df.head(limite_filas)

# df['_time'] = pd.to_datetime(df['_time'])

# acc = df[['Ax', 'Ay', 'Az']].to_numpy()  # en m/s²
# gyr = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180  # deg/s -> rad/s

# # --- PREPARACIÓN DE DATOS PARA EL SENSOR ---
# in_data = {
#     'rate': frecuencia,
#     'acc': acc,
#     'omega': gyr,
# }

# if usar_magnetometro:
#     mag = df[['Mx', 'My', 'Mz']].to_numpy()
#     in_data['mag'] = mag

# sensor = MyOwnSensor(in_file="Datos manuales", in_data=in_data)

# # --- CÁLCULO DE POSICIÓN (puede tardar) ---
# print("Calculando posición...")
# start_time = time.time()
# sensor.calc_position()
# print(f"Cálculo completado en {time.time() - start_time:.2f} segundos")

# # --- GUARDAR RESULTADOS EN EL DATAFRAME ---
# df['Px'] = sensor.pos[:, 0]
# df['Py'] = sensor.pos[:, 1]
# df['Pz'] = sensor.pos[:, 2]
# df['Vx'] = sensor.vel[:, 0]
# df['Vy'] = sensor.vel[:, 1]
# df['Vz'] = sensor.vel[:, 2]

# # --- GRÁFICO DE TRAYECTORIA ---
# plt.figure(figsize=(10, 6))
# plt.plot(df['Px'], df['Py'], label='Trayectoria Estimada (IMU)', color='orange')
# plt.xlabel('Posición X (m)')
# plt.ylabel('Posición Y (m)')
# plt.title('Trayectoria Estimada con Skinematics (Reducido)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # --- GUARDAR RESULTADOS ---
# salida = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\transfornm_data\imu_resultado_skinematics.xlsx"
# df.to_excel(salida, index=False)
# print(f"Resultados guardados en: {salida}")







import pandas as pd
import numpy as np
from skinematics.imu import IMU
import matplotlib.pyplot as plt

# === 1. Cargar datos desde Excel ===
ruta_archivo = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_reducido.xlsx"
df = pd.read_excel(ruta_archivo)
df.replace({',': '.'}, regex=True, inplace=True)
df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
df['_time'] = pd.to_datetime(df['_time'])
df['dt'] = df['_time'].diff().dt.total_seconds().fillna(0.02)

# === 2. Preparar datos para skinematics ===
data = {
    'rate': int(1 / df['dt'].mean()),
    'Acc': df[['Ax', 'Ay', 'Az']].to_numpy(),
    'Gyr': df[['Gx', 'Gy', 'Gz']].to_numpy(),
    'Mag': df[['Mx', 'My', 'Mz']].to_numpy()
}

# === 3. Calcular orientación, velocidad y posición ===
imu = IMU_Base(data=data)
imu.calc_orientation()
imu.calc_position()

# === 4. Guardar resultados ===
df[['Vel_X', 'Vel_Y', 'Vel_Z']] = imu.vel
    
df[['Pos_X', 'Pos_Y', 'Pos_Z']] = imu.pos
df.to_excel('salida_skinematics.xlsx', index=False)

# === 5. Gráfico ===
plt.figure(figsize=(8, 6))
plt.plot(df['Pos_X'], df['Pos_Y'], label='Trayectoria (Skinematics)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Posición estimada - Skinematics')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('grafico_skinematics.png')
plt.show()

df.to_excel(r'C:/Users/Gliglo/OneDrive - Universidad Politécnica de Madrid/Documentos/UPM/TFG/Proyecto_TFG/AFERNANDEZ_ms_2024/transfornm_data/imu_resultado_skinematics.xlsx')