import pandas as pd
import numpy as np
import folium

# Carga y preparacion de datos de InfluxDB 
ruta = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_reducido.xlsx"
df = pd.read_excel(ruta)
df.columns = [col.strip() for col in df.columns]

for col in df.columns[1:]:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

df["_time"] = pd.to_datetime(df["_time"])
df["dt"] = df["_time"].diff().dt.total_seconds().fillna(0.01)

# Suavizado y corrección de bias
for axis in ["Ax", "Ay", "Az"]:
    df[axis] = df[axis].rolling(window=5, center=True, min_periods=1).mean()
    df[axis] -= df[axis].iloc[:100].mean()

# Heading desde magnetómetro
df["heading"] = np.arctan2(df["My"], df["Mx"])  # en radianes

# Inicializar arrays
N = len(df)
velocity = np.zeros((N, 3))   # vx, vy, vz
position = np.zeros((N, 3))   # x, y, z

# Integración con rotación
for i in range(1, N):
    dt = df["dt"].iloc[i]
    ax, ay, az = df.loc[i, ["Ax", "Ay", "Az"]]
    heading = df["heading"].iloc[i]

    acc_x = ax * np.cos(heading) - ay * np.sin(heading)
    acc_y = ax * np.sin(heading) + ay * np.cos(heading)
    acc_z = az  # vertical relativa

    acc_global = np.array([acc_x, acc_y, acc_z])

    velocity[i] = velocity[i-1] + acc_global * dt
    position[i] = position[i-1] + velocity[i] * dt

# Convertir a coordenadas GPS (referencia de Madrid)
start_lat = 40.4529  
start_lon = -3.7266

latitudes = start_lat + position[:,1] / 111000
longitudes = start_lon + position[:,0] / (111000 * np.cos(np.radians(start_lat)))
altitudes = position[:,2]

# Detección de eventos de apoyo
umbral = 50  #modificar para ajustar (hacer mas pruebas)
pasos_S0 = df["S0"] > umbral
pasos_S1 = df["S1"] > umbral
pasos_S2 = df["S2"] > umbral

# Crear mapa con folium
m = folium.Map(location=[start_lat, start_lon], zoom_start=18)

for i, (lat, lon) in enumerate(zip(latitudes, longitudes)):
    if pasos_S0.iloc[i]:
        color = "blue" 
    elif pasos_S1.iloc[i]:
        color = "green" 
    elif pasos_S2.iloc[i]:
        color = "red"  
    else:
        color = "gray" 

    folium.CircleMarker(
        location=[lat, lon],
        radius=2,
        color=color,
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

# Guardar mapa
m.save(r'C:/Users/Gliglo/OneDrive - Universidad Politécnica de Madrid/Documentos/UPM/TFG/Proyecto_TFG/AFERNANDEZ_ms_2024/transfornm_data/trayectoria_sensores_contacto.html')
print("Mapa guardado correctamente'")