import pandas as pd
import folium

# Cargar datos
ruta_archivo = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_prueba10.xlsx"
df = pd.read_excel(ruta_archivo)

# Asegurar que la columna de tiempo es tipo datetime y esta ordenada
df['_time'] = pd.to_datetime(df['_time'])
df = df.sort_values(by='_time')

# Filtrar filas en las que cambian la latitud o longitud
df = df.loc[(df['Latitude'].shift() != df['Latitude']) | (df['Longitude'].shift() != df['Longitude'])]

# Reiniciar índice por orden
df = df.reset_index(drop=True)

# Crear mapa centrado en el primer punto
inicio = [df.loc[0, 'Latitude'], df.loc[0, 'Longitude']]
mapa = folium.Map(location=inicio, zoom_start=18)

# Dibujar la trayectoria como una línea
coordenadas = df[['Latitude', 'Longitude']].values.tolist()
folium.PolyLine(coordenadas, color='blue', weight=4).add_to(mapa)

# Marcar inicio y fin
folium.Marker(location=coordenadas[0], popup="Inicio", icon=folium.Icon(color='green')).add_to(mapa)
folium.Marker(location=coordenadas[-1], popup="Fin", icon=folium.Icon(color='red')).add_to(mapa)

# Guardar mapa a archivo HTML
mapa.save('trayectoria_gps.html')
