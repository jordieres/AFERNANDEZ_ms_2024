from DbClassInflux import InfluxDB

# Parámetros de conexión
bucket = "Gait/autogen"
org = "UPM"
token = "Zx2jR8PD6h3YlS7HVsY5Han1SzF_iz7uk8n5z9BYRZ5q50lk8r1L18N-nFZiGCa57oowLgl8656pVpCig-GANg=="
url = "https://138.100.82.178:8086"  

# Inicializar cliente
influx = InfluxDB(bucket, org, token, url)

# Realizar la consulta pasando las fechas desde/hasta
from_date = "2023-01-01T00:00:00Z"
to_date = "2023-02-01T00:00:00Z"

# Consulta básica
df = influx.query_data(from_date, to_date)
print(df)

# Consulta con agregación
window_size = "10m"  # Ejemplo de ventana de agregación de 10 minutos
df_agg = influx.query_with_aggregate_window(from_date, to_date, window_size)
print(df_agg)
