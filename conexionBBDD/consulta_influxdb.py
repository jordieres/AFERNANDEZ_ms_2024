import yaml
from DbClassInflux import InfluxDB

# Leer la configuración desde el archivo YAML
with open('conexionBBDD\config_db.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Obtener los parámetros de conexión
bucket = config['influxdb']['bucket']
org = config['influxdb']['org']
token = config['influxdb']['token']
url = config['influxdb']['url']

# Inicializar el cliente
influx = InfluxDB(bucket, org, token, url)

# Realizar la consulta con las fechas de:  desde/hasta
# from_date = "2023-01-01T00:00:00Z"
# to_date = "2023-02-01T00:00:00Z"
from_date = "2024-10-01T00:00:00Z"
to_date = "2024-10-21T00:00:00Z"

# Consulta de influx
df = influx.query_data(from_date, to_date)
print(df)

# Consulta con agregación
# window_size = "10m"  
# df_agg = influx.query_with_aggregate_window(from_date, to_date, window_size)
# print(df_agg)
