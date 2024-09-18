import sys
import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient
from datetime import datetime
import urllib3

# Desactivar las advertencias de SSL (solo para desarrollo)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class InfluxDB:
    def __init__(self, bucketv2, orgv2, tokenv2, url='https://138.100.82.178:8086'):
        self.bucket = bucketv2
        self.org = orgv2
        # Aumentar timeout a 60 segundos
        self.client = InfluxDBClient(url=url, token=tokenv2, org=self.org, verify_ssl=False, timeout=60_000)
        if self.bucket == 'Gait/autogen':
            self.measurement = ['Gait']
    
    def query_data(self):
        # Consulta con rango limitado (últimos 30 días)
        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: -100d)
        |> filter(fn: (r) => r._measurement == "Gait")
        |> map(fn: (r) => ({{r with _value: float(v: r._value)}}))
        |> keep(columns: ["_time", "_value", "_measurement", "_field"])
        |> limit(n: 5)
    '''

        # Realizar la consulta
        result = self.client.query_api().query(org=self.org, query=query)

        # Procesar los resultados
        data = []
        for table in result:
            for record in table.records:
                data.append([record.get_time(), record.get_value()])

        # Crear DataFrame
        df = pd.DataFrame(data, columns=["Time", "Value"])

        return df


# Parámetros de conexión
bucket = "Gait/autogen"
org = "UPM"
token = "Zx2jR8PD6h3YlS7HVsY5Han1SzF_iz7uk8n5z9BYRZ5q50lk8r1L18N-nFZiGCa57oowLgl8656pVpCig-GANg=="
url = "https://138.100.82.178:8086"

# Inicializar el cliente de InfluxDB
influx = InfluxDB(bucket, org, token, url)

# Realizar la consulta y mostrar los resultados
df = influx.query_data()
print(df)
