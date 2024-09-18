import sys
import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient
from datetime import datetime

class InfluxDB:
    def __init__(self, bucketv2, orgv2, tokenv2, url='https://138.100.82.178:8086'):
        self.bucket = bucketv2
        self.org = orgv2
        self.client = InfluxDBClient(url=url, token=tokenv2, org=self.org)
        if self.bucket == 'Gait/autogen':
            self.measurement = ['Gait']
    
    def query_data(self):
        # Crea la consulta para obtener datos de tiempo en el measurement Gait
        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: 0)
        |> filter(fn: (r) => r._measurement == "Gait")
        |> keep(columns: ["_time", "_value"])
        '''
        result = self.client.query_api().query(org=self.org, query=query)
        
        # Procesar los resultados en un dataframe
        times = []
        for table in result:
            for record in table.records:
                times.append([record.get_time(), record.get_value()])

        df = pd.DataFrame(times, columns=["Time", "Value"])
        return df

# Parámetros de conexión
bucket = "Gait/autogen"
org = "UPM"
token = "Zx2jR8PD6h3YlS7HVsY5Han1SzF_iz7uk8n5z9BYRZ5q50lk8r1L18N-nFZiGCa57oowLgl8656pVpCig-GANg=="
url = "https://138.100.82.178:8086"

# Inicializar el cliente de InfluxDB
influx = InfluxDB(bucket, org, token, url)

# Realizar la consulta y mostrar los tiempos
df = influx.query_data()
print(df)
