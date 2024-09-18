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
        # Añadir verify_ssl=False para desactivar la verificación SSL
        self.client = InfluxDBClient(url=url, token=tokenv2, org=self.org, verify_ssl=False)
        if self.bucket == 'Gait/autogen':
            self.measurement = ['Gait']
    
    def query_data(self):
        # Consulta para valores de tipo float
        query_float = f'''
        from(bucket: "{self.bucket}")
        |> range(start: 0)
        |> filter(fn: (r) => r._measurement == "Gait")
        |> filter(fn: (r) => r._value is float)
        |> keep(columns: ["_time", "_value"])
        '''
        
        # Consulta para valores de tipo int
        query_int = f'''
        from(bucket: "{self.bucket}")
        |> range(start: 0)
        |> filter(fn: (r) => r._measurement == "Gait")
        |> filter(fn: (r) => r._value is int)
        |> keep(columns: ["_time", "_value"])
        '''
        
        result_float = self.client.query_api().query(org=self.org, query=query_float)
        result_int = self.client.query_api().query(org=self.org, query=query_int)
        
        # Procesar los resultados de ambos en un dataframe
        times = []
        for table in result_float + result_int:
            for record in table.records:
                times.append([record.get_time(), record.get_value()])
        
        df = pd.DataFrame(times, columns=["Time", "Value"])
        
        # Convertir todos los valores a float para evitar la colisión de tipos
        df["Value"] = df["Value"].astype(float)
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
