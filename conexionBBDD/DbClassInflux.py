import sys
import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient
from datetime import datetime
import urllib3

# Desactivar las advertencias de SSL (solo para desarrollo)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class InfluxDB:
    def __init__(self, bucketv2: str, orgv2: str, tokenv2: str, url: str, timeout: int = 500_000):
        """
        Inicializa la clase InfluxDB con los parámetros necesarios para conectarse a la base de datos.

        :param bucketv2: Nombre del bucket en InfluxDB.
        :type bucketv2: str
        :param orgv2: Organización de InfluxDB. 
        :type orgv2: str
        :param tokenv2: Token de autenticación para InfluxDB.
        :type tokenv2: str
        :param url: URL del servidor de InfluxDB.
        :type url: str
        :param timeout: Timeout para la conexión en milisegundos.
        :type timeout: int
        """
        self.bucket = bucketv2
        self.org = orgv2
        self.client = InfluxDBClient(url=url, token=tokenv2, org=self.org, verify_ssl=False, timeout=timeout)
        
        # Asignar measurement según el formato del bucket
        if '/' in self.bucket:
            self.measurement = self.bucket.split("/")[0]  
        else:
            self.measurement = self.bucket  
    def query_data(self, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Consulta datos en InfluxDB, pivotando los resultados para obtener las métricas en columnas.

        :param from_date: Fecha de inicio (formato ISO 8601: 'YYYY-MM-DDTHH:MM:SSZ').
        :param to_date: Fecha de fin (formato ISO 8601: 'YYYY-MM-DDTHH:MM:SSZ').
        :return: DataFrame con las métricas pivotadas en columnas.
        :rtype: pd.DataFrame
        """
        metrics = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz', 'S0', 'S1', 'S2']
        metrics_str = ' or '.join([f'r._field == "{metric}"' for metric in metrics])
        columns_str = ', '.join([f'"{metric}"' for metric in metrics])

        
        query = f'''
            from(bucket: "Gait/autogen")
            |> range(start: 2023-01-01T00:00:00Z, stop: 2023-02-01T00:00:00Z)
            |> filter(fn: (r) => r._measurement == "Gait")
            |> filter(fn: (r) => r._field in ["Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Mx", "My", "Mz", "S0", "S1", "S2"])
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> keep(columns: ["_time", "Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Mx", "My", "Mz", "S0", "S1", "S2"])

        '''
        print(f"Consulta generada: {query}")
        try:
            result = self.client.query_api().query(org=self.org, query=query)
        except Exception as e:
            print(f"Error en la consulta: {str(e)}")
            raise

        data = []
        for table in result:
            for record in table.records:
                row = {
                    "Time": record.get_time(),
                    **{field: record.get_value_by_key(field) for field in metrics if record.get_value_by_key(field) is not None}
                }
                data.append(row)

        df = pd.DataFrame(data)
        return df


    def query_with_aggregate_window(self, from_date: str, to_date: str, window_size: str) -> pd.DataFrame:
        """
        Consulta datos agregados en ventanas de tiempo para optimizar las consultas cuando hay grandes rangos de fechas.

        :param from_date: Fecha de inicio en formato ISO 8601.
        :param to_date: Fecha de fin en formato ISO 8601.
        :param window_size: Tamaño de la ventana de agregación (ejemplo: "10m" para 10 minutos).
        :return: DataFrame con los datos agregados.
        """
        metrics = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz', 'S0', 'S1', 'S2']
        metrics_str = ' or '.join([f'r._field == "{metric}"' for metric in metrics])
        columns_str = ', '.join([f'"{metric}"' for metric in metrics])

        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: "{from_date}", stop: "{to_date}")
        |> filter(fn: (r) => r._measurement == "{self.measurement}")
        |> filter(fn: (r) => {metrics_str})
        |> aggregateWindow(every: {window_size}, fn: mean, createEmpty: false)
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> keep(columns: ["_time", {columns_str}])
        '''
        print(f"Consulta generada con ventana de agregación: {query}")
        
        try:
            result = self.client.query_api().query(org=self.org, query=query)
        except Exception as e:
            print(f"Error en la consulta: {str(e)}")
            raise

        data = []
        for table in result:
            for record in table.records:
                row = {
                    "Time": record.get_time(),
                    **{field: record.get_value_by_key(field) for field in metrics if record.get_value_by_key(field) is not None}
                }
                data.append(row)

        df = pd.DataFrame(data)
        return df
