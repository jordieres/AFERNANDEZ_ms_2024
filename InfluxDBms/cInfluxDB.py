import pandas as pd
from influxdb_client import InfluxDBClient
from datetime import datetime
import urllib3

# Desactiva las advertencias de SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class cInfluxDB:
    def __init__(self, bucket: str, org: str, token: str, url: str, timeout: int = 500_000):
        """
        
        Initialises the InfluxDB class with the necessary parameters to connect to the database.

        :param bucket: Name of the bucket in InfluxDB.
        type bucket: str
        :param org: InfluxDB organisation. 
        :type org: str
        :param token: Authentication token for InfluxDB.
        :type token: str
        :param url: InfluxDB server URL.
        :type url: str
        :param timeout: Connection timeout in milliseconds.
        :type timeout: int

        """

        self.bucket = bucket
        self.org = org
        self.client = InfluxDBClient(url=url, token=token, org=self.org, verify_ssl=False, timeout=timeout)
        self.measurement = self.bucket.split("/")[0] if '/' in self.bucket else self.bucket  

    def query_data(self, from_date: datetime, to_date: datetime) -> pd.DataFrame:
        
        """
        Query data in InfluxDB, pivoting the results to get the metrics in columns.

        :param from_date: Start date (ISO 8601 format: 'YYYYY'-MM-DDTHH:MM:SSZ).
        :type from_date: datetime
        :param to_date: End date (ISO 8601 format: 'YYYYY'-MM-DDTHH:MM:SSZ).
        :type to_date: datetime
        :return: DataFrame with the metrics pivoted on columns.
        rtype pd.DataFrame
        """

        from_date_str = from_date.strftime('%Y-%m-%dT%H:%M:%SZ')  # UTC con 'Z'
        to_date_str = to_date.strftime('%Y-%m-%dT%H:%M:%SZ')

        metrics = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz', 'S0', 'S1', 'S2']
        metrics_str = ' or '.join([f'r._field == "{metric}"' for metric in metrics])
        columns_str = ', '.join([f'"{metric}"' for metric in metrics])

        #from(bucket: "{self.bucket}")
        #|> filter(fn: (r) => r._measurement == "{self.measurement}")
        #|> filter(fn: (r) => {metrics_str})
        query = f'''
        from(bucket: "Gait/autogen") 
        |> range(start: time(v: "{from_date_str}"), stop: time(v: "{to_date_str}"))
        |> filter(fn: (r) => r._measurement == "Gait")
        |> filter(fn: (r) => r._field == "Ax" or r._field == "Ay" or r._field == "Az" or r._field == "Gx" or r._field == "Gy" or r._field == "Gz" or r._field == "Mx" or r._field == "My" or r._field == "Mz" or r._field == "S0" or r._field == "S1" or r._field == "S2")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> keep(columns: ["_time", {columns_str}])
        '''
        print(f"Query generated: {query}")

        try:
            result = self.client.query_api().query(org=self.org, query=query)
        except Exception as e:
            print(f"Error in the query: {str(e)}")
            raise

        # Process the results in a aFrame
        data = []
        for table in result:
            for record in table.records:
                row = {
                    "Time": record.get_time(),
                    **{record.get_field(): record.get_value() for field in metrics if field == record.get_field()}
                }
                data.append(row)

        return pd.DataFrame(data)

#     def query_with_aggregate_window(self, from_date: str, to_date: str, window_size: str) -> pd.DataFrame:
#         """        
#         Query aggregated data in time windows to optimise queries when there are large date ranges.

#         :param from_date: Start date in ISO 8601 format.
#         :param to_date: End date in ISO 8601 format.
#         :param window_size: Size of the aggregation window (e.g. '10m' for 10 minutes).
#         :return: DataFrame with the aggregated data.
#         """
#         metrics = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz', 'S0', 'S1', 'S2']
#         metrics_str = ' or '.join([f'r._field == "{metric}"' for metric in metrics])
#         columns_str = ', '.join([f'"{metric}"' for metric in metrics])

#         query = f'''
#         from(bucket: "{self.bucket}")
#         |> range(start: "{from_date}", stop: "{to_date}")
#         |> filter(fn: (r) => r._measurement == "{self.measurement}")
#         |> filter(fn: (r) => {metrics_str})
#         |> aggregateWindow(every: {window_size}, fn: mean, createEmpty: false)
#         |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
#         |> keep(columns: ["_time", {columns_str}])
#         '''
#         print(f"Query generated with aggregate window: {query}")
        
#         try:
#             result = self.client.query_api().query(org=self.org, query=query)
#         except Exception as e:
#             print(f"Error in the query: {str(e)}")
#             raise

#         data = []
#         for table in result:
#             for record in table.records:
#                 # Usamos get_field() y get_value() para acceder a '_field' y '_value'
#                 field = record.get_field()
#                 value = record.get_value()
                
#                 # Solo procedemos si el campo está en métricas y el valor no es None
#                 if field in metrics and value is not None:
#                     row = {
#                         "Time": record.get_time(),
#                         field: value
#                     }
#                     data.append(row)


#         df = pd.DataFrame(data)
#         return df