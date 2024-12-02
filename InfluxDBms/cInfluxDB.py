import pandas as pd
from influxdb_client import InfluxDBClient
from datetime import datetime
import urllib3
import yaml

# Desactiva las advertencias de SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class cInfluxDB:
    def __init__(self, config_path: str, timeout: int = 500_000):
        """
        Initializes the connection to InfluxDB using a YAML configuration file.

        :param config_path: Path to the YAML configuration file.
        :type config_path: str
        :param timeout: Connection timeout in milliseconds.
        :type timeout: int

        """
        # Carga la configuración desde el archivo YAML
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Extrae los valores necesarios
        self.bucket = config['influxdb']['bucket']
        self.org = config['influxdb']['org']
        self.token = config['influxdb']['token']
        self.url = config['influxdb']['url']

        # Inicializa el cliente de InfluxDB
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org, verify_ssl=False, timeout=timeout)
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

        # from(bucket: "{self.bucket}")
        # |> filter(fn: (r) => r._measurement == "{self.measurement}")
        # |> filter(fn: (r) => {metrics_str})
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


    def query_with_aggregate_window(self, from_date: datetime, to_date: datetime, window_size: str = "1d") -> pd.DataFrame:
        """
        Query data in InfluxDB, pivoting the results to get the metrics in columns.
        Handles cases where there are no data points by using `aggregateWindow`.

        :param from_date: Start date (ISO 8601 format: 'YYYY-MM-DDTHH:MM:SSZ').
        :type from_date: datetime
        :param to_date: End date (ISO 8601 format: 'YYYY-MM-DDTHH:MM:SSZ').
        :type to_date: datetime
        :param window_size: Aggregation window size (e.g., '1s', '1d').
        :type window_size: str
        :return: DataFrame with the metrics pivoted on columns.
        :rtype: pd.DataFrame
        """
        from_date_str = from_date.strftime('%Y-%m-%dT%H:%M:%SZ')  # UTC con 'Z'
        to_date_str = to_date.strftime('%Y-%m-%dT%H:%M:%SZ')

        metrics = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz', 'S0', 'S1', 'S2']
        metrics_str = ' or '.join([f'r._field == "{metric}"' for metric in metrics])
        columns_str = ', '.join([f'"{metric}"' for metric in metrics])

        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: time(v: "{from_date_str}"), stop: time(v: "{to_date_str}"))
        |> filter(fn: (r) => r._measurement == "{self.measurement}")
        |> filter(fn: (r) => {metrics_str})
        |> group(columns: ["_field"])
        |> aggregateWindow(every: {window_size}, fn: last, createEmpty: true)
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> keep(columns: ["_time", {columns_str}])
        '''
        print(f"Query generated with aggregate window: {query}")

        try:
            result = self.client.query_api().query(org=self.org, query=query)
        except Exception as e:
            print(f"Error in the query: {str(e)}")
            raise

        # Process the results into a DataFrame
        data = []
        for table in result:
            for record in table.records:
                row = {"Time": record.get_time()}
                for field in metrics:
                    if field == record.get_field():
                        row[field] = record.get_value()
                data.append(row)

        # Convert the data to a DataFrame
        df = pd.DataFrame(data)

        # Ensure all required columns are present
        for col in ["Time"] + metrics:
            if col not in df:
                df[col] = None  # Fill missing columns with None

        return df
