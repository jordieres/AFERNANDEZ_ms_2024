"""

Module: Test_DB_Connenction to InflixDB for Gait

"""
import yaml,os, sys, pdb
import InfluxDBms
from pathlib import Path, PureWindowsPath

def main():
    """
    Main module testing two queries to the DB

    It does not receive parameters except those in the command line
    
    It does not return specific variables, except those printed out.

    """
    # Leer la configuración desde el archivo YAML
    bp = BatchProcess()
    with open(Path(bp.get_cnf()), 'r') as file:
        config = yaml.safe_load(file)

    # Obtener los parámetros de conexión
    bucket = config['influxdb']['bucket']
    org = config['influxdb']['org']
    token = config['influxdb']['token']
    url = config['influxdb']['url']
    # Inicializar el cliente
    iDB = InfluxDB(bucket,org,token,url)

    # Realizar la consulta con las fechas de:  desde/hasta
    # from_date = "2024-10-01T00:00:00Z"
    # to_date = "2024-10-21T00:00:00Z"
    (f_ini, f_end) = iDB.get_default_dates()

    # Consulta de influx
    df = iDB.query_data(f_ini,f_end)
    print(df)

    # Consulta con agregación
    window_size = "10m"  
    df_agg = iDB.query_with_aggregate_window(f_ini, f_end, window_size)
    print(df_agg)

if __name__ == "__main__":
    main()
