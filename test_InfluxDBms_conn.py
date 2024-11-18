# """

# Module: Test_DB_Connenction to InflixDB for Gait

# """
# import yaml,os, sys, pdb
# import InfluxDBms
# from pathlib import Path, PureWindowsPath

# def main():
#     """
#     Main module testing two queries to the DB

#     It does not receive parameters except those in the command line
    
#     It does not return specific variables, except those printed out.

#     """
#     # Leer la configuración desde el archivo YAML
#     bp = BatchProcess()
#     with open(Path(bp.get_cnf()), 'r') as file:
#         config = yaml.safe_load(file)

#     # Obtener los parámetros de conexión
#     bucket = config['influxdb']['bucket']
#     org = config['influxdb']['org']
#     token = config['influxdb']['token']
#     url = config['influxdb']['url']
#     # Inicializar el cliente
#     iDB = InfluxDB(bucket,org,token,url)

#     # Realizar la consulta con las fechas de:  desde/hasta
#     f_ini = "2024-11-02T18:08:45Z"
#     f_end = "2024-11-02T18:50:00Z"
#     (f_ini, f_end) = iDB.get_default_dates()

#     # Consulta de influx
#     df = iDB.query_data(f_ini,f_end)
#     print(df)

#     # # Consulta con agregación
#     # window_size = "1m"  
#     # df_agg = iDB.query_with_aggregate_window(f_ini, f_end, window_size)
#     # print(df_agg)

# if __name__ == "__main__":
#     main()

from pathlib import Path
from InfluxDBms import *
import argparse
from datetime import *
import yaml,os, sys, pdb
from pathlib import Path, PureWindowsPath


def get_default_dates():
        """
        Obtiene las fechas por defecto para la búsqueda.
        """
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        default_from = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        default_until = yesterday.replace(hour=23, minute=59, second=59, microsecond=0)
        return (default_from.strftime('%Y-%m-%d %H:%M:%S'), \
                default_until.strftime('%Y-%m-%d %H:%M:%S'))


def parse_args():
    """
    Configura y parsea los argumentos de entrada.
    """
    default_from, default_until = get_default_dates()
    parser = argparse.ArgumentParser(description='Ejecución de procesos en batch.')
    parser.add_argument('-f', '--from_time', type=str, default=default_from,
                        help='Fecha de inicio de búsqueda (formato: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('-u', '--until', type=str, default=default_until,
                        help='Fecha de finalización de búsqueda (formato: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('-p', '--path', type=str, default='InfluxDBms\config_db.yaml',  # Proporciona un valor por defecto
                        help='Path al archivo de configuración config.yaml')
    parser.add_argument('-v', '--verbose', nargs='?', action=VAction, dest='verbose',
                        help='Nivel de verbosidad', default=0)
    return parser.parse_args()

def main():
    """
    Modulo principal que prueba dos consultas en la DB.
    """
    args = parse_args()
    bp = BatchProcess(args)  # Pasa los argumentos al BatchProcess

    # Leer la configuración desde el archivo YAML
    config_path = bp.get_cnf()
    with open(Path(config_path), 'r') as file:
        config = yaml.safe_load(file)

    # Obtener los parámetros de conexión
    bucket = config['influxdb']['bucket']
    org = config['influxdb']['org']
    token = config['influxdb']['token']
    url = config['influxdb']['url']

    # Inicializar el cliente
    iDB = cInfluxDB(bucket, org, token, url)

    # Realizar la consulta con las fechas de: desde/hasta
    # f_ini = args.from_time
    # f_end = args.until
    f_ini = "2024-11-02T18:08:45Z"
    f_end = "2024-11-02T18:50:00Z"
    # (f_ini, f_end) = get_default_dates()

    # Consulta de influx
    df = iDB.query_data(f_ini, f_end)
    print(df)

    # Ejecución del proceso principal
    bp.run()

if __name__ == "__main__":
    main()
