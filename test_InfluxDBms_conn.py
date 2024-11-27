"""

Module: Test_DB_Connenction to InflixDB for Gait

"""

import argparse
import yaml
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from pathlib import Path
from InfluxDBms import *

# Función para convertir cadenas a objetos datetime
def parse_datetime(value):
    try:
        return parse_date(value)  # Convierte cadenas ISO 8601 a objetos datetime
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Fecha inválida: {value}. Error: {e}")

# Fechas predeterminadas
def get_default_dates():

    """
    Gets the default dates for the search.
    """
    now = datetime.utcnow()
    return now.isoformat() + "Z", (now + timedelta(minutes=30)).isoformat() + "Z"

# Argumentos del programa
def parse_args():

    """
    Configures and parses input arguments.
    """
    default_from, default_until = get_default_dates()
    parser = argparse.ArgumentParser(description='Ejecución de procesos en batch.')
    parser.add_argument('-f', '--from_time', type=parse_datetime, default=default_from, help='Fecha inicio (ISO 8601)')
    parser.add_argument('-u', '--until', type=parse_datetime, default=default_until, help='Fecha fin (ISO 8601)')
    parser.add_argument('-p', '--path', type=str, default='InfluxDBms/config_db.yaml', help='Path al archivo de configuración')
    return parser.parse_args()

# Función principal
def main():

    """
    Main module that executes an InfluxDB query to obtain a Dataframe with the requested data.    
    """
    args = parse_args()

    # Fechas de entrada (objetos datetime)
    from_time = args.from_time
    until_time = args.until

    # Leer la configuración desde el archivo YAML
    config_path = Path(args.path)
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración en la ruta: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Configuración de conexión a InfluxDB
    bucket = config['influxdb']['bucket']
    org = config['influxdb']['org']
    token = config['influxdb']['token']
    url = config['influxdb']['url']

    # Crear instancia de InfluxDB
    iDB = cInfluxDB(bucket, org, token, url)

    # Realizar consulta
    try:
        df = iDB.query_data(from_time, until_time)
        print("Resultados del query:")
        print(df)
    except Exception as e:
        print(f"Error al consultar datos: {e}")
        return

    # Ejecutar el proceso batch
    bp = BatchProcess(args)
    bp.run()

if __name__ == "__main__":
    main()
