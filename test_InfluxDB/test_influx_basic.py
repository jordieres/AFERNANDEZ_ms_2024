"""
Module: Test_DB_Connection to InfluxDB for Gait
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import argparse
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse as parse_date
import pandas as pd

from InfluxDBms.cInfluxDB import cInfluxDB
from InfluxDBms.fecha_verbose import BatchProcess, VAction
from InfluxDBms.plot_functions_InfluxDB import *

# Función para convertir cadenas en objetos datetime
def parse_datetime(value):
    try:
        return parse_date(value)  
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid date: {value}. Error: {e}")

# Obtener fechas predeterminadas
def get_default_dates():
    now = datetime.now(timezone.utc)
    return now.isoformat() + "Z", (now + timedelta(minutes=30)).isoformat() + "Z"

# Argumentos del programa
def parse_args():
    default_from, default_until = get_default_dates()
    parser = argparse.ArgumentParser(description='Execution of batch processes.')
    parser.add_argument('-f', '--from_time', type=parse_datetime, default=default_from, help='Start date (ISO 8601)')
    parser.add_argument('-u', '--until', type=parse_datetime, default=default_until, help='End date (ISO 8601)')
    parser.add_argument('-q', '--qtok', type=str, required=True, help='CodeID')
    parser.add_argument('-l', '--leg', type=str, required=True, choices=['Left', 'Right'], help='Choice of Left or Right Foot')
    parser.add_argument('-p', '--path', type=str, default='../InfluxDBms/config_db.yaml', help='Path to the configuration file')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Level of verbosity')
    
    return parser.parse_args()

# Función principal
def main():
    args = parse_args()
    
    from_time = args.from_time
    until_time = args.until
    qtok = args.qtok
    pie = args.leg

    try:
        iDB = cInfluxDB(config_path=args.path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error initializing cInfluxDB: {e}")
        return

    # Realizar consulta
    try: 
        
        df = iDB.query_data(from_time, until_time, qtok= qtok , pie=pie)
        print("Results of the query:")
        print(df)
        print(df.shape)
    
    except Exception as e:
        print(f"Error querying data: {e}")
        return

    # Ejecutar batch process
    bp = BatchProcess(args)
    bp.run()

if __name__ == "__main__":
    main()
