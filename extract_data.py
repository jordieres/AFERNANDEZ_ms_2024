"""
Module: Extract_Data from InfluxDB for Gait
"""

import os, argparse
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse as parse_date
import pandas as pd

from InfluxDBms.influxdb_tools import cInfluxDB
from InfluxDBms.fecha_verbose import VAction
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

# Función principal
def main():
    #
    default_from, default_until = get_default_dates()
    parser = argparse.ArgumentParser(description='Execution of Data Extraction.')
    parser.add_argument('-f', '--from_time', type=parse_datetime, default=default_from, help='Start date (ISO 8601)')
    parser.add_argument('-u', '--until', type=parse_datetime, default=default_until, help='End date (ISO 8601)')
    parser.add_argument('-q', '--qtok', type=str, required=True, help='CodeID')
    parser.add_argument('-l', '--leg', type=str, required=True, choices=['Left', 'Right'], help='Choice of Left or Right Foot')
    parser.add_argument('-p', '--path', type=str, default='InfluxDBms/config_db.yaml', help='Path to the configuration file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the excel file as outcome')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Level of verbosity')
    # parser.add_argument('-m', '--metrics', help="Comma-separated list of metrics", default=None)
    args = parser.parse_args()
    # metrics = args.metrics.split(",") if args.metrics else []

    from_time = args.from_time
    until_time = args.until
    qtok = args.qtok
    pie = args.leg
    if not os.path.isdir(args.output):
        print(f"Error: The path {args.output} is not a valid directory!")
        return
    if not args.output.endswith('/'):
        out = args.output + '/'
    else:
        out = args.output
    nam = out + 'data_' + args.qtok + '+' + from_time.strftime("%Y-%m-%d_%H:%M:%S") + '+' + \
        until_time.strftime("%Y-%m-%d_%H:%M:%S") + '+' + pie + '.xlsx'

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
        df = iDB.query_data(from_time, until_time, qtok=qtok, pie=pie)
        gmt_plus_1_fixed = timezone(timedelta(hours=1))
        df['_time'] = df['_time'].dt.tz_convert(gmt_plus_1_fixed).dt.tz_localize(None)
        print(f"Results of the query: Dataset size {df.shape}")
        df_sorted = df.sort_values(by="_time", ascending=False)
        # Guardar el fichero
        df_sorted.to_excel(nam)
    
    except Exception as e:
        print(f"Error querying data: {e}")
        return

#
if __name__ == "__main__":
    main()
