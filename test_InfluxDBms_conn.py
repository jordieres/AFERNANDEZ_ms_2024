"""
Module: Test_DB_Connection to InfluxDB for Gait
"""

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
    parser.add_argument('-p', '--path', type=str, default='InfluxDBms/config_db.yaml', help='Path to the configuration file')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Level of verbosity')
    parser.add_argument('-m', '--metrics', help="Comma-separated list of metrics", default=None)
    
    return parser.parse_args()

# Función principal
def main():
    args = parse_args()
    metrics = args.metrics.split(",") if args.metrics else []

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
        df = iDB.query_with_aggregate_window(from_time, until_time, window_size="1d", qtok=qtok, pie=pie, metrics=metrics)
        #df = iDB.query_data(from_time, until_time, qtok= qtok , pie=pie, metrics=metrics )
        print("Results of the query:")
        print(df)
        print(df.shape)

        # Seleccionar la gráfica en función de las métricas
        if len(metrics) == 1:
            plot_2d(df, "_time", metrics[0], None, f"Time Series Plot of {metrics[0]}")
        elif len(metrics) == 2:
            plot_2d(df, "_time", metrics[0], metrics[1], "2D Time Series Plot")
        elif len(metrics) == 3:
            plot_3d(df, "_time", metrics[0], metrics[1], metrics[2], "3D Visualization")
        elif len(metrics) == 4 and set(metrics).issuperset({"Ax", "Ay", "Gx", "Gy"}):
            plot_dual_3d(df, "_time", ["Ax", "Ay"], ["Gx", "Gy"], "Ax, Ay vs Gx, Gy in 3D")

            #  ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz', 'S0', 'S1', 'S2']
        else:
            print("Unsupported combination of metrics for plotting.")
    
    except Exception as e:
        print(f"Error querying data: {e}")
        return

    # Ejecutar batch process
    bp = BatchProcess(args)
    bp.run()

if __name__ == "__main__":
    main()
