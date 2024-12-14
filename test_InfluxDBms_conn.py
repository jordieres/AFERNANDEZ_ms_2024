# """

# Module: Test_DB_Connenction to InflixDB for Gait

# """

# import argparse
# from datetime import datetime, timedelta
# from dateutil.parser import parse as parse_date


# from InfluxDBms.cInfluxDB import cInfluxDB
# from InfluxDBms.fecha_verbose import BatchProcess, VAction

# # Función para convertir cadenas a objetos datetime
# def parse_datetime(value):
#     try:
#         return parse_date(value)  # Convierte cadenas ISO 8601 a objetos datetime
#     except ValueError as e:
#         raise argparse.ArgumentTypeError(f"Fecha inválida: {value}. Error: {e}")

# # Fechas predeterminadas
# def get_default_dates():

#     """
#     Gets the default dates for the search.
#     """
#     now = datetime.utcnow()
#     return now.isoformat() + "Z", (now + timedelta(minutes=30)).isoformat() + "Z"

# # Argumentos del programa
# def parse_args():

#     """
#     Configures and parses input arguments.
#     """
#     default_from, default_until = get_default_dates()
#     parser = argparse.ArgumentParser(description='Ejecución de procesos en batch.')
#     parser.add_argument('-f', '--from_time', type=parse_datetime, default=default_from, help='Fecha inicio (ISO 8601)')
#     parser.add_argument('-u', '--until', type=parse_datetime, default=default_until, help='Fecha fin (ISO 8601)')
#     parser.add_argument('-q', '--qtok', type=str, required=True, help='CodeID')
#     parser.add_argument('-pi', '--pie', type=str, required=True,choices=['Left', 'Right'], help='Elección de Pie Izquierdo o Derecho')
#     parser.add_argument('-p', '--path', type=str, default='InfluxDBms/config_db.yaml', help='Path al archivo de configuración')
#     parser.add_argument('-v', '--verbose', action='count', default=0, help='Nivel de verbosidad')

#     return parser.parse_args()

# # Función principal
# def main():

#     """
#     Main module that executes an InfluxDB query to obtain a Dataframe with the requested data.    
#     """
#     args = parse_args()

#     # Fechas de entrada (objetos datetime)
#     from_time = args.from_time
#     until_time = args.until
#     qtok = args.qtok
#     pie = args.pie


#     try:
#         iDB = cInfluxDB(config_path=args.path)
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         return
#     except Exception as e:
#         print(f"Error al inicializar cInfluxDB: {e}")
#         return


#     # Realizar consulta
#     try:
#         df = iDB.query_with_aggregate_window(from_time, until_time, window_size="1d", qtok= qtok , pie=pie )
#         print("Resulados de la consulta:")
#         print(df)
#     except Exception as e:
#         print(f"Error al consultar datos: {e}")
#         return

#     # Ejecutar el proceso batch
#     bp = BatchProcess(args)
#     bp.run()

# if __name__ == "__main__":
#     main()

"""

Module: Test_DB_Connenction to InflixDB for Gait

"""

import argparse
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse as parse_date


from InfluxDBms.cInfluxDB import cInfluxDB
from InfluxDBms.fecha_verbose import BatchProcess, VAction

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
    now = datetime.now(timezone.utc)
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
    parser.add_argument('-q', '--qtok', type=str, required=True, help='CodeID')
    parser.add_argument('-l', '--leg', type=str, required=True,choices=['Left', 'Right'], help='Elección de Pie Izquierdo o Derecho')
    parser.add_argument('-p', '--path', type=str, default='InfluxDBms/config_db.yaml', help='Path al archivo de configuración')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Nivel de verbosidad')

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
    qtok = args.qtok
    pie = args.leg


    try:
        iDB = cInfluxDB(config_path=args.path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error al inicializar cInfluxDB: {e}")
        return


    # Realizar consulta
    try:
        df = iDB.query_with_aggregate_window(from_time, until_time, window_size="1d", qtok= qtok , pie=pie )
        #df = iDB.query_data(from_time, until_time, qtok= qtok , pie=pie )
        print("Resulados de la consulta:")
        print(df)
        print(df.shape)
    except Exception as e:
        print(f"Error al consultar datos: {e}")
        return

    # Ejecutar el proceso batch
    bp = BatchProcess(args)
    bp.run()

if __name__ == "__main__":
    main()

