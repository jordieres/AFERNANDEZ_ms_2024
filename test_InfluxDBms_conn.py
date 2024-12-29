"""

Module: Test_DB_Connenction to InflixDB for Gait

"""

import argparse
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse as parse_date


from InfluxDBms.cInfluxDB import cInfluxDB
from InfluxDBms.fecha_verbose import BatchProcess, VAction

# Function to convert strings to datetime objects
def parse_datetime(value):
    try:
        return parse_date(value)  # Converts ISO 8601 strings to datetime objects
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid date: {value}. Error: {e}")

# For predetermined dates
def get_default_dates():

    """
    Gets the default dates for the search.
    """
    now = datetime.now(timezone.utc)
    return now.isoformat() + "Z", (now + timedelta(minutes=30)).isoformat() + "Z"

# Programme arguments
def parse_args():
    """
    Configures and parses input arguments.
    """
    default_from, default_until = get_default_dates()
    parser = argparse.ArgumentParser(description='Execution of batch processes.')
    parser.add_argument('-f', '--from_time', type=parse_datetime, default=default_from, help='Start date (ISO 8601)')
    parser.add_argument('-u', '--until', type=parse_datetime, default=default_until, help='End date (ISO 8601)')
    parser.add_argument('-q', '--qtok', type=str, required=True, help='CodeID')
    parser.add_argument('-l', '--leg', type=str, required=True,choices=['Left', 'Right'], help='Choice of Left or Right Foot')
    parser.add_argument('-p', '--path', type=str, default='InfluxDBms/config_db.yaml', help='Path to the configuration file')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Level of verbosity')

    return parser.parse_args()

# Función principal
def main():

    """
    Main module that executes an InfluxDB query to obtain a Dataframe with the requested data.    
    """
    args = parse_args()

    # Entry dates (datetime objects)
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
        print(f"Error initialising cInfluxDB: {e}")
        return


    # Make an enquiry
    try:
        #df = iDB.query_with_aggregate_window(from_time, until_time, window_size="1d", qtok= qtok , pie=pie )
        df = iDB.query_data(from_time, until_time, qtok= qtok , pie=pie )
        print("Results of the query:")
        print(df)
        print(df.shape)
    except Exception as e:
        print(f"Error when querying data: {e}")
        return

    # Execute the batch process
    bp = BatchProcess(args)
    bp.run()

if __name__ == "__main__":
    main()

