
"""
extract_data.py : Test_DB_Connection to InfluxDB for Gait
"""

import argparse
from InfluxDBms.influxdb_tools import cInfluxDB,InfluxHelper


def parse_args():
    """
    Parse command-line arguments for querying gait data from InfluxDB.

    :return: Parsed arguments object containing all user-specified options.
    :rtype: argparse.Namespace

    **Command-line Arguments:**

    - ``-f``, ``--from_time`` (str): Start timestamp in ISO 8601 format (default: now).
    - ``-u``, ``--until`` (str): End timestamp in ISO 8601 format (default: now + 30min).
    - ``-q``, ``--qtok`` (str): CodeID to identify the subject/session. (required)
    - ``-l``, ``--leg`` (str): Select the foot ("Left" or "Right"). (required)
    - ``-c``, ``--config`` (str): Path to InfluxDB YAML config file (default: ../.config_db.yaml).
    - ``-o``, ``--output`` (str): Output path for the resulting Excel file. (required)
    - ``-v``, ``--verbose`` (int): Verbosity level for console output (default: 0).
    - ``-m``, ``--metrics`` (str): Optional comma-separated list of metrics to query (e.g., "Ax,Ay,Gx,Gy").
    """

    helper = InfluxHelper()
    parser = argparse.ArgumentParser(description='Extract data from InfluxDB and save to Excel.')
    default_from, default_until = helper.get_default_dates()
    parser.add_argument('-f', '--from_time', type=helper.parse_datetime, default=default_from, help='Start date (ISO 8601)')
    parser.add_argument('-u', '--until', type=helper.parse_datetime, default=default_until, help='End date (ISO 8601)')
    parser.add_argument('-q', '--qtok', type=str, required=True, help='CodeID')
    parser.add_argument('-l', '--leg', choices=['Left', 'Right'], required=True, help='Choice of Left or Right Foot')
    parser.add_argument('-c', '--config', type=str, default='../.config_db.yaml', help='Path to the configuration file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the Excel file as output')
    parser.add_argument('-v', '--verbose', type=int, default=0, help='Verbosity level')
    # parser.add_argument('-w', '--window_size', type=str, default="20ms", help="Aggregation window size (default: 20ms)")
    parser.add_argument('-m', '--metrics', help="Comma-separated list of metrics", default=None)
  
    return parser.parse_args()



# Main Function
def main():
    """
    Main execution function to connect to InfluxDB, extract gait data, and export to Excel.

    Workflow:
    ----------
    1. Parse input arguments (dates, subject ID, foot, config file).
    2. Establish connection to InfluxDB using `cInfluxDB` class.
    3. Query the data using raw query (or optionally `aggregateWindow`).
    4. Print metrics and dataset stats based on verbosity level.
    5. Export results to Excel, stripping timezone information.
    6. Close the connection.

    Requires valid configuration via a YAML file and CodeID + foot selection.
    """
    args = parse_args()
    metrics = args.metrics.split(",") if args.metrics else []
    
    if args.verbose >= 1:
        # print(f"Parameters: from_time={args.from_time}, until={args.until}, qtok={args.qtok}, leg={args.leg}, window_size={args.window_size}")
        print(f"Parameters: from_time={args.from_time}, until={args.until}, qtok={args.qtok}, leg={args.leg}")

    # Initialising connection to InfluxDB
    try:
        iDB = cInfluxDB(config_path=args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error initializing InfluxDB connection: {e}")
        return

    try:
        # df = iDB.query_with_aggregate_window(args.from_time, args.until, window_size=args.window_size, qtok=args.qtok, pie=args.leg, metrics=metrics)
        df= iDB.query_data(args.from_time, args.until, qtok= args.qtok , pie=args.leg, metrics=metrics )

        if df.empty:
            print("No data retrieved from InfluxDB.")
            return

        if args.verbose >= 2:
            print(f"Columns in the dataset: {list(df.columns)}")
            print(f"Number of records retrieved: {len(df)}")

        if args.verbose >= 3:
            print("First 5 records:")
            print(df.head())
            print("Last 5 records:")
            print(df.tail())

    except Exception as e:
        print(f"Error querying data: {e}")
        return      

    # Save to Excel
    try:
        print("Columnas disponibles en el DataFrame:", df.columns)
        df["_time"] = df["_time"].dt.tz_localize(None)
        df.to_excel(args.output, index=False)
        iDB.close()
        if args.verbose >= 1:
            print(f"Data successfully saved to {args.output}")
    except Exception as e:
        print(f"Error saving to Excel: {e}")


   # Close DB connection at the end
    iDB.close()


if __name__ == "__main__":
    main() 