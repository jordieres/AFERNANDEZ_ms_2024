"""
extract_data.py : Test_DB_Connection to InfluxDB for Gait
"""

import sys, os, argparse, tables
import pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse as parse_date

# Import specific InfluxDB modules
# Assuming these modules are available in your environment
from InfluxDBms.cInfluxDB import cInfluxDB
from InfluxDBms.fecha_verbose import BatchProcess, VAction
from InfluxDBms.plot_functions_InfluxDB import *

#
# Function to convert strings to datetime objects
def parse_datetime(value):
    """
    Converts a string into a datetime object.

    :param value: String representation of a date/time.
    :type value: str
    :return: Parsed datetime object.
    :rtype: datetime
    :raises argparse.ArgumentTypeError: If the date format is invalid.
    """
    try:
        return parse_date(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid date: {value}. Error: {e}")

# Get default dates
def get_default_dates():
    """
    Retrieves the default start and end dates for querying.

    :return: Tuple with start and end date in ISO 8601 format.
    :rtype: tuple[str, str]
    """
    now = datetime.now(timezone.utc)
    # Adjust default dates to cover a longer period for testing
    return (now - timedelta(days=7)).isoformat() + "Z", now.isoformat() + "Z"

# Parse command-line arguments
def parse_args():
    """
    Parses command-line arguments for batch processing.

    :return: Parsed arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Extract data from InfluxDB and save to Excel/HDF5.')
    default_from, default_until = get_default_dates()
    parser.add_argument('-f', '--from_time', type=parse_datetime, default=default_from, help='Start date (ISO 8601)')
    parser.add_argument('-u', '--until', type=parse_datetime, default=default_until, help='End date (ISO 8601)')
    parser.add_argument('-q', '--qtok', type=str, required=True, help='CodeID')
    parser.add_argument('-c', '--config', type=str, default='../.config_db.yaml', help='Path to the configuration file')
    # parser.add_argument('-o', '--output', type=str, required=True, help='Path to the Excel file as output (will be suffixed with leg)')
    parser.add_argument('-v', '--verbose', type=int, default=0, help='Verbosity level')
    parser.add_argument('-m', '--metrics', help="Comma-separated list of metrics", default=None)
    parser.add_argument('-o', '--hdf5_output', type=str, default='gait_data.h5', help='Path to the HDF5 file as output')

    return parser.parse_args()


# Main Function
def main():
    """
    Main function to execute the InfluxDB query, process the data, and save it to an Excel and HDF5 file.
    It iterates for both 'Left' and 'Right' legs.
    """
    args = parse_args()
    metrics = args.metrics.split(",") if args.metrics else []

    # Define the legs to process
    legs_to_process = ['Left', 'Right']

    if args.verbose >= 1:
        print(f"Parameters: from_time={args.from_time}, until={args.until}, qtok={args.qtok}")
        # print(f"Base Excel Output: {args.output} (will be suffixed with leg)")
        print(f"HDF5 Output: {args.hdf5_output}")

    # Initialising connection to InfluxDB
    try:
        iDB = cInfluxDB(config_path=args.config)
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found at {args.config}. {e}")
        return
    except Exception as e:
        print(f"Error initializing InfluxDB connection: {e}")
        return

    # Loop through each leg
    for current_leg in legs_to_process:
        print(f"\n--- Processing data for Leg: {current_leg} ---")
        df = pd.DataFrame() # Initialize df for each iteration

        try:
            # Query data for the current leg
            df = iDB.query_data(args.from_time, args.until, qtok=args.qtok, pie=current_leg, metrics=metrics)

            if df.empty:
                print(f"No data retrieved from InfluxDB for CodeID: {args.qtok}, Leg: {current_leg} between {args.from_time} and {args.until}.")
                continue # Skip to the next leg if no data
            
            if args.verbose >= 2:
                print(f"Columns in the dataset for {current_leg}: {list(df.columns)}")
                print(f"Number of records retrieved for {current_leg}: {len(df)}")

            if args.verbose >= 3:
                print(f"First 5 records for {current_leg}:")
                print(df.head())
                print(f"Last 5 records for {current_leg}:")
                print(df.tail())

        except Exception as e:
            print(f"Error querying data for {current_leg}: {e}")
            continue # Continue to the next leg on error

        # Determine output file path for Excel (suffixed by leg)
        # Split the base output path into name and extension
        name, ext = os.path.splitext(args.hdf5_output)
        excel_output_path = f"{name}_{current_leg}{ext}"

        # Process and save to Excel
        # try:
        #     if args.verbose >= 1:
        #         print(f"Available columns in DataFrame for Excel ({current_leg}): {list(df.columns)}")
        #         print(f"First rows for Excel ({current_leg}):")
        #         print(df.head())

        #     # Convert to timezone-naive before saving to Excel if '_time' is present and timezone-aware
        #     if '_time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['_time']) and df['_time'].dt.tz is not None:
        #          df["_time"] = df["_time"].dt.tz_localize(None)

        #     df.to_excel(excel_output_path, index=False)
        #     if args.verbose >= 1:
        #         print(f"Data successfully saved to {excel_output_path} (Excel) for {current_leg}")
        # except Exception as e:
        #     print(f"Error saving to Excel for {current_leg}: {e}")

        # Save to HDF5
        try:
            # Format dates to be used in the HDF5 key
            # Convert to ISO 8601 without microseconds and ensure no invalid characters
            # Removing 'Z' from UTC and ':' for path safety
            from_time_str = args.from_time.isoformat(timespec='seconds').replace(':', '-').replace('+', '_').replace('Z', '')
            until_time_str = args.until.isoformat(timespec='seconds').replace(':', '-').replace('+', '_').replace('Z', '')

            # Construct the HDF5 key
            # Using slashes '/' to create a hierarchical structure within the HDF5 file
            hdf5_key = f"/{args.qtok}/{from_time_str}_{until_time_str}/{current_leg}"

            # Open the HDF5 file in append mode ('a') or create if it doesn't exist
            # Using 'a' ensures new dataframes are added without overwriting existing ones.
            with pd.HDFStore(args.hdf5_output, mode='a') as store:
                # Save the DataFrame with the specific key
                # format='table' allows for querying the data later.
                # data_columns=True indexes the columns for efficient searching.
                store.put(hdf5_key, df, format='table', data_columns=True)

            if args.verbose >= 1:
                print(f"Data successfully saved to {args.hdf5_output} under key: {hdf5_key} (HDF5) for {current_leg}")

        except Exception as e:
            print(f"Error saving to HDF5 for {current_leg}: {e}")

    # Ensure the InfluxDB connection is closed after all processing is done
    iDB.close()
    if args.verbose >= 1:
        print("\nInfluxDB connection closed.")


if __name__ == "__main__":
    main()