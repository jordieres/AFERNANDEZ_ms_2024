"""

Module: Test_DB_Connenction to InflixDB for Gait

"""

import argparse
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse as parse_date


from InfluxDBms.cInfluxDB import cInfluxDB
from InfluxDBms.fecha_verbose import BatchProcess, VAction

import matplotlib.pyplot as plt
import pandas as pd
from typing import List

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
    parser.add_argument("-m", "--metrics", help="Comma-separated list of metrics", default=None)


    return parser.parse_args()

def plot_data(df: pd.DataFrame, columns: List[str] = None, title: str = "Data Visualization"):
    """
    Plot selected columns from a DataFrame against time.

    :param df: DataFrame containing the data to plot. Must include a '_time' column.
    :param columns: List of columns to plot. If None, all numerical columns except '_time' are plotted.
    :param title: Title of the plot.
    """
    if '_time' not in df.columns:
        raise ValueError("The DataFrame must contain a '_time' column.")

    # Convert '_time' to datetime if it isn't already
    if not pd.api.types.is_datetime64_any_dtype(df['_time']):
        df['_time'] = pd.to_datetime(df['_time'])

    # Select columns to plot
    if columns is None:
        columns = [col for col in df.columns if col != '_time' and pd.api.types.is_numeric_dtype(df[col])]

    if not columns:
        raise ValueError("No valid columns to plot.")

    plt.figure(figsize=(14, 8))
    for column in columns:
        plt.plot(df['_time'], df[column], label=column, alpha=0.8)

    plt.title(title, fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.legend(title="Metrics", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_individual_variables(df: pd.DataFrame, columns: List[str] = None, title_prefix: str = "Variable Plot"):
    """
    Plot each variable in a DataFrame individually against time.

    :param df: DataFrame containing the data to plot. Must include a '_time' column.
    :param columns: List of columns to plot. If None, all numerical columns except '_time' are plotted.
    :param title_prefix: Prefix for the title of each individual plot.
    """
    if '_time' not in df.columns:
        raise ValueError("The DataFrame must contain a '_time' column.")

    # Convert '_time' to datetime if it isn't already
    if not pd.api.types.is_datetime64_any_dtype(df['_time']):
        df['_time'] = pd.to_datetime(df['_time'])

    # Select columns to plot
    if columns is None:
        columns = [col for col in df.columns if col != '_time' and pd.api.types.is_numeric_dtype(df[col])]

    if not columns:
        raise ValueError("No valid columns to plot.")

    for column in columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['_time'], df[column], label=column, color='b', alpha=0.8)
        plt.title(f"{title_prefix}: {column}", fontsize=16)
        plt.xlabel("Time", fontsize=14)
        plt.ylabel(column, fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()


def plot_4d(df: pd.DataFrame, time_col: str, x_col: str, y_col: str, z_col: str, title: str = "4D Visualization"):
    """
    Create a 3D plot with time as color (4th dimension).

    :param df: DataFrame containing the data to plot.
    :param time_col: Name of the column representing time.
    :param x_col: Name of the column for the x-axis.
    :param y_col: Name of the column for the y-axis.
    :param z_col: Name of the column for the z-axis.
    :param title: Title of the plot.
    """
    if time_col not in df.columns or x_col not in df.columns or y_col not in df.columns or z_col not in df.columns:
        raise ValueError("One or more specified columns are not in the DataFrame.")

    # Convert time column to numeric values for color mapping
    df[time_col] = pd.to_datetime(df[time_col])
    time_numeric = (df[time_col] - df[time_col].min()).dt.total_seconds()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(df[x_col], df[y_col], df[z_col], c=time_numeric, cmap='viridis', alpha=0.8)
    plt.colorbar(sc, label="Time (seconds since start)")

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_zlabel(z_col, fontsize=12)

    plt.tight_layout()
    plt.show()



# Función principal
def main():

    """
    Main module that executes an InfluxDB query to obtain a Dataframe with the requested data.    
    """
    args = parse_args()
    metrics = args.metrics.split(",") if args.metrics else None

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
        df = iDB.query_data(from_time, until_time, qtok= qtok , pie=pie, metrics=metrics )
        print("Results of the query:")
        print(df)
        print(df.shape)
    
        #plot_data(df, columns=['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'], title="Sensor Data Visualization")
        #plot_individual_variables(df, columns=['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'], title_prefix="Sensor Variable")
        #plot_4d(df, time_col='_time', x_col='Ax', y_col='Ay', z_col='Az', title="4D Plot: Ax, Ay, Az vs Time")

    except Exception as e:
        print(f"Error when querying data: {e}")
        return

    # Execute the batch process
    bp = BatchProcess(args)
    bp.run()

if __name__ == "__main__":
    main()

