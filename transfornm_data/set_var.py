import pandas as pd

def compute_time_parameters(file_path):
    """
    Reads an Excel file and computes timing parameters from the '_time' column.

    Parameters:
    - file_path (str): Path to the Excel file.

    Returns:
    - dict with startTime, stopTime, samplePeriod, frequency, and the loaded DataFrame.
    """
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print("Error reading the file:", e)
        return None

    start_time = 0

    if '_time' in df.columns:
        try:
            df['_time'] = pd.to_datetime(df['_time'])
            df = df.sort_values('_time').reset_index(drop=True)
            dt = df['_time'].diff().dt.total_seconds().dropna()
            sample_period = dt.mean()
            frequency = 1 / sample_period
            stop_time = (len(df) - 1) * sample_period

            return {
                "start_time": start_time,
                "stop_time": stop_time,
                "sample_period": sample_period,
                "frequency": frequency,
                "dataframe": df
            }

        except Exception as e:
            print("Error processing the '_time' column:", e)
    else:
        print("'_time' column not found.")


    return None
