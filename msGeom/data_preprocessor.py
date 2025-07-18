
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import CubicSpline
from pyproj import Proj


class DataPreprocessor:
    """
    Class for loading configuration and Excel data, resampling, and preprocessing sensor data.
    """

    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """
        Load YAML configuration file from the specified path.

        :param config_path: Path to the YAML configuration file.
        :type config_path: str
        :return: Dictionary containing configuration data.
        :rtype: dict
        """
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def load_data(self, file_path):
        """
        Load Excel data from the specified path.

        :param file_path: Path to the Excel file.
        :type file_path: str
        :return: DataFrame containing raw data.
        :rtype: pd.DataFrame
        """
        df = pd.read_excel(file_path)
        return df

    def resample_to_40hz(self, df, time_col = '_time', freq_hz = 40, gap_threshold_ms = 200):
        """
        Resample data to the target frequency handling session gaps.

        :param df: Raw DataFrame.
        :type df: pd.DataFrame
        :param time_col: Name of the time column.
        :type time_col: str
        :param freq_hz: Target frequency in Hz.
        :type freq_hz: int
        :param gap_threshold_ms: Gap threshold to split sessions.
        :type gap_threshold_ms: int
        :return: Interpolated DataFrame.
        :rtype: pd.DataFrame
        """
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)
        df['delta'] = df[time_col].diff().dt.total_seconds() * 1000
        df['session'] = (df['delta'] > gap_threshold_ms).cumsum()

        interpolated = []
        for session_id, group in df.groupby('session'):
            group = group.set_index(time_col)
            group = group.sort_index()
            new_index = pd.date_range(start=group.index[0], end=group.index[-1], freq=f'{int(1000/freq_hz)}ms')
            df_interp = pd.DataFrame(index=new_index)

            for col in group.columns.difference(['delta', 'session']):
                clean = group[col].dropna()
                if len(clean) >= 4:
                    t = (clean.index - clean.index[0]).total_seconds().to_numpy()
                    y = clean.to_numpy()
                    cs = CubicSpline(t, y)
                    t_new = (new_index - clean.index[0]).total_seconds().to_numpy()
                    df_interp[col] = cs(t_new)
                else:
                    df_interp[col] = np.nan

            df_interp.reset_index(inplace=True)
            df_interp.rename(columns={'index': time_col}, inplace=True)
            df_interp['session'] = session_id
            interpolated.append(df_interp)

        result = pd.concat(interpolated, ignore_index=True)
        result.dropna(inplace=True)
        return result

    def preprocess_data(self, df):
        """
        Process DataFrame to compute time, sample rate, and sensor arrays.

        :param df: Preprocessed DataFrame.
        :type df: pd.DataFrame
        :return: Tuple of time array, sample rate, gyroscope, accelerometer, magnetometer arrays, sample period and filtered GPS DataFrame..
        :rtype: tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float, pd.DataFrame]
        """
        ...
        df['time'] = (df['_time'] - df['_time'].iloc[0]).dt.total_seconds()
        time = df['time'].to_numpy()
        sample_period = np.mean(np.diff(time))
        sample_rate = 1.0 / sample_period

        gyr = df[['Gx', 'Gy', 'Gz']].to_numpy() * np.pi / 180
        acc = df[['Ax', 'Ay', 'Az']].to_numpy() 
        mag = df[['Mx', 'My', 'Mz']].to_numpy() * 100
        
        

        return time, sample_rate, gyr, acc, mag
    
    def compute_positions(self, df, config):
        """
        Convert GPS coordinates to local Cartesian positions using projection configuration.

        :param df: DataFrame with 'lat' and 'lng' columns.
        :type df: pd.DataFrame
        :param config: Configuration dictionary containing 'Location' section with projection params.
        :type config: dict
        :return: Tuple of GPS position array and final GPS position.
        :rtype: tuple[np.ndarray, np.ndarray]
        :raises KeyError: If required projection parameters are missing in config['Location'].
        """
        location_cfg = config["Location"]
        proj = Proj(
            proj=location_cfg["proj"],
            zone=location_cfg["zone"],
            ellps=location_cfg["ellps"],
            south=location_cfg["south"]
        )

        df_gps = df[['lat', 'lng', 'time']].dropna().reset_index(drop=True)

        lat = df_gps['lat'].to_numpy()
        lng = df_gps['lng'].to_numpy()
        
        x, y = proj(lng, lat)
        gps_pos = np.stack((x - x[0], y - y[0]), axis=1)
        return df_gps,gps_pos, gps_pos[-1]
