import yaml
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline

class DataGait:
    """

    """
    def __init__(self, cnf: str, key: str, vb: int = 0) -> None:
        """

        """
        self.config = cnf
        self.key    = key
        self.verbose= vb
        self.dat    = {}
        self.dat_frq= {}
        
        with open(self.config, 'r') as file:
            self.config_obj = yaml.safe_load(file)

        
    def load_df(self, df: pd.DataFrame, leg:str) -> None:
        """
        sumary_line
        
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        self.dat[leg.upper()] = df
        
    def get_legs(self) -> list:
        """
        Explores the loaded keys in self.dat
        
        Return: dict with leg name and number of rows.
        """
        res = {}
        kys = self.dat.keys()
        for i in kys:
            res[kys] = self.dat[kys].shape[0]
        
        return res
    
    def get_cnf(self,key:str) -> dict | None :
        """
        Returns the properties of the object self.config_obj[key] if exists
        
        Keyword arguments:
        key -- Component of the config element of interest
        Return: dict with the parameters linked to the indicated key.
        """
        if key in self.config_obj.keys():
            return self.config_obj[key]
        else:
            return None
        
        

    def load_data_from_hdf5_leg(self, hdf5_file_path: str, hdf5_full_key: str, leg: str) -> None:
        """
        Loads a DataFrame from a specific HDF5 key within a file.

        :param hdf5_file_path: Path to the HDF5 file.
        :param hdf5_full_key: The full key (path) to the DataFrame within the HDF5 file.
        :return: Loaded Pandas DataFrame.
        :raises KeyError: If the key does not exist in the HDF5 file.
        """
        with pd.HDFStore(hdf5_file_path, mode='r') as store:
            if hdf5_full_key in store:
                df = store.get(hdf5_full_key)
                self.load_df(df,leg)
                if self.verbose >= 3:
                    print(f"Successfully loaded data from HDF5 key: {hdf5_full_key}")
            else:
                raise KeyError(f"Key '{hdf5_full_key}' not found in HDF5 file '{hdf5_file_path}'.")



    def resample2frq_cte(self, time_col='_time', freq_hz=40, gap_threshold_ms=400) -> None:
        """
        Resample data to the target frequency handling session gaps when data is avaliable at self.dat.
        It will be done on both legs

        :param df: Raw DataFrame.
        :param time_col: Name of the time column.
        :type time_col: str
        :param freq_hz: Target frequency in Hz.
        :type freq_hz: int
        :param gap_threshold_ms: Gap threshold to split sessions.
        :type gap_threshold_ms: int
        :return: Interpolated DataFrame.
        :rtype: pd.DataFrame
        """
        
        for leg in self.dat.keys():
            df = self.dat[leg].copy()
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
        self.dat_frq[leg] = result
