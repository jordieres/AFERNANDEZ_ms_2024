import numpy as np
import pandas as pd


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

