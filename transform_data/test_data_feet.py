import numpy as np
import pandas as pd
import argparse
import os
import yaml
from scipy import signal
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ahrs.filters import Madgwick
from ahrs.common.orientation import q_conj, q_rot
from pyproj import Proj


def load_data(file_path):
    """
    Load Excel data from the specified path.

    :param file_path: Path to the Excel file.
    :type file_path: str
    :return: DataFrame containing raw data.
    :rtype: pd.DataFrame
    """
    df = pd.read_excel(file_path)
    return df



def resample_to_40hz(df, time_col='_time', freq_hz=40, gap_threshold_ms=200):
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


def print_summary(df, label):
    """
    Imprime un resumen de los datos: lat/lng inicial y final, tiempo inicial y final, número de muestras.

    :param df: DataFrame de entrada
    :param label: Texto para indicar si es antes o después de interpolar
    """
    print(f"\nResumen {label}:")
    try:
        lat_ini = df['lat'].iloc[0]
        lat_fin = df['lat'].iloc[-1]
        lng_ini = df['lng'].iloc[0]
        lng_fin = df['lng'].iloc[-1]
        time_ini = df['_time'].iloc[0]
        time_fin = df['_time'].iloc[-1]
        n_samples = len(df)
        print(f"  Latitud inicial: {lat_ini}, final: {lat_fin}")
        print(f"  Longitud inicial: {lng_ini}, final: {lng_fin}")
        print(f"  Tiempo inicial: {time_ini}, final: {time_fin}")
        print(f"  Número de muestras: {n_samples}")
    except Exception as e:
        print(f"  Error al imprimir resumen: {e}")


def parse_args():
    """
    Parse command-line arguments for the IMU processing pipeline.

    :return: Parsed command-line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="IMU data processing pipeline")
    parser.add_argument("-f", "--file_paths", type=str, nargs="+", required=True, help="Paths to one or more Excel files")
    parser.add_argument("--threshold", type=float, default=0.1, help="Stationary detection threshold")
    parser.add_argument('-v', '--verbose', type=int, default=3, help='Verbosity level')
    parser.add_argument('-c', '--config', type=str, default='.config.yaml', help='Path to the configuration file')
    parser.add_argument('--output_mode', choices=["screen", "save", "both"], default="screen", help="How to handle output plots: 'screen', 'save', or 'both'")
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='Directory to save output plots')
    return parser.parse_args()





def main():
    """
    Main function to process IMU Excel files and plot estimated trajectories vs GPS.
    """
    args = parse_args()
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for file_path in args.file_paths:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        foot_label = "Left Foot" if "left" in base_name.lower() else "Right Foot" if "right" in base_name.lower() else base_name

        if args.verbose >= 2:
            print(f"\n{'*'*33}  {foot_label}  {'*'*33}\n")
            print(f"{'-'*80}")
            print(f"Processing file: {base_name}...")
            print(f"{'-'*80}\n")

        try:
            df = load_data(file_path)

             # Mostrar resumen antes de interpolar
            print_summary(df, "ANTES de interpolar")
            print(df.head())
            df_interp = resample_to_40hz(df)

            # Mostrar resumen después de interpolar
            print_summary(df_interp, "DESPUÉS de interpolar")
            print(df_interp.head())

            # 1. Comprobar si hay valores nulos
            print("¿Tiene NaNs?")
            print(df_interp.isnull().sum())

            # 2. Ver las estadísticas generales
            print("\nResumen estadístico:")
            print(df_interp.describe())

            # 3. Ver las primeras y últimas filas
            print("\nPrimeras filas:")
            print(df_interp.head())

            print("\nÚltimas filas:")
            print(df_interp.tail())

            # 4. Comprobar que los tiempos estén bien ordenados y espaciados
            print("\nEspaciado temporal:")
            delta_times = df_interp['_time'].diff().dt.total_seconds().dropna()
            print(f"Media del intervalo (s): {delta_times.mean():.4f}")
            print(f"¿Hay intervalos mayores de 0.05s?: {(delta_times > 0.05).sum()}")

            # 5. Ver si hay variación en columnas clave
            print("\n¿Ax tiene valores únicos?:", df_interp['Ax'].nunique())
            print("¿lat/lng tienen variación?:", df_interp['lat'].nunique(), df_interp['lng'].nunique())
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    main()


