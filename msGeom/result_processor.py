import numpy as np
import pandas as pd
import os


class ResultsProcessor:
    """
    Class for preparing and exporting step-wise movement metrics, including position, 
    velocity, and GPS data. Supports exporting results to Excel and computing key metrics.
    """

    def __init__(self):
        pass

    def prepare_step_dataframe(self, time, gps_lat, gps_lng, pos_kalman, vel, df_inter):
        """
        Generate a DataFrame with step-wise position, velocity, and GPS data.

        :param time: Time vector in seconds.
        :type time: np.ndarray
        :param gps_lat: GPS latitude array.
        :type gps_lat: np.ndarray
        :param gps_lng: GPS longitude array.
        :type gps_lng: np.ndarray
        :param pos_kalman: Kalman-filtered position array (N, 2+).
        :type pos_kalman: np.ndarray
        :param vel: Velocity array (N, 3).
        :type vel: np.ndarray
        :return: DataFrame with time, position, velocity, and step distance.
        :rtype: pd.DataFrame
        """
        step_distance = np.zeros(len(time))
        step_distance[1:] = np.linalg.norm(pos_kalman[1:, :2] - pos_kalman[:-1, :2], axis=1)

        df_steps = pd.DataFrame({
            'time': time,
            'lat': gps_lat,
            'lng': gps_lng,
            'pos_x_m': pos_kalman[:, 0],
            'pos_y_m': pos_kalman[:, 1],
            'velocity_m_s': np.linalg.norm(vel, axis=1),
            'step_distance_m': step_distance,
            'datetime': df_inter['_time'].values
        })

        return df_steps
    
    
    def print_metrics(self, name, traj, gps_final):
        """
        Print final error and total distance of a given trajectory compared to GPS final point.

        :param name: Name/label of the trajectory method.
        :type name: str
        :param traj: Estimated position trajectory (N, 2 or 3).
        :type traj: np.ndarray
        :param gps_final: Final reference GPS coordinate [x, y].
        :type gps_final: np.ndarray
        :return: None
        :rtype: None
        """

        final_err = np.linalg.norm(traj[-1, :2] - gps_final)
        total_dist = np.sum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1))
        print(f"- {name}   -> Final error: {final_err:.2f} m | Distance: {total_dist:.2f} m")


    def export_to_excel(self, df, file_path):
        """
        Export a DataFrame to an Excel file.

        :param df: DataFrame to export.
        :type df: pd.DataFrame
        :param file_path: Path where the Excel file should be saved.
        :type file_path: str
        :return: None
        :rtype: None
        :raises Exception: If the Excel file cannot be saved.
        """
        try:
            df.to_excel(file_path, index=False)
            print(f"\n Excel saved: {file_path}")
        except Exception as e:
            print(f" Error saving Excel {file_path}: {e}")

    
    def get_output_path(self, filename, args):
        """
        Construct full output path using provided filename and CLI arguments.

        :param filename: Name of the file to be saved.
        :type filename: str
        :param args: Argument object containing `output_dir`.
        :type args: argparse.Namespace
        :return: Full file path.
        :rtype: str
        """
        return os.path.join(args.output_dir or ".", filename)