import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from pyproj import Proj


class GPSTrajectoryProcessor:
    """
    A class that handles GPS data processing, mapping, and trajectory plotting.
    """

    def __init__(self):
        """
        Constructor. Currently does not require any parameters.
        """
        pass

    def prepare_gps_dataframe(self, excel_path: str) -> pd.DataFrame:
        """
        Loads and prepares a GPS DataFrame from an Excel file.

        :param excel_path: Path to the Excel file containing '_time', 'lat', and 'lng'.
        :return: Cleaned and sorted DataFrame with unique GPS points.
        """
        df = pd.read_excel(excel_path)
        df['_time'] = pd.to_datetime(df['_time'])
        df = df.sort_values(by='_time')
        df = df.loc[(df['lat'].shift() != df['lat']) | (df['lng'].shift() != df['lng'])]
        return df.reset_index(drop=True)

    def generate_gps_map(self, df: pd.DataFrame, output_html_path: str):
        """
        Generates a GPS trajectory map and saves it as an HTML file.

        :param df: DataFrame with 'lat' and 'lng' columns.
        :param output_html_path: Path where the HTML file with the map will be saved.
        """
        start_location = [df.loc[0, 'lat'], df.loc[0, 'lng']]
        map_object = folium.Map(location=start_location, zoom_start=18)

        coordinates = df[['lat', 'lng']].values.tolist()
        folium.PolyLine(locations=coordinates, color='blue', weight=4).add_to(map_object)
        folium.Marker(location=coordinates[0], popup="Start", icon=folium.Icon(color='green')).add_to(map_object)
        folium.Marker(location=coordinates[-1], popup="End", icon=folium.Icon(color='red')).add_to(map_object)

        map_object.save(output_html_path)
        print(f"Map saved successfully at: {output_html_path}")



    def plot_macroscopic_trajectory(self, df: pd.DataFrame, results: dict = None, errors: dict = None, output_path: str = None):
        """
        Converts GPS coordinates to local flat coordinates (using UTM projection),
        and plots the trajectory along with optional estimated trajectories (e.g., from IMU).

        :param df: DataFrame with 'lat' and 'lng' columns representing GPS data.
        :param results: Optional dictionary of estimated trajectories. Format: {name: np.ndarray (N, 2)}.
        :param errors: Optional dictionary of final position errors for each estimated trajectory.
        :param output_path: Optional path to save the plot as a PNG file.
        """
        lat = df['lat'].to_numpy()
        lng = df['lng'].to_numpy()

        # UTM projection (zone 30, WGS84). Adjust zone if needed for your location.
        proj = Proj(proj='utm', zone=30, ellps='WGS84', south=False)
        x_gps, y_gps = proj(lng, lat)

        # Normalize GPS path relative to the starting point
        gps_pos = np.stack((x_gps - x_gps[0], y_gps - y_gps[0]), axis=1)
        gps_final = gps_pos[-1]

        plt.figure(figsize=(10, 8))

        # Plot estimated trajectories if provided
        if results:
            for name, pos in results.items():
                err = errors[name] if errors and name in errors else 0.0
                plt.plot(pos[:, 0], pos[:, 1], label=f"{name} ({err:.2f} m)")

        # Plot GPS reference trajectory
        plt.plot(gps_pos[:, 0], gps_pos[:, 1], 'k--', label="GPS (reference)")
        plt.plot(gps_final[0], gps_final[1], 'ko', label="Final GPS position")
        plt.title("Trajectory Comparison (Estimated vs GPS)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.grid()
        plt.legend()

        if output_path:
            plt.savefig(output_path)
            print(f"Macroscopic trajectory saved to: {output_path}")

        plt.show()


    def calculate_total_distance(self, df: pd.DataFrame) -> float:
        """
        Calculates the total distance walked (in meters) using the haversine formula.

        :param df: DataFrame with 'lat' and 'lng' columns.
        :return: Total distance in meters.
        """
        R = 6371000  # Earth radius in meters

        lat_rad = np.radians(df['lat'].values)
        lng_rad = np.radians(df['lng'].values)

        delta_lat = lat_rad[1:] - lat_rad[:-1]
        delta_lng = lng_rad[1:] - lng_rad[:-1]

        a = np.sin(delta_lat / 2)**2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(delta_lng / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distances = R * c

        total_distance = np.sum(distances)
        return total_distance
