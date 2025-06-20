import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium


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

    def plot_macroscopic_trajectory(self, df: pd.DataFrame, output_path: str = None):
        """
        Converts GPS coordinates to a macroscopic trajectory in local flat coordinates and plots it.

        :param df: DataFrame with 'lat' and 'lng' columns.
        :param output_path: Path where the PNG file with the plot will be saved.
        """
        lat = df['lat'].values
        lng = df['lng'].values
        lat_rad = np.radians(lat)
        lng_rad = np.radians(lng)

        lat0 = lat_rad[0]
        lng0 = lng_rad[0]
        R = 6371000

        x = R * (lng_rad - lng0) * np.cos(lat0)
        y = R * (lat_rad - lat0)

        plt.figure(figsize=(8, 6))
        plt.plot(x, y, marker='o', linestyle='-', label='GPS')
        plt.title("Macroscopic GPS Trajectory")
        plt.xlabel("East (m)")
        plt.ylabel("North (m)")
        plt.axis('equal')
        plt.grid()
        plt.legend()

        if output_path:
            plt.savefig(output_path)
            print(f"Macroscopic trajectory saved to: {output_path}")

        plt.show()
