# gps_map_generator.py

import pandas as pd
import folium

def generate_gps_map_from_excel(excel_path: str, output_html_path: str):
    """
    Generates a GPS trajectory map from an Excel file and saves it as an HTML file.

    :param excel_path: Path to the Excel file containing '_time', 'Latitude', and 'Longitude'.
    :type excel_path: str
    :param output_html_path: Path where the HTML file with the map will be saved.
    :type output_html_path: str
    """

    # Load GPS data from the Excel file
    df = pd.read_excel(excel_path)

    # Ensure '_time' is datetime and sort by time
    df['_time'] = pd.to_datetime(df['_time'])
    df = df.sort_values(by='_time')

    # Remove duplicate coordinates
    df = df.loc[(df['Latitude'].shift() != df['Latitude']) | (df['Longitude'].shift() != df['Longitude'])]
    df = df.reset_index(drop=True)

    # Create a map centered on the first coordinate
    start_location = [df.loc[0, 'Latitude'], df.loc[0, 'Longitude']]
    map_object = folium.Map(location=start_location, zoom_start=18)

    # Add polyline of the trajectory
    coordinates = df[['Latitude', 'Longitude']].values.tolist()
    folium.PolyLine(locations=coordinates, color='blue', weight=4).add_to(map_object)

    # Add start and end markers
    folium.Marker(location=coordinates[0], popup="Start", icon=folium.Icon(color='green')).add_to(map_object)
    folium.Marker(location=coordinates[-1], popup="End", icon=folium.Icon(color='red')).add_to(map_object)

    # Save to HTML
    map_object.save(output_html_path)
    print(f"Map saved successfully at: {output_html_path}")

