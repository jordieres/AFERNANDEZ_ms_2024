# run_gps_map.py
from gps_trajectory.gps_map_generator import *

# Define paths
input_excel = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_tabuenca_right.xlsx"
output_html = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\gps_trajectory\out_gps\trayectoria_tabuenca_right_1.html"

# Prepare GPS data
df = prepare_gps_dataframe(input_excel)

# Generate HTML map
generate_gps_map(df, output_html)

# Plot macroscopic trajectory
plot_macroscopic_trajectory(df)