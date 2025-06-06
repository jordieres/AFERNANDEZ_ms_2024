# run_gps_map.py
from gps_trajectory.gps_map_generator import generate_gps_map_from_excel

# Define paths
input_excel = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_prueba10.xlsx"
output_html = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\gps_trajectory\out_gps\trayectoria_gps1.html"

# Generate the map
generate_gps_map_from_excel(input_excel, output_html)
