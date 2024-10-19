import sys
import os
sys.path.insert(0, os.path.abspath('../../conexionBBDD'))

# Intentar importar los módulos
try:
    import conexionBBDD.DbClassInflux
    import conexionBBDD.fecha_verbose
    print("Importaciones exitosas.")
except ImportError as e:
    print(f"Error en la importación: {e}")
