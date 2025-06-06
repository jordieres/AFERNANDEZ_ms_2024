from InfluxDBms.cInfluxDB import cInfluxDB
from datetime import datetime

# # Crear instancia
# iDB = cInfluxDB(config_path=config_path)

# # Llamar a la función debug_fields
# iDB.debug_fields()



config_path = "InfluxDBms/config_db.yaml"
iDB = cInfluxDB(config_path=config_path)

iDB.show_raw_sample(
    from_date=datetime(2024, 11, 2, 16, 0, 0),
    to_date=datetime(2024, 11, 2, 16, 15, 0),
    qtok="JOM20241031-104",
    pie="Left"
)