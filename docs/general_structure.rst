General Structure
=================

The application includes:

* A package named ``InfluxDBms`` for managing and querying time-series data.
* A package named ``msGeom`` for IMU + GPS fusion, orientation estimation, stride detection,
  trajectory reconstruction, and gait metric calculation.
* A main execution script ``test_InfluxDB/extract_data.py`` to retrieve raw data from InfluxDB.
* A main execution script ``transform_data/stride_measurement.py`` to run the full processing pipeline,
  from preprocessing to gait metrics output.

Project Layout
--------------

.. code-block:: text

   AFERNANDEZ_ms_2024/
   │
   ├── InfluxDBms/            # InfluxDB data access and querying
   ├── msGeom/                # Sensor fusion, trajectory estimation, visualization
   ├── test_InfluxDB/         # Query InfluxDB and export to Excel
   ├── transform_data/        # Motion analysis pipeline (main script)
   │
   ├── docs/                  # Sphinx documentation
   ├── dist/                  # Distribution artifacts
   ├── .vscode/               # VSCode settings
   │
   ├── config.yaml            # General project configuration
   │
   ├── LICENSE
   ├── README.md
   ├── poetry.lock
   └── pyproject.toml
