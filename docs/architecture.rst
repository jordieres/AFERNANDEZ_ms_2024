Architecture
============

End-to-end data flow
--------------------
1. Sensoria socks (IMU + pressure + GPS) -> BLE -> Mobile app
2. App -> InfluxDB (time-series storage)
3. ``test_InfluxDB/extract_data.py`` -> export to Excel
4. ``transform_data/stride_measurement.py``:
   - Preprocessing & resampling
   - Orientation estimation (Madgwick)
   - GPS+IMU fusion (Kalman)
   - Step/stride detection (modG peaks)
   - Stride validation & quality checks
   - Plots, Excel export, optional HTML map

Main components
---------------
* ``InfluxDBms``: InfluxDB connectivity and query helpers.
* ``msGeom``: preprocessing, filters (Madgwick/Mahony), Kalman fusion, peak detection,
  stride stats, plotting, and exporting.
* Scripts:
  * ``test_InfluxDB/extract_data.py``
  * ``transform_data/stride_measurement.py``

.. note::
   Coordinate projection parameters are read from ``.config.yaml`` (UTM zone, ellipsoid, etc.)
   to project geographic coordinates to a local Cartesian frame.

