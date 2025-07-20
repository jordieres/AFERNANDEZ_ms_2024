Introduction
============

The main purpose of this package is to assist in the identification and analysis of gait features.
The ambition is to characterize the fine-grain attributes of the walking process, particularly for
individuals affected by Multiple Sclerosis (MS).

Motivation
==========

The project focuses on seamlessly integrating data collected at the edge. High-frequency sensor data
from Sensoria Health© instrumented socks is transmitted via BLE through custom Android or iOS apps
and uploaded into an InfluxDB time-series database.

Additional tools analyze this data to identify valid activity periods and store them in a PostgreSQL
relational database. Using specialized algorithms, walking periods longer than 7 seconds are detected
and further analyzed.

The current goal is to extract detailed low-level gait features, such as stride length, cadence,
swing distance, and other movement metrics to enable precise assessment of mobility impairments.

General Structure
=================

The application includes:

* A package named ``InfluxDBms`` containing classes for managing and querying InfluxDB.
* A package named ``msGeom`` containing processing tools for IMU + GPS fusion, step detection, 
  stride analysis, and visualization.
* A main execution script ``test_InfluxDB\extract_data.py`` that extract data from InfluxDB.
* A main execution script ``transform_data/stride_measurement.py`` to run the full processing pipeline.

Detailed information can be found in the modules listed below.