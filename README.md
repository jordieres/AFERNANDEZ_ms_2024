# AFERNANDEZ\_ms\_2024

## Position calculation using IMU9
This project processes, transforms, and stores data from IMU9 sensors for motion tracking, trajectory estimation, and analysis using time-series databases such as InfluxDB.

## Introduction

The main purpose of this project is to assist in the identification and analysis of gait features. The ambition is to characterize fine-grain attributes of the walking process, particularly for individuals affected by Multiple Sclerosis (MS).

## Motivation

This project focuses on seamlessly integrating data collected at the edge. High-frequency sensor data from Sensoria Health© instrumented socks is transmitted via BLE through custom Android or iOS apps and uploaded into an InfluxDB time-series database.

Additional tools analyze this data to identify valid activity periods and store them in a PostgreSQL relational database. Using specialized algorithms, walking periods longer than 7 seconds are detected and further analyzed.

The current goal is to extract detailed low-level gait features, such as stride length, cadence, swing distance, and other movement metrics to enable precise assessment of mobility impairments.

## Project Structure

```
AFERNANDEZ_ms_2024/
│
├── InfluxDBms/            # Python package for InfluxDB data access and querying
├── msGeom/                # Core module for sensor fusion, trajectory estimation, and visualization
├── test_InfluxDB/         # Scripts for querying InfluxDB and exporting to Excel
├── transform_data/        # Data transformation and motion analysis pipeline
│
├── docs/                  # Project documentation
├── dist/                  # Distribution artifacts
├── .vscode/               # VSCode settings
│
├── config.yaml            # General project configuration
│
├── LICENSE
├── README.md              # This file
├── poetry.lock            # Dependency lock file
├── pyproject.toml         # Poetry project definition
├── .gitignore
```

## Requirements

This project uses [Poetry](https://python-poetry.org/) for dependency management:

```bash
pip install poetry
```

To install dependencies:

```bash
poetry install
```

## Module Descriptions

### `transform_data`

This folder contains scripts and classes for processing IMU and GPS data. It includes orientation estimation filters (Madgwick, Mahony), position refinement methods (Kalman, EKF, complementary filters), and tools for drift correction. It also supports static plots and interactive maps for data visualization.

### `test_InfluxDB`

This module provides tools to test and validate the InfluxDB querying process. The main script, `extract_data.py`, creates an instance of the `cInfluxDB` class from `influxdb_tools` to retrieve sensor data and export it as an Excel file. It supports CLI configuration for time ranges and measurement parameters. Outputs are stored in the `out/` folder.

### `msGeom`

This module provides core classes for preprocessing sensor data, step detection, sensor fusion, and visualization. It supports IMU+GPS analysis pipelines including:

* Resampling and preprocessing raw sensor data
* Orientation filtering (Madgwick, Mahony)
* Kalman filtering for position estimation
* Step and stride detection
* Metric computation and visualization using Plotly, Matplotlib, or Folium

### `InfluxDBms`

A dedicated package for handling InfluxDB connections and queries. It includes the `influxdb_tools.py` module, which defines the `cInfluxDB` class for querying and retrieving data into `pandas` DataFrames. It's tightly integrated with `test_InfluxDB/extract_data.py` for streamlined querying.

## Usage

1. **Configure the system**:

   * Edit `config.yaml` with appropriate parameters

2. **Extract data**:

   * Use the script `transform_data/extract_data.py`  for export data from InfluxDB


3. **Transform and analyze data**:

   * Use scripts in `transform_data/` and `msGeom/` for motion processing, visualization, and export

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
