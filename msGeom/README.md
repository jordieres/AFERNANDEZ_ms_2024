# README - msGeom

## Description

This module contains all the core classes for preprocessing, sensor fusion, step detection, trajectory estimation, and visualization of movement data from IMU and GPS sensors.

The main goal of the `msGeom` package is to provide a modular, reusable, and extensible set of tools for analyzing pedestrian locomotion, evaluating step-level and trajectory-level metrics, and aligning inertial data with GPS traces.

It supports multiple stages in the processing pipeline, including:
- Loading and resampling data from structured Excel sheets.
- Preprocessing raw IMU data (acceleration, gyroscope, magnetometer).
- Estimating position with orientation filters (Madgwick/Mahony) and Kalman filtering.
- Detecting peaks in sensor signals related to foot strikes.
- Computing stride-level statistics and quality metrics.
- Visualizing results using interactive or static visualizations (Folium, Plotly, Matplotlib).
- Exporting clean, structured results for further analysis.


## Structure

```text
msGeom/
├── data_preprocessor.py        
├── imu_processor.py
├── kalman_processor.py  
├── peak_detector.py  
├── stride_processor.py  
├── result_processor.py                      
├── plot_processor.py   
└── README.md               
```

---

## Usage

The set of classes defined in `msGeom/` is used to execute the processing pipeline implemented in the script:

```
transform_data/stride_measurement.py
```

This script orchestrates the full IMU + GPS processing workflow. It loads the data, runs preprocessing, orientation estimation, sensor fusion, step detection, stride validation, and visualization, using the various classes in `msGeom`. It supports command-line arguments for configuration, verbosity level, exporting results, and map generation.

You can run it with:

```bash
python transform_data/stride_measurement.py -f path/to/your_file.xlsx -c path/to/.config.yaml -o results/ -om both -e yes -m yes
```

---

## Class Overview


- **`data_preprocessor.py`**
  - `DataPreprocessor`: Loads configuration and Excel data, performs resampling, and prepares sensor signals for analysis.

- **`imu_processor.py`**
  - `IMUProcessor`: Processes accelerometer, gyroscope, and magnetometer data. Includes gravity vector estimation, motion/stationary detection, and position estimation using Madgwick or Mahony algorithms with ZUPT/ZUPH corrections.

- **`kalman_processor.py`**
  - `KalmanProcessor`: Implements a 2D Kalman Filter to fuse GPS and IMU data, estimating system position and velocity.

- **`peak_detector.py`**
  - `DetectPeaks`: Detects step-related peaks in IMU signals and computes stride statistics.

- **`stride_processor.py`**
  - `StrideProcessor`: Filters and validates stride-related data. Includes tools for spatial alignment and segment-level diagnostics.

- **`result_processor.py`**
  - `ResultsProcessor`: Exports key per-stride metrics such as position and velocity to Excel, and computes relevant movement metrics.

- **`plot_processor.py`**
  - `PlotProcessor`: Provides visualization tools using Matplotlib, Plotly, and Folium. Enables comparison between estimated trajectories and GPS data, and generates interactive or static visual outputs.

---

## Setup – Install

1. Ensure you have the following prerequisites installed:
   - **Python 3.8+**
   - The following Python libraries:

      * `numpy`
      * `pandas`
      * `pyyaml`
      * `matplotlib`
      * `plotly`
      * `folium`
      * `ahrs`
      * `pyproj`
      * `filterpy`
      * `scipy`
      * `geopy`

- To install the required Python libraries, you have **two options**:

  - **Option 1: Use `requirements.txt`**:

    ```bash
    pip install -r requirements.txt
    ```

  - **Option 2: Manually**:

    ```bash
    pip install numpy pandas pyyaml matplotlib plotly folium ahrs pyproj filterpy scipy geopy
    ```

2. Configure your environment:

   Ensure the file `.config.yaml` contains the correct projection and zone parameters for your coordinate system. See the **Configuration** section below for details.

---

## Configuration

Create or modify the `.config.yaml` file with the following structure:

```yaml
Location:
  proj: 'utm'
  zone: 30
  ellps: 'WGS84'
  south: false
```

This configuration is required to correctly project geographic coordinates (latitude/longitude) into a local coordinate system for position estimation and spatial analysis.