# README - transform_data

## Description

The `transform_data` folder contains scripts and classes designed to process IMU (Inertial Measurement Unit) and GPS data for motion tracking and trajectory estimation. The pipeline includes orientation estimation using filters such as **Madgwick** and **Mahony**, and position refinement via **Kalman**, **EKF**, complementary filters, and drift correction.

It also supports generation of static plots and interactive maps, making it a versatile tool for both analysis and visualization of movement data.

## Folder Structure

```text
transform_data/
│
├── class_madwick.py              # Madgwick-based orientation + position estimation
├── class_transform_imu.py        # Data preprocessing, filtering, and fusion functions
├── test_data_feet.py             # Data inspection and diagnostic script
├── transform_imu_madwick_v1.py   # Full Madgwick pipeline + ZUPT/ZUPH + filtering
├── transform_imu_madwick_v2.py   # Kalman prediction test with split trajectory
├── transform_imu_unif.py         # Unified comparison of Madgwick/Mahony with/without magnetometer
│
├── done/                         # Prototypes and tests using ahrs and skinematics
└── out_transform_data/           # Output folder for plots and maps
```


##  File Descriptions

### `class_madwick.py`
Implements orientation and position estimation from IMU data using:
- Adaptive Madgwick filter (with ZUPT/ZUPH)
- Drift correction strategies:
  - Kalman filter
  - EKF 2D
  - Complementary filter
  - Progressive linear correction
- Visualization of acceleration, position, velocity, and 2D/3D trajectories

### `class_transform_imu.py`
Support functions for the full processing pipeline:
- File loading and resampling to 40 Hz
- Stationary period detection
- Orientation estimation (Madgwick/Mahony)
- Application of multiple filters (Kalman, EKF, Complementary)
- Plotting with `matplotlib`, `plotly`, and mapping with `folium`

### `test_data_feet.py`
Diagnostic tool for IMU Excel files:
- Verifies file structure, timestamps, null values, and outliers
- Interpolates and compares data before and after resampling
- Useful for validating raw sensor data

---

## Main Scripts

### `transform_imu_madwick_v1.py`
Processes IMU + GPS files using Madgwick (with magnetometer):
- Applies ZUPT/ZUPH strategies
- Estimates IMU trajectory and fuses it with GPS
- Compares multiple drift correction methods
- Saves and/or displays visual outputs

### `transform_imu_madwick_v2.py`
Splits the dataset in two halves:
- First half uses Kalman filter with GPS corrections
- Second half estimates position without GPS (prediction mode)
- Designed to test Kalman filter robustness

### `transform_imu_unif.py`
Unified algorithm comparator:
- Runs multiple filters (Madgwick/Mahony) with and without magnetometer
- Applies Kalman filtering to each estimation
- Plots error metrics and trajectories
- Supports interactive map visualization



## Dependencies

To install all required packages, run:

```bash
pip install numpy pandas scipy matplotlib ahrs pyproj plotly folium filterpy pyyaml openpyxl
```

## Notes
- All scripts support the --help flag to display usage and parameters.
- Use --output_mode both if you want to both show and save the output figures.
- Output plots will be stored in the out_transform_data/ folder if the --output_dir parameter is specified.
- The system assumes input data is in Excel format (.xlsx) with IMU and GPS columns like Ax, Gx, lat, lng, _time, etc.