# README - transform_data

## Description

## Description

This folder is intended for processing and evaluating gait data using IMU and GPS signals. It contains the script `stride_measurement.py`, which serves as the main entry point for executing a complete gait analysis pipeline—including sensor fusion, stride detection, and trajectory evaluation.

The script leverages multiple processing classes from the `msGeom` package to perform operations such as data resampling, Kalman filtering, peak detection, and stride validation. Users can control the pipeline through command-line arguments, specifying input files, configuration options, and output preferences.

The folder also includes an `out_transform_data/ `directory, which is used to store the generated plots, evaluation reports, and optional Excel exports.


## Structure

```text
transform_data/
│
├── __init__.py               # Package initializer
├── stride_measurement.py     # Main script for IMU+GPS data processing                             
├── out_transform_data/       # Output directory for plots and exported data
└── README.md            
```

## How It Works

The main script, `stride_measurement.py`, uses command-line arguments to control processing options such as:

- `--file_path` → Path to the input Excel file containing IMU and GPS data
- `--verbose` → Verbosity level from 1 to 3 (default: 3)
- `--config` → Path to the YAML configuration file
- `--output_mode` → Plot display/save mode: screen, save, or both
- `--output_dir` → Directory to save plots and data exports
- `--export_excel` → Whether to export stride data to Excel (yes / no)
- `--map_html` → Whether to generate an interactive HTML map (yes / no)

---

## Example Usage
```bash
python transform_data/stride_measurement.py \
  -f  file_path\test.xlsx \
  -v 3 \
  -c config.yaml \
  -om screen \
  -o transform_data/out_transform_data \
  -e no \
  -m no
```

---

## Debugging with VS Code

The script is designed to run either from terminal or through a debugger (e.g., VSCode). Example debug configuration:

```json
{
  "name": "Python Debugger: stride_measurement",
  "type": "debugpy",
  "request": "launch",
  "program": "${file}",
  "python": "${workspaceFolder}/venv/Scripts/python.exe",
  "args": [
    "-f", "${workspaceFolder}/file/path/test.xlsx",
    "-v", "3",
    "-c", "${workspaceFolder}/config.yaml",
    "-om", "screen",
    "-o", "${workspaceFolder}/transform_data/out_transform_data",
    "-e", "no",
    "-m", "no"
  ],
  "console": "integratedTerminal",
  "cwd": "${workspaceFolder}"
}
```

## Output

Depending on selected options, the script can produce:

- **Trajectory Plots**: IMU vs. Kalman vs. GPS
- **Stride Statistics**: Per-step and per-minute metrics
- **Stride Filtering Summary**: Valid/invalid stride lists
- **Evaluation Results**: Quality score per segment
- **Excel Reports** (if `-e yes` is enabled)
- **Interactive HTML Map** (if `-m yes` is enabled)

All outputs (except screen-only plots) are saved to `transform_data/out_transform_data/`.

---



#  Requirements

Make sure the following Python packages are installed:

```bash
pip install numpy pandas matplotlib tabulate argsparse
```

Additionally, the script relies on an internal package named `msGeom` which must include:

- `DataPreprocessor`
- `IMUProcessor`
- `KalmanProcessor`
- `DetectPeaks`
- `StrideProcessor`
- `ResultsProcessor`
- `PlotProcessor`

---

##  Notes

- The GPS–IMU alignment assumes a walking distance of ~430 meters (can be configured).
- Be sure to customize `.config.yaml` based on your sensor setup and desired filters.
- Compatible with both Windows and Linux environments.
