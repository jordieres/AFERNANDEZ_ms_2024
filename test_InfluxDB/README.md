# README - test_InfluxDB

## Description

This folder is intended for testing and validating the InfluxDB data extraction functionality implemented in the `InfluxDBms` package. It contains the script `extract_data.py`, which serves as the entry point for querying time-series data from an InfluxDB database and exporting the results to an Excel file.

This script creates an instance of the `cInfluxDB` class from the `influxdb_tools` module, allowing users to easily define date ranges, metrics, and other query parameters via command-line arguments.

The folder also includes an `out/` directory, which is used to store the resulting `.xlsx` files generated after each execution.

---

## Structure

```text
test_InfluxDB/
├── extract_data.py         # Script to extract data from InfluxDB and export to Excel
├── out/                    # Output folder for resulting Excel files
└── README.md               
```

---

## How It Works

The script `extract_data.py` connects to InfluxDB, queries sensor data for a specific CodeID, date range, and foot side, and writes the resulting dataset to an Excel file.

It uses command-line arguments to control input parameters such as:

- `--from_time` and `--until` → Start and end datetime (ISO 8601)
- `--qtok` → Subject or record identifier (CodeID)
- `--leg` → Foot (Left or Right)
- `--config` → Path to YAML configuration file
- `--output` → Destination path for Excel output
- `--metrics` → Comma-separated list of metrics (e.g., Ax,Gx,My)
- `--verbose` → Verbosity level from 0 to 3

---

## Example Usage

You can run the script directly from the terminal:

```bash
python extract_data.py \
  -f "2024-04-25T14:20:52Z" \
  -u "2024-04-25T14:21:52Z" \
  -l Right \
  -q "********" \
  -c .config.yaml \
  -o test_InfluxDB/out/test.xlsx \
  -v 3 \
  -m Ax,Ay,Az,Gx,Gy,Gz,Mx,My,Mz
```

---

## Debugging with VS Code

To debug the script using Visual Studio Code, ensure you have the following configuration in `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python Debugger: extract_data",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/test_InfluxDB/extract_data.py",
      "python": "${workspaceFolder}/venv/Scripts/python.exe",
      "args": [
        "-f", "2024-04-25T14:20:52Z",
        "-u", "2024-04-25T14:21:52Z",
        "-l", "Right",
        "-q", "******",
        "-c", ".config.yaml",
        "-o", "test_InfluxDB/out/test.xlsx",
        "-v", "3",
        "-m", "Ax,Ay,Az,Gx,Gy,Gz,Mx,My,Mz"
      ],
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}"
    }
  ]
}
```

---

## Output

All generated Excel files will be saved inside the `out/` directory. Each file contains queried time-series sensor data, ready for analysis or visualization.

---

## Requirements

Make sure the `InfluxDBms` package is installed and the `config_db.yaml` file is properly configured with InfluxDB credentials before running the script.
Additionally, ensure that the `argparse` module is available (it is included by default in standard Python distributions).

```bash
    pip install argparse
```

## Notes

The `extract_data.py` script includes a few commented-out lines that can be activated to use the `aggregateWindow` functionality provided by InfluxDB.

```python
# parser.add_argument('-w', '--window_size', type=str, default="20ms", help="Aggregation window size (default: 20ms)")
# df = iDB.query_with_aggregate_window(args.from_time, args.until, window_size=args.window_size, qtok=args.qtok, pie=args.leg, metrics=metrics)
```

The `aggregateWindow` feature is useful for reducing data density by grouping values over fixed intervals (e.g., 20 milliseconds) and applying aggregation functions (like `last`, `mean`, or `sum`). This is especially helpful when working with high-frequency time-series data where downsampling improves performance, simplifies visualization, and can reduce noise.

By enabling this feature, you can obtain a more manageable dataset while preserving the relevant signal trends.

To use it:
1. Uncomment the `-w / --window_size` argument in the parser.
2. Replace the call to `query_data(...)` with `query_with_aggregate_window(...)` in the `main()` function.

This makes the script more flexible for exploratory analysis and large datasets.