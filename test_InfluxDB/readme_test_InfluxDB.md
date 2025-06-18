# README - test_influxDB

## Description

The `test_InfluxDB` folder contains test scripts designed to verify the functionality of querying and retrieving data from an InfluxDB database. It also includes a directory for storing the test results in Excel format.

These scripts are particularly useful for debugging and validating data access parameters, metrics, and aggregation behaviors using the `InfluxDBms` module.


## Folder Structure

```text
test_InfluxDB/
│
├── test_influx_basic.py                
├── test_influx_basic_conMetrics.py    
├── test_influx_aggWindow.py           
├── extract_data.py                    
│
└── out/                               # Folder for output Excel files
```

## Test Files

-  **`test_influx_basic.py`**:

        - Description: Executes a basic query to the InfluxDB database.
        - Execution command:
        ```bash
            python test_influx_basic.py -f "2024-11-02T15:08:45Z" -u "2024-11-02T20:50:00Z" -q "JOM20241031-104" -l "Left"
        ```

- **`test_influx_basic_conMetrics.py`**:
    Description: Executes a query including additional metric parameters.
    Execution command:
        ```bash
            python test_influx_basic_conMetrics.py -f "2024-11-02T15:08:45Z" -u "2024-11-02T20:50:00Z" -q "JOM20241031-104" -l "Left" -m "Ax,Ay,Az"
        ```
- **`test_influx_aggWindow.py`**:

        - Description: Executes queries with a defined aggregation window.
        - Execution command:
        ```bash
            python test_influx_aggWindow.py -f "2024-11-02T15:08:45Z" -u "2024-11-02T20:50:00Z" -q "JOM20241031-104" -l "Left" -m "Ax,Ay,Az" -w "100ms"
        ```

- **`extract_data.py`**

        - Description: Extracts data from InfluxDB and saves it to an Excel file.
        - Execution command:
        ```bash
            python extract_data.py \
            -f "2024-11-02T15:08:45Z" \
            -u "2024-11-02T20:50:00Z" \
            -q JOM20241031-104 \
            -l Left \
            -p ../InfluxDBms/config_db.yaml \
            -o "C:/.../test_InfluxDB/out/dat_2024_prueba6.xlsx" \
            -v 2 \
            -m Ax,Ay,Az,Gx,Gy,Gz,Mx,My,Mz,S0,S1,S2

        

## Output Folder

**`out/`** folder stores the files generated after executing the scripts. Each file is saved in Excel (.xlsx) format for further analysis.

## Dependencies
- To install the necessary dependencies, run:
```bash
    pip install pandas python-dateutil influxdb-client
```

## Notes

- It is recommended to check the database configuration in ../InfluxDBms/config_db.yaml before running the scripts.
- For more details on execution parameters, use --help with each script.