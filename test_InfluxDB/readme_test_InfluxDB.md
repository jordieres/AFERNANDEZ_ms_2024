# README - test:influxDE

## Description

The test:influxDE folder contains test files for connecting to and querying data from an InfluxDB database. It includes four Python test files and a folder named resultados_test, where Excel files with the results obtained after running the tests are stored.

## Folder Structure

 test_InfluxDB/
 |-- test_influx_basic.py
 |-- test_influx_basic_conMetrics.py
 |-- test_influx_aggWindow.py
 |-- extract_data.py
 |-- resultados_test/  # Folder where results are stored in Excel format

## Test Files

1. test_influx_basic.py

    - Description: Executes a basic query to the InfluxDB database.
    - Execution command:
        python test_influx_basic.py -f "2024-11-02T15:08:45Z" -u "2024-11-02T20:50:00Z" -q "JOM20241031-104" -l "Left"

2. test_influx_basic_conMetrics.py

    - Description: Executes a query including additional metric parameters.
    - Execution command:
        python test_influx_basic_conMetrics.py -f "2024-11-02T15:08:45Z" -u "2024-11-02T20:50:00Z" -q "JOM20241031-104" -l "Left" -m "Ax,Ay,Az"

3. test_influx_aggWindow.py

    - Description: Executes queries with a defined aggregation window.
    - Execution command:
        python test_influx_aggWindow.py -f "2024-11-02T15:08:45Z" -u "2024-11-02T20:50:00Z" -q "JOM20241031-104" -l "Left" -m "Ax,Ay,Az" -w "100ms"

4. extract_data.py

    - Description: Extracts data from InfluxDB and saves it to an Excel file.
    - Execution command:
        python extract_data.py -f "2024-11-02T15:08:45Z" -u "2024-11-02T20:50:00Z" -q "JOM20241031-104" -l "Left" -o "resultados_test/salida.xlsx" -w "20ms"

## resultados_test Folder

    This folder stores the files generated after executing the scripts. Each file is saved in Excel (.xlsx) format for further analysis.

## Dependencies

    The scripts require the following Python libraries:
        - argparse, pandas, datetime, dateutil

## InfluxDBms (Custom modules for connecting to InfluxDB)

    - To install the necessary dependencies, run:

        pip install pandas python-dateutil influxdb-client

## Notes

    - It is recommended to check the database configuration in ../InfluxDBms/config_db.yaml before running the scripts.
    - For more details on execution parameters, use --help with each script.