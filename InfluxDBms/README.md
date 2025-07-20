# README - InfluxDBms

## Description


This folder contains a Python package designed to simplify interaction with an **InfluxDB** time-series database. It includes a core module (`influxdb_tools.py`) that provides a class for managing connections, querying sensor data, and retrieving structured results in `pandas` DataFrames.

The main goal of this package is to streamline access to and processing of telemetry or biometric data stored in InfluxDB — particularly for analytical workflows.

This functionality is directly integrated into the script `extract_data.py`, located in the `test_influxdb/` directory. By running that script, an instance of the `cInfluxDB` class from `influxdb_tools` is created and executed to query the database and export the results as an Excel file.

This setup allows for quick extraction of data from InfluxDB using predefined filters (e.g., time range, patient ID, foot side), making it easy to retrieve and analyze relevant subsets of your dataset.


## Setup – Install
1. Ensure you have the following prerequisites installed:
   - **Python 3.8+**
   - The following Python libraries:

      * `pandas`
      * `influxdb-client`
      * `pyyaml`
      * `urllib3`
      * `matplotlib`

- To install the required Python libraries, you have **two options**:

  - **Option 1: Use `requirements.txt`**:

    ```bash
    pip install -r requirements.txt
    ```

  - **Option 2: Manually**:

    ```bash
    pip install pandas influxdb-client pyyaml urllib3 matplotlib
    ```


2. Configure your environment:

  Ensure the file `config_db.yaml` contains the correct credentials and parameters for your InfluxDB instance. See **Configuration** section below for details.


## Configuration

Create or modify the `config_db.yaml` file with the following structure:

```yaml
influxdb:
  bucket: "your_bucket"
  org: "your_organization"
  token: "your_api_token"
  url: "http://your-influxdb-url:8086"
```


## Usage

This folder includes the following file of interest:

- **`influxdb_tools.py`**:  
  Main module that handles connection and advanced queries to InfluxDB. It contains:

  - `InfluxHelper`:
    * Parses string inputs to `datetime` objects.
    * Generates default ISO 8601 datetime ranges.

  - `cInfluxDB`: Core class that wraps the InfluxDB client, with methods to:
    * `query_data()` → Queries raw time series data and pivots metrics into columns.
    * `query_with_aggregate_window()` → Queries aggregated data using `aggregateWindow` for resampling.
    * `debug_fields()` → Prints available field names in the bucket.
    * `show_raw_sample()` → Prints the first 5 records from a sample query.
    * `close()` → Closes the InfluxDB connection.

## Example Workflow

```python
from influxdb_tools import cInfluxDB
from datetime import datetime

client = cInfluxDB("config_db.yaml")
df = client.query_data(datetime(2024, 1, 1), datetime(2024, 1, 2), qtok="ID123", pie="Right")
print(df.head())
```
