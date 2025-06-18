# InfluxDBms

## Description
This folder contains the tools and configurations needed to interact with an InfluxDB database. It includes scripts for querying, processing, and visualizing data, as well as specific configuration files for database access.

The main goal of this folder is to simplify access to and management of data stored in InfluxDB for projects related to data analysis and processing.

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
This folder includes the following files and their purposes:

- **`cInfluxDB.py`**:  
  Main class that manages the connection and queries to InfluxDB. It provides methods to:
  - `query_data()`--> Retrieves sensor data for a specific date range and foot side.
  - `query_with_aggregate_window() `--> Retrieves data with aggregation using `aggregateWindow`, helpful for smoothing or resampling.
  - `debug_fields()` --> Prints available metric fields in your InfluxDB bucket.
  - `show_raw_sample()` --> Fetches a few raw entries for quick inspection.

- **`fecha_verbose.py`**:  
  A utility script to manage batch execution with configurable verbosity:
  * Use `-v`, `-vv`, or `-vvv` to control output detail.
  * Accepts arguments for config path and others, designed for extension.

- **`plot_functions.py`**:  
  Utilities to visualize InfluxDB time series data using matplotlib:
  * `plot_2d`: 2D time series plot of one or two metrics over time.
  * `plot_3d`: 3D scatter plot of three variables with time coloring.
  * `plot_4d`: 3D plot with time as a color gradient (4D feel).
  * `plot_dual_3d`: Side-by-side 3D plots for comparative analysis.
  Useful for exploratory data analysis and feature inspection.

- **`orden.sh`**:  
  A script file with instructions to install additional dependencies required to interact with InfluxDB.

- **`config_db.yaml`**:  
  Configuration file containing the credentials and parameters required to connect to the InfluxDB database.

- **`setup.py`**:  
  Allows packaging and potential distribution of the module as a Python package.


## Example Workflow

1. Query data from InfluxDB:

   ```python
   from cInfluxDB import cInfluxDB
   from datetime import datetime

   client = cInfluxDB("config_db.yaml")
   df = client.query_data(datetime(2024, 1, 1), datetime(2024, 1, 2), qtok="ID123", pie="Right")
   ```

2. Plot results:

   ```python
   from plot_functions import plot_2d
   plot_2d(df, "_time", "Ax", "Ay")
   ```
