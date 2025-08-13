Installation
============

This section describes how to set up the environment and install all dependencies required to run the project.

Prerequisites
-------------

- Python 3.10 or higher
- pip (Python package manager)
- Virtual environment tool (`venv` or `conda`)

Clone the repository:

.. code-block:: bash

    git clone https://github.com/jordieres/AFERNANDEZ_ms_2024
    cd AFERNANDEZ_ms_2024

Create and activate a virtual environment:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate     # Linux / MacOS
    venv\Scripts\activate        # Windows

Install the required packages:

.. code-block:: bash

    pip install -r requirements.txt

Requirements
------------

The main dependencies for this project are:

.. code-block:: text

    numpy>=1.24.0
    pandas>=2.2.3
    scipy>=1.12.0
    matplotlib>=3.10.1
    PyYAML>=6.0.2
    plotly>=5.20.0
    folium>=0.15.1
    ahrs>=0.4.0
    filterpy>=1.4.5
    pyproj>=3.6.1
    geopy>=2.4.0
    tabulate

Once the dependencies are installed, you are ready to run the processing pipeline and other modules.
