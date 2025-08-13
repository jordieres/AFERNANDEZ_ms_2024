Motivation
==========

Multiple Sclerosis is a chronic neurological disorder that can cause significant gait impairments, including reduced
balance, altered cadence, and shorter stride length. Traditional gait analysis methods —often performed in controlled
laboratory settings— are limited by their high cost, sporadic measurements, and lack of representation of a patient’s
everyday mobility.

This project focuses on seamlessly integrating data collected at the edge. High-frequency sensor data from
Sensoria Health© instrumented socks, which embed IMUs, pressure sensors, and GPS modules, is transmitted via
Bluetooth Low Energy (BLE) through custom Android or iOS applications and uploaded into an InfluxDB time-series database.

This wearable technology enables continuous, non-invasive gait monitoring in the patient’s daily environment.
The data pipeline developed in collaboration between the Polytechnic University of Madrid and the Public Hospital of
Getafe integrates high-frequency inertial and positional data to:

* Capture raw acceleration, angular velocity, magnetic field, pressure, and GPS coordinates.
* Synchronize and preprocess multi-sensor data for accurate alignment.
* Estimate foot orientation using sensor fusion algorithms (e.g., Madgwick filter).
* Compute 3D foot trajectory and correct drift using techniques such as ZUPT and ZUPH.
* Detect and segment strides, calculating spatiotemporal gait metrics.

The system enables early detection of gait alterations, supports clinical decision-making, and facilitates
personalized rehabilitation plans.
