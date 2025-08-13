Introduction
============

The main purpose of this repository is to support the estimation of foot orientation and trajectory during walking, 
using data from inertial measurement units (IMUs) and pressure sensors embedded in smart socks. 
The system is designed to enable detailed gait analysis for individuals affected by Multiple Sclerosis (MS), 
providing accurate measurements of stride length, stride duration, and other spatiotemporal metrics.  
By combining IMU and GPS data, the solution aims to deliver reliable mobility assessments in real-world conditions, 
supporting both clinical evaluation and rehabilitation monitoring.

Motivation
==========

Multiple Sclerosis is a chronic neurological disorder that can cause significant gait impairments, 
including reduced balance, altered cadence, and shorter stride length.  
Traditional gait analysis methods —often performed in controlled laboratory settings— 
are limited by their high cost, sporadic measurements, and lack of representation of a patient’s 
everyday mobility.

Wearable technology, such as instrumented socks with embedded IMUs, pressure sensors, and GPS modules, 
opens new possibilities for continuous, non-invasive gait monitoring in the patient’s daily environment.  

This project, developed in collaboration between the Universidad Politécnica de Madrid and the Hospital Público de Getafe, 
integrates high-frequency inertial and positional data to build a processing pipeline that:

* Captures raw acceleration, angular velocity, magnetic field, pressure, and GPS coordinates.
* Synchronizes and preprocesses multi-sensor data for accurate alignment.
* Estimates foot orientation using sensor fusion algorithms (e.g., Madgwick filter).
* Computes 3D foot trajectory and corrects drift using techniques such as ZUPT and ZUPH.
* Detects and segments strides, calculating spatiotemporal gait metrics.

The system enables early detection of gait alterations, supports clinical decision-making, and 
facilitates personalized rehabilitation plans.

General Structure
=================

The application includes:

* A package named ``InfluxDBms`` for managing and querying time-series data.
* A package named ``msGeom`` for IMU + GPS fusion, orientation estimation, stride detection, 
  trajectory reconstruction, and gait metric calculation.
* A main execution script ``test_InfluxDB/extract_data.py`` to retrieve raw data from InfluxDB.
* A main execution script ``transform_data/stride_measurement.py`` to run the full processing pipeline, 
  from preprocessing to gait metrics output.

Detailed descriptions of the modules and algorithms can be found in the following sections.
