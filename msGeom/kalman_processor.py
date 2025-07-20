import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class KalmanProcessor:
    """
    Implements a 2D Kalman Filter for fusing GPS and IMU (accelerometer) data to estimate position and velocity.

    This class models the system state as [x, y, vx, vy] and supports both prediction using IMU acceleration and
    correction using GPS measurements. It's useful for sensor fusion in pedestrian navigation or robotics.
    """

    def __init__(self, dt, q, r):
        """
        Initialize the Kalman filter with system parameters.

        :param dt: Time step between measurements (in seconds).
        :type dt: float
        :param q: Process noise covariance (controls trust in motion model).
        :type q: float
        :param r: Measurement noise covariance (controls trust in GPS).
        :type r: float
        """
        self.state = np.zeros(4)  # Initial state [x, y, vx, vy]
        self.P = np.eye(4) * 1000  # Initial state covariance (high uncertainty)

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])  # State transition matrix

        self.B = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]
        ])  # Control input matrix (for acceleration input)

        self.Q = np.eye(4) * q  # Process noise covariance
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])  # Observation matrix (GPS measures x, y)

        self.R = np.eye(2) * r  # Measurement noise covariance

    def initialize(self, initial_gps_pos):
        """
        Initialize the filter state using the first GPS position (assumes zero initial velocity).

        :param initial_gps_pos: Initial position from GPS [x, y].
        :type initial_gps_pos: array-like
        """
        self.state = np.array([initial_gps_pos[0], initial_gps_pos[1], 0.0, 0.0])
        self.P = np.eye(4) * 1.0  # Lower uncertainty for known initial position

    def predict(self, acc_imu=None):
        """
        Predict the next state using the motion model and optional control input (IMU acceleration).

        :param acc_imu: Acceleration input in earth frame [ax, ay]. If None, uses only velocity model.
        :type acc_imu: array-like or None
        """
        if acc_imu is not None:
            u = np.array([acc_imu[0], acc_imu[1]])
            self.state = self.F @ self.state + self.B @ u
        else:
            self.state = self.F @ self.state

        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement_gps):
        """
        Update the filter state using a new GPS measurement.

        :param measurement_gps: GPS position measurement [x, y].
        :type measurement_gps: array-like
        :return: Updated position estimate [x, y].
        :rtype: np.ndarray
        """
        y = measurement_gps - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + (K @ y)
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

        return self.state[:2]

    def initialize_kalman(self, gps_pos, sample_rate):
        """
        Convenience method to create and initialize a Kalman filter instance.

        :param gps_pos: GPS position array, where the first entry is used for initialization.
        :type gps_pos: np.ndarray
        :param sample_rate: Sampling rate of the data (Hz).
        :type sample_rate: float
        :return: Initialized KalmanProcessor instance.
        :rtype: KalmanProcessor
        """
        dt = 1.0 / sample_rate
        kf = KalmanProcessor(dt=dt, q=0.1, r=0.5)
        kf.initialize(gps_pos[0])
        return kf

    def run_filter_with_acc_and_gps(self, acc_earth, gps_pos):
        """
        Run the Kalman filter using IMU acceleration and GPS measurements over time.

        :param acc_earth: Array of 2D acceleration values (in earth frame), shape (N, 2).
        :type acc_earth: np.ndarray
        :param gps_pos: Array of GPS positions, shape (N, 2).
        :type gps_pos: np.ndarray
        :return: Array of fused position estimates, shape (N, 2).
        :rtype: np.ndarray
        """
        fused = []
        for i in range(len(acc_earth)):
            self.predict(acc_imu=acc_earth[i, :2])
            self.update(gps_pos[i])
            fused.append(self.state[:2])
        return np.array(fused)
