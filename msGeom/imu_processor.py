import numpy as np
from scipy import signal
from ahrs.filters import Madgwick, Mahony
from ahrs.common.orientation import q_conj, q_rot, axang2quat


class IMUProcessor:
    """
    Class for processing IMU data: gravity estimation, motion detection, and position estimation.
    """

    def __init__(self):
        pass

    def estimate_gravity_vector(self, acc, alpha = 0.9):
        """
        Estimate normalized gravity vector from accelerometer data using exponential moving average.

        :param acc: Acceleration data array with shape (N, 3).
        :type acc: np.ndarray
        :param alpha: Smoothing coefficient (0 < alpha < 1).
        :type alpha: float
        :return: Normalized gravity vectors.
        :rtype: np.ndarray
        :raises ValueError: If input shape is invalid or alpha is out of bounds.
        """
        if acc.ndim != 2 or acc.shape[1] != 3:
            raise ValueError("Input array must have shape (N, 3) with columns [Ax, Ay, Az].")
        if not (0 < alpha < 1):
            raise ValueError("Parameter alpha must be in the range (0, 1).")

        gravity = np.zeros_like(acc)
        gravity[0] = acc[0]

        for i in range(1, len(acc)):
            gravity[i] = alpha * gravity[i - 1] + (1 - alpha) * acc[i]

        # Normalización segura
        norm = np.linalg.norm(gravity, axis=1, keepdims=True)
        norm[norm == 0] = 1.0  # Evita división por cero
        gravity_normalized = gravity / norm

        return gravity_normalized

    def detect_stationary(self, acc, sample_rate):
        """
        Detect stationary periods from accelerometer data using filtering and thresholding.

        :param acc: Acceleration data array with shape (N, 3).
        :type acc: np.ndarray
        :param sample_rate: Sampling rate of the accelerometer signal in Hz.
        :type sample_rate: float
        :return: A tuple containing stationary: Boolean array indicating stationary (True) or moving (False) states.
        :rtype: tuple[np.ndarray]
        :raises ValueError: If input shape is invalid or sample_rate is non-positive.
        """
        acc_mag = np.linalg.norm(acc, axis=1)
        acc_mag_clipped = np.clip(acc_mag, 0, 20)

        b, a = signal.butter(1, 0.01 / (sample_rate / 2), 'highpass')
        acc_hp = signal.filtfilt(b, a, acc_mag_clipped)

        b, a = signal.butter(1, 5.0 / (sample_rate / 2), 'lowpass')
        acc_lp = signal.filtfilt(b, a, np.abs(acc_hp))

        threshold = np.percentile(acc_lp, 15)
        stationary = acc_lp < threshold

        return stationary

    # Agregación de ZUPH
    def detect_no_rotation(self, gyr, threshold = 0.05, duration_samples = 5):
        """
        Detect periods where there is negligible angular velocity on Z-axis (i.e., no yaw rotation).

        :param gyr: Gyroscope data array of shape (N, 3), in rad/s.
        :type gyr: np.ndarray
        :param threshold: Threshold for Z-axis angular velocity (in rad/s) to define no rotation.
        :type threshold: float
        :param duration_samples: Minimum number of consecutive samples below the threshold to validate no rotation.
        :type duration_samples: int
        :return: Boolean array of shape (N,), where True indicates no rotation around the Z-axis.
        :rtype: np.ndarray
        """

        gz = np.abs(gyr[:, 2])  # Only Z axis
        mask = gz < threshold
        stable = np.copy(mask)
        for i in range(len(mask)):
            if not mask[i]:
                continue
            if i + duration_samples <= len(mask) and np.all(mask[i:i + duration_samples]):
                stable[i:i + duration_samples] = True
        return stable

    def estimate_position_generic(self, method, use_mag, gyr, acc, mag, time, sample_rate, stationary):
        """
        Estimate the 2D position from IMU sensor data using either the Madgwick or Mahony filter.

        This method computes orientation quaternions using an AHRS filter, rotates the 
        acceleration to the earth frame, removes gravity, integrates to velocity and 
        position, and compensates for drift during stationary periods.

        :param method: Algorithm to use ('madgwick' or 'mahony').
        :type method: str
        :param use_mag: Whether to use magnetometer data (True for MARG, False for IMU-only).
        :type use_mag: bool
        :param gyr: Gyroscope data array of shape (N, 3), in rad/s.
        :type gyr: np.ndarray
        :param acc: Accelerometer data array of shape (N, 3), in m/s^2.
        :type acc: np.ndarray
        :param mag: Magnetometer data array of shape (N, 3), in arbitrary units.
        :type mag: np.ndarray
        :param time: Time vector of shape (N,), in seconds.
        :type time: np.ndarray
        :param sample_rate: Sampling frequency in Hz.
        :type sample_rate: float
        :param stationary: Boolean array of shape (N,) indicating stationary periods.
        :type stationary: np.ndarray
        :return: Estimated position array of shape (N, 3).
        :rtype: np.ndarray
        :raises ValueError: If the method is not recognized.
        """

        if method == "madgwick":
            base_gain = 0.005
            filter_ = Madgwick(frequency=sample_rate, gain=base_gain)
            q = axang2quat([0, 0, 1], np.deg2rad(45))
        elif method == "mahony":
            base_kp = 1.5
            filter_ = Mahony(Kp=base_kp, Ki=0.01, frequency=sample_rate)
            q = np.array([1.0, 0.0, 0.0, 0.0])
            acc_init = np.median(acc[time <= time[0] + 2], axis=0)
            for _ in range(2000):
                q = filter_.updateIMU(q, gyr=np.zeros(3), acc=acc_init)
        else:
            raise ValueError("Unknown method. Use 'madgwick' or 'mahony'.")

        quats = np.zeros((len(time), 4))
        no_rotation = self.detect_no_rotation(gyr)
        no_motion = stationary & no_rotation

        for t in range(len(time)):
            if method == "madgwick":
                filter_.gain = 0.001 if no_motion[t] else base_gain
                q_new = filter_.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t]) if use_mag else filter_.updateIMU(q, gyr=gyr[t], acc=acc[t])
            else:
                filter_.Kp = base_kp * 0.1 if no_motion[t] else (base_kp if stationary[t] else 0.0)
                q_new = filter_.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t]) if use_mag else filter_.updateIMU(q, gyr=gyr[t], acc=acc[t])
            if q_new is not None:
                q = q_new
            quats[t] = q

        acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)])
        acc_earth -= self.estimate_gravity_vector(acc, 0.95)
        acc_earth *= 9.81

        vel = np.zeros_like(acc_earth)
        for t in range(1, len(vel)):
            vel[t] = vel[t - 1] + acc_earth[t] * (1 / sample_rate)
            if stationary[t]:
                vel[t] = 0

        drift = np.zeros_like(vel)
        starts = np.where(np.diff(stationary.astype(int)) == -1)[0] + 1
        ends = np.where(np.diff(stationary.astype(int)) == 1)[0] + 1
        for s, e in zip(starts, ends):
            drift_rate = vel[e - 1] / (e - s)
            drift[s:e] = np.outer(np.arange(e - s), drift_rate)
        vel -= drift

        pos = np.zeros_like(vel)
        for t in range(1, len(pos)):
            pos[t] = pos[t - 1] + vel[t] * (1 / sample_rate)

        return pos




class PositionVelocityEstimator:
    """
    Estimates orientation, acceleration, velocity, and position from IMU sensors (gyroscope, accelerometer, magnetometer).
    Uses the Madgwick filter with adaptive gain for ZUPH and velocity corrections using ZUPT.
    """
    def __init__(self, sample_rate, sample_period):
        self.sample_rate = sample_rate
        self.sample_period = sample_period
        self.base_gain = 0.041
        self.low_gain = 0.001

    def estimate_orientation_and_position(self, time, gyr, acc, mag, stationary):
        """
        Estimate orientation, linear acceleration, velocity, and position from sensor data using a Madgwick filter
        and zero-velocity updates.

        Applies orientation estimation with adaptive gain based on motion state (ZUPH),
        and velocity correction during stationary periods (ZUPT).

        :param time: Array of time stamps.
        :type time: np.ndarray
        :param gyr: Gyroscope data array with shape (N, 3), in rad/s.
        :type gyr: np.ndarray
        :param acc: Accelerometer data array with shape (N, 3), in m/s².
        :type acc: np.ndarray
        :param mag: Magnetometer data array with shape (N, 3), in µT.
        :type mag: np.ndarray
        :param sample_period: Time between samples, in seconds.
        :type sample_period: float
        :param sample_rate: Sampling rate in Hz.
        :type sample_rate: float
        :param stationary: Boolean array indicating stationary states (True for stationary).
        :type stationary: np.ndarray
        :return: Tuple (quats, acc_earth, vel, pos) where:
                - quats: Quaternion orientation estimates.
                - acc_earth: Linear acceleration in earth frame (gravity removed).
                - vel: Estimated velocity with ZUPT correction.
                - pos: Estimated position.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        :raises ValueError: If input shapes are inconsistent.
        """
        madgwick = Madgwick(frequency=self.sample_rate, gain=self.base_gain)
        q = np.array([1.0, 0.0, 0.0, 0.0])
        quats = np.zeros((len(time), 4))
        quats[0] = q

        gyro_norm = np.linalg.norm(gyr, axis=1)
        no_rotation = gyro_norm < 0.1
        no_motion = stationary & no_rotation

        for t in range(1, len(time)):
            madgwick.gain = self.low_gain if no_motion[t] else self.base_gain
            q = madgwick.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t])
            quats[t] = q

        imu_proc = IMUProcessor()
        gravity = imu_proc.estimate_gravity_vector(acc, 0.95)

        acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)])
        acc_earth -= gravity
        acc_earth *= 9.81
        
        vel = np.zeros_like(acc_earth)
        for t in range(1, len(vel)):
            vel[t] = vel[t - 1] + acc_earth[t] * self.sample_period
            if stationary[t] and not stationary[t - 1]:
                vel[t] = 0  

        vel_drift = np.zeros_like(vel)
        starts = np.where(np.diff(stationary.astype(int)) == -1)[0] + 1
        ends = np.where(np.diff(stationary.astype(int)) == 1)[0] + 1
        for s, e in zip(starts, ends):
            if e > s:
                drift_rate = vel[e - 1] / (e - s)
                vel_drift[s:e] = np.outer(np.arange(e - s), drift_rate)
        vel -= vel_drift

        pos = np.zeros_like(vel)
        for t in range(1, len(pos)):
            pos[t] = pos[t - 1] + vel[t] * self.sample_period

        return quats, acc_earth, vel, pos

