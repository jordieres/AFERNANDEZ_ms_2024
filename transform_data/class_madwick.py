
import numpy as np
from ahrs.filters import Madgwick
from ahrs.common.orientation import q_conj, q_rot

from class_transform_imu import *

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

        # acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)])
        # acc_earth -= IMUProcessor.estimate_gravity_vector(acc, 0.95)
        # acc_earth *= 9.81
        gravity = IMUProcessor.estimate_gravity_vector(acc, 0.95)
        print("➡️ Vector gravedad estimado:", gravity)

        acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)])
        acc_earth -= gravity
        acc_earth *= 9.81

        print("🔍 Normas de acc_earth (primeros 10):", np.linalg.norm(acc_earth[:10], axis=1))
        print("acc_earth media total:", np.mean(np.linalg.norm(acc_earth, axis=1)))
        
        
        vel = np.zeros_like(acc_earth)
        for t in range(1, len(vel)):
            vel[t] = vel[t - 1] + acc_earth[t] * self.sample_period
            if stationary[t]:
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

        print("🔍 Resumen de flags:")
        print(f"Samples estacionarios: {np.sum(stationary)} / {len(stationary)} ({np.mean(stationary)*100:.2f}%)")
        print(f"acc_earth media (primeros 5):\n{acc_earth[:5]}")
        print(f"vel media (primeros 5):\n{vel[:5]}")
        print(f"pos media (primeros 5):\n{pos[:5]}")

        return quats, acc_earth, vel, pos



class SensorFusionFilters:
    """
    Class containing filtering and correction methods for sensor fusion,
    including an Extended Kalman Filter (EKF), complementary filter, and linear drift correction.
    """

    def __init__(self, alpha: float = 0.98):
        """
        Initialize the filter with a default weighting factor for complementary fusion.

        :param alpha: Weighting factor for the complementary filter (0 < alpha < 1).
        """
        self.alpha = alpha



    

    def ekf_fusion_2d(self,gps_pos, imu_acc, time):
        """
        Extended Kalman Filter (EKF) for 2D position estimation using GPS and IMU acceleration.

        :param gps_pos: GPS positions, shape (N, 2).
        :type gps_pos: np.ndarray
        :param imu_acc: IMU accelerations, shape (N, 2).
        :type imu_acc: np.ndarray
        :param time: Time stamps, shape (N,).
        :type time: np.ndarray
        :return: Estimated 2D positions, shape (N, 2).
        :rtype: np.ndarray
        """
        N = len(time)
        dt = np.mean(np.diff(time))

        # Initial state: [x, y, vx, vy, ax, ay]
        x = np.array([gps_pos[0, 0], gps_pos[0, 1], 0.0, 0.0, imu_acc[0, 0], imu_acc[0, 1]])
        P = np.eye(6) * 10.0

        Q = np.diag([0.05, 0.05, 0.05, 0.05, 0.1, 0.1])  # Process noise
        R = np.eye(2) * 3.0                             # Measurement noise

        estimates = np.zeros((N, 6))

        for k in range(1, N):
            u = imu_acc[k]

            # Prediction step
            x_pred = np.zeros(6)
            x_pred[0] = x[0] + x[2]*dt + 0.5*u[0]*dt**2
            x_pred[1] = x[1] + x[3]*dt + 0.5*u[1]*dt**2
            x_pred[2] = x[2] + u[0]*dt
            x_pred[3] = x[3] + u[1]*dt
            x_pred[4] = 0.9 * x[4] + 0.1 * u[0]  # Smooth update of acceleration
            x_pred[5] = 0.9 * x[5] + 0.1 * u[1]

            # Jacobian of motion model
            F = np.array([
                [1, 0, dt, 0, 0.5*dt**2, 0],
                [0, 1, 0, dt, 0, 0.5*dt**2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])

            P = F @ P @ F.T + Q

            # Correction step
            H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0]
            ])
            z = gps_pos[k]
            y = z - H @ x_pred
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x_pred + K @ y
            P = (np.eye(6) - K @ H) @ P

            # Optional: clip extreme values
            x = np.clip(x, -1e6, 1e6)

            estimates[k] = x

        return estimates[:, :2]

    def complementary_filter(self,pos_imu, gps_pos):
        """
        Fuse IMU and GPS positions using a complementary filter.

        :param pos_imu: IMU-estimated positions, shape (N, 3).
        :type pos_imu: np.ndarray
        :param gps_pos: GPS positions, shape (N, 2).
        :type gps_pos: np.ndarray
        :param alpha: Weighting factor for IMU data (0 < alpha < 1). Default is 0.98.
        :type alpha: float
        :return: Fused position estimate, shape (N, 3).
        :rtype: np.ndarray
        """
        fused = np.zeros_like(pos_imu)
        fused[:, :2] = self.alpha * pos_imu[:, :2] + (1 - self.alpha) * gps_pos
        fused[:, 2] = pos_imu[:, 2]  # Retain IMU Z-axis
        return fused


        
    def linear_drift_correction(self,pos_imu, gps_start, gps_end):
        """
        Apply linear correction to IMU position to reduce drift between known GPS start and end points.

        The correction is applied progressively across the time series to match GPS anchors.

        :param pos_imu: IMU-estimated positions (N, 3).
        :type pos_imu: np.ndarray
        :param gps_start: 2D GPS start position (2,).
        :type gps_start: np.ndarray
        :param gps_end: 2D GPS end position (2,).
        :type gps_end: np.ndarray
        :return: Drift-corrected IMU position array (N, 3).
        :rtype: np.ndarray
        """
        N = len(pos_imu)
        imu_start = pos_imu[0, :2]
        imu_end = pos_imu[-1, :2]
        drift = (gps_end - gps_start) - (imu_end - imu_start)
        correction = np.linspace(0, 1, N).reshape(-1, 1) * drift
        corrected = pos_imu.copy()
        corrected[:, :2] += correction
        return corrected



class KalmanFilter2D:
    """
    Filtro de Kalman para estimación de posición y velocidad en 2D.
    
    Implementa un filtro discreto con modelo lineal de estado:
    estado = [x, y, vx, vy]. Observa únicamente la posición (x, y).
    
    El filtro puede inicializarse una vez y aplicarse a múltiples secuencias
    de datos sin redefinir sus matrices internas.
    
    Attributes:
        dt (float): Intervalo temporal constante entre muestras.
        F (ndarray): Matriz de transición del estado.
        H (ndarray): Matriz de observación.
        Q (ndarray): Covarianza del ruido del proceso.
        R (ndarray): Covarianza del ruido de la observación.
        P (ndarray): Matriz de covarianza del error estimado.
        x (ndarray): Estado actual del filtro [x, y, vx, vy].
    """
 
    def __init__(self, dt, q=0.05, r=5.0, p0=1.0):
        """
        Inicializa el filtro con parámetros de dinámica y ruido.

        Args:
            dt (float): Intervalo de tiempo entre muestras.
            q (float, optional): Varianza del ruido de proceso. Default = 0.05.
            r (float, optional): Varianza del ruido de observación GPS. Default = 5.0.
            p0 (float, optional): Valor inicial para la matriz de covarianza P. Default = 1.0.
        """
        self.dt = dt

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.Q = np.eye(4) * q
        self.R = np.eye(2) * r
        self.P = np.eye(4) * p0
        self.x = np.zeros(4)

    def initialize(self, pos0_xy, vel0_xy=(0.0, 0.0)):
        """
        Establece el estado inicial del filtro.

        Args:
            pos0_xy (array-like): Posición inicial [x, y].
            vel0_xy (array-like, optional): Velocidad inicial [vx, vy]. Default = (0.0, 0.0).
        """
        self.x[:2] = pos0_xy
        self.x[2:] = vel0_xy

    def reset_covariance(self, p0=1.0):
        """
        Reinicia la matriz de covarianza P del filtro.

        Args:
            p0 (float): Valor escalar para la nueva matriz P = p0 * I.
        """
        self.P = np.eye(4) * p0

    def step(self, z=None):
        """
        Realiza un paso de predicción y corrección (si hay observación).

        Args:
            z (array-like or None): Observación [x, y] del GPS. Si es None, no se corrige.

        Returns:
            ndarray: Estado posterior estimado [x, y, vx, vy].
        """
        # Predicción
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Corrección si hay observación
        if z is not None:
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)

            self.x += K @ y
            self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.x.copy()

    def filter_sequence(self, gps_positions):
        """
        Aplica el filtro a una secuencia de observaciones GPS.

        Args:
            gps_positions (array-like): Lista o array (N, 2) de observaciones [x, y].
                Se pueden usar `None` o vectores con `np.nan` para pasos sin observación.

        Returns:
            ndarray: Matriz (N, 4) con los estados estimados [x, y, vx, vy] en cada paso.
        """
        filtered = []
        for z in gps_positions:
            if z is None or (isinstance(z, np.ndarray) and np.isnan(z).any()):
                z = None
            filtered.append(self.step(z))
        return np.vstack(filtered)