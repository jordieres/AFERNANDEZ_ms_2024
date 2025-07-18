import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


class KalmanProcessor:
    """
    Class for applying Kalman filters to fuse IMU and GPS position data.
    """

    def apply_filter1(self, pos_imu, gps_pos, sample_rate):
        """
        Apply a Kalman filter using filterpy to fuse IMU and GPS data.

        :param pos_imu: IMU-estimated positions array of shape (N, 2).
        :type pos_imu: np.ndarray
        :param gps_pos: GPS positions array of shape (N, 2).
        :type gps_pos: np.ndarray
        :param sample_rate: Sampling rate in Hz.
        :type sample_rate: float
        :return: Kalman-filtered fused positions of shape (N, 2).
        :rtype: np.ndarray
        """

        n = pos_imu.shape[0]
        dt = 1.0 / sample_rate

        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1,  0],
                            [0, 0, 0,  1]])
        kf.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        kf.R = np.eye(2) * 20.0
        kf.P = np.eye(4) * 10.0
        kf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=2)

        # Estado inicial: posición y velocidad estimada por IMU
        vel_imu = np.gradient(pos_imu, dt, axis=0)
        kf.x = np.zeros((4, 1))
        kf.x[:2] = pos_imu[0, :2].reshape(2, 1)
        kf.x[2:] = vel_imu[0, :2].reshape(2, 1)

        fused = np.zeros((n, 2))
        for i in range(n):
            # Predicción basada en modelo (usa estado anterior)
            kf.predict()

            # Corrección basada en GPS si disponible
            if i < gps_pos.shape[0]:
                z = gps_pos[i].reshape(2, 1)
                kf.update(z)

            fused[i] = kf.x[:2, 0]

        return fused


    def apply_filter2(self, pos_imu, gps_pos, time):
        """
        Apply a 2D Kalman filter to adjust IMU trajectory using GPS corrections.

        :param pos_imu: IMU-estimated positions array of shape (N, 3).
        :type pos_imu: np.ndarray
        :param gps_pos: GPS projected positions array of shape (N, 2).
        :type gps_pos: np.ndarray
        :param time: Time vector array of shape (N,).
        :type time: np.ndarray
        :return: Kalman-corrected position array of shape (N, 3).
        :rtype: np.ndarray
        """
        dt = np.mean(np.diff(time))
        n = len(time)

        # Estado: [x, y, vx, vy]
        X = np.zeros((4, n))
        X[:2, 0] = gps_pos[0]  # posición inicial GPS
        X[2:, 0] = 0  # velocidad inicial 0

        # Matriz de transición del estado
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0 ],
            [0, 0, 0, 1 ]
        ])

        # Matriz de observación (solo medimos x, y del GPS)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Matrices de covarianza
        Q = np.eye(4) * 0.05   # Ruido del proceso (más alto si confías poco en la IMU)
        R = np.eye(2) * 5.0    # Ruido de la medición GPS
        P = np.eye(4) * 1.0    # Inicialización de incertidumbre

        estimated = np.zeros((n, 4))
        for k in range(1, n):
            # Predicción
            X[:, k] = F @ X[:, k - 1]
            P = F @ P @ F.T + Q

            # Solo corregimos si hay dato GPS
            if k < len(gps_pos):
                Z = gps_pos[k]  # medición GPS
                y = Z - H @ X[:, k]  # error de innovación
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)  # ganancia de Kalman
                X[:, k] += K @ y
                P = (np.eye(4) - K @ H) @ P

            estimated[k] = X[:, k]

        pos_kalman = np.copy(pos_imu)
        pos_kalman[:, 0] = estimated[:, 0]
        pos_kalman[:, 1] = estimated[:, 1]
        return pos_kalman
    


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



import numpy as np

class KalmanFilter2D:
    def __init__(self, dt, q, r):
        # dt: tiempo entre muestras
        # q: covarianza del ruido del proceso (modelo de movimiento del IMU)
        # r: covarianza del ruido de la medición (GPS)

        # Estado inicial (ej. [x, y, vx, vy])
        self.state = np.zeros(4)
        # Covarianza del estado
        self.P = np.eye(4) * 1000 # Gran incertidumbre inicial

        # Matriz de Transición de Estado (F) - Modelo de movimiento (ej. Posición = Posición anterior + Velocidad*dt)
        # Asumiendo un modelo de velocidad constante o aceleración constante
        self.F = np.array([
            [1, 0, dt, 0],  # x = x + vx*dt
            [0, 1, 0, dt],  # y = y + vy*dt
            [0, 0, 1, 0],   # vx = vx
            [0, 0, 0, 1]    # vy = vy
        ])

        # Matriz de Control (B) - Si aplicas entradas de control (aceleraciones del IMU)
        # Si usas aceleraciones del IMU para predecir, B sería:
        self.B = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]
        ])
        # Y necesitarías pasar 'acc' a la predicción. 

        # Matriz de Ruido del Proceso (Q) - Incertidumbre en nuestro modelo de movimiento
        self.Q = np.eye(4) * q

        # Matriz de Observación (H) - Cómo las mediciones se relacionan con el estado (GPS mide [x, y])
        self.H = np.array([
            [1, 0, 0, 0],  # Medición de x
            [0, 1, 0, 0]   # Medición de y
        ])

        # Matriz de Ruido de la Medición (R) - Incertidumbre del sensor GPS
        self.R = np.eye(2) * r

    def initialize(self, initial_gps_pos):
        # Inicializa el estado con la primera posición del GPS. Velocidad inicial cero.
        self.state = np.array([initial_gps_pos[0], initial_gps_pos[1], 0.0, 0.0])
        # Resetea la incertidumbre para la inicialización
        self.P = np.eye(4) * 1.0 # Menor incertidumbre si estamos seguros del punto de partida

    def predict(self, acc_imu=None):
        # Predicción del estado: x_k = F * x_{k-1} (+ B * u)
        # Si usaras las aceleraciones del IMU directamente en el modelo de predicción:
        if acc_imu is not None:
            u = np.array([acc_imu[0], acc_imu[1]]) # Asumiendo acc_earth ya en 2D
            self.state = self.F @ self.state + self.B @ u  

        else:
          self.state = self.F @ self.state

        # Predicción de la covarianza: P_k = F * P_{k-1} * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement_gps):
        # Innovación/Error de Medición: y = z - H * x_k
        y = measurement_gps - (self.H @ self.state)

        # Covarianza de Innovación: S = H * P_k * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Ganancia de Kalman: K = P_k * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Actualización del estado: x_k = x_k + K * y
        self.state = self.state + (K @ y)

        # Actualización de la covarianza: P_k = (I - K * H) * P_k
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

        return self.state[:2] # Retorna la posición (x, y) estimada
