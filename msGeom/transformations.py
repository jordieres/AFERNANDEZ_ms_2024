import warnings
import numpy as np
from numpy.linalg import norm
from msGeom.transformations import Quaternion

class Quaternion:
    """
    A simple class implementing basic quaternion arithmetic.
    """
    def __init__(self, w_or_q, x=None, y=None, z=None):
        """
        Initializes a Quaternion object
        :param w_or_q: A scalar representing the real part of the quaternion, another Quaternion object or a
                    four-element array containing the quaternion values
        :param x: The first imaginary part if w_or_q is a scalar
        :param y: The second imaginary part if w_or_q is a scalar
        :param z: The third imaginary part if w_or_q is a scalar
        """
        self._q = np.array([1, 0, 0, 0])

        if x is not None and y is not None and z is not None:
            w = w_or_q
            q = np.array([w, x, y, z])
        elif isinstance(w_or_q, Quaternion):
            q = np.array(w_or_q.q)
        else:
            q = np.array(w_or_q)
            if len(q) != 4:
                raise ValueError("Expecting a 4-element array or w x y z as parameters")

        self.q = q

    # Quaternion specific interfaces

class MadgwickAHRS:
    samplePeriod = 1/256
    quaternion = Quaternion(1, 0, 0, 0)
    beta = 1
    zeta = 0

    def __init__(self, sampleperiod=None, quaternion=None, beta=None, zeta=None):
        """
        Initialize the class with the given parameters.
        :param sampleperiod: The sample period
        :param quaternion: Initial quaternion
        :param beta: Algorithm gain beta
        :param beta: Algorithm gain zeta
        :return:
        """
        if sampleperiod is not None:
            self.samplePeriod = sampleperiod
        if quaternion is not None:
            self.quaternion = quaternion
        if beta is not None:
            self.beta = beta
        if zeta is not None:
            self.zeta = zeta

    def update(self, gyroscope, accelerometer, magnetometer):
        """
        Perform one update step with data from a AHRS sensor array
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        :param magnetometer: A three-element array containing the magnetometer data. Can be any unit since a normalized value is used.
        :return:
        """
        q = self.quaternion

        gyroscope = np.array(gyroscope, dtype=float).flatten()
        accelerometer = np.array(accelerometer, dtype=float).flatten()
        magnetometer = np.array(magnetometer, dtype=float).flatten()

        # Normalise accelerometer measurement
        if norm(accelerometer) == 0:
            warnings.warn("accelerometer is zero")
            return
        accelerometer /= norm(accelerometer)

        # Normalise magnetometer measurement
        if norm(magnetometer) == 0:
            warnings.warn("magnetometer is zero")
            return
        magnetometer /= norm(magnetometer)

        h = q * (Quaternion(0, magnetometer[0], magnetometer[1], magnetometer[2]) * q.conj())
        b = np.array([0, norm(h[1:3]), 0, h[3]])

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2],
            2*b[1]*(0.5 - q[2]**2 - q[3]**2) + 2*b[3]*(q[1]*q[3] - q[0]*q[2]) - magnetometer[0],
            2*b[1]*(q[1]*q[2] - q[0]*q[3]) + 2*b[3]*(q[0]*q[1] + q[2]*q[3]) - magnetometer[1],
            2*b[1]*(q[0]*q[2] + q[1]*q[3]) + 2*b[3]*(0.5 - q[1]**2 - q[2]**2) - magnetometer[2]
        ])
        j = np.array([
            [-2*q[2],                  2*q[3],                  -2*q[0],                  2*q[1]],
            [2*q[1],                   2*q[0],                  2*q[3],                   2*q[2]],
            [0,                        -4*q[1],                 -4*q[2],                  0],
            [-2*b[3]*q[2],             2*b[3]*q[3],             -4*b[1]*q[2]-2*b[3]*q[0], -4*b[1]*q[3]+2*b[3]*q[1]],
            [-2*b[1]*q[3]+2*b[3]*q[1], 2*b[1]*q[2]+2*b[3]*q[0], 2*b[1]*q[1]+2*b[3]*q[3],  -2*b[1]*q[0]+2*b[3]*q[2]],
            [2*b[1]*q[2],              2*b[1]*q[3]-4*b[3]*q[1], 2*b[1]*q[0]-4*b[3]*q[2],  2*b[1]*q[1]]
        ])
        step = j.T.dot(f)
        step /= norm(step)  # normalise step magnitude

        # Gyroscope compensation drift
        gyroscopeQuat = Quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])
        stepQuat = Quaternion(step.T[0], step.T[1], step.T[2], step.T[3])

        gyroscopeQuat = gyroscopeQuat + (q.conj() * stepQuat) * 2 * self.samplePeriod * self.zeta * -1

        # Compute rate of change of quaternion
        qdot = (q * gyroscopeQuat) * 0.5 - self.beta * step.T

        # Integrate to yield quaternion
        q += qdot * self.samplePeriod
        self.quaternion = Quaternion(q / norm(q))  # normalise quaternion

    def update_imu(self, gyroscope, accelerometer):
        """
        Perform one update step with data from a IMU sensor array
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        """
        q = self.quaternion

        gyroscope = np.array(gyroscope, dtype=float).flatten()
        accelerometer = np.array(accelerometer, dtype=float).flatten()

        # Normalise accelerometer measurement
        if norm(accelerometer) == 0:
            warnings.warn("accelerometer is zero")
            return
        accelerometer /= norm(accelerometer)

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2]
        ])
        j = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
            [0, -4*q[1], -4*q[2], 0]
        ])
        step = j.T.dot(f)
        step /= norm(step)  # normalise step magnitude

        # Compute rate of change of quaternion
        qdot = (q * Quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])) * 0.5 - self.beta * step.T

        # Integrate to yield quaternion
        q += qdot * self.samplePeriod
        self.quaternion = Quaternion(q / norm(q))  # normalise quaternion


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
 

         