import numpy as np



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
 


# PARA EL MAIN
 
# dt = np.mean(np.diff(df_1))               # ∆t del primer dataset
# kf = KalmanFilter2D(dt, q=0.05, r=5.0)
# kf.initialize(gps_pos_1[0])                   # estado inicial = primer fix GPS
 
# # Filtra el primer conjunto de datos ===
# traj_filtrada_1 = kf.filter_sequence(gps_pos_1)
# pos_kalman_1 = pos_imu_1.copy()
# pos_kalman_1[:, :2] = traj_filtrada_1[:, :2]
 
# # Segundo recorrido  SIN  recalibrar Q, R, etc.===
# # Basta con reiniciar el estado inicial (¡sin reinstanciar la clase!)
# kf.initialize(gps_pos_2[0])
# kf.reset_covariance(p0=1.0)
 
# traj_filtrada_2 = kf.filter_sequence(gps_pos_2)
# pos_kalman_2 = pos_imu_2.copy()
# pos_kalman_2[:, :2] = traj_filtrada_2[:, :2]
 