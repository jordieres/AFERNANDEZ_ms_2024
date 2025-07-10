
import numpy as np
import pandas as pd
import argparse
import os
import yaml
from scipy import signal
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ahrs.filters import Madgwick
from ahrs.common.orientation import q_conj, q_rot

from class_transform_imu import *



def parse_args_M():
    """
    Parse command-line arguments for the IMU processing pipeline.

    :return: Parsed command-line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="IMU data processing pipeline")
    parser.add_argument("-f", "--file_paths", type=str, nargs="+", required=True, help="Paths to one or more Excel files")
    parser.add_argument("--threshold", type=float, default=0.1, help="Stationary detection threshold")
    parser.add_argument('-v', '--verbose', type=int, default=3, help='Verbosity level')
    parser.add_argument('-c', '--config', type=str, default='.config.yaml', help='Path to the configuration file')
    parser.add_argument('--output_mode', choices=["screen", "save", "both"], default="screen", help="How to handle output plots: 'screen', 'save', or 'both'")
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='Directory to save output plots')
    return parser.parse_args()



def estimate_orientation_and_position(time, gyr, acc, mag, sample_period, sample_rate, stationary):
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
    base_gain = 0.041  # Ganancia base de Madgwick
    low_gain = 0.001   # Ganancia reducida para ZUPH
    madgwick = Madgwick(frequency=sample_rate, gain=base_gain)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    quats = np.zeros((len(time), 4))
    quats[0] = q

    # Detecta zonas sin rotación (giro bajo)
    gyro_norm = np.linalg.norm(gyr, axis=1)
    no_rotation = gyro_norm < 0.1  # umbral en rad/s
    no_motion = stationary & no_rotation  # ZUPH: sin movimiento ni rotación

    for t in range(1, len(time)):
        # Aplica ZUPH: reduce el gain si estamos en zona estacionaria sin rotación
        madgwick.gain = low_gain if no_motion[t] else base_gain
        q = madgwick.updateMARG(q, gyr=gyr[t], acc=acc[t], mag=mag[t])
        quats[t] = q

    # Convertir aceleraciones al marco terrestre
    acc_earth = np.array([q_rot(q_conj(qt), a) for qt, a in zip(quats, acc)])
    acc_earth -= estimate_gravity_vector(acc, 0.95)
    acc_earth *= 9.81

    # Integración de velocidad con ZUPT
    vel = np.zeros_like(acc_earth)
    for t in range(1, len(vel)):
        vel[t] = vel[t - 1] + acc_earth[t] * sample_period
        if stationary[t]:
            vel[t] = 0  # ZUPT aplicado: velocidad puesta a cero si está estacionario

    # Corrección de deriva por interpolación entre zonas ZUPT
    vel_drift = np.zeros_like(vel)
    starts = np.where(np.diff(stationary.astype(int)) == -1)[0] + 1
    ends = np.where(np.diff(stationary.astype(int)) == 1)[0] + 1
    for s, e in zip(starts, ends):
        if e > s:
            drift_rate = vel[e - 1] / (e - s)
            vel_drift[s:e] = np.outer(np.arange(e - s), drift_rate)
    vel -= vel_drift

    # Integración para obtener posición
    pos = np.zeros_like(vel)
    for t in range(1, len(pos)):
        pos[t] = pos[t - 1] + vel[t] * sample_period

    return quats, acc_earth, vel, pos



def ekf_fusion_2d(gps_pos: np.ndarray, imu_acc: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    Extended Kalman Filter (EKF) for 2D position estimation using GPS and IMU acceleration.

    :param gps_pos: Nx2 array of GPS positions [x, y].
    :param imu_acc: Nx2 array of IMU accelerations [ax, ay].
    :param time: Array of timestamps (N,).
    :return: Nx2 array of estimated positions.
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

def complementary_filter(pos_imu: np.ndarray, gps_pos: np.ndarray, alpha: float = 0.98) -> np.ndarray:
    """
    Fuse IMU and GPS positions using a complementary filter.

    :param pos_imu: IMU-estimated positions, shape (N, 3).
    :param gps_pos: GPS positions, shape (N, 2).
    :param alpha: Weighting factor for IMU data (0 < alpha < 1).
    :return: Fused position estimate, shape (N, 3).
    """
    fused = np.zeros_like(pos_imu)
    fused[:, :2] = alpha * pos_imu[:, :2] + (1 - alpha) * gps_pos
    fused[:, 2] = pos_imu[:, 2]  # Retain IMU Z-axis
    return fused


def linear_drift_correction(pos_imu: np.ndarray, gps_start: np.ndarray, gps_end: np.ndarray) -> np.ndarray:
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



def plot_results(time, acc_lp, threshold, pos, vel, gps_pos=None, output_dir=None, title="Trajectory Comparison", base_name="trajectory", verbose=3, traj_label="IMU" ):
    """
    Plot diagnostic and trajectory figures for IMU data, and optionally save them.

    This function generates several plots based on filtered acceleration, position,
    velocity, 2D and 3D trajectory. If GPS data is provided, it includes a comparative plot. 
    Depending on the verbosity level and output directory, plots can also be saved.

    :param time: Array of timestamps in seconds.
    :type time: np.ndarray
    :param acc_lp: Low-pass filtered acceleration magnitude.
    :type acc_lp: np.ndarray
    :param threshold: Threshold value used to detect stationary periods.
    :type threshold: float
    :param pos: Estimated position array of shape (N, 3).
    :type pos: np.ndarray
    :param vel: Estimated velocity array of shape (N, 3).
    :type vel: np.ndarray
    :param gps_pos: Optional GPS position array of shape (N, 2). Defaults to None.
    :type gps_pos: np.ndarray or None
    :param output_dir: Directory to save plots. If None, figures are displayed but not saved.
    :type output_dir: str or None
    :param title: Title used in trajectory comparison plots.
    :type title: str
    :param base_name: Base filename for saving output plots (without extension).
    :type base_name: str
    :param verbose: Verbosity level:
                    - 1: No plots
                    - 2: Show/save trajectory plots
                    - 3: Show/save all plots (acceleration, velocity, etc.)
    :type verbose: int
    """
    def save_figure(title):
        if output_dir:
            filename = f"{base_name}_{title.lower().replace(' ', '_')}.png"
            path = os.path.join(output_dir, filename)
            plt.savefig(path, bbox_inches='tight')

    # Filtered acceleration
    if verbose == 3:
        plt.figure(figsize=(10, 4))
        plt.plot(time, acc_lp, label='acc_lp')
        plt.axhline(threshold, color='r', linestyle='--')
        plt.title("Filtered Acceleration")
        plt.xlabel("Time (s)")
        plt.grid()
        plt.legend()
        save_figure("Filtered Acceleration")

        # Position
        plt.figure(figsize=(15, 5))
        plt.plot(time, pos[:, 0], 'r', label='x')
        plt.plot(time, pos[:, 1], 'g', label='y')
        plt.plot(time, pos[:, 2], 'b', label='z')
        plt.title("Position")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.legend()
        plt.grid()
        save_figure("Position")

        # Velocity
        plt.figure(figsize=(15, 5))
        plt.plot(time, vel[:, 0], 'r', label='x')
        plt.plot(time, vel[:, 1], 'g', label='y')
        plt.plot(time, vel[:, 2], 'b', label='z')
        plt.title("Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.legend()
        plt.grid()
        save_figure("Velocity")

        # XY Trajectory
        plt.figure()
        plt.plot(pos[:, 0], pos[:, 1])
        plt.axis('equal')
        plt.title("XY Trajectory")
        plt.grid()
        save_figure("XY Trajectory")

        # 3D Trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])
        ax.set_title("3D Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        if output_dir:
            filename = f"{base_name}_3d_trajectory.png"
            path = os.path.join(output_dir, filename)
            plt.savefig(path, bbox_inches='tight')

    # IMU vs GPS comparison (always in verbose 2 or 3)
    if gps_pos is not None and verbose >= 2:
        plt.figure(figsize=(10, 8))
        plt.plot(pos[:, 0], pos[:, 1], label=f'{traj_label} Trajectory')
        plt.plot(gps_pos[:, 0], gps_pos[:, 1], 'k--', label='GPS Reference')
        plt.plot(gps_pos[-1, 0], gps_pos[-1, 1], 'ko', label='Final GPS')
        plt.title(title)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.grid()
        plt.legend()
        save_figure(f"Trajectory Comparison ({traj_label} vs GPS)")


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