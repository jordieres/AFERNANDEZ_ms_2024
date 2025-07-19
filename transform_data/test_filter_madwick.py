import numpy as np
import matplotlib.pyplot as plt
import argparse
from class_transform_imu import *

def parse_args():
    parser = argparse.ArgumentParser(description="Validador Madgwick")
    parser.add_argument("-f", "--file_path", required=True, help="Ruta al archivo Excel")
    parser.add_argument("-c", "--config", type=str, default=".config.yaml", help="Archivo de configuración YAML")
    return parser.parse_args()

def validate_madgwick(file_path, config_path):
    # Inicializa clases necesarias
    preprocessor = DataPreprocessor(config_path)
    estimator = PositionVelocityEstimator(sample_rate=40, sample_period=1/40)
    plotter = PlotProcessor()
    imu_proc = IMUProcessor()

    # Preprocesamiento
    df = preprocessor.load_data(file_path)
    df_interp = preprocessor.resample_to_40hz(df)
    time, sample_rate, gyr, acc, mag = preprocessor.preprocess_data(df_interp)
    stationary = imu_proc.detect_stationary(acc, sample_rate)

    # Estimación con Madgwick
    quats, acc_earth, vel, pos = estimator.estimate_orientation_and_position(time, gyr, acc, mag, stationary)

    # Proyecciones GPS
    df_gps, gps_pos, gps_final = preprocessor.compute_positions(df_interp, preprocessor.config)

    # ==================
    # VALIDACIONES VISUALES
    # ==================

    # 1. Trayectoria estimada vs GPS
    plt.figure(figsize=(10, 8))
    plt.plot(pos[:, 0], pos[:, 1], label="Madgwick Estimado")
    plt.plot(gps_pos[:, 0], gps_pos[:, 1], 'k--', label="GPS")
    plt.plot(gps_final[0], gps_final[1], 'ko', label="GPS Final")
    plt.title("Trayectoria IMU (Madgwick) vs GPS")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.axis("equal")
    plt.grid()

    # 2. Velocidad
    plt.figure(figsize=(10, 4))
    plt.plot(time, np.linalg.norm(vel, axis=1), label="Velocidad")
    plt.title("Velocidad Magnitud (Madgwick)")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Velocidad (m/s)")
    plt.grid()

    # 3. Aceleración terrestre
    plt.figure(figsize=(10, 4))
    plt.plot(time, acc_earth[:, 0], label="Ax_earth")
    plt.plot(time, acc_earth[:, 1], label="Ay_earth")
    plt.plot(time, acc_earth[:, 2], label="Az_earth")
    plt.title("Aceleración en marco terrestre")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Aceleración (m/s²)")
    plt.legend()
    plt.grid()

    # 4. Orientación (cuaterniones)
    plt.figure(figsize=(10, 4))
    plt.plot(time, quats[:, 0], label="w")
    plt.plot(time, quats[:, 1], label="x")
    plt.plot(time, quats[:, 2], label="y")
    plt.plot(time, quats[:, 3], label="z")
    plt.title("Componentes del Cuaternión (Orientación)")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid()

    # 5. Comparación de distancia recorrida
    dist_imu = np.sum(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1))
    dist_gps = np.sum(np.linalg.norm(np.diff(gps_pos, axis=0), axis=1))
    print(f"\nDistancia estimada por Madgwick: {dist_imu:.2f} m")
    print(f"Distancia medida por GPS:        {dist_gps:.2f} m")
    print(f"Diferencia relativa:             {abs(dist_imu - dist_gps) / dist_gps * 100:.2f}%\n")

    plt.show()

if __name__ == "__main__":
    args = parse_args()
    validate_madgwick(args.file_path, args.config)
