import numpy as np
import os
import argparse
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from class_transform_imu import *

def parse_args():
    parser = argparse.ArgumentParser(description="IMU data processing pipeline")
    parser.add_argument("-f", "--file_path", type=str, required=True, help="Path to the Excel file")
    parser.add_argument('-v', '--verbose', type=int, default=3, help='Verbosity level')
    parser.add_argument('-c', '--config', type=str, default='.config.yaml', help='Path to the configuration file')
    parser.add_argument('--output_mode', choices=["screen", "save", "both"], default="screen", help="How to handle output plots")
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='Directory to save output plots')
    parser.add_argument('-e','--export_excel', choices=["yes", "no"], default="yes", help="Export Excel with time, lat, lng, pos, vel, distance")
    return parser.parse_args()

def tune_and_evaluate_kalman(time, acc_earth, gps_pos, df_gps, KalmanFilter2D):
    results = []

    gps_time = df_gps["time"].values
    interp_x = interp1d(gps_time, gps_pos[:, 0], kind="linear", fill_value="extrapolate")
    interp_y = interp1d(gps_time, gps_pos[:, 1], kind="linear", fill_value="extrapolate")
    gps_interp = np.stack((interp_x(time), interp_y(time)), axis=1)

    acc_bias_estimate = np.mean(acc_earth[-200:, :2], axis=0)
    acc_earth_corrected = acc_earth[:, :2] - acc_bias_estimate

    dt = np.mean(np.diff(time))
    q_values = [0.01, 0.05, 0.1]
    r_values = [0.5, 1.0, 5.0]

    for q in q_values:
        for r in r_values:
            kf = KalmanFilter2D(dt=dt, q=q, r=r)
            kf.initialize(gps_interp[0])
            fused = []
            for i in range(len(time)):
                kf.predict(acc_earth_corrected[i])
                kf.update(gps_interp[i])
                fused.append(kf.state[:2])

            traj = np.array(fused)
            kalman_dist = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
            gps_dist = np.sum(np.linalg.norm(np.diff(gps_interp, axis=0), axis=1))
            final_error = np.linalg.norm(traj[-1] - gps_interp[-1])
            results.append((q, r, kalman_dist, gps_dist, final_error))

            kalman_error = np.linalg.norm(traj - gps_interp, axis=1)
            plt.figure(figsize=(8, 3))
            plt.plot(time, kalman_error)
            plt.title(f"Kalman Error over Time (Q={q}, R={r})")
            plt.xlabel("Time [s]")
            plt.ylabel("Error [m]")
            plt.grid(True)
            plt.tight_layout()
            

    df_results = pd.DataFrame(results, columns=["Q", "R", "Kalman_Distance", "GPS_Distance", "Final_Error"])


    print("\nResultados del ajuste de Kalman:")
    print(df_results)

    return df_results

def main():
    args = parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    preprocessor = DataPreprocessor(args.config)
    imu_processor = IMUProcessor()
    estimator = PositionVelocityEstimator(sample_rate=40, sample_period=1/40)
    plotter = PlotProcessor()
    dp = DetectPeaks()
    kf = None

    file_path = args.file_path
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    foot_label = "Left Foot" if "left" in base_name.lower() else "Right Foot" if "right" in base_name.lower() else base_name

    if args.verbose >= 2:
        print(f"\n{'*'*33}  {foot_label}  {'*'*33}\n")
        print(f"{'-'*80}")
        print(f"Processing file: {base_name}...")
        print(f"{'-'*80}\n")

    try:
        df = preprocessor.load_data(file_path)
        df_inter = preprocessor.resample_to_40hz(df)
        time, sample_rate, gyr, acc, mag = preprocessor.preprocess_data(df_inter)

        stationary = imu_processor.detect_stationary(acc, sample_rate)
        quats, acc_earth, vel, pos = estimator.estimate_orientation_and_position(time, gyr, acc, mag, stationary)
        df_gps, gps_pos, gps_final = preprocessor.compute_positions(df_inter, preprocessor.config)

        tune_and_evaluate_kalman(time, acc_earth, gps_pos, df_gps, KalmanFilter2D)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    if args.output_dir and args.output_mode in ("save", "both"):
        print(f"{'-'*80}")
        print(f"\nTrajectory plots saved to: {args.output_dir}")
    if args.output_mode in ("screen", "both"):
        plt.show()

if __name__ == "__main__":
    main()