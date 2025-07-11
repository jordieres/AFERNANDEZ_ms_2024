"""
Madwick con Magnetometro aplicando ZUPT,ZUPH y lo de la rotación.
Además se aplica el kalman simple unicamente pero en este caso se divide el df en 2 en el primero se deriva con kalman y en el segundo se estima a partir de la primera deriva a ver cuanto de fiable es el filtro. 

"""

import numpy as np
import os
import argparse
from matplotlib import pyplot as plt

from class_transform_imu import *
from class_madwick import *



def parse_args():
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



def main():
    """
    Main function to process IMU Excel files and plot estimated trajectories vs GPS.
    """
    args = parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    preprocessor = DataPreprocessor(args.config)
    imu_processor = IMUProcessor()
    estimator = PositionVelocityEstimator(sample_rate=40, sample_period=1/40)
    plotter = PlotProcessor()

    kf = None  # KalmanFilter2D instance

    for file_path in args.file_paths:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        foot_label = "Left Foot" if "left" in base_name.lower() else "Right Foot" if "right" in base_name.lower() else base_name

        if args.verbose >= 2:
            print(f"\n{'*'*33}  {foot_label}  {'*'*33}\n")
            print(f"{'-'*80}")
            print(f"Processing file: {base_name}...")
            print(f"{'-'*80}\n")

        try:
            df = preprocessor.resample_to_40hz(df)
            time, sample_rate, gyr, acc, mag, sample_period = preprocessor.preprocess_data(df)
            stationary, acc_lp, threshold = imu_processor.detect_stationary(acc, sample_rate)
            quats, acc_earth, vel, pos = estimator.estimate_orientation_and_position(
                time, gyr, acc, mag, sample_period, sample_rate, stationary
            )

            gps_pos, gps_final = preprocessor.compute_positions(df, preprocessor.config)
            

            # División para validación con Kalman
            n = len(gps_pos)
            mid = n // 2
            gps_pos_1, gps_pos_2 = gps_pos[:mid], gps_pos[mid:]
            pos_imu_1, pos_imu_2 = pos[:mid], pos[mid:]

            dt = np.mean(np.diff(time))
            if kf is None:
                kf = KalmanFilter2D(dt=dt, q=0.05, r=5.0)

            # Primera mitad con observación
            kf.initialize(gps_pos_1[0])
            traj_filtrada_1 = kf.filter_sequence(gps_pos_1)
            pos_kalman_1 = pos_imu_1.copy()
            pos_kalman_1[:, :2] = traj_filtrada_1[:, :2]

            # Segunda mitad sin observación
            kf.initialize(gps_pos_2[0])
            kf.reset_covariance(p0=1.0)
            traj_filtrada_2 = kf.filter_sequence(gps_pos_2)
            pos_kalman_2 = pos_imu_2.copy()
            pos_kalman_2[:, :2] = traj_filtrada_2[:, :2]

            # Unir trayectorias
            pos_kalman = np.vstack((pos_kalman_1, pos_kalman_2))

            dist_gps = np.sum(np.linalg.norm(np.diff(gps_pos, axis=0), axis=1)) if gps_pos is not None else 0

            if args.verbose >= 2:
                def print_metrics(name, traj):
                    final_err = np.linalg.norm(traj[-1, :2] - gps_final)
                    total_dist = np.sum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1))
                    print(f"- {name}   -> Final error: {final_err:.2f} m | Distance: {total_dist:.2f} m")

                print("\n Quantitative Comparison: ")
                print(f"- Total GPS distance: {dist_gps:.2f} m")
                print_metrics("IMU", pos)
                print_metrics("Kalman", pos_kalman)
                print(f"{'-'*80}")
            elif args.verbose == 3:
                print("Diagnostics:")
                print(f"- Samples: {len(df)}")
                print(f"- Frequency (Hz): {sample_rate:.2f}")
                print(f"- Stationary samples: {np.sum(stationary)}")
                print(f"- Final position: {pos[-1]}")
                print(f"- Final velocity: {vel[-1]}")

            
            if args.output_dir and args.output_mode in ("save", "both"):
                return os.path.join(args.output_dir, f"{base_name}.png")

            plotter.plot_results_madwick(
                time, acc_lp, threshold, pos, vel, gps_pos=gps_pos,
                output_dir=args.output_dir if args.output_mode in ("save", "both") else None,
                title=f"Trajectory Comparison - {foot_label} (IMU vs GPS)",
                base_name=base_name + "_pre_kalman",
                verbose=args.verbose,
                traj_label="IMU"
            )

            plotter.plot_results_madwick(
                time, acc_lp, threshold, pos_kalman, vel, gps_pos=gps_pos,
                output_dir=args.output_dir if args.output_mode in ("save", "both") else None,
                title=f"Trajectory Comparison - {foot_label} (Kalman vs GPS)",
                base_name=base_name + "_post_kalman",
                verbose=args.verbose,
                traj_label="Kalman"
            )

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if args.output_dir and args.output_mode in ("save", "both"):
        print(f"{'-'*80}")
        print(f"\nTrajectory plots saved to: {args.output_dir}")
    if args.output_mode in ("screen", "both"):
        plt.show()


if __name__ == "__main__":
    main()
