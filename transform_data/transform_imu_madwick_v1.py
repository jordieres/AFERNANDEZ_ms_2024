import os
import argparse
import numpy as np

from class_transform_imu import*
from class_madwick import *


def parse_args():
    parser = argparse.ArgumentParser(description="IMU data processing pipeline")
    parser.add_argument("-f", "--file_paths", type=str, nargs="+", required=True, help="Paths to one or more Excel files")
    parser.add_argument("--threshold", type=float, default=0.1, help="Stationary detection threshold")
    parser.add_argument('-v', '--verbose', type=int, default=3, help='Verbosity level')
    parser.add_argument('-c', '--config', type=str, default='.config.yaml', help='Path to the configuration file')
    parser.add_argument('--output_mode', choices=["screen", "save", "both"], default="screen", help="How to handle output plots")
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
    fusion = SensorFusionFilters(alpha=0.98)
    kalman = KalmanProcessor()
    plotter = PlotProcessor()

    for file_path in args.file_paths:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        foot_label = "Left Foot" if "left" in base_name.lower() else "Right Foot" if "right" in base_name.lower() else base_name
        if args.verbose >= 2:
            print(f"\n{'*'*33}  {foot_label}  {'*'*33}\n")
            print(f"{'-'*80}")
            print(f"Processing file: {base_name}...")
            print(f"{'-'*80}\n")

        try:
            df_raw = preprocessor.load_data(file_path)
            df = preprocessor.resample_to_40hz(df_raw)
            time, sample_rate, gyr, acc, mag, df_gps = preprocessor.preprocess_data(df)
            stationary = imu_processor.detect_stationary(acc, sample_rate)
            gps_pos, gps_final = preprocessor.compute_positions(df, preprocessor.config)

            quats, acc_earth, vel, pos = estimator.estimate_orientation_and_position(
                time, gyr, acc, mag, stationary
            )

            pos_kalman = kalman.apply_filter2(pos, gps_pos, time)
            pos_ekf_2d = fusion.ekf_fusion_2d(gps_pos=gps_pos, imu_acc=acc_earth[:, :2], time=time)
            pos_complementary = fusion.complementary_filter(pos, gps_pos)
            pos_drift_corrected = fusion.linear_drift_correction(pos, gps_start=gps_pos[0], gps_end=gps_pos[-1])

            def print_metrics(name, traj):
                final_err = np.linalg.norm(traj[-1, :2] - gps_final)
                total_dist = np.sum(np.linalg.norm(np.diff(traj[:, :2]), axis=1))
                print(f"- {name:<15} -> Final error: {final_err:.2f} m | Distance: {total_dist:.2f} m")

            if args.verbose >= 2:
                dist_gps = np.sum(np.linalg.norm(np.diff(gps_pos, axis=0), axis=1))
                print("\n Quantitative Comparison: ")
                print(f"- Total GPS distance: {dist_gps:.2f} m")
                print_metrics("IMU", pos)
                print_metrics("Kalman", pos_kalman)
                print_metrics("EKF 2D", np.hstack((pos_ekf_2d, pos[:, 2:3])))
                print_metrics("Complementary", pos_complementary)
                print_metrics("Linear Drift", pos_drift_corrected)


            if args.output_dir and args.output_mode in ("save", "both"):
                return os.path.join(args.output_dir, f"{base_name}.png")
            

            # Plot all variations
            plotter.plot_results_madwick(time, acc_lp=None, threshold=args.threshold, pos=pos, vel=vel,
                                         gps_pos=gps_pos, output_dir=args.output_dir, base_name=base_name + "_pre_kalman",
                                         verbose=args.verbose, title=f"{foot_label} - IMU", traj_label="IMU")

            plotter.plot_results_madwick(time, acc_lp=None, threshold=args.threshold, pos=pos_kalman, vel=vel,
                                         gps_pos=gps_pos, output_dir=args.output_dir, base_name=base_name + "_kalman",
                                         verbose=args.verbose, title=f"{foot_label} - Kalman", traj_label="Kalman")

            plotter.plot_results_madwick(time, acc_lp=None, threshold=args.threshold, 
                                         pos=np.hstack((pos_ekf_2d, pos[:, 2:3])), vel=vel,
                                         gps_pos=gps_pos, output_dir=args.output_dir, base_name=base_name + "_ekf2d",
                                         verbose=args.verbose, title=f"{foot_label} - EKF 2D", traj_label="EKF 2D")

            plotter.plot_results_madwick(time, acc_lp=None, threshold=args.threshold, pos=pos_complementary, vel=vel,
                                         gps_pos=gps_pos, output_dir=args.output_dir, base_name=base_name + "_complementary",
                                         verbose=args.verbose, title=f"{foot_label} - Complementary", traj_label="Complementary")

            plotter.plot_results_madwick(time, acc_lp=None, threshold=args.threshold, pos=pos_drift_corrected, vel=vel,
                                         gps_pos=gps_pos, output_dir=args.output_dir, base_name=base_name + "_drift",
                                         verbose=args.verbose, title=f"{foot_label} - Linear Drift", traj_label="LinearDrift")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if args.output_dir and args.output_mode in ("save", "both"):
        print(f"\nTrajectory plots saved to: {args.output_dir}")
    if args.output_mode in ("screen", "both"):
        plt.show()


if __name__ == "__main__":
    main()
