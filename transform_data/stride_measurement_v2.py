import numpy as np
import os
import argparse
import pandas as pd
from matplotlib import pyplot as plt

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
        # print(" Primeras posiciones estimadas (pos):", pos[:5])
        # print(" Primeras velocidades estimadas (vel):", vel[:5])

        df_gps,gps_pos, gps_final = preprocessor.compute_positions(df_inter, preprocessor.config)
        n = len(gps_pos)
        mid = n // 2
        gps_pos_1, gps_pos_2 = gps_pos[:mid], gps_pos[mid:]
        pos_imu_1, pos_imu_2 = pos[:mid], pos[mid:]

        dt = np.mean(np.diff(time))
        if kf is None:
            kf = KalmanFilter2D(dt=dt, q=0.05, r=5.0)

        kf.initialize(gps_pos_1[0])
        traj_filtrada_1 = kf.filter_sequence(gps_pos_1)
        pos_kalman_1 = pos_imu_1.copy()
        pos_kalman_1[:, :2] = traj_filtrada_1[:, :2]

        kf.initialize(gps_pos_2[0])
        kf.reset_covariance(p0=1.0)
        traj_filtrada_2 = kf.filter_sequence(gps_pos_2)
        pos_kalman_2 = pos_imu_2.copy()
        pos_kalman_2[:, :2] = traj_filtrada_2[:, :2]

        pos_kalman = np.vstack((pos_kalman_1, pos_kalman_2))

        df_inter["modG"] = np.linalg.norm(gyr, axis=1)
        df_inter["modA"] = np.linalg.norm(acc, axis=1)
        df_inter["time"] = time

        triplets_gyr = dp.detect_triplet_peaks(df_inter, column='modG')
        dp.plot_peaks(df_inter, signal_column='modG', peak_df=triplets_gyr, signal_name='modG')

        triplets_acc = dp.detect_triplet_peaks(df_inter, column='modA')
        dp.plot_peaks(df_inter, signal_column='modA', peak_df=triplets_acc, signal_name='modA')

        def print_metrics(name, traj):
            final_err = np.linalg.norm(traj[-1, :2] - gps_final)
            total_dist = np.sum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1))
            print(f"- {name}   -> Final error: {final_err:.2f} m | Distance: {total_dist:.2f} m")

        if args.verbose >= 2:
            print("\nQuantitative Comparison:")
            print(f"- Total GPS distance: {np.sum(np.linalg.norm(np.diff(gps_pos, axis=0), axis=1)):.2f} m")
            print_metrics("IMU", pos)
            print_metrics("Kalman", pos_kalman)
            print(f"Pasos detectados (modG): {len(triplets_gyr[triplets_gyr['peak_type'] == 'main'])}")
            print(f"Pasos detectados (modA): {len(triplets_acc[triplets_acc['peak_type'] == 'main'])}")
            # total_time = time[-1]
            # print(f"\nValidación de pasos detectados en ventanas de {int(total_time)}s para {triplets_gyr.columns.tolist()}")
            # print(triplets_gyr.head())  # o muestra por tipo "main"
            # dp.analyze_step_robustness(triplets_gyr, "modG", total_time)
            # dp.analyze_step_robustness(triplets_acc, "modA", total_time)

            print("pos antes de Kalman (primeros 3):", pos[:3])
            print("pos después de Kalman (primeros 3):", pos_kalman[:3])


        plotter.plot_results_madwick(
            time, pos, vel, gps_pos=gps_pos,
            output_dir=args.output_dir if args.output_mode in ("save", "both") else None,
            title=f"Trajectory - {foot_label} (IMU vs GPS)",
            base_name=base_name + "_pre_kalman",
            verbose=args.verbose,
            traj_label="IMU"
        )

        plotter.plot_results_madwick(
            time, pos_kalman, vel, gps_pos=gps_pos,
            output_dir=args.output_dir if args.output_mode in ("save", "both") else None,
            title=f"Trajectory - {foot_label} (Kalman vs GPS)",
            base_name=base_name + "_post_kalman",
            verbose=args.verbose,
            traj_label="Kalman"
        )

        # === Exportar Excel con zancadas ===
        gps_lat = df_gps['lat'] if not df_gps.empty else np.full(len(pos_kalman), np.nan)
        gps_lng = df_gps['lng'] if not df_gps.empty else np.full(len(pos_kalman), np.nan)

        gps_lat = gps_lat[:len(time)] if len(gps_lat) >= len(time) else np.pad(gps_lat, (0, len(time) - len(gps_lat)), constant_values=np.nan)
        gps_lng = gps_lng[:len(time)] if len(gps_lng) >= len(time) else np.pad(gps_lng, (0, len(time) - len(gps_lng)), constant_values=np.nan)
        
        pos_kalman = pos_kalman.astype(float)
        step_distance = np.zeros(len(time))
        step_distance[1:] = np.linalg.norm(pos_kalman[1:, :2] - pos_kalman[:-1, :2], axis=1)
        
        vel_magnitude = np.linalg.norm(vel, axis=1)

        df_steps = pd.DataFrame({
            'time_s': time,
            'lat': gps_lat,
            'lng': gps_lng,
            'pos_x_m': pos_kalman[:, 0],
            'pos_y_m': pos_kalman[:, 1],
            'velocity_m_s': vel_magnitude,
            'step_distance_m': step_distance
        })

        print("\n Primeras 5 filas del resultado (stride measurement):")
        print(df_steps.head())

        if args.export_excel == "yes":
            excel_path = os.path.join(args.output_dir or ".", f"{base_name}_stride.xlsx")
            df_steps.to_excel(excel_path, index=False)
            print(f"\n Excel saved: {excel_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    if args.output_dir and args.output_mode in ("save", "both"):
        print(f"{'-'*80}")
        print(f"\nTrajectory plots saved to: {args.output_dir}")
    if args.output_mode in ("screen", "both"):
        plt.show()


if __name__ == "__main__":
    main()
