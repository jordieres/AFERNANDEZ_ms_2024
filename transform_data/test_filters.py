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
    parser.add_argument('-om','--output_mode', choices=["screen", "save", "both"], default="screen", help="How to handle output plots")
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='Directory to save output plots')
    parser.add_argument('-e','--export_excel', choices=["yes", "no"], default="yes", help="Export Excel with time, lat, lng, pos, vel, distance")
    parser.add_argument('-m','--map_html', choices=["yes", "no"], default="no", help="Generate interactive map with trajectories")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    preprocessor = DataPreprocessor(args.config)
    imu_processor = IMUProcessor()
    plotter = PlotProcessor()
    dp = DetectPeaks()
    rp = ResultsProcessor()

    file_path = args.file_path
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    foot_label = "Left Foot" if "left" in base_name.lower() else "Right Foot" if "right" in base_name.lower() else base_name

    try:
        # --- Carga y preprocesado ---
        df = preprocessor.load_data(file_path)
        df_inter = preprocessor.resample_to_40hz(df)
        time, sample_rate, gyr, acc, mag = preprocessor.preprocess_data(df_inter)

        stationary = imu_processor.detect_stationary(acc, sample_rate)

        # --- Estimación con y sin magnetómetro ---
        pos_madgwick_m = imu_processor.estimate_position_generic("madgwick", True, gyr, acc, mag, time, sample_rate, stationary)
        pos_madgwick   = imu_processor.estimate_position_generic("madgwick", False, gyr, acc, mag, time, sample_rate, stationary)

        pos_mahony_m = imu_processor.estimate_position_generic("mahony", True, gyr, acc, mag, time, sample_rate, stationary)
        pos_mahony   = imu_processor.estimate_position_generic("mahony", False, gyr, acc, mag, time, sample_rate, stationary)

        # --- GPS proyectado ---
        df_gps, gps_pos, gps_final = preprocessor.compute_positions(df_inter, preprocessor.config)

        # --- Kalman solo sobre trayectorias con magnetómetro ---
        dt = np.mean(np.diff(time))
        acc_earth_mg_m = np.gradient(pos_madgwick_m, axis=0) * sample_rate
        acc_earth_mh_m = np.gradient(pos_mahony_m, axis=0) * sample_rate

        kp_m = KalmanProcessor(dt=dt, q=0.1, r=0.5)
        kp_m.initialize(gps_pos[0])
        fused_madgwick = kp_m.run_filter_with_acc_and_gps(acc_earth_mg_m, gps_pos)

        kp_h = KalmanProcessor(dt=dt, q=0.1, r=0.5)
        kp_h.initialize(gps_pos[0])
        fused_mahony = kp_h.run_filter_with_acc_and_gps(acc_earth_mh_m, gps_pos)

        # --- Preparación para análisis ---
        df_inter["modG"] = np.linalg.norm(gyr, axis=1)
        df_inter["modA"] = np.linalg.norm(acc, axis=1)
        df_inter["time"] = time

        triplets_gyr = dp.detect_triplet_peaks(df_inter, column='modG')
        dp.plot_peaks(df_inter, signal_column='modG', peak_df=triplets_gyr, signal_name='modG')

        gps_lat = df_gps['lat'] if not df_gps.empty else np.full(len(time), np.nan)
        gps_lng = df_gps['lng'] if not df_gps.empty else np.full(len(time), np.nan)
        gps_lat = gps_lat[:len(time)] if len(gps_lat) >= len(time) else np.pad(gps_lat, (0, len(time) - len(gps_lat)), constant_values=np.nan)
        gps_lng = gps_lng[:len(time)] if len(gps_lng) >= len(time) else np.pad(gps_lng, (0, len(time) - len(gps_lng)), constant_values=np.nan)

        df_steps = rp.prepare_step_dataframe(time, gps_lat, gps_lng, fused_madgwick, acc_earth_mg_m, df_inter)

        step_peaks = triplets_gyr.loc[triplets_gyr['peak_type'] == 'main', 'orig_index'].dropna().astype(int).values
        df_stride_stats, df_stride_raw = dp.compute_stride_stats_per_minute(df_steps, fused_madgwick, step_peaks)

        cleaner = StrideCleaner(min_stride=0.2, max_stride=2.5)
        df_stride_raw_clean = cleaner.clean_stride_data(df_stride_raw)
        df_stride_stats_clean = cleaner.recompute_stats_per_minute(df_stride_raw_clean)

        # --- Reporte ---
        if args.verbose >= 2:
            print(f"\n{'*'*33}  {foot_label}  {'*'*33}\n")
            print(f"{'-'*80}")
            print(f"Processing file: {base_name}...")
            print(f"{'-'*80}\n")

            total_gps_dist = np.sum(np.linalg.norm(np.diff(gps_pos, axis=0), axis=1))
            print("Quantitative Comparison:")
            print(f"- Total GPS distance: {total_gps_dist:.2f} m")
            rp.print_metrics("Madgwick (MARG)", pos_madgwick_m, gps_final)
            rp.print_metrics("Madgwick (IMU)",  pos_madgwick, gps_final)
            rp.print_metrics("Mahony (MARG)",   pos_mahony_m, gps_final)
            rp.print_metrics("Mahony (IMU)",    pos_mahony, gps_final)
            rp.print_metrics("Kalman (Madgwick MARG)", fused_madgwick, gps_final)
            rp.print_metrics("Kalman (Mahony MARG)",   fused_mahony, gps_final)

            step_count = len(triplets_gyr[triplets_gyr['peak_type'] == 'main'])
            print(f"- Steps detected (modG): {step_count}")

            # --- Gráficas ---
            if args.output_mode in ("screen", "both") or args.output_dir:
                def maybe_plot(name, pos):
                    plotter.plot_macroscopic_comparision(pos, gps_pos,
                        output_dir=args.output_dir if args.output_mode in ("save", "both") else None,
                        title=f"Trajectory - {foot_label} ({name})",
                        base_name=f"{base_name}_{name.lower().replace(' ', '_')}",
                        traj_label=name)

                maybe_plot("Madgwick (MARG)", pos_madgwick_m)
                maybe_plot("Madgwick (IMU)",  pos_madgwick)
                maybe_plot("Mahony (MARG)",   pos_mahony_m)
                maybe_plot("Mahony (IMU)",    pos_mahony)
                maybe_plot("Kalman Madgwick (MARG)", fused_madgwick)
                maybe_plot("Kalman Mahony (MARG)",   fused_mahony)

            if args.verbose >= 3:
                print("\nFirst 5 stride stats (filtered):")
                print(df_stride_stats_clean.head())
                print("\nFirst 5 individual strides (filtered):")
                print(df_stride_raw_clean.head())

        if args.export_excel == "yes":
            rp.export_to_excel(df_steps, rp.get_output_path(f"{base_name}_stride.xlsx", args))

        if args.map_html == "yes":
            map_out_path = os.path.join(args.output_dir or ".", f"{base_name}_map.html")
            resultados_mapa = {
                "Madgwick (MARG)": pos_madgwick_m[:, :2],
                "Madgwick (IMU)":  pos_madgwick[:, :2],
                "Mahony (MARG)":   pos_mahony_m[:, :2],
                "Mahony (IMU)":    pos_mahony[:, :2],
                "Kalman Madgwick": fused_madgwick[:, :2],
                "Kalman Mahony":   fused_mahony[:, :2]
            }
            plotter.generate_map_with_estimates(df_gps, resultados_mapa, map_out_path, preprocessor.config)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


    if args.output_dir and args.output_mode in ("save", "both"):
        print(f"{'-'*80}")
        print(f"\nTrajectory plots saved to: {args.output_dir}")

    if args.output_mode in ("screen", "both"):
        plt.show()



if __name__ == "__main__":
    main()