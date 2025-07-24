
import numpy as np
import os
import argparse
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate

from msGeom.data_preprocessor import DataPreprocessor
from msGeom.imu_processor import IMUProcessor
from msGeom.kalman_processor import KalmanProcessor
from msGeom.peak_detector import DetectPeaks
from msGeom.stride_processor import StrideProcessor
from msGeom.result_processor import ResultsProcessor
from msGeom.plot_processor import PlotProcessor


def parse_args():
    """
    Parse command-line arguments for the IMU data processing pipeline.

    :return: Parsed command-line arguments.
    :rtype: argparse.Namespace

    **Command-line Arguments:**

    - ``-f``, ``--file_path`` (str, required): Path to the input Excel file.
    - ``-v``, ``--verbose`` (int, optional): Verbosity level (default: 3).
    - ``-c``, ``--config`` (str, optional): Path to the YAML configuration file (default: .config.yaml).
    - ``-om``, ``--output_mode`` (str, optional): How to handle output plots. Options: "screen", "save", "both" (default: screen).
    - ``-o``, ``--output_dir`` (str, optional): Directory where output plots will be saved.
    - ``-e``, ``--export_excel`` (str, optional): Whether to export results to Excel. Options: "yes", "no" (default: yes).
    - ``-m``, ``--map_html`` (str, optional): Whether to generate an interactive HTML map. Options: "yes", "no" (default: no).
    """

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
    """
    Main execution function for the IMU + GPS gait analysis pipeline.

    This function performs the following steps:
    - Parses command-line arguments.
    - Loads and resamples sensor data from Excel.
    - Detects stationary periods.
    - Estimates orientation and position using Madgwick filter.
    - Fuses GPS and IMU data using a Kalman Filter.
    - Computes stride-level metrics and filters invalid steps.
    - Evaluates trajectory quality and segment-level metrics.
    - Optionally exports data to Excel and generates plots/maps.

    Dependencies include several processors from the ``msGeom`` package, including:
    - DataPreprocessor
    - IMUProcessor
    - KalmanProcessor
    - DetectPeaks
    - ResultsProcessor
    - StrideProcessor
    - PlotProcessor
    """
    args = parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)


    sample_rate = 40
    sample_period = 1 / sample_rate
    preprocessor = DataPreprocessor(args.config)
    imu_processor = IMUProcessor(sample_rate, sample_period)
    plotter = PlotProcessor()
    dp = DetectPeaks()
    rp = ResultsProcessor()
    sp = StrideProcessor(min_stride=0.2, max_stride=2.5, window_sec=1.5) # VENTANA DE (+-)3 -->DURACION DE LA MARCHA


    file_path = args.file_path
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    foot_label = "Left Foot" if "left" in base_name.lower() else "Right Foot" if "right" in base_name.lower() else base_name

    try:
        # Load and preprocess
        df = preprocessor.load_data(file_path)
        df_inter = preprocessor.resample_to_40hz(df)
        time, sample_rate, gyr, acc, mag = preprocessor.preprocess_data(df_inter)

        stationary = imu_processor.detect_stationary(acc, sample_rate)
        quats, acc_earth, vel, pos = imu_processor.estimate_position_madwick(time, gyr, acc, mag, stationary)

        df_gps, gps_pos, gps_final = preprocessor.compute_positions(df_inter, preprocessor.config)

        # Kalman fusion
        kp = KalmanProcessor(dt=sample_period, q=0.1, r=0.5)
        kp.initialize(gps_pos[0])
        fused_trajectory = kp.run_filter_with_acc_and_gps(acc_earth, gps_pos)

        pos_kalman = pos.copy()
        pos_kalman[:, :2] = fused_trajectory

        df_inter["modG"] = np.linalg.norm(gyr, axis=1)
        df_inter["time"] = time

        df_imu = df_inter[["time", "Ax", "Ay", "Az", "Gx", "Gy", "Gz"]].copy()
        triplets_gyr = dp.detect_triplet_peaks(df_inter, column='modG')

        gps_lat = df_gps['lat'] if not df_gps.empty else np.full(len(pos_kalman), np.nan)
        gps_lng = df_gps['lng'] if not df_gps.empty else np.full(len(pos_kalman), np.nan)

        gps_lat = gps_lat[:len(time)] if len(gps_lat) >= len(time) else np.pad(gps_lat, (0, len(time) - len(gps_lat)), constant_values=np.nan)
        gps_lng = gps_lng[:len(time)] if len(gps_lng) >= len(time) else np.pad(gps_lng, (0, len(time) - len(gps_lng)), constant_values=np.nan)

        df_steps = rp.prepare_step_dataframe(time, gps_lat, gps_lng, pos_kalman, vel, df_inter)
        df_steps["minute"] = (df_steps["time"] // 60).astype(int)

        # Step filtering
        step_peaks = (triplets_gyr.loc[triplets_gyr['peak_type'] == 'main', 'orig_index']
                      .dropna().astype(int).values)
        step_peaks = np.sort(np.unique(step_peaks))

        df_stride_stats, df_stride_raw = dp.compute_stride_stats_per_minute(df_steps, pos_kalman, step_peaks)
        df_stride_raw_clean = sp.clean_stride_data(df_stride_raw)
        df_stride_raw_clean["minute"] = (df_stride_raw_clean["time"] // 60).astype(int)
        df_stride_stats_clean = sp.recompute_stats_per_minute(df_stride_raw_clean)

        # Quality checks
        df_stats_checked = sp.check_distance_similarity(df_stride_stats_clean, gps_distance=430.04)
        df_stats_checked = sp.check_stride_length_range(df_stats_checked)
        df_steps_checked = sp.check_trajectory_smoothness(df_steps)
        alignment_mask, percent_ok = sp.check_spatial_alignment(pos_kalman[:, :2], gps_pos)
        df_eval = sp.evaluate_quality_segments(df_stats_checked, df_steps, gps_pos, imu_pos=pos_kalman[:, :2], gps_distance=430.04)

        df_stride_valid = df_stride_raw_clean[["time"]].copy()
        valid_times_set = set(df_stride_valid["time"])
        df_stride_invalid = df_stride_raw[~df_stride_raw["time"].isin(valid_times_set)][["time"]].copy()

        kalman_gps_error, jump_indices = sp.compute_kalman_gps_error_and_jumps(time, pos_kalman, gps_pos)
        jump_times = time[jump_indices]

        # Verbose outputs and plots
        if args.verbose >= 2:
            print(f"\n{'*'*58}  {foot_label}  {'*'*59}\n")
            print(f"{'-'*130}")
            print(f"Processing file: {base_name}...")
            print(f"{'-'*130}\n")

            print("Quantitative Comparison:\n")
            total_gps_dist = np.sum(np.linalg.norm(np.diff(gps_pos, axis=0), axis=1))
            print(f"- Total GPS distance: {total_gps_dist:.2f} m")
            rp.print_metrics("IMU", pos, gps_final)
            rp.print_metrics("Kalman", pos_kalman, gps_final)

            step_count = len(triplets_gyr[triplets_gyr['peak_type'] == 'main'])
            print(f"- Steps detected (modG): {step_count}")

            print("\nGPS jumps detected:\n")
            for idx in jump_indices:
                print(f"- Time: {time[idx]:.2f} s | Kalman-GPS Error: {kalman_gps_error[idx]:.2f} m")


            plotter.plot_macroscopic_comparision(
                pos, gps_pos,
                output_dir=args.output_dir if args.output_mode in ("save", "both") else None,
                title=f"Trajectory - {foot_label} (IMU vs GPS)",
                base_name=base_name + "_pre_kalman",
                traj_label="IMU"
            )

            plotter.plot_macroscopic_comparision(
                pos_kalman, gps_pos,
                output_dir=args.output_dir if args.output_mode in ("save", "both") else None,
                title=f"Trajectory - {foot_label} (Kalman vs GPS)",
                base_name=base_name + "_post_kalman",
                traj_label="Kalman"
            )

           
            dp.plot_peaks(df_inter, signal_column='modG', peak_df=triplets_gyr, signal_name='modG')
            
            sp.plot_gps_jumps(time, kalman_gps_error, jump_indices)

            sp.plot_jumps_and_strides(
                time=time,
                kalman_gps_error=kalman_gps_error,
                jump_times=jump_times,
                stride_times={30: 39.10, 31: 40.08, 32: 41.25, 33: 42.40},
                valid_strides=[31, 32, 33],
                threshold=7.0
            )
            
            # Extra details for verbose = 3
            if args.verbose >= 3:
                print("\n\nAdditional Metrics and Debug Info (verbose 3):\n")
                print(f"- Total recorded time: {time[-1] - time[0]:.2f} s")
                print(f"- Total number of samples: {len(time)}")
                print(f"- Average step length (IMU): {np.mean(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)):.3f} m")
                print(f"- Average step length (Kalman): {np.mean(np.linalg.norm(np.diff(pos_kalman[:, :2], axis=0), axis=1)):.3f} m")
                print(f"- Average velocity (IMU): {np.mean(np.linalg.norm(vel, axis=1)):.3f} m/s")
                print(f"\n{'-'*130}")

                vel_magnitudes = np.linalg.norm(vel, axis=1)
                acc_magnitudes = np.linalg.norm(acc_earth, axis=1)

                print(f"\n{'*'*52} DYNAMICS CHECK {'*'*52}")
                print(f"- Max velocity (IMU): {np.max(vel_magnitudes):.2f} m/s")
                print(f"- Max acceleration (earth frame): {np.max(acc_magnitudes):.2f} m/s²")
                print(f"- % of samples with velocity > 3.0 m/s: {(vel_magnitudes > 3.0).sum() / len(vel_magnitudes) * 100:.1f}%")
                print(f"\n{'-'*130}")

                errors = np.linalg.norm(pos_kalman[:, :2] - gps_pos, axis=1)
                print(f"\n{'*'*52} SPATIAL ERROR ANALYSIS {'*'*52}")
                print(f"- Mean Kalman-GPS error: {np.mean(errors):.2f} m")
                print(f"- Median error: {np.median(errors):.2f} m")
                print(f"- Max error: {np.max(errors):.2f} m")
                print(f"- Std dev of error: {np.std(errors):.2f} m")
                print(f"- % of points with error < 5.0 m: {(errors < 5.0).sum() / len(errors) * 100:.1f}%")
                print(f"- % of points with error < 2.0 m: {(errors < 2.0).sum() / len(errors) * 100:.1f}%")
                print(f"\n{'-'*130}")


        if args.export_excel == "yes":
            rp.export_to_excel(df_steps, rp.get_output_path(f"{base_name}_stride.xlsx", args))

        if args.map_html == "yes":
            map_out_path = os.path.join(args.output_dir or ".", f"{base_name}_map.html")
            resultados_mapa = {"IMU": pos[:, :2], "Kalman": pos_kalman[:, :2]}
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
