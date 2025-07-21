

import numpy as np
import os
import argparse
import pandas as pd
from matplotlib import pyplot as plt

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
    sp = StrideProcessor(min_stride=0.2, max_stride=2.5, window_sec=6.0)


    file_path = args.file_path
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    foot_label = "Left Foot" if "left" in base_name.lower() else "Right Foot" if "right" in base_name.lower() else base_name

    try:
        df = preprocessor.load_data(file_path)
        df_inter = preprocessor.resample_to_40hz(df)
        time, sample_rate, gyr, acc, mag = preprocessor.preprocess_data(df_inter)

        stationary = imu_processor.detect_stationary(acc, sample_rate)
        quats, acc_earth, vel, pos = imu_processor.estimate_position_madwick(time, gyr, acc, mag, stationary)

        df_gps, gps_pos, gps_final = preprocessor.compute_positions(df_inter, preprocessor.config)
        
        kp = KalmanProcessor(dt=sample_period, q=0.1, r=0.5)
        kp.initialize(gps_pos[0])
        fused_trajectory = kp.run_filter_with_acc_and_gps(acc_earth, gps_pos)


        pos_kalman = pos.copy()
        pos_kalman[:, :2] = fused_trajectory

        df_inter["modG"] = np.linalg.norm(gyr, axis=1)
        df_inter["modA"] = np.linalg.norm(acc, axis=1)
        df_inter["time"] = time

        df_imu = df_inter[["time", "Ax", "Ay", "Az", "Gx", "Gy", "Gz"]].copy()      # Para posteriormente analizar la zancada

        triplets_gyr = dp.detect_triplet_peaks(df_inter, column='modG')
        dp.plot_peaks(df_inter, signal_column='modG', peak_df=triplets_gyr, signal_name='modG')

        gps_lat = df_gps['lat'] if not df_gps.empty else np.full(len(pos_kalman), np.nan)
        gps_lng = df_gps['lng'] if not df_gps.empty else np.full(len(pos_kalman), np.nan)

        gps_lat = gps_lat[:len(time)] if len(gps_lat) >= len(time) else np.pad(gps_lat, (0, len(time) - len(gps_lat)), constant_values=np.nan)
        gps_lng = gps_lng[:len(time)] if len(gps_lng) >= len(time) else np.pad(gps_lng, (0, len(time) - len(gps_lng)), constant_values=np.nan)

        df_steps = rp.prepare_step_dataframe(time, gps_lat, gps_lng, pos_kalman, vel, df_inter)
        df_steps["minute"] = (df_steps["time"] // 60).astype(int)  # Marcar minuto en datos de pasos


        # FILTRACION
        # === Paso 1: Detección de zancadas ===
        step_peaks = (triplets_gyr.loc[triplets_gyr['peak_type'] == 'main', 'orig_index'].dropna().astype(int).values)
        step_peaks = np.sort(np.unique(step_peaks))  # Elimina duplicados y ordena

        # === Paso 2: Cálculo de estadísticas por minuto ===
        df_stride_stats, df_stride_raw = dp.compute_stride_stats_per_minute(df_steps, pos_kalman, step_peaks)

        # === Paso 3: Filtro por longitud de zancada válida ===
        
        df_stride_raw_clean = sp.clean_stride_data(df_stride_raw)

        # Añadir columna 'minute' a zancadas filtradas
        df_stride_raw_clean["minute"] = (df_stride_raw_clean["time"] // 60).astype(int)

        # Estadísticas limpias por minuto
        df_stride_stats_clean = sp.recompute_stats_per_minute(df_stride_raw_clean)

        # === Paso 4: Comprobaciones de calidad ===
        df_stats_checked = sp.check_distance_similarity(df_stride_stats_clean, gps_distance=430.04)
        df_stats_checked = sp.check_stride_length_range(df_stats_checked)

        # Comprobación de suavidad de la trayectoria (velocidad razonable)
        df_steps_checked = sp.check_trajectory_smoothness(df_steps)



        # Comprobación de alineación espacial con GPS
        alignment_mask, percent_ok = sp.check_spatial_alignment(pos_kalman[:, :2], gps_pos)
        # === Paso 5: Evaluación conjunta por minuto ===
        df_eval = sp.evaluate_quality_segments(df_stats_checked,df_steps,gps_pos,imu_pos=pos_kalman[:, :2],gps_distance=430.04)

        # === Paso 6: Visualización de tramos buenos/malos ===
        sp.plot_stride_filtering(df_stride_raw, df_stride_raw_clean,  y_max=3.0)
        # Zancadas válidas (ya filtradas)
        df_stride_valid = df_stride_raw_clean[["time"]].copy()

        # Zancadas inválidas = todas menos las válidas
        valid_times_set = set(df_stride_valid["time"])
        df_stride_invalid = df_stride_raw[~df_stride_raw["time"].isin(valid_times_set)][["time"]].copy()
       

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


            # Extra details for verbose = 3
            if args.verbose >= 3:
                print("\n\nAdditional Metrics and Debug Info (verbose 3):\n")
                print(f"- Total recorded time: {time[-1] - time[0]:.2f} s")
                print(f"- Total number of samples: {len(time)}")
                print(f"- Average step length (IMU): {np.mean(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)):.3f} m")
                print(f"- Average step length (Kalman): {np.mean(np.linalg.norm(np.diff(pos_kalman[:, :2], axis=0), axis=1)):.3f} m")
                print(f"- Average velocity (IMU): {np.mean(np.linalg.norm(vel, axis=1)):.3f} m/s")
                print(f"\n{'-'*130}")

                print(f"\n{'*'*56} RAW OUTPUT TABLES {'*'*55}")
                print("\nFirst 5 rows of stride measurement results (position, velocity, distance per step):\n")
                print(df_steps.head())

                print("\n\nFirst 5 rows of per-minute stride statistics (summary table):\n")
                print(df_stride_stats.head())

                print("\n\nirst 5 rows of raw stride data (individual strides):\n")
                print(df_stride_raw.head())
                print(f"{'-'*130}\n")

                print(f"\n{'*'*52} STRIDE CLEANING SUMMARY {'*'*53}") 
                print(f"\n- Filtered out {len(df_stride_raw) - len(df_stride_raw_clean)} strides outside the valid range (0.2–2.5 m).")
                print(f"- Estimated distance / GPS ratio: {df_stride_stats_clean['distance_m'].sum() / 430.04:.2f} ({'OK' if abs(df_stride_stats_clean['distance_m'].sum() / 430.04 - 1) <= 0.15 else 'NO'})")

                print(f"- Detected {df_steps_checked['velocity_spike'].sum()} velocity spikes (> 3.0 m/s).")

                print(f"- {percent_ok:.1f}% of points have spatial error < 10.0 m.")
                print(f"\n- Points well aligned with GPS: {alignment_mask.sum()} / {len(alignment_mask)} ({percent_ok:.1f}%)")

                well_aligned = df_eval[df_eval["all_criteria_ok"] == True]
                if not well_aligned.empty:
                    print("- Well-aligned segments detected:")
                    print(well_aligned)
                else:
                    print("- No well-aligned segments detected.")
                print("- Number of valid strides:", len(df_stride_raw_clean))
                print(f"\n{'-'*130}")
                print(f"\n{'*'*54} VALID STRIDE DETAIL {'*'*55}") 
                print("\nValid stride times and lengths:\n")
                print(df_stride_raw_clean[["time", "stride_length_m"]])
                print("\n\nFirst 5 rows of filtered per-minute stride statistics:\n")
                print(df_stride_stats_clean.head())
                print("\n\nFirst 5 rows of filtered individual strides:\n")
                print(df_stride_raw_clean.head())
                print(f"\n{'-'*130}")


                # Extraer regiones para inspección manual en Grafana 
                valid_results = sp.analyze_strides(df_imu, df_gps, df_stride_valid,  stride_type="valid")
                invalid_results = sp.analyze_strides(df_imu, df_gps, df_stride_invalid,  stride_type="invalid")
                print(f"\n{'*'*51} GRAFANA INSPECTION WINDOWS {'*'*51}\n")

                valid_dists = [r["gps_distance_m"] for r in valid_results if r["gps_distance_m"] is not None]
                invalid_dists = [r["gps_distance_m"] for r in invalid_results if r["gps_distance_m"] is not None]

                print(f"- Average GPS distance (valid strides): {np.mean(valid_dists):.2f} m")
                print(f"- Average GPS distance (invalid strides): {np.mean(invalid_dists):.2f} m")
                print("\nVALID STRIDES:")
                for r in valid_results:
                    stride_idx = np.searchsorted(df_inter["time"].values, r["stride_time"])
                    stride_dt = df_inter["_time"].values[stride_idx]  
                    print(f"- Stride #{r['stride_index']:<3} | Time: {r['stride_time']:6.2f}s | Datetime: {stride_dt:<26} | Window: [{r['time_window_start']:6.2f}s – {r['time_window_end']:6.2f}s] | GPS Distance: {r['gps_distance_m']:6.2f} m")

                print("\nINVALID STRIDES:")
                for r in invalid_results:
                    stride_idx = np.searchsorted(df_inter["time"].values, r["stride_time"])
                    stride_dt = df_inter["_time"].values[stride_idx]  
                    print(f"- Stride #{r['stride_index']:<3} | Time: {r['stride_time']:6.2f}s | Datetime: {stride_dt:<26} | Window: [{r['time_window_start']:6.2f}s – {r['time_window_end']:6.2f}s] | GPS Distance: {r['gps_distance_m']:6.2f} m")

            
                print(f"{'-'*130}\n")

        if args.export_excel == "yes":
            rp.export_to_excel(df_steps, rp.get_output_path(f"{base_name}_stride.xlsx", args))
            # rp.export_to_excel(df_stride_stats, rp.get_output_path(f"{base_name}_stride_stats_per_minute.xlsx", args))
            # rp.export_to_excel(df_stride_raw, rp.get_output_path(f"{base_name}_individual_strides.xlsx", args))

        if args.map_html == "yes":
            map_out_path = os.path.join(args.output_dir or ".", f"{base_name}_map.html")
            resultados_mapa = {
                "IMU": pos[:, :2],
                "Kalman": pos_kalman[:, :2]
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
