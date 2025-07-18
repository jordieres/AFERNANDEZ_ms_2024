import numpy as np
import argparse
import os
from class_transform_imu import *
from scipy.interpolate import interp1d


def parse_args():
    """
    Parse command-line arguments for the IMU processing pipeline.

    :return: Parsed command-line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="IMU data processing pipeline")
    parser.add_argument("-f", "--file_paths", type=str, nargs="+", required=True, help="Paths to one or more Excel files")
    parser.add_argument('-v', '--verbose', type=int, default=3, help='Verbosity level')
    parser.add_argument('-c', '--config', type=str, default='.config.yaml', help='Path to the configuration file')
    parser.add_argument('-om','--output_mode', choices=["screen", "save", "both"], default="screen", help="How to handle output plots: 'screen', 'save', or 'both'")
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='Directory to save output plots')
    parser.add_argument('-m','--methods', nargs='+', choices=['madgwick_imu', 'madgwick_marg', 'mahony_imu', 'mahony_marg'], required=True, help="Algoritmos a ejecutar (elige uno o varios)")
    parser.add_argument('-g','--plot_mode', choices=['split', 'all', 'interactive'], default='split', help="How to plot trajectories: 'split' (default), 'all', or 'interactive'")
    return parser.parse_args()


def main():
    args = parse_args()
    data_proc = DataPreprocessor(args.config)
    imu_proc = IMUProcessor()
    kalman_proc = KalmanProcessor()
    plot_proc = PlotProcessor()

    config = data_proc.config

    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    method_configs = {
        "madgwick_imu":  {"method": "madgwick", "use_mag": False, "title": "Madgwick without Magnetometer"},
        "madgwick_marg": {"method": "madgwick", "use_mag": True,  "title": "Madgwick with Magnetometer"},
        "mahony_imu":    {"method": "mahony",   "use_mag": False, "title": "Mahony without Magnetometer"},
        "mahony_marg":   {"method": "mahony",   "use_mag": True,  "title": "Mahony with Magnetometer"},
    }

    for file_path in args.file_paths:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        foot_label = "Left Foot" if "left" in base_name.lower() else "Right Foot" if "right" in base_name.lower() else base_name

        if args.verbose >= 2:
            print(f"\n{'*'*33}  {foot_label}  {'*'*33}\n")
            print(f"{'-'*80}")
            print(f"Processing file: {base_name}...")
            print(f"{'-'*80}\n")

        try:
            df = data_proc.load_data(file_path)
            df = data_proc.resample_to_40hz(df)
            time, sample_rate, gyr, acc, mag, df_gps = data_proc.preprocess_data(df)
            gps_pos, gps_final = data_proc.compute_positions(df_gps, config)
            stationary = imu_proc.detect_stationary(acc, sample_rate)
            imu_time = df['time'].to_numpy()
            gps_time = df_gps['time'].to_numpy()
            gps_interp_fn = interp1d(gps_time, gps_pos, axis=0, bounds_error=False, fill_value="extrapolate")

            resultados = {}
            errores = {}

            
            for method_key in args.methods:
                m_conf = method_configs[method_key]
                method_name = m_conf["title"]

                pos = imu_proc.estimate_position_generic(
                    method=m_conf["method"],
                    use_mag=m_conf["use_mag"],
                    gyr=gyr, acc=acc, mag=mag,
                    time=time, sample_rate=sample_rate,
                    stationary=stationary
                )

                gps_interp = gps_interp_fn(imu_time[:len(pos)])
                kalman_fused = kalman_proc.apply_filter2(pos, gps_interp, time)

                base_error = np.mean(np.linalg.norm(pos[:, :2] - gps_interp[:len(pos), :2], axis=1))
                kalman_error = np.mean(np.linalg.norm(kalman_fused[:, :2] - gps_interp[:len(pos), :2], axis=1))

                def path_length(traj):
                    return np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))

                dist_imu = path_length(pos[:, :2])
                dist_kalman = path_length(kalman_fused)
                dist_gps = path_length(gps_pos)


                resultados[f"{method_name}"] = pos[:, :2]
                resultados[f"{method_name} + Kalman"] = kalman_fused
                errores[f"{method_name}"] = base_error
                errores[f"{method_name} + Kalman"] = kalman_error

                if args.verbose >= 2:
                    print(f"\nMethod: {method_name}")
                    print(f"IMU Distance             : {dist_imu:.2f} m")
                    print(f"Kalman Distance          : {dist_kalman:.2f} m")
                    print(f"GPS Distance             : {dist_gps:.2f} m")
                    print(f"Mean IMU Error           : {base_error:.2f} m")
                    print(f"Mean Kalman Error        : {kalman_error:.2f} m")

                if args.verbose >= 3:
                    duration = time[-1] - time[0]
                    mean_speed = dist_imu / duration
                    print(f"Total Duration           : {duration:.2f} s")
                    print(f"Average Speed (IMU)      : {mean_speed:.2f} m/s")

                
            print(f"{'-'*80}")

            if args.output_mode in ("screen", "both"):
                if args.plot_mode == 'split':
                    for method_key in args.methods:
                        method_title = method_configs[method_key]["title"]
                        plot_proc.plot_trajectories_split(
                            {k: v for k, v in resultados.items() if method_title in k},
                            {k: v for k, v in errores.items() if method_title in k},
                            gps_pos, gps_final,
                            title=f"Trajectory - {method_title} ({foot_label})"
                        )
                elif args.plot_mode == 'all':
                    all_data = {k: (resultados[k], errores[k]) for k in resultados}
                    plot_proc.plot_trajectories_all(all_data, gps_pos, gps_final, title=f"Trajectory ({foot_label})")
                elif args.plot_mode == 'interactive':
                    if output_dir:
                        output_map = os.path.join(output_dir, f"{base_name}_map_estimates.html")
                        plot_proc.generate_map_with_estimates(df_gps, resultados, output_map, config)

            if output_dir and args.output_mode in ("save", "both"):
                for method_key in args.methods:
                    m_conf = method_configs[method_key]
                    method_name = m_conf["title"]
                    output_file = os.path.join(output_dir, f"{base_name}_{method_key}.png")
                    plot_proc.plot_trajectories(
                        resultados={f"{method_name}": resultados[f"{method_name}"], f"{method_name} + Kalman": resultados[f"{method_name} + Kalman"]},
                        errores={f"{method_name}": errores[f"{method_name}"], f"{method_name} + Kalman": errores[f"{method_name} + Kalman"]},
                        gps_pos=gps_pos,
                        gps_final=gps_final,
                        title=f"Trajectory ({foot_label})",
                        save_path=output_file
                    )

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if args.output_mode in ("screen", "both"):
        plt.show()



if __name__ == "__main__":
    main()