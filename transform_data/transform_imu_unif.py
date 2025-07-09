import numpy as np
from class_transform_imu import *
from scipy.interpolate import interp1d


def main():
    args = parse_args()
    config = load_config(args.config)
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
            df = load_data(file_path)
            df = resample_to_40hz(df)
            time, sample_rate, gyr, acc, mag, gps_df = preprocess_data(df)
            stationary = detect_stationary(acc, sample_rate)

            gps_pos, gps_final = compute_gps_positions(df, config)
            imu_time = df['time'].to_numpy()
            gps_time = df.loc[~df['lat'].isna(), 'time'].to_numpy()
            gps_interp_fn = interp1d(gps_time, gps_pos, axis=0, bounds_error=False, fill_value="extrapolate")

            resultados = {}
            errores = {}

            
            for method_key in args.methods:
                m_conf = method_configs[method_key]
                method_name = m_conf["title"]

                pos = estimate_position_generic(
                    method=m_conf["method"],
                    use_mag=m_conf["use_mag"],
                    gyr=gyr, acc=acc, mag=mag,
                    time=time, sample_rate=sample_rate,
                    stationary=stationary
                )

                gps_interp = gps_interp_fn(imu_time[:len(pos)])
                kalman_fused = apply_kalman_filter2(pos, gps_interp, time)

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

                 

                # Guardar imagen si procede
                if output_dir and args.output_mode in ("save", "both"):
                    output_file = os.path.join(output_dir, f"{base_name}_{method_key}.png")

                    plot_trajectories(
                        resultados={f"{method_name}": pos[:, :2], f"{method_name} + Kalman": kalman_fused},
                        errores={f"{method_name}": base_error, f"{method_name} + Kalman": kalman_error},
                        gps_pos=gps_pos,
                        gps_final=gps_final,
                        title=f"Trajectory  ({foot_label})",
                        save_path=output_file
                    )

            print(f"{'-'*80}")

            if args.output_mode in ("screen", "both"):
                if args.plot_mode == 'split':
                    for method_key in args.methods:
                        method_title = method_configs[method_key]["title"]
                        plot_trajectories_split(
                            {k: v for k, v in resultados.items() if method_title in k},
                            {k: v for k, v in errores.items() if method_title in k},
                            gps_pos, gps_final,
                            title=f"Trajectory - {method_title} ({foot_label})"
                        )
                elif args.plot_mode == 'all':
                    all_data = {k: (resultados[k], errores[k]) for k in resultados}
                    plot_trajectories_all(all_data, gps_pos, gps_final, title=f"Trajectory ({foot_label})")

                elif args.plot_mode == 'interactive':
                    if output_dir:
                        output_map = os.path.join(output_dir, f"{base_name}_map_estimates.html")
                        generate_map_with_estimates(gps_df, resultados, output_map, config)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


    if args.output_mode in ("screen", "both"):
        plt.show()



if __name__ == "__main__":
    main()