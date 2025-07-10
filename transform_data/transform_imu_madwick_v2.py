"""
Madwick con Magnetometro aplicando ZUPT,ZUPH y lo de la rotación.
Además se aplica el kalman simple unicamente pero en este caso se divide el df en 2 en el primero se deriva con kalman y en el segundo se estima a partir de la primera deriva a ver cuanto de fiable es el filtro. 

"""

import numpy as np
import os
from matplotlib import pyplot as plt

from class_transform_imu import *
from class_madwick import *

prev_gps_latlng = None
prev_gps_pos = None

def main():
    """
    Main function to process IMU Excel files and plot estimated trajectories vs GPS.
    """
    global prev_gps_latlng, prev_gps_pos
    args = parse_args_M()
    config = load_config(args.config)
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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
            df = load_data(file_path)

            if 'lat' in df.columns and 'lng' in df.columns:
                current_gps_latlng = df[['lat', 'lng']].to_numpy()
                use_prev_gps = False

                if prev_gps_latlng is None:
                    print("Este es el primer archivo con GPS cargado.")
                else:
                    if current_gps_latlng.shape == prev_gps_latlng.shape:
                        same_gps = np.allclose(current_gps_latlng, prev_gps_latlng, atol=1e-6)
                        print(f"¿GPS idéntico a archivo anterior?: {same_gps}")
                        if same_gps:
                            use_prev_gps = True
                    else:
                        print("GPS no comparable: diferente número de muestras.")
                prev_gps_latlng = current_gps_latlng

            df = resample_to_40hz(df)
            time, sample_rate, gyr, acc, mag, sample_period = preprocess_data(df)
            stationary, acc_lp, threshold = detect_stationary(acc, sample_rate)
            quats, acc_earth, vel, pos = estimate_orientation_and_position(
                time, gyr, acc, mag, sample_period, sample_rate, stationary
            )

            if use_prev_gps and prev_gps_pos is not None:
                gps_pos = prev_gps_pos
                gps_final = gps_pos[-1]
                print("Se reutilizó el GPS del primer archivo.")
            else:
                gps_pos, gps_final = compute_gps_positions(df, config)
                prev_gps_pos = gps_pos

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

            save_path = None
            if output_dir and args.output_mode in ("save", "both"):
                save_path = os.path.join(output_dir, f"{base_name}_trajectory.png")

            plot_results(
                time, acc_lp, threshold, pos, vel, gps_pos=gps_pos,
                output_dir=output_dir if args.output_mode in ("save", "both") else None,
                title=f"Trajectory Comparison - {foot_label} (IMU vs GPS)",
                base_name=base_name + "_pre_kalman",
                verbose=args.verbose,
                traj_label="IMU"
            )

            plot_results(
                time, acc_lp, threshold, pos_kalman, vel, gps_pos=gps_pos,
                output_dir=output_dir if args.output_mode in ("save", "both") else None,
                title=f"Trajectory Comparison - {foot_label} (Kalman vs GPS)",
                base_name=base_name + "_post_kalman",
                verbose=args.verbose,
                traj_label="Kalman"
            )

            if save_path:
                plt.savefig(save_path)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if output_dir and args.output_mode in ("save", "both"):
        print(f"{'-'*80}")
        print(f"\nTrajectory plots successfully saved to:\n{output_dir}\n")

    if args.output_mode in ("screen", "both"):
        plt.show()


if __name__ == "__main__":
    main()
