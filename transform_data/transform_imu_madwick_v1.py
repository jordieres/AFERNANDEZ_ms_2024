"""

Madwick con Magnetometro aplicando ZUPT,ZUPH y lo de la rotación.
Además se ven distintas formas de ajustar la deriva aunque la mejor es la de kalman basico.

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
                elif current_gps_latlng.shape == prev_gps_latlng.shape:
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

            # Apply filters
            pos_kalman = apply_kalman_filter2(pos, gps_pos, time)
            pos_ekf_2d = ekf_fusion_2d(gps_pos=gps_pos, imu_acc=acc_earth[:, :2], time=time)
            pos_complementary = complementary_filter(pos, gps_pos, alpha=0.98)
            pos_drift_corrected = linear_drift_correction(pos, gps_start=gps_pos[0], gps_end=gps_pos[-1])

            dist_gps = np.sum(np.linalg.norm(np.diff(gps_pos, axis=0), axis=1)) if gps_pos is not None else 0

            if args.verbose >= 2:
                def print_metrics(name, traj):
                    final_err = np.linalg.norm(traj[-1, :2] - gps_final)
                    total_dist = np.sum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1))
                    print(f"- {name}    -> Final error: {final_err:.2f} m | Distance: {total_dist:.2f} m")

                print("\n Quantitative Comparison: ")
                print(f"- Total GPS distance: {dist_gps:.2f} m")
                print_metrics("IMU", pos)
                print_metrics("Kalman", pos_kalman)
                print_metrics("EKF 2D", np.hstack((pos_ekf_2d, pos[:, 2:3])))
                print_metrics("Complementary", pos_complementary)
                print_metrics("Linear Drift", pos_drift_corrected)
                print(f"{'-'*80}")

            save_path = None
            if output_dir and args.output_mode in ("save", "both"):
                save_path = os.path.join(output_dir, f"{base_name}_trajectory.png")

            # Plots
            plot_results(time, acc_lp, threshold, pos, vel, gps_pos=gps_pos,
                         output_dir=output_dir if args.output_mode in ("save", "both") else None,
                         title=f"Trajectory Comparison - {foot_label} (IMU vs GPS)",
                         base_name=base_name + "_pre_kalman",
                         verbose=args.verbose, traj_label="IMU")

            plot_results(time, acc_lp, threshold, pos_kalman, vel, gps_pos=gps_pos,
                         output_dir=output_dir if args.output_mode in ("save", "both") else None,
                         title=f"Trajectory Comparison - {foot_label} (Kalman vs GPS)",
                         base_name=base_name + "_post_kalman",
                         verbose=args.verbose, traj_label="Kalman")

            plot_results(time, acc_lp, threshold, np.hstack((pos_ekf_2d, pos[:, 2:3])), vel, gps_pos=gps_pos,
                         output_dir=output_dir if args.output_mode in ("save", "both") else None,
                         title=f"Trajectory Comparison - {foot_label} (EKF 2D vs GPS)",
                         base_name=base_name + "_post_ekf2d",
                         verbose=args.verbose, traj_label="EKF 2D")

            plot_results(time, acc_lp, threshold, pos_complementary, vel, gps_pos=gps_pos,
                         output_dir=output_dir if args.output_mode in ("save", "both") else None,
                         title=f"Trajectory Comparison - {foot_label} (Complementary Filter vs GPS)",
                         base_name=base_name + "_post_complementary",
                         verbose=args.verbose, traj_label="Complementary")


            plot_results(time, acc_lp, threshold, pos_drift_corrected, vel, gps_pos=gps_pos,
                         output_dir=output_dir if args.output_mode in ("save", "both") else None,
                         title=f"Trajectory Comparison - {foot_label} (Linear Drift Correction vs GPS)",
                         base_name=base_name + "_post_drift",
                         verbose=args.verbose, traj_label="LinearDrift")

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