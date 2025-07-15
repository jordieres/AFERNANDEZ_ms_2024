
import numpy as np
import os
import argparse
from matplotlib import pyplot as plt

from class_transform_imu import *




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
    dp = DetectPeaks()

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
            df = preprocessor.load_data(file_path)
            df_inter = preprocessor.resample_to_40hz(df)
            df_inter["modG"] = np.sqrt(df_inter["Gx"]**2 + df_inter["Gy"]**2 + df_inter["Gz"]**2)
            time, sample_rate, gyr, acc, mag, df_gps = preprocessor.preprocess_data(df_inter)
            triplets = dp.detect_gyro_triplet_peaks(df_inter, column='modG')
            dp.plot_peaks(df_inter, signal_column='modG', peak_df=triplets)
    

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if args.output_dir and args.output_mode in ("save", "both"):
        print(f"{'-'*80}")
        print(f"\nTrajectory plots saved to: {args.output_dir}")
    if args.output_mode in ("screen", "both"):
        plt.show()


if __name__ == "__main__":
    main()
