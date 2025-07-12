import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from msGeom.transformations import MadgwickAHRS
from msGeom.transformations import Quaternion

def load_data(file_path: str) -> pd.DataFrame:
    """Load and convert sensor data from a CSV file.

    Args:
        file_path (str): Path to the tab-delimited CSV file.

    Returns:
        pd.DataFrame: Converted sensor data.
    """
    df = pd.read_excel(file_path)
    # for axis in ["Gx", "Gy", "Gz", "Ax", "Ay", "Az", "Mx", "My", "Mz"]:
    #     df[axis] = df[axis].str.replace(",", ".").astype(float)
    df["Gx"] *= np.pi / 180
    df["Gy"] *= np.pi / 180
    df["Gz"] *= np.pi / 180
    return df

def compute_orientation(df: pd.DataFrame, sample_period: float = 1/256, beta: float = 0.1) -> pd.DataFrame:
    """Compute orientation using the Madgwick filter.

    Args:
        df (pd.DataFrame): DataFrame with columns Gx, Gy, Gz, Ax, Ay, Az, Mx, My, Mz.
        sample_period (float): Sampling period in seconds.
        beta (float): Filter gain parameter.

    Returns:
        pd.DataFrame: DataFrame with additional columns: Roll, Pitch, Yaw.
    """
    madgwick = MadgwickAHRS(sampleperiod=sample_period, beta=beta)
    orientations = []

    for _, row in df.iterrows():
        gyro = [row["Gx"], row["Gy"], row["Gz"]]
        accel = [row["Ax"], row["Ay"], row["Az"]]
        mag = [row["Mx"], row["My"], row["Mz"]]
        madgwick.update(gyroscope=gyro, accelerometer=accel, magnetometer=mag)
        roll, pitch, yaw = madgwick.quaternion.to_euler_angles()
        orientations.append([roll, pitch, yaw])

    orientation_df = pd.DataFrame(orientations, columns=["Roll", "Pitch", "Yaw"])
    return pd.concat([df, orientation_df], axis=1)

def plot_orientation(df: pd.DataFrame) -> None:
    """Plot the evolution of orientation over time.

    Args:
        df (pd.DataFrame): DataFrame containing '_time', 'Roll', 'Pitch', and 'Yaw' columns.
    """
    df["_time"] = pd.to_datetime(df["_time"], errors="coerce")
    plt.figure(figsize=(10, 5))
    plt.plot(df["_time"], df["Yaw"], label="Yaw")
    plt.plot(df["_time"], df["Pitch"], label="Pitch")
    plt.plot(df["_time"], df["Roll"], label="Roll")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Angles (rad)")
    plt.title("Right Foot Orientation Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the orientation computation pipeline."""
    parser = argparse.ArgumentParser(description=
                        "Compute foot orientation from IMU data using Madgwick filter.")
    parser.add_argument("-i","-input_file", type=str, required=True, dest="input_file",
                        help="Path to the input CSV file (tab-delimited)")
    args = parser.parse_args()

    df = load_data(args.input_file)
    df_with_orientation = compute_orientation(df)
    plot_orientation(df_with_orientation)

if __name__ == "__main__":
    main()
