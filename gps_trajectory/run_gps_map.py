import argparse
from gps_trajectory.gps_map_generator import GPSTrajectoryProcessor


def main():
        
    """
    Main function to run the GPS map and trajectory generation.
    """
    parser = argparse.ArgumentParser(description="Generate GPS map and trajectory plot.")
    parser.add_argument("-i", "--input", required=True, help="Path to input Excel file.")
    parser.add_argument("-o", "--output", required=True, help="Path to output HTML map.")
    parser.add_argument("-p", "--plot", required=True, help="Path to output trajectory PNG plot.")
    args = parser.parse_args()

    processor = GPSTrajectoryProcessor()
    df = processor.prepare_gps_dataframe(args.input)
    processor.generate_gps_map(df, args.output)
    processor.plot_macroscopic_trajectory(df, args.plot)
    distancia_total = processor.calculate_total_distance(df)
    print(f"Distancia total recorrida: {distancia_total:.2f} metros")


if __name__ == "__main__":
    main()