import plotly.graph_objects as go
import folium
import os

from matplotlib import pyplot as plt
from pyproj import Proj, Transformer


class PlotProcessor:
    """
    Class for visualizing trajectories using Matplotlib, Plotly, and Folium.

    This class provides a variety of plotting methods to compare estimated trajectories with ground truth GPS data,
    display diagnostic plots, and export results to interactive maps or static images.
    """

    def plot_trajectories_interactive(self, results, errors, gps_pos, gps_final, title="Trajectory Comparison", save_path=None):
        """
        Create an interactive Plotly plot comparing estimated and GPS trajectories.

        :param results: Dictionary of estimated positions.
        :type results: dict[str, np.ndarray]
        :param errors: Dictionary of position errors.
        :type errors: dict[str, float]
        :param gps_pos: GPS positions array.
        :type gps_pos: np.ndarray
        :param gps_final: Final GPS position.
        :type gps_final: np.ndarray
        :param title: Plot title.
        :type title: str
        :param save_path: Optional path to save the HTML plot.
        :type save_path: str or None
        """
        fig = go.Figure()
        for name, pos in results.items():
            fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode='lines', name=f"{name} ({errors[name]:.2f} m)"))

        fig.add_trace(go.Scatter(x=gps_pos[:, 0], y=gps_pos[:, 1], mode='lines', name="GPS (reference)", line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=[gps_final[0]], y=[gps_final[1]], mode='markers', name="GPS final", marker=dict(color='black', size=10)))

        fig.update_layout(title=title, xaxis_title='X (m)', yaxis_title='Y (m)', width=900, height=700)
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive plot saved to: {save_path}")

    def plot_trajectories_all(self, all_results, gps_pos, gps_final, title="Trajectory Comparison"):
        """
        Plot all estimated trajectories together using Matplotlib.

        :param all_results: Dictionary with keys as method names and values as (positions, error).
        :type all_results: dict[str, tuple[np.ndarray, float]]
        :param gps_pos: GPS positions array.
        :type gps_pos: np.ndarray
        :param gps_final: Final GPS position.
        :type gps_final: np.ndarray
        :param title: Plot title.
        :type title: str
        """
        plt.figure(figsize=(10, 8))
        for label, (pos, err) in all_results.items():
            linestyle = '--' if 'Kalman' in label else '-'
            alpha = 0.6 if 'Kalman' not in label else 1.0
            plt.plot(pos[:, 0], pos[:, 1], linestyle, alpha=alpha, label=f"{label} ({err:.2f} m)")
        plt.plot(gps_pos[:, 0], gps_pos[:, 1], 'k--', label="GPS (reference)")
        plt.plot(gps_final[0], gps_final[1], 'ko', label="GPS final")
        plt.title(title)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.legend()
        plt.grid()

    def plot_trajectories_split(self, results, errors, gps_pos, gps_final, title, save_path=None):
        """
        Plot each estimated trajectory separately using Matplotlib.

        :param results: Dictionary of estimated positions.
        :type results: dict[str, np.ndarray]
        :param errors: Dictionary of errors per method.
        :type errors: dict[str, float]
        :param gps_pos: GPS positions array.
        :type gps_pos: np.ndarray
        :param gps_final: Final GPS point.
        :type gps_final: np.ndarray
        :param title: Plot title.
        :type title: str
        :param save_path: Optional path to save plot.
        :type save_path: str or None
        """
        plt.figure(figsize=(10, 8))
        for name, pos in results.items():
            plt.plot(pos[:, 0], pos[:, 1], label=f"{name} ({errors[name]:.2f} m)")
        plt.plot(gps_pos[:, 0], gps_pos[:, 1], 'k--', label="GPS (reference)")
        plt.plot(gps_final[0], gps_final[1], 'ko', label="GPS final")
        plt.title(title)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.legend()
        plt.grid()
        if save_path:
            plt.savefig(save_path)

    def generate_map_with_estimates(self, df_gps, results, output_html_path, config):
        """
        Generates an interactive map with GPS and estimated IMU/Kalman trajectories using Folium.

        :param df_gps: DataFrame with 'lat' and 'lng' columns.
        :type df_gps: pd.DataFrame
        :param results: Dictionary of trajectory name to estimated XY positions.
        :type results: dict[str, np.ndarray]
        :param output_html_path: Path to save HTML map.
        :type output_html_path: str
        :param config: YAML configuration with 'Location' projection info.
        :type config: dict
        """
        location_cfg = config["Location"]
        proj = Proj(proj=location_cfg["proj"], zone=location_cfg["zone"], ellps=location_cfg["ellps"], south=location_cfg["south"])
        ref_code = location_cfg["code"]
        transformer = Transformer.from_proj(proj, ref_code, always_xy=True)

        lat0, lon0 = df_gps.loc[0, 'lat'], df_gps.loc[0, 'lng']
        fmap = folium.Map(location=[lat0, lon0], zoom_start=18)

        gps_coords = df_gps[['lat', 'lng']].values.tolist()
        gps_group = folium.FeatureGroup(name='GPS (reference)')
        folium.PolyLine(gps_coords, color='grey', weight=4).add_to(gps_group)
        folium.Marker(location=gps_coords[0], popup="Start", icon=folium.Icon(color='green')).add_to(gps_group)
        folium.Marker(location=gps_coords[-1], popup="End", icon=folium.Icon(color='red')).add_to(gps_group)
        gps_group.add_to(fmap)

        color_list = ['cadetblue','#F04BF0','#FAA43A', "#056641",'#F17CB0','#DECF3F','#F15854','#5DA5DA']
        for i, (name, traj) in enumerate(results.items()):
            x_coords = traj[:, 0]
            y_coords = traj[:, 1]
            lon_est, lat_est = transformer.transform(
                x_coords + proj(lon0, lat0)[0],
                y_coords + proj(lon0, lat0)[1]
            )
            path = list(zip(lat_est, lon_est))
            color = color_list[i % len(color_list)]
            group = folium.FeatureGroup(name=name)
            folium.PolyLine(path, color=color, weight=3).add_to(group)
            group.add_to(fmap)

        folium.LayerControl(collapsed=False).add_to(fmap)
        fmap.save(output_html_path)
        print(f"Interactive map saved to: {output_html_path}")



    def plot_trajectories(self, results, errores, gps_pos, gps_final, title="Trajectory Comparison", save_path=None):
        """
        Plot estimated and GPS trajectories.

        :param results: Dictionary of estimated positions.
        :type results: dict[str, np.ndarray]
        :param errors: Dictionary of position errors.
        :type errors: dict[str, float]
        :param gps_pos: GPS positions array.
        :type gps_pos: np.ndarray
        :param gps_final: Final GPS position.
        :type gps_final: np.ndarray
        :param title: Title for the plot.
        :type title: str
        :param save_path: If given, path to save the plot image.
        :type save_path: str or None
        """
        plt.figure(figsize=(10, 8))
        for name, pos in results.items():
            plt.plot(pos[:, 0], pos[:, 1], label=f"{name} ({errores[name]:.2f} m)")
        plt.plot(gps_pos[:, 0], gps_pos[:, 1], 'k--', label="GPS (reference)")
        plt.plot(gps_final[0], gps_final[1], 'ko', label="GPS final")
        plt.title(title)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.legend()
        plt.grid()
        if save_path:
            plt.savefig(save_path)

        
    def plot_macroscopic_comparision(self, pos, gps_pos=None, output_dir=None, title="Trajectory Comparison", base_name="trajectory", traj_label= None ):
        """
        Plot diagnostic and trajectory figures for IMU data, and optionally save them.

        This function generates several plots based on filtered acceleration, position,
        velocity, 2D and 3D trajectory. If GPS data is provided, it includes a comparative plot. 
        Depending on the verbosity level and output directory, plots can also be saved.


        :param pos: Estimated position array of shape (N, 3).
        :type pos: np.ndarray
        :param gps_pos: Optional GPS position array of shape (N, 2). Defaults to None.
        :type gps_pos: np.ndarray or None
        :param title: Title used in trajectory comparison plots.
        :type title: str

        """
        if gps_pos is not None:
            plt.figure(figsize=(10, 8))
            plt.plot(pos[:, 0], pos[:, 1], label=f'{traj_label} Trajectory')
            plt.plot(gps_pos[:, 0], gps_pos[:, 1], 'k--', label='GPS Reference')
            plt.plot(gps_pos[-1, 0], gps_pos[-1, 1], 'ko', label='Final GPS')
            plt.title(title)
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.axis("equal")
            plt.grid()
            plt.legend()

    def plot_resume(self, time, pos, vel, output_dir=None, base_name="summary"):
        """
        Plot position, velocity, and 3D trajectory of the estimated movement.

        This function summarizes motion-related variables: 1D time series of position and velocity
        (in x, y, z), and a 3D trajectory plot in space.

        :param time: Time vector in seconds.
        :type time: np.ndarray
        :param pos: Estimated position array of shape (N, 3).
        :type pos: np.ndarray
        :param vel: Estimated velocity array of shape (N, 3).
        :type vel: np.ndarray
        :param output_dir: (Not used anymore; figures are not saved).
        :type output_dir: str or None
        :param base_name: (Unused). Name that would be used for saving if enabled.
        :type base_name: str
        """

        # Position over time
        plt.figure(figsize=(15, 5))
        plt.plot(time, pos[:, 0], 'r', label='x')
        plt.plot(time, pos[:, 1], 'g', label='y')
        plt.plot(time, pos[:, 2], 'b', label='z')
        plt.title("Position over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.legend()
        plt.grid()

        # Velocity over time
        plt.figure(figsize=(15, 5))
        plt.plot(time, vel[:, 0], 'r', label='vx')
        plt.plot(time, vel[:, 1], 'g', label='vy')
        plt.plot(time, vel[:, 2], 'b', label='vz')
        plt.title("Velocity over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.legend()
        plt.grid()

        # 3D Trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])
        ax.set_title("3D Trajectory")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")



