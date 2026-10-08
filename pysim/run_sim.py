"""Run the simulation entry point."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from pyfrontend import pysim_runner  # pylint: disable=no-name-in-module


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

OUTPUT_CSV = "sim_output.csv"

if __name__ == "__main__":
    epoch_0 = "2019-07-15T11:23:27.30 UTC"
    x_0 = np.array([
        1.28237, 0, 0,
        0, 0.0000617543738092, 0,
    ])
    t, states, _ = pysim_runner(epoch_0, x_0, (0.0, 130474.456227), 60)

    state_names = ["x", "y", "z", "x_dot", "y_dot", "z_dot"]
    state_units = ["km", "km", "km", "km/s", "km/s", "km/s"]
    state_fig, state_axes = plt.subplots(3, 2, sharex=True, figsize=(12, 8))
    for index, state_axis in enumerate(state_axes.flat):
        state_axis.plot(t, states[:, index])
        state_axis.set_title(state_names[index])
        state_axis.set_ylabel(state_units[index])
        state_axis.grid(True, alpha=0.3)

    state_axes[-1, 0].set_xlabel("Time (s)")
    state_axes[-1, 1].set_xlabel("Time (s)")
    state_fig.suptitle("Simulation State History")
    state_fig.tight_layout()

    fig = plt.figure(figsize=(10, 8))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(states[:, 0], states[:, 1], states[:, 2], label="Spacecraft orbit")
    axis.scatter(*states[0, :3], color="green", s=50, label="Start")
    axis.scatter(*states[-1, :3], color="red", s=50, label="End")
    axis.set_title("3D Orbit")
    axis.set_xlabel("x (km)")
    axis.set_ylabel("y (km)")
    axis.set_zlabel("z (km)")
    axis.set_box_aspect((1, 1, 1))
    axis.legend()
    set_axes_equal(axis)
    def disable_vert_rotation(event):
        azim = axis.azim
        elev = axis.elev
        axis.view_init(elev=elev, azim=azim, roll=0.0)
    fig.tight_layout()
    fig.canvas.mpl_connect('motion_notify_event', disable_vert_rotation)
    plt.show()


    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"])
        for t_i, row in zip(t, states):
            writer.writerow([t_i, *row])

    print(f"Wrote {len(t)} rows to {OUTPUT_CSV}")
