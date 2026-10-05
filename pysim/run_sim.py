"""Run the simulation entry point."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from pyfrontend import pysim_runner  # pylint: disable=no-name-in-module

OUTPUT_CSV = "sim_output.csv"

if __name__ == "__main__":
    epoch_0 = "2019-07-15T11:23:27.30 UTC"
    x_0 = np.array([
        -194.33826150101773, 824.8947002999065, 1653.703391999927,
        .08094043359034313, -1.4478938749999684, .731723312100025,
    ])
    t, states, _ = pysim_runner(epoch_0, x_0, (0.0, 28800), 0.1)

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
    fig.tight_layout()
    plt.show()


    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"])
        for t_i, row in zip(t, states):
            writer.writerow([t_i, *row])

    print(f"Wrote {len(t)} rows to {OUTPUT_CSV}")
