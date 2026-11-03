"""Run the simulation entry point."""
import csv
import numpy as np
import matplotlib.pyplot as plt

from config import PlotConfig, SimConfig
from plots.orbit_3d import plot_orbit_3d
from plots.state_history import plot_state_history
from plots.three_sigma import plot_three_sigma_sawtooth

from pyfrontend import pysim_runner  # pylint: disable=no-name-in-module


OUTPUT_CSV = "sim_output.csv"

if __name__ == "__main__":
    epoch_0 = "2019-07-15T11:23:27.30 UTC"
    x_0 = np.array([
        [1282.37, 0.0, 0.0, 0.0, 0.0617543738092, 0.0],
        [0.0, 1282.37, 0.0, -0.0617543738092, 0.0, 0.0]])
    x_0 = np.atleast_2d(np.asarray(x_0, dtype=np.float64))
    n_craft = x_0.shape[0]
    p_0 = np.diag([2**2, 2**2, 2**2, .00003**2, .00003**2, .00003**2])
    p_0 = np.asarray(p_0, dtype=np.float64)
    p_0 = np.repeat(p_0[None], n_craft, axis=0)
    I = np.eye(3)

    dt = 10
    q = 1e-7**2 * np.block([
        [dt**3/3 * I, dt**2/2 * I],
        [dt**2/2 * I, dt * I]
    ])
    q = np.asarray(q, dtype=np.float64)
    q = np.repeat(q[None], n_craft, axis=0)
    harmonics = "pysim/chesley_coeffs.csv"
    if p_0.ndim == 2: p_0 = p_0[None]                        # (n_craft, n, n)
    if q.ndim == 2:   q = q[None]

    t, states, estimated_states, covariance = pysim_runner(
        epoch_0, x_0, x_0, p_0, q, harmonics, (0.0, 200000), 10, 42
    )

    plots = PlotConfig(length_unit="m")



    plot_orbit_3d(states, (10, 8), plots)
    n_craft = states.shape[1]
    for c in range(n_craft):
        plot_state_history(t, states, estimated_states, plots, craft=c)
        plot_three_sigma_sawtooth(t, states, estimated_states, covariance, plots, (12, 8), craft=c)


    plt.show()


    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"])
        for t_i, row in zip(t, states):
            writer.writerow([t_i, *row])

    print(f"Wrote {len(t)} rows to {OUTPUT_CSV}")
