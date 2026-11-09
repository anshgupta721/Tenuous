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

# def run_monte_carlo(n_runs, epoch_0, x_0, P0, Q0, harmonics, t_span, dt, base_seed=0):
#     n_craft = x_0.shape()[0]
#     p_0 = np.repeat(P0[None], n_craft, axis=0)
#     q = np.repeat(Q0[None], n_craft, axis=0)
#     errors, sigmas, nees = [], [], []

#     for r in range(n_runs):
#         seed = base_seed + r
#         rng = np.random.default_rng(seed)
#         # Vary the estimated states
#         est_x_0 = x_0 + rng.multivariate_normal(np.zeros(6), P0, size=n_craft)

#         t, states, estimated_states, covariance = pysim_runner(epoch_0, x_0, est_x_0, p_0, q, harmonics, t_span, seed)

#         post = covariance[1::2]



#     return t, np.stack

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


    Q0 = np.zeros((6, 6))
    Q0[:3, :3] = 1e-7**2 * np.eye(3)
    q = np.repeat(Q0[None], n_craft, axis=0)
    dt = 10
   
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
