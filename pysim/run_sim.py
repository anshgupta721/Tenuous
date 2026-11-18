"""Run the simulation entry point."""
import csv
import numpy as np
import matplotlib.pyplot as plt

from config import PlotConfig, SimConfig
from plots.orbit_3d import plot_orbit_3d
from plots.state_history import plot_state_history
from plots.three_sigma import plot_state_sigma, plot_param_sigma
from gravity_field import load_gravity_field
from scipy.linalg import block_diag

from pyfrontend import pysim_runner  # pylint: disable=no-name-in-module


OUTPUT_CSV = "sim_output.csv"

def run_monte_carlo(n_runs, epoch_0, x_0, sc_P0, Q0, mu, grav_p0, harmonics, estimated_degree, t_span, dt, save_every, base_seed=0):
    n_craft = x_0.shape[0]
    CRAFT_STATES = x_0.shape[0][0]
    GRAV_STATES = grav_p0.shape[0]
    sc_p0 = np.repeat(sc_P0[None], n_craft, axis=0)
    grav_p0 = np.repeat(grav_p0[None], n_craft, axis=0)
    q = np.repeat(Q0[None], n_craft, axis=0)
    errors, sigmas, nees = [], [], []

    # Define the initial covariance matrices for the estimator of each spacecraft
    p_single = np.asarray(block_diag(sc_p0, grav_p0), dtype=np.float64)
    p_0 = np.repeat(p_single[None], n_craft, axis=0)
    estimates, variances = [], []
    for r in range(n_runs):
        seed = base_seed + r
        rng = np.random.default_rng(seed)
        # Vary the estimated states
        est_x_0 = x_0 + rng.multivariate_normal(np.zeros(CRAFT_STATES), sc_p0, size=n_craft)
        # Vary the gravitational field that gets passed into each estimator, keeping the prior guess the same across spacecraft
        estimated_gravity_field = np.concatenate(
            (np.atleast_1d(mu), load_gravity_field(harmonics, estimated_degree))
        )
        gravity_field_variance = rng.multivariate_normal(np.zeros(GRAV_STATES), grav_p0)
        estimated_gravity_field = estimated_gravity_field + gravity_field_variance

        # Add a estimated gravity field to each different spacecraft
        gravity_field_columns = np.repeat(
            estimated_gravity_field[None, :], n_craft, axis=0
        )
        est_x_0 = np.concatenate((est_x_0, gravity_field_columns), axis=1)
        # Construct a uncertainty matrix for each spacecraft for each estimator
        
        t, states, estimated_states, covariance = pysim_runner(epoch_0, x_0, est_x_0, p_0, q, harmonics, t_span, save_every, seed)

        post = covariance[1::2]
        
        estimates.append(estimated_states)
        variances.append(np.diagonal(post, axis1=2, axis2=3))   # (T, n_craft, n)



    return t, np.stack

if __name__ == "__main__":
    epoch_0 = "2019-07-15T11:23:27.30 UTC"
    truth_x_0 = np.array([
        [1282.37, 0.0, 0.0, 0.0, 0.0617543738092, 0.0],
        [0.0, 1282.37, 0.0, -0.0617543738092, 0.0, 0.0]])
    truth_x_0 = np.atleast_2d(np.asarray(truth_x_0, dtype=np.float64))
    n_craft = truth_x_0.shape[0]
    est_x_0 = np.array([
        [1282.37, 0.0, 0.0, 0.0, 0.0617543738092, 0.0, 4.89044967462],
        [0.0, 1282.37, 0.0, -0.0617543738092, 0.0, 0.0, 4.89044967462]])

    sigma_mu = 0.05 * 4.89044967462

    
    sc_p_0 = np.diag([2**2, 2**2, 2**2, .00003**2, .00003**2, .00003**2, sigma_mu**2])
    sc_p_0 = np.asarray(sc_p_0, dtype=np.float64)
    I = np.eye(3)


    Q0 = np.zeros((7, 7))
    Q0[:3, :3] = 1e-7**2 * np.eye(3)
    q = np.repeat(Q0[None], n_craft, axis=0)
    dt = 10
   
    harmonics = "pysim/chesley_coeffs.csv"

    t, states, estimated_states, covariance = pysim_runner(
        epoch_0, truth_x_0, est_x_0, sc_p_0, q, harmonics, (0.0, 200000), 10, 42
    )

    plots = PlotConfig(length_unit="m")



    plot_orbit_3d(states, (10, 8), plots)
    n_craft = states.shape[1]
    for c in range(n_craft):
        plot_state_history(t, states, estimated_states[..., :6], plots, craft=c)
        plot_state_sigma(t, states, estimated_states, covariance, plots, (12, 8), craft=c)
        plot_param_sigma(
            t, estimated_states, covariance,
            truth_params=[4.89044967462], labels=["mu"], units=["m³/s²"],
            fig_size=(12, 8), craft=c,
        )

    plt.show()


    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"])
        for t_i, row in zip(t, states):
            writer.writerow([t_i, *row])

    print(f"Wrote {len(t)} rows to {OUTPUT_CSV}")
