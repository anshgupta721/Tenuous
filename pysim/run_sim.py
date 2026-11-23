# """Run the simulation entry point."""
# import csv
# import numpy as np
# import matplotlib.pyplot as plt

# from config import PlotConfig, SimConfig
# from plots.orbit_3d import plot_orbit_3d
# from plots.state_history import plot_state_history
# from plots.three_sigma import plot_state_sigma, plot_param_sigma
# from gravity_field import load_gravity_field
# from scipy.linalg import block_diag

# from pyfrontend import pysim_runner  # pylint: disable=no-name-in-module # type: ignore[attr-defined]

# OUTPUT_CSV = "sim_output.csv"

# if __name__ == "__main__":
#     epoch_0 = "2019-07-15T11:23:27.30 UTC"
#     harmonics = "pysim/chesley_coeffs.csv"
#     truth_x_0 = np.array([
#         [1200.37, 0.0, 0.0, 0.0, 0.0617543738092, 0.0],
#         [0.0, 1200.37, 0.0, -0.0617543738092, 0.0, 0.0]])
#     truth_x_0 = np.atleast_2d(np.asarray(truth_x_0, dtype=np.float64))
#     mu = 4.89044967462
#     true_gravity_field = np.concatenate(([mu], load_gravity_field(harmonics, 2)))
#     n_grav = len(true_gravity_field)
#     n_craft = truth_x_0.shape[0]
#     est_x_0 = np.array([
#         [1282.37, 0.0, 0.0, 0.0, 0.0617543738092, 0.0],
#         [0.0, 1282.37, 0.0, -0.0617543738092, 0.0, 0.0]])
#     estimated_gravity_field = true_gravity_field

#     gravity_field_columns = np.repeat(
#                 estimated_gravity_field[None, :], n_craft, axis=0
#             )
#     est_x_0 = np.concatenate((est_x_0, gravity_field_columns), axis=1)

#     sigma_mu = 0.05 * 4.89044967462


#     sc_p_0 = np.diag([2**2, 2**2, 2**2, .00003**2, .00003**2, .00003**2])
#     sc_p_0 = np.asarray(sc_p_0, dtype=np.float64)
#     grav_p_0 = np.diag([sigma_mu**2, 9.0e-8**2, 9.0e-8**2, 9.0e-8**2, 9.0e-8**2, 9.0e-8**2])
#     I = np.eye(3)

#     p_single = np.asarray(block_diag(sc_p_0, grav_p_0), dtype=np.float64)
#     p_0 = np.repeat(p_single[None], n_craft, axis=0)
#     Q0 = np.zeros((12, 12))
#     Q0[:3, :3] = 1e-7**2 * np.eye(3)
#     q = np.repeat(Q0[None], n_craft, axis=0)
#     dt = 10


#     dict = pysim_runner(
#         epoch_0, truth_x_0, est_x_0, p_0, q, harmonics, (0.0, 200000), 10, 10, True, 42
#     )

#     plots = PlotConfig(length_unit="m")
#     t = dict["t"]                        # (T,)
#     states = dict["truth"]               # (T, n_craft, 6)
#     estimated_states = dict["est"]       # (T, n_craft, 6 + n_grav), post-update
#     covariance = dict["cov_full"]        # (2T, n_craft, n, n), interleaved [predict, update, ...]


#     plot_orbit_3d(states, (10, 8), plots)
#     n_craft = states.shape[1]
#     grav_units = ["m³/s²"] + ["-"] * (n_grav - 1)
#     for c in range(n_craft):
#         plot_state_history(t, states, estimated_states[..., :6], plots, craft=c)
#         plot_state_sigma(t, states, estimated_states, covariance, plots, (12, 8), craft=c)
#         plot_param_sigma(
#             t, estimated_states, covariance,
#             truth_params=true_gravity_field, labels = plots.grav_state_names, units=grav_units,
#             fig_size=(12, 8), craft=c,
#         )

#     plt.show()


#     with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["t_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"])
#         for t_i, row in zip(t, states):
#             writer.writerow([t_i, *row])

#     print(f"Wrote {len(t)} rows to {OUTPUT_CSV}")
