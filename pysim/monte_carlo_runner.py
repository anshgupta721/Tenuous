"""Run the simulation entry point."""

import numpy as np
import matplotlib.pyplot as plt

from config import PlotConfig
from plots.orbit_3d import plot_orbit_3d
from plots.three_sigma import plot_run_overlay
from plots.consistency import plot_chi2
from gravity_field import (
    load_gravity_field,
    load_gravity_field_covariance,
    length_from_degree,
)
from scipy.linalg import block_diag

from pyfrontend import pysim_runner  # pylint: disable=no-name-in-module  # type: ignore[attr-defined]

# OUTPUT_CSV = "sim_output.csv"


def run_monte_carlo(
    n_runs,
    epoch_0,
    x_0,
    p0,
    q0,
    estimated_degree,
    nominal_gravity,
    nominal_gravity_cov,
    t_span,
    dt,
    save_every,
    base_seed=0,
):
    crafts, craft_states = x_0.shape
    results, true_fields = [], []
    nominal_est_gravity = np.repeat(
        nominal_gravity[None, : length_from_degree(estimated_degree) + 1],
        crafts,
        axis=0,
    )
    for run in range(n_runs):
        seed = base_seed + run
        rng = np.random.default_rng(seed)

        # The initial estimate of the gravitational field should be the nominal field
        # True gravitational field should be generated per run along a gaussian
        # SC
        true_gravity = rng.multivariate_normal(nominal_gravity, nominal_gravity_cov)

        sc_noise = np.array(
            [
                rng.multivariate_normal(
                    np.zeros(craft_states), p0[craft, :craft_states, :craft_states]
                )
                for craft in range(crafts)
            ]
        )
        est_x_0 = np.concatenate((x_0 + sc_noise, nominal_est_gravity), axis=1)

        results.append(
            pysim_runner(
                epoch_0,
                x_0,
                est_x_0,
                p0,
                q0,
                true_gravity,
                t_span,
                dt,
                save_every,
                True,
                seed,
            )
        )
        true_fields.append(true_gravity)

    return results, true_fields


if __name__ == "__main__":
    epoch_0 = "2019-07-13T11:23:27.30 UTC"
    MEAN_COEFFS_STR = "pysim/data/chesley_coeffs.csv"
    COVAR_COEFFS_STR = "pysim/data/chesley_coeffs_covariance.csv"
    truth_degree = 10
    estimated_degree = 2
    sc_states = 6
    truth_x_0 = np.array(
        [
            [1282.37, 0.0, 0.0, 0.0, 0.0617543738092, 0.0],
            [0.0, 1282.37, 0.0, -0.0617543738092, 0.0, 0.0],
        ]
    )
    truth_x_0 = np.atleast_2d(np.asarray(truth_x_0, dtype=np.float64))

    # Extract gravitational mean values and covariance
    mu = 4.89044967462
    nominal_gravity_field = np.concatenate(
        ([mu], load_gravity_field(MEAN_COEFFS_STR, truth_degree))
    )
    truth_gravity_cov = load_gravity_field_covariance(COVAR_COEFFS_STR, truth_degree)
    CRAFT_STATES = truth_x_0.shape[1]
    EST_GRAV_STATES = (
        length_from_degree(estimated_degree) + 1
    )  # Plus 1 to account for Mu
    CRAFTS = truth_x_0.shape[0]
    N_STATES = CRAFT_STATES + EST_GRAV_STATES

    gravity_p_0 = truth_gravity_cov[:EST_GRAV_STATES, :EST_GRAV_STATES]

    sc_p_0 = np.diag(
        [2**2, 2**2, 2**2, 0.00003**2, 0.00003**2, 0.00003**2]
    )  # defined variances for SC states
    sc_p_0 = np.asarray(sc_p_0, dtype=np.float64)
    I = np.eye(3)

    p_single = np.asarray(block_diag(sc_p_0, gravity_p_0), dtype=np.float64)

    # print(p_single)

    p_0 = np.repeat(p_single[None], CRAFTS, axis=0)
    Q0 = np.zeros((12, 12))
    Q0[:3, :3] = 1e-7**2 * np.eye(3)
    q = np.repeat(Q0[None], CRAFTS, axis=0)

    dt = 360
    save_every = 360
    t_span = (0.0, 5000000)
    seed = 42

    n_runs = 1
    runs_to_plot = 1

    dicts, true_fields = run_monte_carlo(
        n_runs,
        epoch_0,
        truth_x_0,
        p_0,
        q,
        estimated_degree,
        nominal_gravity_field,
        truth_gravity_cov,
        t_span,
        dt,
        save_every,
        seed,
    )


    for key in ("nees_full", "nees_sc", "nees_grav_par", "nis_pos_vel"):
        a = np.asarray(dicts[0][key])            # (T, n_craft)
        bad = np.where(np.isnan(a).any(axis=1))[0]
        print(key, "first NaN step:", bad[0] if len(bad) else None, "of", a.shape[0])
    # take a few random runs and put them all on one plot rather than individual runs to make sure all converge
    t = []
    truth_states = []
    estimate_states = []
    variances = []
    spacecraft_nees = []
    gravity_nees = []
    full_nees = []
    snapshot_index = []
    covariance_full = []

    for item in dicts:
        truth_states.append(item["truth"])
        estimate_states.append(item["estimates"])
        variances.append(item["var"])
        spacecraft_nees.append(item["nees_sc"])
        gravity_nees.append(item["nees_grav_par"])
        full_nees.append(item["nees_full"])
        snapshot_index.append(item["snap_idx"])
        covariance_full.append(item["cov_snap"])

    plots = PlotConfig(length_unit="m")
    t = dicts[0]["t"]

    # Plot orbits of one run
    plot_orbit_3d(truth_states[0], (10, 8), plots)
    grav_units = ["m³/s²"] + ["-"] * (EST_GRAV_STATES - 1)

    # Chooses runs_to_plot number of values
    picks = np.sort(
        np.random.default_rng(0).choice(
            n_runs, size=min(runs_to_plot, n_runs), replace=False
        )
    )

    estimate_states_array = np.asarray(estimate_states)
    truth_states_array = np.asarray(truth_states)
    variances_array = np.asarray(variances)
    for craft in range(CRAFTS):
        err_sc = (
            estimate_states_array[picks, :, craft, :CRAFT_STATES]
            - truth_states_array[picks, :, craft, :CRAFT_STATES]
        )
        sig_sc = np.sqrt(variances_array[picks, :, craft, :CRAFT_STATES].mean(axis=0))
        plot_run_overlay(
            t,
            err_sc,
            sig_sc,
            plots.sc_state_names,
            plots.sc_units,
            f"Spacecraft error, craft {craft}",
            (12, 8),
        )

        err_g = (
            estimate_states_array[picks, :, craft, CRAFT_STATES:]
            - np.asarray(true_fields)[picks, None, :EST_GRAV_STATES]
        )
        sig_g = np.sqrt(variances_array[picks, :, craft, 6:].mean(axis=0))
        plot_run_overlay(
            t,
            err_g,
            sig_g,
            plots.grav_state_names,
            grav_units,
            f"Gravity error, craft {craft}",
            (12, 8),
        )
        nees = {
            "spacecraft": np.stack(spacecraft_nees),
            "gravity":    np.stack(gravity_nees),
            "full":       np.stack(full_nees),
        }
        dims = {"spacecraft": CRAFT_STATES, "gravity": EST_GRAV_STATES,
                "full": CRAFT_STATES + EST_GRAV_STATES}
        plot_chi2(t, nees, dims, "NEES")

    plt.show()

    # with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["t_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"])
    #     for t_i, row in zip(t, states):
    #         writer.writerow([t_i, *row])

    # print(f"Wrote {len(t)} rows to {OUTPUT_CSV}")
