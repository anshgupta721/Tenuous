use hifitime::Epoch;
use numpy::ndarray::{Array3, Array4, Axis};
use numpy::{
    IntoPyArray, PyArray1, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods, PyReadonlyArray3
};

use dynamics::gravity::spherical_harmonics::HarmonicCoeffs;

use onboard_software::swarm_navigation::SwarmNavigation;
use onboard_software::independent_swarm::IndependentSwarm;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::str::FromStr;


use onboard_software::config::{EST_STATES, EstimatorMatrix, EstimatorVector};
use onboard_software::ekf::BaseEKF;
use simulation::plant::{StateVector, NX};
use simulation::stepper::sim_stepper;

fn to_est_matrix(name: &str, array: &PyReadonlyArray3<f64>) -> PyResult<Vec<EstimatorMatrix>> {
    let shape = array.shape();
    if shape[1] != EST_STATES || shape[2] != EST_STATES {
        return Err(PyValueError::new_err(format!(
            "{name} must be {EST_STATES}x{EST_STATES}, got {}x{}",
            shape[1], shape[2]
        )));
    }
    Ok(array.as_array()
        .axis_iter(Axis(0))
        .map(|m| {
            let flat: Vec<f64> = m.iter().copied().collect(); // row-major
            EstimatorMatrix::from_row_slice(&flat)
        })
        .collect())
}

/// This function will return a time array, simulation truth state vector array, and estimator array
#[pyfunction]
#[pyo3(signature = (epoch_0, true_x_0, est_x_0, p_0, q, harmonics, t_span, dt, seed))]
pub fn pysim_runner<'py>(
    py: Python<'py>,
    epoch_0: &str,
    true_x_0: PyReadonlyArray2<'py, f64>, 
    est_x_0: PyReadonlyArray2<'py, f64>,
    p_0: PyReadonlyArray3<'py, f64>,
    q: PyReadonlyArray3<'py, f64>,
    harmonics: &str,
    t_span: [f64; 2],
    dt: f64,
    seed: u64,
) -> PyResult<(
    Py<PyArray1<f64>>, // Time
    Py<PyArray3<f64>>, // Truth 
    Py<PyArray3<f64>>, // Estimated Mean
    Py<PyArray4<f64>>, // Estimated Covariance
)> {
    let n_craft = true_x_0.shape()[0];

    let shape_true_x0 = true_x_0.shape();
    if shape_true_x0[1] != NX {
        return Err(PyValueError::new_err(format!(
            "true_x_0 must have shape (n_craft, {NX}), got ({}, {})",
            shape_true_x0[0], shape_true_x0[1]
        )));
    }
    let true_x_0: Vec<StateVector> = true_x_0
        .as_array()
        .axis_iter(Axis(0))
        .map(|row| StateVector::from_row_slice(&row.to_vec()))
        .collect();


    let shape_est_x0 = est_x_0.shape();
    if shape_est_x0[1] != EST_STATES {
        return Err(PyValueError::new_err(format!(
            "est_x_0 must have shape (n_craft, {EST_STATES}), got ({}, {})",
            shape_est_x0[0], shape_est_x0[1]
        )));
    }
    let est_x_0: Vec<EstimatorVector> = est_x_0
        .as_array()
        .axis_iter(Axis(0))
        .map(|row| StateVector::from_row_slice(&row.to_vec()))
        .collect();

    let p_0 = to_est_matrix("p_0", &p_0)?;
    let q = to_est_matrix("q", &q)?;


    // Estimator should eat the initial state estimate, covariance, and process noise

    // Convert the &str from python into an hifitime::Epoch type
    let epoch_0 = Epoch::from_str(epoch_0)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid epoch: {e}")))?;

    // Load the harmonic coefficients from a csv
    let harmonics = HarmonicCoeffs::from_csv(harmonics, Some(10))
        .map_err(|e| PyValueError::new_err(format!("failed to load harmonic coefficients: {e}")))?;
    // run the stepper
    let history =
        sim_stepper(epoch_0, true_x_0, est_x_0, p_0, q, t_span, dt, harmonics, seed);
    let n_steps = history.t.len();   
    let t_arr = history.t.into_pyarray(py).into();
    // Take t_hist vector and vectors of nalgebra arrays (x_hist and u_hist) and convert them to numpy arrays
    let x_flat: Vec<f64> = history.truth
        .iter()
        .flat_map(|x| x.as_slice().iter().copied())
        .collect();
    let x_arr = Array3::from_shape_vec((n_steps, n_craft, NX), x_flat)
    .map_err(|e| PyValueError::new_err(format!("bad truth shape: {e}")))?
    .into_pyarray(py)
    .into();



    let n_est = history.est.len();
    let est_flat: Vec<f64> = history.est
        .iter()
        .flat_map(|x| x.as_slice().iter().copied())
        .collect();
    let est_arr = Array3::from_shape_vec((n_steps, n_craft, EST_STATES), est_flat)
    .map_err(|e| PyValueError::new_err(format!("bad estimate shape: {e}")))?
    .into_pyarray(py)
    .into();

    let est_cov_flat: Vec<f64> = history.cov
        .iter()
        .flat_map(|p| p.transpose().as_slice().to_vec())
        .collect();
    let est_cov_arr = Array4::from_shape_vec((n_steps * 2, n_craft, EST_STATES, EST_STATES), est_cov_flat)
    .map_err(|e| PyValueError::new_err(format!("bad covariance shape: {e}")))?
    .into_pyarray(py)
    .into();

    Ok((t_arr, x_arr, est_arr, est_cov_arr))
}

// #[pyfunction]
// #[pyo3(signature = (x_0, u_0, t_span, dt, n_runs, x0_std, process_noise_std, seed=None))]
// fn monte_carlo_sim(
//     py: Python<'py>,
//     x_0: [f64; NX],
// )
