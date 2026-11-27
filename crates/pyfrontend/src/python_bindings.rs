use hifitime::Epoch;
use numpy::ndarray::{ArrayD, Axis, IxDyn};
use numpy::{
    IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};

use dynamics::gravity::spherical_harmonics::HarmonicCoeffs;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::str::FromStr;

use nalgebra::SVector;

use onboard_software::config::{EST_STATES, EstimatorMatrix, EstimatorVector};
use simulation::plant::{NX, StateVector};
use simulation::stepper::sim_stepper;

fn to_est_matrix(name: &str, array: &PyReadonlyArray3<f64>) -> PyResult<Vec<EstimatorMatrix>> {
    let shape = array.shape();
    if shape[1] != EST_STATES || shape[2] != EST_STATES {
        return Err(PyValueError::new_err(format!(
            "{name} must be {EST_STATES}x{EST_STATES}, got {}x{}",
            shape[1], shape[2]
        )));
    }
    Ok(array
        .as_array()
        .axis_iter(Axis(0))
        .map(|m| {
            let flat: Vec<f64> = m.iter().copied().collect(); // row-major
            EstimatorMatrix::from_row_slice(&flat)
        })
        .collect())
}

fn flatten_vectors<const N: usize>(v: &[SVector<f64, N>]) -> Vec<f64> {
    v.iter()
        .flat_map(|x| x.as_slice().iter().copied())
        .collect()
}

/// Since nalgebra is column-major, transpose before sending it to numpy
fn flatten_matrices(m: &[EstimatorMatrix]) -> Vec<f64> {
    let mut out = Vec::with_capacity(m.len() * EST_STATES * EST_STATES);
    for p in m {
        for i in 0..EST_STATES {
            for j in 0..EST_STATES {
                out.push(p[(i, j)]);
            }
        }
    }
    out
}

fn to_py<'py>(
    py: Python<'py>,
    name: &str,
    shape: &[usize],
    data: Vec<f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let arr = ArrayD::from_shape_vec(IxDyn(shape), data)
        .map_err(|e| PyValueError::new_err(format!("bad {name} shape {shape:?}: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// This function will return a time array, simulation truth state vector array, and estimator array
#[pyfunction]
#[pyo3(signature = (epoch_0, true_x_0, est_x_0, p_0, q, truth_gravity_field, t_span, dt, save_every, full_covariance_flag, seed))]
pub fn pysim_runner<'py>(
    py: Python<'py>,
    epoch_0: &str,
    true_x_0: PyReadonlyArray2<'py, f64>,
    est_x_0: PyReadonlyArray2<'py, f64>,
    p_0: PyReadonlyArray3<'py, f64>,
    q: PyReadonlyArray3<'py, f64>,
    truth_gravity_field: PyReadonlyArray1<'py, f64>,
    t_span: [f64; 2],
    dt: f64,
    save_every: f64,
    full_covariance_flag: bool,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
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
        .map(|row| EstimatorVector::from_row_slice(&row.to_vec()))
        .collect();

    let p_0 = to_est_matrix("p_0", &p_0)?;
    let q = to_est_matrix("q", &q)?;

    let truth_gravity_field: &[f64] = truth_gravity_field.as_slice()?;

    // Estimator should eat the initial state estimate, covariance, and process noise

    // Convert the &str from python into an hifitime::Epoch type
    let epoch_0 = Epoch::from_str(epoch_0)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid epoch: {e}")))?;

    // Load the harmonic coefficients from a csv
    let (mu, harmonics) = HarmonicCoeffs::from_gravity_vector(truth_gravity_field);

    // run the stepper
    let history = sim_stepper(
        epoch_0,
        true_x_0,
        est_x_0,
        p_0,
        q,
        t_span,
        dt,
        mu,
        harmonics,
        save_every,
        full_covariance_flag,
        seed,
    );

    let n_steps = history.t.len();
    let n_snapshots = history.snap_idx.len();
    let n_cov_full = if n_craft > 0 {
        history.cov_full.len() / n_craft
    } else {
        0
    };
    // Take all returned items and put them in a python dictionary for further analysis in Python

    let history_dict = PyDict::new(py);
    history_dict.set_item("t", history.t.into_pyarray(py))?;
    history_dict.set_item(
        "truth",
        to_py(
            py,
            "truth",
            &[n_steps, n_craft, NX],
            flatten_vectors(&history.truth),
        )?,
    )?;
    history_dict.set_item(
        "estimates",
        to_py(
            py,
            "estimates",
            &[n_steps, n_craft, EST_STATES],
            flatten_vectors(&history.est),
        )?,
    )?;
    history_dict.set_item(
        "var",
        to_py(
            py,
            "var",
            &[n_steps, n_craft, EST_STATES],
            flatten_vectors(&history.var),
        )?,
    )?;
    history_dict.set_item(
        "nees_sc",
        to_py(py, "nees_sc", &[n_steps, n_craft], history.nees_sc)?,
    )?;
    history_dict.set_item(
        "nees_grav_par",
        to_py(py, "nees_par", &[n_steps, n_craft], history.nees_grav_par)?,
    )?;
    history_dict.set_item(
        "nees_full",
        to_py(
            py,
            "nees_full",
            &[n_steps, n_craft],
            history.nees_full_matrix,
        )?,
    )?;
    history_dict.set_item(
        "nis_pos_vel",
        to_py(py, "nis_pos_vel", &[n_steps, n_craft], history.nis_pos_vel)?,
    )?;
    history_dict.set_item("snap_idx", history.snap_idx.clone().into_pyarray(py))?;
    history_dict.set_item(
        "cov_snap",
        to_py(
            py,
            "cov_snap",
            &[n_snapshots, n_craft, EST_STATES, EST_STATES],
            flatten_matrices(&history.cov_snap),
        )?,
    )?;
    history_dict.set_item(
        "cov_full",
        to_py(
            py,
            "cov_full",
            &[n_cov_full, n_craft, EST_STATES, EST_STATES],
            flatten_matrices(&history.cov_full),
        )?,
    )?;

    Ok(history_dict)
}
