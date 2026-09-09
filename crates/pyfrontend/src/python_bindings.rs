use hifitime::Epoch;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use std::str::FromStr;

use onboard_software::config::{ControlVector, EST_STATES, NU};
use onboard_software::control_stack::ControlStack;
use simulation::config::{NX, StateVector};
use simulation::stepper::sim_stepper;

// struct TrialResult {
//     t: Vec<f64>,
//     x: Vec<StateVector>,
//     u: Vec<ControlVector>,
// }

#[pyfunction]
#[pyo3(signature = (epoch_0, x_0, t_span, dt, seed=None))]
pub fn pysim_runner<'py>(
    py: Python<'py>,
    epoch_0: &str,
    x_0: PyReadonlyArray1<'py, f64>,
    // u_0: PyReadonlyArray1<'py, f64>,
    t_span: [f64; 2],
    dt: f64,
    seed: Option<u64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let x_0_slice = x_0.as_slice()?;
    if x_0_slice.len() != NX {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x_0 must contain {NX} elements, got {}",
            x_0_slice.len()
        )));
    }
    let x_0 = StateVector::from_row_slice(x_0_slice);
    // let u_0 = ControlVector::from_row_slice(u_0.as_slice()?);
    // Instantiate the control stack
    let gnc: ControlStack = ControlStack::new(/*Configure the estimator here in future*/);

    // Convert the &str from python into an hifitime::Epoch type
    let epoch_0 = Epoch::from_str(epoch_0)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid epoch: {e}")))?;

    // run the stepper
    let (t_hist, x_hist, est_hist) = sim_stepper(epoch_0, x_0, t_span, dt, gnc);
    let t_arr = t_hist.into_pyarray(py).into();
    // Take t_hist vector and vectors of nalgebra arrays (x_hist and u_hist) and convert them to numpy arrays
    let n1 = x_hist.len();
    let x_flat: Vec<f64> = x_hist
        .iter()
        .flat_map(|x| x.as_slice().iter().copied())
        .collect();
    let x_arr = Array2::from_shape_vec((n1, NX), x_flat)
        .unwrap()
        .into_pyarray(py)
        .into();

    // let n2 = u_hist.len();
    // let u_flat: Vec<f64> = u_hist
    //     .iter()
    //     .flat_map(|x| x.as_slice().iter().copied())
    //     .collect();
    // let u_arr = Array2::from_shape_vec((n2, NU), u_flat)
    //     .unwrap()
    //     .into_pyarray(py)
    //     .into();

    let n3 = est_hist.len();
    let est_flat: Vec<f64> = est_hist
        .iter()
        .flat_map(|x| x.as_slice().iter().copied())
        .collect();
    let est_arr = Array2::from_shape_vec((n3, EST_STATES), est_flat)
        .unwrap()
        .into_pyarray(py)
        .into();

    Ok((t_arr, x_arr, est_arr))
}

// #[pyfunction]
// #[pyo3(signature = (x_0, u_0, t_span, dt, n_runs, x0_std, process_noise_std, seed=None))]
// fn monte_carlo_sim(
//     py: Python<'py>,
//     x_0: [f64; NX],
//     u_0: [f64]
// )
