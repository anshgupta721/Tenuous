use nalgebra::{SMatrix, SVector};

use dynamics::models::state_space_model::{LTVSystem, StateSpace};

pub const EST_STATES: usize = 6;
// pub const NU: usize = 3;

// pub type ControlVector = SVector<f64, NU>;
pub type EstimatorVector = SVector<f64, EST_STATES>;
pub type SensorVector = SVector<f64, 6>;
// Estimator system model
pub struct navigation;
impl navigation {
    fn a(&self, t: f64) -> SMatrix<f64, EST_STATES, EST_STATES> {
        // nonlinear equations of motion

        SMatrix::<f64, EST_STATES, EST_STATES>::zeros()
    }
    fn b(&self, t: f64) -> SMatrix<f64, EST_STATES, 1> {
        SMatrix::<f64, EST_STATES, 1>::zeros()
    }
}
