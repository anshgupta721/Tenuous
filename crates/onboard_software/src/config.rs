use nalgebra::{SMatrix, SVector};

use dynamics::models::state_space_model::{LTVSystem, StateSpace};

pub const EST_STATES: usize = 6;
pub const MEASUREMENTS: usize = 6;
// pub const THETA_STATES: usize = 121;
// pub const NU: usize = 3;

// pub type ControlVector = SVector<f64, NU>;
pub type EstimatorVector = SVector<f64, EST_STATES>;
pub type SensorVector = SVector<f64, MEASUREMENTS>;
// Estimator system model
pub struct Navigation;
impl Navigation {
    // The dynamics here should output the jacobian of the state vecotr
    pub fn system_dynamics(&self, t: f64, m: EstimatorVector) -> EstimatorVector {
        // propagation of values
        EstimatorVector::zeros()
    }

    pub fn system_jacobian(
        &self,
        t: f64,
        m: EstimatorVector,
    ) -> SMatrix<f64, EST_STATES, EST_STATES> {
        // jacobian of f with respect to state vector
        SMatrix::<f64, EST_STATES, EST_STATES>::zeros()
    }

    // Maps internal state space into the expected measurement values
    pub fn observation(&self, mean: EstimatorVector) -> SensorVector {
        SensorVector::zeros()
    }

    // Linearizes the non-linear measurement function around the current predicted state
    pub fn observation_jacobian() -> SMatrix<f64, MEASUREMENTS, EST_STATES> {
        SMatrix::<f64, MEASUREMENTS, EST_STATES>::zeros()
    }
}
