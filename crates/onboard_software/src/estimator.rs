use crate::config::{EST_STATES, EstimatorVector, SensorVector};
use nalgebra::{SMatrix, SVector};

// use dynamics::models::state_space_model::{LTVSystem, StateSpace};

pub struct Estimator {
    x: EstimatorVector,
    p: SMatrix<f64, EST_STATES, EST_STATES>,
    process_noise: SMatrix<f64, EST_STATES, EST_STATES>,
}

impl Estimator {
    pub fn new() -> Estimator {
        Estimator {
            x: EstimatorVector::zeros(),
            p: SMatrix::<f64, EST_STATES, EST_STATES>::zeros(),
            process_noise: SMatrix::<f64, EST_STATES, EST_STATES>::zeros(),
        }
    }

    pub fn initialize(&self, sensor_data: SensorVector) {}

    pub fn get_estimate(&self) -> EstimatorVector {
        self.x
    }

    pub fn get_covariance(&self) -> SMatrix<f64, EST_STATES, EST_STATES> {
        self.p
    }

    pub fn predict(&mut self) {}

    pub fn update(&mut self, sensor_data: SensorVector) {}
}
