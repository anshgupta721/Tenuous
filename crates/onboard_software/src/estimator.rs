use crate::config::{EST_STATES, EstimatorVector, MEASUREMENTS, Navigation, SensorVector};
use nalgebra::{SMatrix, SVector};

// use dynamics::models::state_space_model::{LTVSystem, StateSpace};

pub struct Estimator {
    x: EstimatorVector,
    p_own: SMatrix<f64, EST_STATES, EST_STATES>,
    // p_indep_theta: SMatrix<f64, EST_STATES, EST_STATES>,
    // p_correl_theta: SMatrix<f64, EST_STATES, EST_STATES>,
    process_noise: SMatrix<f64, EST_STATES, EST_STATES>,
    p_vv: SMatrix<f64, MEASUREMENTS, MEASUREMENTS>,
    nav: Navigation,
}

impl Estimator {
    pub fn new(
        initial_state_guess: EstimatorVector,
        initial_state_covariance: SMatrix<f64, EST_STATES, EST_STATES>,
    ) -> Estimator {
        Estimator {
            x: EstimatorVector::zeros(),
            p_own: SMatrix::<f64, EST_STATES, EST_STATES>::zeros(),
            process_noise: SMatrix::<f64, EST_STATES, EST_STATES>::zeros(),
            p_vv: SMatrix::<f64, MEASUREMENTS, MEASUREMENTS>::zeros(),
            nav: Navigation,
        }
    }

    pub fn initialize(&mut self, sensor_data: SensorVector) {
        self.x = sensor_data
    }

    pub fn get_estimate(&self) -> EstimatorVector {
        self.x
    }

    pub fn get_covariance(&self) -> SMatrix<f64, EST_STATES, EST_STATES> {
        self.p_own
    }

    pub fn predict(&mut self, t: f64, dt: f64) {
        let m_dot = self.nav.system_dynamics(t, self.x);
        self.x += m_dot * dt;
    }

    pub fn update(&mut self, sensor_data: SensorVector) {}
}
