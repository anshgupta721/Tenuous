use nalgebra::{SMatrix, SVector};

use crate::config::{EST_STATES, EstimatorVector};

pub trait MeasurementModel<const M: usize> {
    /// h(x) function
    fn predict(&self, m_x: &EstimatorVector) -> SVector<f64, M>;
    /// H function
    fn jacobian(&self, _m_x: &EstimatorVector) -> SMatrix<f64, M, EST_STATES>;
    /// R matrix
    fn noise(&self, _m_x: &EstimatorVector) -> SMatrix<f64, M, M>;
}

pub struct LinearMeasurement<const M: usize> {
    h: SMatrix<f64, M, EST_STATES>,
    r: SMatrix<f64, M, M>,
}

impl<const M: usize> MeasurementModel<M> for LinearMeasurement<M> {
    fn predict(&self, m_x: &EstimatorVector) -> SVector<f64, M> {
        self.h * m_x
    }

    fn jacobian(&self, _m_x: &EstimatorVector) -> SMatrix<f64, M, EST_STATES> {
        self.h
    }

    fn noise(&self, _m_x: &EstimatorVector) -> SMatrix<f64, M, M> {
        self.r
    }
}

// Convenience constructor for a 6-measurement [pos, vel] sensor
impl LinearMeasurement<6> {
    pub fn position_velocity(pos_sigma: f64, vel_sigma: f64) -> Self {
        let mut h = SMatrix::<f64, 6, EST_STATES>::zeros();
        for i in 0..6 {
            h[(i, i)] = 1.0; // measurement i reads state i directly
        }

        let mut r = SMatrix::<f64, 6, 6>::zeros();
        for i in 0..3 {
            r[(i, i)] = pos_sigma * pos_sigma;
            r[(i + 3, i + 3)] = vel_sigma * vel_sigma;
        }

        Self { h, r }
    }
}
