use dynamics::gravity::gravity::SphericalHarmonics;
use dynamics::gravity::spherical_harmonics::HarmonicCoeffs;
use dynamics::integrators::rk4::rk4;
use nalgebra::{SMatrix, SVector};

use anise::frames::Frame;
use anise::prelude::*;

use crate::config::{
    AugmentedVector, EST_MAT_LENGTH, EST_STATES, EstimatorMatrix, EstimatorVector, FIXED_FRAME,
    INER_FRAME, REF_RADIUS_SHM, SC_STATES, SHM_DEGREE, SHM_PARAMS,
};
use crate::filters::filter::{Context, Filter};
use crate::measurements::MeasurementModel;
use crate::propagator::GravityDynamics;

// MeanEstimate = [x, y, z, vx, vy, vz, theta]
// theta = [mu, c20, c21, s21, c22, s22..]
pub struct BaseEKF {
    mx: EstimatorVector,
    px: EstimatorMatrix,
    process_noise: EstimatorMatrix,
    // h: SMatrix<f64, MEASUREMENTS, EST_STATES>,
    dynamics: GravityDynamics,
}

impl BaseEKF {
    pub fn get_estimate(&self) -> EstimatorVector {
        self.mx
    }

    pub fn get_covariance(&self) -> EstimatorMatrix {
        self.px
    }

    fn process_noise_jacobian(&self) -> EstimatorMatrix {
        //  [1 0 0]
        //  [0 1 0]  *  dt (process noise scaled by dt)
        //  [0 0 1]

        let mut fw = EstimatorMatrix::zeros();
        fw.fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&SMatrix::<f64, 3, 3>::identity());
        fw
    }
}

impl Filter for BaseEKF {
    fn new(
        initial_state_guess: EstimatorVector,
        initial_state_covariance: EstimatorMatrix,
        process_noise: EstimatorMatrix,
    ) -> Self {
        BaseEKF {
            mx: initial_state_guess,
            px: initial_state_covariance,
            process_noise: process_noise,
            dynamics: GravityDynamics::new(),
        }
    }

    fn predict(&mut self, epoch: Epoch, dt: f64, almanac: &Almanac) {
        // Propagate mean estimate and state transition matrix forward using gravity dynamics
        let (mx_next, phi) = self.dynamics.propagate(&self.mx, epoch, dt, almanac);
        let noise_jacobian = self.process_noise_jacobian();
        let discrete_process_noise =
            noise_jacobian * self.process_noise * noise_jacobian.transpose() * dt; // Potentially upgrade to rk4
        self.mx = mx_next;
        self.px = phi * self.px * phi.transpose() + discrete_process_noise;
    }

    fn update<const M: usize, Meas: MeasurementModel<M>>(
        &mut self,
        meas_model: &Meas,
        z: &SVector<f64, M>,
    ) -> SVector<f64, M> {
        // mean estimate transformed from state domain into measurement domain
        let mz = meas_model.predict(&self.mx);
        let innov = z - mz;
        // H_x calculated from measurement model
        let observation_jacobian = meas_model.jacobian(&self.mx);
        // Measurement noise matrix assigned
        let r = meas_model.noise(&self.mx);
        // Cross covariance calculated: P_{xz,k}^- = P_{xx,k}^i * H_x^T
        let pxz = self.px * observation_jacobian.transpose(); // m x n matrix
        let pzz = observation_jacobian * self.px * observation_jacobian.transpose() + r; // R should eventually change to noise jacobian when noise isn't additive
        let kalman_gain = pxz * pzz.try_inverse().expect("P_zz is singular");
        self.mx = self.mx + kalman_gain * (z - mz);
        self.px = self.px - pxz * kalman_gain.transpose() - kalman_gain * pxz.transpose()
            + kalman_gain * pzz * kalman_gain.transpose();

        innov
    }

    fn set_estimate(&mut self, x: EstimatorVector, p: EstimatorMatrix) {
        self.mx = x;
        self.px = p;
    }

    fn estimate(&self) -> EstimatorVector {
        BaseEKF::get_estimate(&self)
    }
    fn covariance(&self) -> EstimatorMatrix {
        BaseEKF::get_covariance(&self)
    }
}
