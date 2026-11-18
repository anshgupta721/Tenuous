use anise::frames::Frame;
use anise::prelude::*;
use nalgebra::{DMatrix, DVector, SMatrix, SVector};

use dynamics::gravity::gravity::SphericalHarmonics;
use dynamics::gravity::spherical_harmonics::HarmonicCoeffs;

use crate::config::{
    AugmentedVector, EST_MAT_LENGTH, EST_STATES, EstimatorMatrix, EstimatorVector, FIXED_FRAME,
    GRAV_STATES, INER_FRAME, REF_RADIUS_SHM, SC_STATES, SHM_DEGREE,
};
use crate::measurements::MeasurementModel;
use crate::propagator::GravityDynamics;

fn craft_index(i: usize) -> usize {
    GRAV_STATES + SC_STATES * i
}

// MeanEstimate = [theta, spacecraft1, spacecraft2..]
// theta = [mu, c10, c11, s11, c20, c21, s21, c22, s22..]
// spacecraftn = [x, y, z, vx, vy, vz]
pub struct CentralizedEKF {
    mx: DVector<f64>,
    px: DMatrix<f64>,
    process_noise: Vec<SMatrix<f64, SC_STATES, SC_STATES>>,
    dynamics: GravityDynamics,
}

impl CentralizedEKF {
    fn n_craft(&self) -> usize {
        (self.mx.len() - GRAV_STATES) / SC_STATES
    }

    /// Takes in the craft number and outputs the estimator vector
    pub fn local_state(&self, i: usize) -> EstimatorVector {
        let mut vec = EstimatorVector::zeros();
        vec.fixed_rows_mut::<SC_STATES>(0)
            .copy_from(&self.mx.rows(craft_index(i), SC_STATES));
        vec.fixed_rows_mut::<GRAV_STATES>(SC_STATES)
            .copy_from(&self.mx.rows(0, GRAV_STATES));
        vec
    }

    /// Find the local covariance matrix from the large px
    pub fn local_covariance(&self, i: usize) -> EstimatorMatrix {
        let sc_state_index = craft_index(i);

        // let map: std::iter::Chain<std::ops::Range<usize>, _> = (sc_state_index..(sc_state_index + SC_STATES)).chain(other)

        let mut local_covariance = EstimatorMatrix::zeros();

        local_covariance.view_mut((0,0), (SC_STATES, SC_STATES)).copy_from(&self.px.fixed_view::<SC_STATES, SC_STATES>(sc_state_index, sc_state_index));
        local_covariance.view_mut((SC_STATES, SC_STATES), (GRAV_STATES, GRAV_STATES)).copy_from(&self.px.fixed_view::<GRAV_STATES, GRAV_STATES>(0, 0));

        local_covariance.view_mut((0, SC_STATES), (SC_STATES, GRAV_STATES)).copy_from(&self.px.fixed_view::<SC_STATES, GRAV_STATES>(sc_state_index, 0));
        local_covariance.view_mut((SC_STATES, 0), (GRAV_STATES, SC_STATES)).copy_from(&self.px.fixed_view::<GRAV_STATES, SC_STATES>(0, sc_state_index));

        local_covariance
    }

    pub fn get_estimate(&self) -> Vec<EstimatorVector> {
        
        (0..self.n_craft()).map(|i| self.local_state(i)).collect()
    }

    pub fn get_covariance(&self) -> Vec<EstimatorMatrix> {
        (0..self.n_craft()).map(|i| self.local_covariance(i)).collect()
    }

    fn process_noise_jacobian(&self) -> SMatrix<f64, SC_STATES, SC_STATES> {
        //  [1 0 0]
        //  [0 1 0]  *  dt (process noise scaled by dt)
        //  [0 0 1]

        let mut fw = SMatrix::<f64, SC_STATES, SC_STATES>::zeros();
        fw.fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&SMatrix::<f64, 3, 3>::identity());
        fw
    }

    pub fn new(
        initial_state_guess: DVector<f64>,
        initial_state_covariance: DMatrix<f64>,
        process_noise: Vec<SMatrix<f64, SC_STATES, SC_STATES>>,
    ) -> Self {
        Self {
            mx: initial_state_guess,
            px: initial_state_covariance,
            process_noise: process_noise,
            dynamics: GravityDynamics::new(),
        }
    }

    pub fn predict(&mut self, epoch: Epoch, dt: f64, almanac: &Almanac) {
        let n = self.mx.len();
        // instantiate the state transition matrix and discrete process noise matrices
        let mut phi = DMatrix::<f64>::identity(n, n);
        let mut discrete_process_noise = DMatrix::<f64>::zeros(n, n);

        for i in 0..self.n_craft() {
            let sc_state_index = craft_index(i);
            // Propagate each spacecraft forward then set up new mx and px
            let (next, local_phi) =
                self.dynamics
                    .propagate(&self.local_state(sc_state_index), epoch, dt, almanac);

            self.mx
                .rows_mut(sc_state_index, SC_STATES)
                .copy_from(&next.fixed_rows::<SC_STATES>(0));

            phi.view_mut((sc_state_index, sc_state_index), (SC_STATES, SC_STATES))
                .copy_from(&local_phi.fixed_view::<SC_STATES, SC_STATES>(0, 0));

            phi.view_mut((sc_state_index, 0), (SC_STATES, GRAV_STATES))
                .copy_from(&local_phi.fixed_view::<SC_STATES, GRAV_STATES>(0, SC_STATES));

            let process_noise = self.process_noise[i];
            let noise_jacobian = self.process_noise_jacobian();
            let local_qd = noise_jacobian * process_noise * noise_jacobian.transpose() * dt;

            discrete_process_noise
                .view_mut((sc_state_index, sc_state_index), (SC_STATES, SC_STATES))
                .copy_from(&local_qd);
        }

        self.px = &phi * &self.px * &phi.transpose() + discrete_process_noise;
    }

    fn local_correct(
        &mut self,
        h: &DMatrix<f64>,     // M x STATES
        innov: &DVector<f64>, // M
        r: &DMatrix<f64>,
    ) {
        let pxz = &self.px * h.transpose();
        let pzz = h * &self.px * h.transpose() + r;
        let kalman_gain = &pxz * pzz.clone().try_inverse().expect("P_zz is singular");
        self.mx = &self.mx + &kalman_gain * innov;
        self.px = &self.px - &pxz * &kalman_gain.transpose() - &kalman_gain * &pxz.transpose()
            + &kalman_gain * &pzz * &kalman_gain.transpose();
    }

    pub fn correct<const M: usize, Meas: MeasurementModel<M>>(
        &mut self,
        meas_model: &Meas,
        z: &[SVector<f64, M>],
    ) {
        // process the measurements of every spacecraft and calculate cross and innovation covariance
        let states = self.mx.len();
        for i in 0..self.n_craft() {
            let local_state = self.local_state(i);

            let mz_local = meas_model.predict(&local_state);
            // Observation jacobian
            let h_local = meas_model.jacobian(&local_state);
            let r = meas_model.noise(&local_state);
            let innov = z[i] - meas_model.predict(&local_state);
            let sc_state_index = craft_index(i);
            let row = M * i;
            let mut h = DMatrix::<f64>::zeros(M, states);

            h.view_mut((row, sc_state_index), (M, states))
                .copy_from(&h_local.fixed_columns::<SC_STATES>(0));
            h.view_mut((0, 0), (M, GRAV_STATES))
                .copy_from(&h_local.fixed_columns::<GRAV_STATES>(SC_STATES));

            self.local_correct(
                &h,
                &DVector::from_column_slice(innov.as_slice()),
                &DMatrix::from_column_slice(M, M, r.as_slice()),
            );
        }
    }
}
