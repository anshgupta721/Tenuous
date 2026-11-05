use dynamics::gravity::gravity::SphericalHarmonics;
use dynamics::gravity::spherical_harmonics::HarmonicCoeffs;
use dynamics::integrators::rk4::rk4;
use nalgebra::{SMatrix, SVector};

use anise::frames::Frame;
use anise::prelude::*;

use crate::config::{
    AugmentedVector, EST_MAT_LENGTH, EST_STATES, EstimatorMatrix, EstimatorVector, FIXED_FRAME,
    INER_FRAME, MU_BENNU_GUESS, REF_RADIUS_SHM, SHM_DEGREE,
};
use crate::filter::{Context, Filter};
use crate::measurements::MeasurementModel;
// use dynamics::models::state_space_model::{LTVSystem, StateSpace};

fn stack_state_and_stm(
    x: &EstimatorVector,
    phi: &SMatrix<f64, EST_STATES, EST_STATES>,
) -> AugmentedVector {
    let mut vec = AugmentedVector::zeros();
    vec.fixed_rows_mut::<EST_STATES>(0).copy_from(x);
    vec.fixed_rows_mut::<EST_MAT_LENGTH>(EST_STATES)
        .copy_from_slice(phi.as_slice());
    vec
}

fn split_state_and_stm(vec: AugmentedVector) -> (EstimatorVector, EstimatorMatrix) {
    let x = vec.fixed_rows::<EST_STATES>(0).into_owned();
    let phi = SMatrix::<f64, EST_STATES, EST_STATES>::from_column_slice(
        vec.fixed_rows::<EST_MAT_LENGTH>(EST_STATES).as_slice(),
    );
    (x, phi)
}

// MeanEstimate = [x, y, z, vx, vy, vz, mu, c10, ]
pub struct BaseEKF {
    mx: EstimatorVector,
    p_own: EstimatorMatrix,
    // p_indep_theta: SMatrix<f64, EST_STATES, EST_STATES>,
    // p_correl_theta: SMatrix<f64, EST_STATES, EST_STATES>,
    process_noise: EstimatorMatrix,
    // h: SMatrix<f64, MEASUREMENTS, EST_STATES>,
    gravity: SphericalHarmonics,
    inertial_frame: Frame,
    body_fixed_frame: Frame,
}

impl BaseEKF {
    // pub fn initialize(
    //     &mut self,
    //     sensor_data: SensorVector,
    //     covariance0: SMatrix<f64, EST_STATES, EST_STATES>,
    // ) {
    //     self.mx = sensor_data;
    //     self.p_own = covariance0;
    // }

    pub fn get_estimate(&self) -> EstimatorVector {
        self.mx
    }

    pub fn get_covariance(&self) -> SMatrix<f64, EST_STATES, EST_STATES> {
        self.p_own
    }

    fn dynamics_and_jacobian(
        &self,
        x: EstimatorVector,
        epoch: Epoch,
        almanac: &Almanac,
        coeffs: &HarmonicCoeffs,
    ) -> (EstimatorVector, EstimatorMatrix) {
        let r = SVector::<f64, 3>::new(x[0], x[1], x[2]);
        let v = SVector::<f64, 3>::new(x[3], x[4], x[5]);

        let dcm_i2b = almanac
            .rotate(self.inertial_frame, self.body_fixed_frame, epoch)
            .unwrap()
            .rot_mat;

        let r_bf = dcm_i2b * r;

        let (accel, gradient) = self.gravity.acceleration_and_gradient(r_bf, &coeffs);
        let accel_inertial = dcm_i2b.transpose() * accel;
        let gradient_inertial: SMatrix<f64, 3, 3> = dcm_i2b.transpose() * gradient * dcm_i2b;
        // Build vector of r_dot, v_dot, mu_dot, coeffs_dot
        let derivative = EstimatorVector::new(
            v[0],
            v[1],
            v[2],
            accel_inertial[0],
            accel_inertial[1],
            accel_inertial[2],
        );

        //  [dv/dr,

        let mut jacobian = SMatrix::<f64, EST_STATES, EST_STATES>::zeros();

        jacobian
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&SMatrix::<f64, 3, 3>::identity());
        jacobian
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&gradient_inertial);

        // [[0, 0, 0, 0, 0, 0]
        // ]]

        (derivative, jacobian)
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

    pub fn augmented_dynamics(
        &self,
        vec: AugmentedVector,
        epoch: Epoch,
        almanac: &Almanac,
        coeffs: &HarmonicCoeffs,
    ) -> AugmentedVector {
        let (x, phi) = split_state_and_stm(vec);
        let (x_dot, jac) = self.dynamics_and_jacobian(x, epoch, almanac, coeffs);
        stack_state_and_stm(&x_dot, &(jac * phi))
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
            p_own: initial_state_covariance,
            process_noise: process_noise,
            gravity: SphericalHarmonics::new(MU_BENNU_GUESS, REF_RADIUS_SHM, SHM_DEGREE),
            inertial_frame: INER_FRAME,
            body_fixed_frame: FIXED_FRAME,
        }
    }

    fn predict(&mut self, epoch: Epoch, dt: f64, ctx: &crate::filter::Context) {
        let vec0 = stack_state_and_stm(
            &self.mx,
            &SMatrix::<f64, EST_STATES, EST_STATES>::identity(),
        );

        let f = |time: f64, vec: AugmentedVector| {
            self.augmented_dynamics(
                vec,
                epoch + time * Unit::Second,
                ctx.almanac,
                &ctx.harmonics,
            )
        };
        let (mx_next, phi) = split_state_and_stm(rk4(&f, 0.0, dt, vec0));
        // println!("{:?}", phi.determinant());
        let noise_jacobian = self.process_noise_jacobian();
        let discrete_process_noise =
            noise_jacobian * self.process_noise * noise_jacobian.transpose() * dt; // Potentially upgrade to rk4

        self.mx = mx_next;
        self.p_own = phi * self.p_own * phi.transpose() + discrete_process_noise;
    }
    fn update<const M: usize, Meas: MeasurementModel<M>>(
        &mut self,
        meas_model: &Meas,
        y: &SVector<f64, M>,
    ) {
        let mz = meas_model.predict(&self.mx);
        let observation_jacobian = meas_model.jacobian(&self.mx);
        let r = meas_model.noise(&self.mx);
        let pxz = self.p_own * observation_jacobian.transpose(); // m x n matrix
        let pzz = observation_jacobian * self.p_own * observation_jacobian.transpose() + r; // R should eventually change to noise jacobian when noise isn't additive
        let kalman_gain = pxz * pzz.try_inverse().expect("P_zz is singular");
        self.mx = self.mx + kalman_gain * (y - mz);
        self.p_own = self.p_own - pxz * kalman_gain.transpose() - kalman_gain * pxz.transpose()
            + kalman_gain * pzz * kalman_gain.transpose();
    }
    fn estimate(&self) -> EstimatorVector {
        BaseEKF::get_estimate(&self)
    }
    fn covariance(&self) -> EstimatorMatrix {
        BaseEKF::get_covariance(&self)
    }
}
