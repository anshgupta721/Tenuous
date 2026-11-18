use anise::prelude::*;
use nalgebra::{SMatrix, SVector};

use dynamics::gravity::gravity::SphericalHarmonics;
use dynamics::gravity::spherical_harmonics::HarmonicCoeffs;
use dynamics::integrators::rk4::rk4;

use crate::config::{
    AugmentedVector, EST_MAT_LENGTH, EST_STATES, EstimatorMatrix, EstimatorVector, FIXED_FRAME,
    GRAV_STATES, INER_FRAME, REF_RADIUS_SHM, SC_STATES, SHM_DEGREE, SHM_PARAMS,
};

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

pub struct GravityDynamics {
    gravity: SphericalHarmonics,
    inertial_frame: Frame,
    body_fixed_frame: Frame,
}

impl GravityDynamics {
    pub fn new() -> Self {
        Self {
            gravity: SphericalHarmonics::new(REF_RADIUS_SHM, SHM_DEGREE),
            inertial_frame: INER_FRAME,
            body_fixed_frame: FIXED_FRAME,
        }
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
        let mu = x[6];

        let dcm_i2b = almanac
            .rotate(self.inertial_frame, self.body_fixed_frame, epoch)
            .unwrap()
            .rot_mat;

        let r_bf = dcm_i2b * r;

        let (accel, gradient) = self.gravity.acceleration_and_gradient(r_bf, mu, &coeffs);
        let accel_inertial = dcm_i2b.transpose() * accel;
        let gradient_inertial: SMatrix<f64, 3, 3> = dcm_i2b.transpose() * gradient * dcm_i2b;
        let accel_partials = self.gravity.acceleration_partials::<SHM_PARAMS>(r_bf, mu);

        let da_dmu = accel_inertial / mu;
        // Build vector of r_dot, v_dot, mu_dot, coeffs_dot
        let mut derivative = EstimatorVector::zeros();
        derivative.fixed_rows_mut::<3>(0).copy_from(&v);
        derivative.fixed_rows_mut::<3>(3).copy_from(&accel_inertial);

        //  [dv/dr, dv/dv, dv/dmu, dv/d(Cnm, Snm)]

        let mut jacobian = SMatrix::<f64, EST_STATES, EST_STATES>::zeros();

        jacobian
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&SMatrix::<f64, 3, 3>::identity());
        jacobian
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&gradient_inertial);
        jacobian.fixed_view_mut::<3, 1>(3, 6).copy_from(&da_dmu);
        jacobian
            .fixed_view_mut::<3, SHM_PARAMS>(3, SC_STATES + 1)
            .copy_from(&(dcm_i2b.transpose() * accel_partials));

        // [[0, 0, 0, 0, 0, 0]
        // ]]

        (derivative, jacobian)
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

    pub fn propagate(
        &self,
        x0: &EstimatorVector,
        epoch: Epoch,
        dt: f64,
        almanac: &Almanac,
    ) -> (EstimatorVector, EstimatorMatrix) {
        let coeffs = HarmonicCoeffs::from_parameters(SHM_DEGREE, &x0.as_slice()[SC_STATES + 1..]);

        let vec0 = stack_state_and_stm(x0, &EstimatorMatrix::identity());
        let f = |t: f64, vec: AugmentedVector| {
            self.augmented_dynamics(vec, epoch + t * Unit::Second, almanac, &coeffs)
        };
        split_state_and_stm(rk4(&f, 0.0, dt, vec0))
    }
}
