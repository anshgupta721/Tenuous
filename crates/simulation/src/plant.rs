use anise::constants::{frames::SUN_J2000, orientations};
use anise::frames::Frame;
use anise::prelude::*;
// use rand::SeedableRng;
// use rand_chacha::ChaCha12Rng;

use dynamics::forces::srp::srp_acceleration;
use nalgebra::{SMatrix, SVector};

use crate::config::{INER_FRAME, SC_CS_AREA, SC_MASS, SC_REFLECTIVITY, SUN_MU};
use crate::helper::jacobian;

// use dynamics::forces::srp::srp_acceleration;
use dynamics::gravity::gravity::SphericalHarmonics;
use dynamics::gravity::spherical_harmonics::HarmonicCoeffs;
// This simulations statevector:

//[x, y, z, x_dot, y_dot, z_dot]
pub const NX: usize = 6;
pub type StateVector = SVector<f64, NX>;

fn point_mass_acceleration(mu: f64, pos_body_fixed: SVector<f64, 3>) -> SVector<f64, 3> {
    let r = pos_body_fixed.norm();
    let common = -mu / r.powf(3.0);
    pos_body_fixed * common
}

pub struct Plant {
    gravity: SphericalHarmonics,
    inertial_frame: Frame,
    body_fixed_frame: Frame,
    mu: f64,
    coeffs: HarmonicCoeffs,
}

impl Plant {
    pub fn new(
        gravity: SphericalHarmonics,
        inertial_frame: Frame,
        body_fixed_frame: Frame,
        mu: f64,
        coeffs: HarmonicCoeffs,
    ) -> Plant {
        Plant {
            gravity,
            inertial_frame,
            body_fixed_frame,
            mu,
            coeffs,
        }
    }
    pub fn derivative(&self, x: StateVector, almanac: &Almanac, epoch: Epoch) -> StateVector {
        let pos_inertial = SVector::<f64, 3>::from_row_slice(&[x[0], x[1], x[2]]);

        let dcm_i2b = almanac
            .rotate(self.inertial_frame, self.body_fixed_frame, epoch)
            .unwrap();
        let dcm_b2i = almanac
            .rotate(self.body_fixed_frame, self.inertial_frame, epoch)
            .unwrap();
        let pos_body_fixed = dcm_i2b.rot_mat * pos_inertial;
        // For SRP, need to pass in sun's position wrt to Bennu
        let sun_pos_wrt_bennu = almanac
            .translate(SUN_J2000, INER_FRAME, epoch, None)
            .unwrap()
            .radius_km;

        let (bennu_grav_accel, _) =
            self.gravity
                .acceleration_and_gradient(pos_body_fixed, self.mu, &self.coeffs);

        let bennu_grav_accel_inertial = dcm_b2i.rot_mat * bennu_grav_accel;
        let srp_acceleration = srp_acceleration(
            sun_pos_wrt_bennu,
            pos_inertial,
            SC_CS_AREA / SC_MASS,
            SC_REFLECTIVITY,
        );
        let sun_grav_accel_inertial = point_mass_acceleration(SUN_MU, sun_pos_wrt_bennu);
        // Acceleration in the Bennu J2000 frame,
        let accel_inertial = bennu_grav_accel_inertial; //+ srp_acceleration(sun_pos_wrt_bennu, pos_inertial, SC_CS_AREA / SC_MASS,SC_REFLECTIVITY)
        StateVector::from_row_slice(&[
            x[3],
            x[4],
            x[5],
            accel_inertial[0],
            accel_inertial[1],
            accel_inertial[2],
        ])
    }

    pub fn gravity_gradients(
        &self,
        x: &StateVector,
        almanac: &Almanac,
        epoch: Epoch,
        h: f64,
    ) -> (SMatrix<f64, 3, 3>, SMatrix<f64, 3, 3>) {
        let pos_inertial = SVector::<f64, 3>::new(x[0], x[1], x[2]);
        let dcm_i2b = almanac
            .rotate(self.inertial_frame, self.body_fixed_frame, epoch)
            .unwrap();
        let pos_body_fixed = dcm_i2b * pos_inertial;

        let (_, analytic_grad) =
            self.gravity
                .acceleration_and_gradient(pos_body_fixed, self.mu, &self.coeffs);
        let finite_difference_grad = jacobian(
            |&p| {
                self.gravity
                    .acceleration_and_gradient(p, self.mu, &self.coeffs)
            },
            pos_body_fixed,
            h,
        );
        (analytic_grad, finite_difference_grad)
    }
}
