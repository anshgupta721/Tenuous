use anise::constants::{frames::SUN_J2000, orientations};
use anise::frames::Frame;
use anise::prelude::*;

use nalgebra::{SMatrix, SVector};

use crate::config::{INER_FRAME, NX};
use crate::helper::jacobian;

use dynamics::forces::srp::srp_acceleration;
use dynamics::gravity::gravity::SphericalHarmonics;
use dynamics::gravity::spherical_harmonics::HarmonicCoeffs;
// This simulations statevector:

//[x, y, z, x_dot, y_dot, z_dot]
pub type StateVector = SVector<f64, NX>;

pub struct Plant {
    gravity: SphericalHarmonics,
    inertial_frame: Frame,
    body_fixed_frame: Frame,
    coeffs: HarmonicCoeffs,
}

impl Plant {
    pub fn new(
        gravity: SphericalHarmonics,
        inertial_frame: Frame,
        body_fixed_frame: Frame,
        coeffs: HarmonicCoeffs,
    ) -> Plant {
        Plant {
            gravity,
            inertial_frame,
            body_fixed_frame,
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
        let (accel, _) = self
            .gravity
            .acceleration_and_gradient(pos_body_fixed, &self.coeffs);
        // Acceleration in the Bennu J2000 frame,
        let accel_inertial = dcm_b2i.rot_mat * accel; //+ srp_acceleration(sun_pos_wrt_bennu, pos_inertial, SC_CS_AREA / SC_MASS,SC_REFLECTIVITY)
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

        let (_, analytic_grad) = self
            .gravity
            .acceleration_and_gradient(pos_body_fixed, &self.coeffs);
        let finite_difference_grad = jacobian(
            |&p| self.gravity.acceleration_and_gradient(p, &self.coeffs),
            pos_body_fixed,
            h,
        );
        (analytic_grad, finite_difference_grad)
    }
}
