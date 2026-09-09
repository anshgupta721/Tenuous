use anise::constants::frames::{MOON_J2000, MOON_PA_DE440_FRAME};
use anise::frames::Frame;
use anise::prelude::*;
use nalgebra::{SMatrix, SVector};

use dynamics::gravity::gravity::GravityField;
use dynamics::gravity::spherical_harmonics::HarmonicCoeffs;

// use dynamics::models::state_space_model::{LTVSystem, StateSpace};
/// This file contains the simulation configuration for the plant dynamics configuration
/// NOT anything that would be done for Monte Carlo simulatino
/// i.e. state space models

pub const BSP_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/de440s.bsp");
pub const PCA_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/pck11.pca");
pub const MOON_PA_FILE: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/data/moon_pa_de440_200625.bpc");

pub const NX: usize = 6;

pub const MOON_MU: f64 = 4902.801076;
pub const INER_FRAME: Frame = MOON_J2000;
pub const CEL_FIXED_FRAME: Frame = MOON_PA_DE440_FRAME;

pub const REF_RADIUS_SHM: f64 = 1738.0;

const SHM_DEGREE: usize = 8;

pub const COEFFS: &[(usize, usize, f64, f64)] = &[
    (0, 0, 1.0, 0.0),
    (2, 0, -9.0882923650770995e-05, 0.0000000000000000E+00),
    (2, 1, 8.4954064857652003e-11, 9.7726994478962992E-10),
    (2, 2, 3.4670944268755999E-05, -2.4064244523445002E-10),
    (3, 0, -3.1974039070980999E-06, 0.0000000000000000E+00),
    (3, 1, 2.6367948585301000E-05, 5.4545621847343999E-06),
    (3, 2, 1.4171538267300000E-05, 4.8779443647939002E-06),
    (3, 3, 1.2275202024552999E-05, -1.7743747889595000E-06),
    (4, 0, 3.2347924522417001E-06, 0.0000000000000000E+00),
    (4, 1, -6.0134612265187000E-06, 1.6643327702085001E-06),
    (4, 2, -7.1161737190123004E-06, -6.7770230383513997E-06),
    (4, 3, -1.3499173662211000E-06, -1.3444991147298001E-05),
    (4, 4, -6.0069115393814999E-06, 3.9266111458205004E-06),
    (5, 0, -2.2378186921629999E-07, 0.0000000000000000E+00),
    (5, 1, -1.0116123460056000E-06, -4.1189034337586998E-06),
    (5, 2, 4.3995350687644004E-06, 1.0571040222894000E-06),
    (5, 3, 4.6619443222844998E-07, 8.6988722731551006E-06),
    (5, 4, 2.7542004508341999E-06, 6.7624331748839005E-08),
    (5, 5, 3.1107298179927999E-06, -2.7545431341824001E-06),
    (6, 0, 3.8184191011809003E-06, 0.0000000000000000E+00),
    (6, 1, 1.5282769193243000E-06, -2.5995934726698998E-06),
    (6, 2, -4.3973005721118001E-06, -2.1676869353336998E-06),
    (6, 3, -3.3175420296933002E-06, -3.4274097833201001E-06),
    (6, 4, 3.4122057961179000E-07, -4.0580193556196003E-06),
    (6, 5, 1.4543741845814999E-06, -1.0341804065343001E-05),
    (6, 6, -4.6842414195666002E-06, 7.2298542235022004E-06),
    (7, 0, 5.5933881887178998E-06, 0.0000000000000000E+00),
    (7, 1, 7.4716915273386999E-06, -1.1974837275945000E-07),
    (7, 2, -6.5012294054409005E-07, 2.4111005879336999E-06),
    (7, 3, 5.9942806310729004E-07, 2.3573269328809000E-06),
    (7, 4, -8.4369454620858004E-07, 7.5651532118737996E-07),
    (7, 5, -2.0682821988196001E-07, 1.0693087986910999E-06),
    (7, 6, -1.0653045588942000E-06, 1.1004282428110999E-06),
    (7, 7, -1.8202645695563999E-06, -1.5999965833572001E-06),
    (8, 0, 2.3468300594816001E-06, 0.0000000000000000E+00),
    (8, 1, 4.1735904924526999E-09, 1.0980290732908000E-06),
    (8, 2, 3.0093221489839999E-06, 1.9305655599869999E-06),
    (8, 3, -1.8890482916189001E-06, 9.5448280394918990E-07),
    (8, 4, 3.4086972282042999E-06, -5.2824948185726003E-07),
    (8, 5, -1.2480443032891000E-06, 2.9185712820189999E-06),
    (8, 6, -1.6604388534337000E-06, -2.1146631041151002E-06),
    (8, 7, -1.5096535588051000E-06, 3.2688520883962001E-06),
    (8, 8, -2.4856047110162002E-06, 2.1164273548015999E-06),
];

pub fn load() -> HarmonicCoeffs {
    let mut harmonics = HarmonicCoeffs::zeros(SHM_DEGREE);
    for &(n, m, c, s) in COEFFS {
        harmonics.set(n, m, c, s);
    }
    harmonics
}

// This simulations statevector:

//[x, y, z, x_dot, y_dot, z_dot]
pub type StateVector = SVector<f64, NX>;

pub struct Plant {
    gravity: Box<dyn GravityField>,
    inertial_frame: Frame,
    body_fixed_frame: Frame,
}

impl Plant {
    pub fn new(
        gravity: Box<dyn GravityField>,
        inertial_frame: Frame,
        body_fixed_frame: Frame,
    ) -> Plant {
        Plant {
            gravity,
            inertial_frame,
            body_fixed_frame,
        }
    }
    pub fn derivative(
        &self,
        t: f64,
        x: StateVector,
        almanac: &Almanac,
        epoch: Epoch,
    ) -> StateVector {
        let pos_inertial = SVector::<f64, 3>::from_row_slice(&[x[0], x[1], x[2]]);

        let dcm_i2b = almanac
            .rotate(self.inertial_frame, self.body_fixed_frame, epoch)
            .unwrap();
        let dcm_b2i = almanac
            .rotate(self.body_fixed_frame, self.inertial_frame, epoch)
            .unwrap();
        let pos_body_fixed = dcm_i2b.rot_mat * pos_inertial;
        let accel = self.gravity.acceleration(pos_body_fixed);
        let accel_inertial = dcm_b2i.rot_mat * accel;
        StateVector::from_row_slice(&[
            x[3],
            x[4],
            x[5],
            accel_inertial[0],
            accel_inertial[1],
            accel_inertial[2],
        ])
    }
}
