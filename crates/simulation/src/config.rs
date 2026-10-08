
use anise::frames::{Frame};

use anise::constants::{frames::SUN_J2000, orientations};
use anise::prelude::*;
use nalgebra::{SMatrix, SVector};

use dynamics::gravity::gravity::GravityField;
use dynamics::gravity::spherical_harmonics::HarmonicCoeffs;
use dynamics::forces::srp::srp_acceleration;

// use dynamics::models::state_space_model::{LTVSystem, StateSpace};
/// This file contains the simulation configuration for the plant dynamics configuration
/// NOT anything that would be done for Monte Carlo simulatino
/// i.e. state space models

pub const BSP_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/de440s.bsp");
pub const PCA_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/pck11.pca");
pub const BENNU_BSP_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/sb-101955-118.bsp");
pub const BENNU_PCA_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/bennu_v14.pca");



pub const NX: usize = 6;

pub const BENNU_MU: f64 = 4.89044967462e-09; // km^3 / s^2

pub const INER_FRAME: Frame = Frame::new(2101955, orientations::J2000);
pub const SUN_FRAME: Frame = SUN_J2000;
const BENNU_ID: i32 = 2101955;
pub const FIXED_FRAME: Frame = Frame::new(BENNU_ID, BENNU_ID);

pub fn fix_bennu_parent(mut almanac: Almanac) -> Almanac {
    let mut bennu = almanac
        .get_planetary_data_from_id(BENNU_ID)
        .expect("Bennu missing from PCA (no GM when it was converted?)");
    bennu.parent_id = orientations::J2000;
    almanac.set_planetary_data_from_id(BENNU_ID, bennu).unwrap();
    almanac
}


pub const REF_RADIUS_SHM: f64 = 0.29;

const SHM_DEGREE: usize = 10;

pub const SC_MASS: f64 = 1500.0; // kg
pub const SC_CS_AREA: f64 = 2e-5; // km^2
pub const SC_REFLECTIVITY: f64 = 1.4;

pub const COEFFS: &[(usize, usize, f64, f64)] = &[
    (0, 0, 1.0, 0.0),
    (2, 0, 0.019261012209376163, 0.0000000000000000E+00),
    (2, 1, -2.1782173147855912e-14, 3.0009695268217895e-15),
    (2, 2, 0.00306499464152612, -0.00109450399573948),
    (3, 0, -0.0012219404640668086, 0.0000000000000000E+00),
    (3, 1, 0.0008148921217387432, -0.0005434579977478096),
    (3, 2, -0.000934922673655136, -0.0005377851962265501),
    (3, 3, 0.0011710305387050103, -0.00031001193429507437),
    (4, 0, -0.006496001836889563, 0.0000000000000000E+00),
    (4, 1, -0.0008821561290796273, -0.0005752155149983148),
    (4, 2, -0.0008707051950524519, -8.400092905051768e-05),
    (4, 3, -7.621121755654963e-05, -0.0003878715548433681),
    (4, 4, 0.0007748481150705528, 0.0022464919895245237),
    // (5, 0, , 0.0000000000000000E+00),
    // (5, 1, , ),
    // (5, 2, , ),
    // (5, 3, , ),
    // (5, 4, , ),
    // (5, 5, , ),
    // (6, 0, , 0.0000000000000000E+00),
    // (6, 1, , ),
    // (6, 2, , ),
    // (6, 3, , ),
    // (6, 4, , ),
    // (6, 5, , ),
    // (6, 6, , ),
    // (7, 0, , 0.0000000000000000E+00),
    // (7, 1, , ),
    // (7, 2, , ),
    // (7, 3, , ),
    // (7, 4, , ),
    // (7, 5, , ),
    // (7, 6, , ),
    // (7, 7, , ),
    // (8, 0, , 0.0000000000000000E+00),
    // (8, 1, , ),
    // (8, 2, , ),
    // (8, 3, , ),
    // (8, 4, , ),
    // (8, 5, , ),
    // (8, 6, , ),
    // (8, 7, , ),
    // (8, 8, , ),
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
        // For SRP, need to pass in sun's position wrt to Bennu
        let sun_pos_wrt_bennu = almanac.translate(SUN_J2000, INER_FRAME, epoch, None).unwrap().radius_km;
        let accel = self.gravity.acceleration(pos_body_fixed);
        // Acceleration in the Bennu J2000 frame, 
        let accel_inertial = dcm_b2i.rot_mat * accel + srp_acceleration(sun_pos_wrt_bennu, pos_inertial, SC_CS_AREA / SC_MASS,SC_REFLECTIVITY);
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
