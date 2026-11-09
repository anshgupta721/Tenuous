use anise::frames::Frame;

use anise::constants::{frames::SUN_J2000, orientations};
use anise::prelude::*;
use dynamics::gravity::spherical_harmonics::HarmonicCoeffs;
use nalgebra::{SMatrix, SVector};

// use dynamics::models::state_space_model::{LTVSystem, StateSpace};
/// This file contains the simulation configuration for the plant dynamics configuration
/// NOT anything that would be done for Monte Carlo simulatino
/// i.e. state space models

pub const BSP_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/de440s.bsp");
pub const PCA_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/pck11.pca");
pub const BENNU_BSP_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/sb-101955-118.bsp");
pub const BENNU_PCA_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/bennu_v14.pca");

pub const BENNU_MU: f64 = 4.89044967462; // m^3 / s^2

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

pub const REF_RADIUS_SHM: f64 = 290.0;

pub const SHM_DEGREE: usize = 10;

pub const SC_MASS: f64 = 1500.0; // kg
pub const SC_CS_AREA: f64 = 20.0; // m^2 changed
pub const SC_REFLECTIVITY: f64 = 1.4;

// pub fn load() -> HarmonicCoeffs {
//     let mut harmonics = HarmonicCoeffs::zeros(SHM_DEGREE);
//     for &(n, m, c, s) in COEFFS {
//         harmonics.set(n, m, c, s);
//     }
//     harmonics
// }
