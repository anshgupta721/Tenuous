use anise::constants::{frames::SUN_J2000, orientations};
use nalgebra::{SMatrix, SVector};

use anise::prelude::*;

pub const SC_STATES: usize = 6;
pub const SHM_DEGREE: usize = 2;
pub const SHM_PARAMS: usize = (SHM_DEGREE + 1) * (SHM_DEGREE + 1) - 4; // Estimator SHM degree
pub const GRAV_STATES: usize = 1 + SHM_PARAMS;
pub const EST_STATES: usize = SC_STATES + GRAV_STATES;

pub const MU_BENNU_GUESS: f64 = 4.89044967462; // m
pub const REF_RADIUS_SHM: f64 = 290.0;

pub const INER_FRAME: Frame = Frame::new(2101955, orientations::J2000);
pub const SUN_FRAME: Frame = SUN_J2000;
const BENNU_ID: i32 = 2101955;
pub const FIXED_FRAME: Frame = Frame::new(BENNU_ID, BENNU_ID);

// pub const NU: usize = 3;

pub const EST_MAT_LENGTH: usize = EST_STATES * EST_STATES;
const AUG_LENGTH: usize = EST_MAT_LENGTH + EST_STATES;
pub type EstimatorVector = SVector<f64, EST_STATES>;
pub type EstimatorMatrix = SMatrix<f64, EST_STATES, EST_STATES>;
pub type AugmentedVector = SVector<f64, AUG_LENGTH>;
pub type GravityVector = SVector<f64, GRAV_STATES>;
pub type GravityMatrix = SMatrix<f64, GRAV_STATES, GRAV_STATES>;
