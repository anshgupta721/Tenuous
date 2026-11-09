use nalgebra::SVector;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use anise::prelude::*;
use hifitime::Epoch;

use dynamics::gravity::gravity::SphericalHarmonics;
use dynamics::gravity::spherical_harmonics::HarmonicCoeffs;
use dynamics::integrators::rk4::rk4;

use crate::plant::{Plant, StateVector};

use crate::config::{
    BENNU_BSP_FILE, BENNU_MU, BENNU_PCA_FILE, BSP_FILE, FIXED_FRAME, INER_FRAME, PCA_FILE,
    REF_RADIUS_SHM, SHM_DEGREE, fix_bennu_parent,
};

pub struct Observables {
    specific_force: SVector<f64, 3>,
}

impl Observables {
    pub fn new() -> Self {
        Self {
            specific_force: SVector::<f64, 3>::zeros(),
        }
    }
}

pub struct Spacecraft {
    pub x: StateVector,
    pub y: Observables,
}

pub struct World {
    pub almanac: Almanac,
    pub harmonics: HarmonicCoeffs,
    plant: Plant,
    epoch_0: Epoch,
    pub t: f64,
    pub crafts: Vec<Spacecraft>,
}

impl World {
    pub fn new(
        epoch_0: Epoch,
        x_0: &[StateVector],
        t_0: f64,
        seed: u64,
        harmonics: HarmonicCoeffs,
    ) -> Self {
        let almanac = Almanac::new(BSP_FILE)
            .unwrap()
            .load(PCA_FILE)
            .unwrap()
            .load(BENNU_BSP_FILE)
            .unwrap()
            .load(BENNU_PCA_FILE)
            .unwrap();
        let almanac = fix_bennu_parent(almanac);

        let plant = Plant::new(
            SphericalHarmonics::new(BENNU_MU, REF_RADIUS_SHM, SHM_DEGREE),
            INER_FRAME,
            FIXED_FRAME,
            harmonics.clone(),
        );

        let crafts: Vec<Spacecraft> = x_0
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let mut rng = ChaCha12Rng::seed_from_u64(seed);
                rng.set_stream(i as u64);
                Spacecraft {
                    x,
                    y: Observables::new(),
                }
            })
            .collect();
        Self {
            almanac,
            harmonics,
            plant,
            epoch_0,
            t: t_0,
            crafts,
        }
    }

    pub fn step(&mut self, dt: f64) -> Epoch {
        let epoch = self.epoch();
        for i in &mut self.crafts {
            i.x = rk4(
                &(|t_local: f64, x: StateVector| {
                    let e = self.epoch_0 + Unit::Second * t_local;
                    self.plant.derivative(x, &self.almanac, e)
                }),
                self.t,
                dt,
                i.x,
            );
            self.t += dt;
        }
        epoch
    }

    pub fn epoch(&self) -> Epoch {
        self.epoch_0 + Unit::Second * self.t
    }
}
