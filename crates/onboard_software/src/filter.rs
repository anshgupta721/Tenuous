use anise::prelude::*;
use nalgebra::SVector;

use dynamics::gravity::spherical_harmonics::HarmonicCoeffs;

use crate::config::{EstimatorMatrix, EstimatorVector};
use crate::measurements::MeasurementModel;

#[derive(Clone, Copy)]
pub struct Context<'a> {
    pub almanac: &'a Almanac,
    pub harmonics: &'a HarmonicCoeffs, // whatever type load() returns
}

pub trait Filter {
    fn new(
        initial_state_guess: EstimatorVector,
        initial_state_covariance: EstimatorMatrix,
        process_noise: EstimatorMatrix,
    ) -> Self;
    fn predict(&mut self, epoch: Epoch, dt: f64, ctx: &Context);
    fn update<const M: usize, Meas: MeasurementModel<M>>(
        &mut self,
        model: &Meas,
        y: &SVector<f64, M>,
    );
    fn estimate(&self) -> EstimatorVector;
    fn covariance(&self) -> EstimatorMatrix;
}
