use anise::prelude::*;
use nalgebra::SVector;

use crate::config::{EstimatorMatrix, EstimatorVector};
use crate::filter::{Context, Filter};
use crate::measurements::MeasurementModel;

pub trait SwarmNavigation<F: Filter> {
    fn new(
        initial_estimates: Vec<EstimatorVector>,
        initial_covariances: Vec<EstimatorMatrix>,
        q_matrices: Vec<EstimatorMatrix>,
    ) -> Self;
    fn predict(&mut self, epoch: Epoch, dt: f64, ctx: Context);
    fn correct<const M: usize, Meas: MeasurementModel<M>>(
        &mut self,
        model: &Meas,
        y: &Vec<SVector<f64, M>>,
    );
    fn estimates(&self) -> Vec<EstimatorVector>;
    fn covariance(&self) -> Vec<EstimatorMatrix>;
}
