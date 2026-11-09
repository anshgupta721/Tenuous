use anise::prelude::*;
use nalgebra::SVector;

use crate::config::{EstimatorMatrix, EstimatorVector};
use crate::filter::{Context, Filter};
use crate::measurements::MeasurementModel;
use crate::swarm_navigation::SwarmNavigation;

pub struct IndependentSwarm<F> {
    filters: Vec<F>,
}

impl<F: Filter> SwarmNavigation<F> for IndependentSwarm<F> {
    fn new(
        initial_estimates: Vec<EstimatorVector>,
        initial_covariances: Vec<EstimatorMatrix>,
        q_matrices: Vec<EstimatorMatrix>,
    ) -> Self {
        let mut filters: Vec<F> = Vec::with_capacity(initial_estimates.len());
        for i in 0..initial_estimates.len() {
            filters.push(F::new(
                initial_estimates[i],
                initial_covariances[i],
                q_matrices[i],
            ));
        }

        Self { filters }
    }

    fn predict(&mut self, epoch: Epoch, dt: f64, ctx: Context) {
        for i in self.filters.iter_mut() {
            F::predict(i, epoch, dt, &ctx);
        }
    }

    fn correct<const M: usize, Meas: MeasurementModel<M>>(
        &mut self,
        model: &Meas,
        y: &Vec<SVector<f64, M>>,
    ) {
        let measurements: Vec<&SVector<f64, M>> = y.iter().collect();
        for (i, filter) in self.filters.iter_mut().enumerate() {
            filter.update::<M, Meas>(model, measurements[i]);
        }
    }

    fn estimates(&self) -> Vec<EstimatorVector> {
        let mut ests: Vec<EstimatorVector> = Vec::new();
        for i in self.filters.iter() {
            ests.push(i.estimate());
        }
        ests
    }

    fn covariance(&self) -> Vec<EstimatorMatrix> {
        let mut covs: Vec<EstimatorMatrix> = Vec::new();
        for i in self.filters.iter() {
            covs.push(i.covariance());
        }
        covs
    }
}
