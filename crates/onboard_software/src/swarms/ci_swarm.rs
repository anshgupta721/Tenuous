use nalgebra::SVector;

use crate::config::{EstimatorMatrix, EstimatorVector, GravityVector, GravityMatrix};
use crate::filters::filter::Filter;
use crate::swarms::swarm_navigation::SwarmNavigation;

pub struct CovarianceIntersectionSwarm<F: Filter> {
    filters: Vec<F>,
}

impl<F: Filter> SwarmNavigation for CovarianceIntersectionSwarm<F> {
    fn new(
        initial_estimates: Vec<crate::config::EstimatorVector>,
        initial_covariances: Vec<crate::config::EstimatorMatrix>,
        q_matrices: Vec<crate::config::EstimatorMatrix>,
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

    fn estimates(&self) -> Vec<crate::config::EstimatorVector> {
        let mut ests: Vec<EstimatorVector> = Vec::new();
        for i in self.filters.iter() {
            ests.push(i.estimate());
        }
        ests
    }

    fn covariance(&self) -> Vec<crate::config::EstimatorMatrix> {
        let mut covs: Vec<EstimatorMatrix> = Vec::new();
        for i in self.filters.iter() {
            covs.push(i.covariance());
        }
        covs
    }

    fn predict(
        &mut self,
        epoch: anise::prelude::Epoch,
        dt: f64,
        almanac: &anise::prelude::Almanac,
    ) {
        for filter in self.filters.iter_mut() {
            filter.predict(epoch, dt, almanac);
        }
    }

    fn correct<const M: usize, Meas: crate::measurements::MeasurementModel<M>>(
        &mut self,
        model: &Meas,
        y: &Vec<SVector<f64, M>>,
    ) {
        let measurements: Vec<&SVector<f64, M>> = y.iter().collect();
        let mut nis_vector: Vec<f64> = Vec::new();
        for (i, filter) in self.filters.iter_mut().enumerate() {
            nis_vector.push(filter.update::<M, Meas>(model, measurements[i]));
        }
    }
}

impl<F: Filter> CovarianceIntersectionSwarm<F> {
    pub fn fuse(&mut self, neighbors: &[Vec<usize>]) {
        let crafts = self.filters.len();
        for (i, filter) in self.filters.iter_mut().enumerate() {
            let estimate = filter.estimate();
            let covariance = filter.covariance();
            for craft in 0..crafts {
                if i == craft {
                    continue
                }
                // Fuse using covariance intersection
                
            }
        }
    }
}