// use anise::prelude::*;
// use nalgebra::SVector;

// use crate::config::{EstimatorVector, EstimatorMatrix};
// use crate::swarm_navigation::SwarmNavigation;
// use crate::measurements::MeasurementModel;
// use crate::filter::{Filter, Context};

// pub struct CentralizedSwarm<F> {
//     filters: Vec<F>,
// }

// impl<F: Filter> SwarmNavigation<F> for CentralizedSwarm<F> {
//     fn predict(&mut self, epoch: Epoch, dt: f64, ctx: Context) {
//         self.filters[0].predict(epoch, dt, &ctx);
//     }

//     fn correct<const M: usize>(&mut self, y: &SVector<f64, M>) {
//         self.filters[0].update(y);
//     }

//     fn estimates(&self) -> Vec<EstimatorVector>{
//         let mut vec: Vec<EstimatorVector> = Vec::new();
//         vec.push(self.filters[0].estimate());
//         vec
//     }

//     fn covariance(&self) -> Vec<EstimatorMatrix> {
//         let mut vec: Vec<EstimatorMatrix> = Vec::new();
//         vec.push(self.filters[0].covariance());
//         vec
//     }
// }
