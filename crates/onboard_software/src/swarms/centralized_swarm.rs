use anise::prelude::*;
use nalgebra::{DMatrix, DVector, SMatrix, SVector};

use crate::config::{EstimatorMatrix, EstimatorVector, GRAV_STATES, SC_STATES};
use crate::filters::centralized_ekf::CentralizedEKF;
use crate::measurements::MeasurementModel;
use crate::swarms::swarm_navigation::SwarmNavigation;

pub struct CentralizedSwarm {
    filter: CentralizedEKF,
}

impl SwarmNavigation for CentralizedSwarm {
    fn new(
        initial_estimates: Vec<EstimatorVector>,
        initial_covariances: Vec<EstimatorMatrix>,
        q_matrices: Vec<EstimatorMatrix>,
    ) -> Self {
        let crafts = initial_estimates.len();
        let n = 6 * crafts;

        // Build dynamic vectors and matrices to pass into the centralized filter
        let mut x0 = DVector::<f64>::zeros(n);
        let mut p0 = DMatrix::<f64>::zeros(n, n);

        // Populate the new containers with the gravitational field first then each spacecraft's state
        x0.rows_mut(0, GRAV_STATES)
            .copy_from(&initial_estimates[0].fixed_rows::<GRAV_STATES>(SC_STATES));

        p0.view_mut((0, 0), (GRAV_STATES, GRAV_STATES)).copy_from(
            &initial_covariances[0].fixed_view::<GRAV_STATES, GRAV_STATES>(SC_STATES, SC_STATES),
        );

        for craft in 0..crafts {
            let f = GRAV_STATES + craft * SC_STATES;
            p0.view_mut((f, f), (SC_STATES, SC_STATES))
                .copy_from(&initial_covariances[craft].fixed_view::<SC_STATES, SC_STATES>(0, 0));
        }

        let q_sc: Vec<SMatrix<f64, SC_STATES, SC_STATES>> = q_matrices
            .iter()
            .map(|q| q.fixed_view::<SC_STATES, SC_STATES>(0, 0).into_owned())
            .collect();

        let filter: CentralizedEKF = CentralizedEKF::new(x0, p0, q_sc);
        Self { filter }
    }

    fn predict(&mut self, epoch: Epoch, dt: f64, almanac: &Almanac) {
        self.filter.predict(epoch, dt, almanac);
    }

    fn correct<const M: usize, Meas: MeasurementModel<M>>(
        &mut self,
        model: &Meas,
        y: &Vec<SVector<f64, M>>,
    ) -> Vec<f64> {
        self.filter.correct(model, y)
    }

    fn estimates(&self) -> Vec<EstimatorVector> {
        self.filter.get_estimate()
    }

    fn covariance(&self) -> Vec<EstimatorMatrix> {
        self.filter.get_covariance()
    }
}
