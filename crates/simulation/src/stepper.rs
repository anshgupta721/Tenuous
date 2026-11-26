use anise::prelude::*;

use dynamics::gravity::spherical_harmonics::HarmonicCoeffs;
use estimation::analysis::statistics::squared_mahalanobis_distance;

use onboard_software::config::{EstimatorMatrix, EstimatorVector, GRAV_STATES, SC_STATES};
use onboard_software::filters::ekf::BaseEKF;
use onboard_software::measurements::{LinearMeasurement, Models};
use onboard_software::swarms::independent_swarm::IndependentSwarm;
use onboard_software::swarms::swarm_navigation::SwarmNavigation;

use crate::graph::Graph;
use crate::plant::StateVector;
use crate::sensors::{
    POS_SIGMA, PosVelSensor, Sensor, SensorID, SpacecraftSuite, SwarmSensors, VEL_SIGMA,
    sensor_rng_generator,
};
use crate::world::World;

pub struct History {
    // pub n_craft: usize,
    pub t: Vec<f64>,
    pub truth: Vec<StateVector>, // [step][craft][NX]       Spacecraft1-tk(x, y, z, vx, vy, vz)----
    pub est: Vec<EstimatorVector>, // [step][craft][EST_STATES]
    pub var: Vec<EstimatorVector>, // [step][craft][update][row][col]
    pub nees_sc: Vec<f64>,       // [step][craft]
    pub nees_grav_par: Vec<f64>, // [step][craft]
    pub nees_full_matrix: Vec<f64>, // [step][craft]
    pub nis_pos_vel: Vec<f64>,   // [step][craft]
    pub snap_idx: Vec<usize>,    // step indices where full covariances are saved
    pub cov_snap: Vec<EstimatorMatrix>, // snap_idx.len() * n_craft, post-update
    pub cov_full: Vec<EstimatorMatrix>, // interleaved predict/update, only if full_cov is true
}

impl History {
    pub fn with_capacity(steps: usize, n_craft: usize) -> Self {
        Self {
            // n_craft,
            t: Vec::with_capacity(steps),
            truth: Vec::with_capacity(steps * n_craft),
            est: Vec::with_capacity(steps * n_craft),
            var: Vec::with_capacity(steps * n_craft),
            nees_sc: Vec::with_capacity(steps * n_craft),
            nees_grav_par: Vec::with_capacity(steps * n_craft),
            nees_full_matrix: Vec::with_capacity(steps * n_craft),
            nis_pos_vel: Vec::with_capacity(steps * n_craft),
            snap_idx: Vec::new(),
            cov_snap: Vec::new(),
            cov_full: Vec::new(),
        }
    }

    pub fn append_full_covariance() {}

    pub fn record<SwrmNav: SwarmNavigation>(
        &mut self,
        world: &World,
        swarm_navigation: &SwrmNav,
        mut nis: Vec<f64>,
        full_covariance_flag: bool,
    ) {
        let n_craft = world.crafts.len();
        let swarm_estimates = swarm_navigation.estimates();
        let swarm_covariances = swarm_navigation.covariance();
        let gravity_parameters = world.harmonics.to_parameters();
        let mut true_gravity = Vec::with_capacity(GRAV_STATES);
        true_gravity.push(world.mu);
        true_gravity.extend_from_slice(&gravity_parameters[..GRAV_STATES - 1]);
        self.nis_pos_vel.append(&mut nis);
        for craft in 0..n_craft {
            let x_hat = &swarm_estimates[craft];
            let p = &swarm_covariances[craft];
            let mut full_truth_vector = EstimatorVector::zeros();
            full_truth_vector
                .fixed_rows_mut::<SC_STATES>(0)
                .copy_from(&world.crafts[craft].x);
            full_truth_vector
                .fixed_rows_mut::<GRAV_STATES>(6)
                .copy_from_slice(&true_gravity[..GRAV_STATES]);
            let full_error = full_truth_vector - x_hat;

            self.nees_full_matrix
                .push(squared_mahalanobis_distance(full_error, *p).unwrap_or(f64::NAN));

            let sc_error = full_error.fixed_rows::<SC_STATES>(0).into_owned();
            let p_sc = p.fixed_view::<SC_STATES, SC_STATES>(0, 0).into_owned();
            self.nees_sc
                .push(squared_mahalanobis_distance(sc_error, p_sc).unwrap_or(f64::NAN));

            let grav_error = full_error.fixed_rows::<GRAV_STATES>(SC_STATES).into_owned();
            let p_grav = p
                .fixed_view::<GRAV_STATES, GRAV_STATES>(SC_STATES, SC_STATES)
                .into_owned();
            self.nees_grav_par
                .push(squared_mahalanobis_distance(grav_error, p_grav).unwrap_or(f64::NAN));
            // Add the correct step covariance matrix to full covariance history
            self.var.push(p.diagonal());
        }

        if full_covariance_flag {
            self.cov_full
                .extend((0..n_craft).map(|i| swarm_navigation.covariance()[i]));
        }
        // Save data from each spacecraft
        // Save truth and estimates
        self.t.push(world.t);
        self.truth.extend(world.crafts.iter().map(|c| c.x));
        self.est.extend(swarm_estimates.iter().copied());
    }
}

pub fn run<SwrmNav: SwarmNavigation>(
    world: &mut World,
    sensors: &mut SwarmSensors,
    swarm_navigation: &mut SwrmNav,
    models: &Models,
    t_end: f64,
    dt: f64,
    save_every: f64,
    full_covariance_flag: bool,
) -> History {
    let n = ((t_end - world.t) / dt).ceil() as usize + 1;
    let mut history = History::with_capacity(n, world.crafts.len());
    let n_craft = world.crafts.len();
    if full_covariance_flag {
        let p0 = swarm_navigation.covariance();
        history.cov_full.extend(p0.iter().copied()); // stands in for the step-0 "predict" entry
    }
    history.record(
        world,
        swarm_navigation,
        vec![f64::NAN; n_craft],
        full_covariance_flag,
    );
    for _ in 0..n {
        // Propagate all spacecraft from n to n+1
        let epoch = world.step(dt);

        // Update pose of the shape
        world.update_shape_pose();

        // Generate measurements for each spacecraft
        let meas = sensors.measurements(&world.crafts);

        // Propagate every spacecraft forward
        swarm_navigation.predict(epoch, dt, &world.almanac);

        // Add the predict step covariance matrix to full covariance history
        if full_covariance_flag {
            history
                .cov_full
                .extend((0..n_craft).map(|i| swarm_navigation.covariance()[i]));
        }

        // Correct step using linear measurement
        let nis = swarm_navigation.correct(&models.pos_vel, &meas);

        // Extract data and send it out
        history.record(world, swarm_navigation, nis, full_covariance_flag);
    }
    history
}

pub fn sim_stepper(
    epoch_0: Epoch,
    x_0: Vec<StateVector>,
    est_x_0: Vec<EstimatorVector>,
    p_0: Vec<EstimatorMatrix>,
    q: Vec<EstimatorMatrix>,
    t_span: [f64; 2],
    dt: f64,
    mu: f64,
    harmonics: HarmonicCoeffs,
    save_every: f64,
    full_covariance_flag: bool,
    seed: u64,
) -> History {
    let mut world = World::new(epoch_0, &x_0, t_span[0], seed, mu, harmonics);
    let suites: Vec<SpacecraftSuite> = (0..x_0.len())
        .map(|c| {
            SpacecraftSuite::new(
                vec![Box::new(PosVelSensor::new(SensorID::new(0))) as Box<dyn Sensor>],
                sensor_rng_generator(seed, c, 0),
            )
        })
        .collect();

    let mut swarm = SwarmSensors::new(suites);

    // Measurement models defined here
    let models = Models {
        pos_vel: LinearMeasurement::position_velocity(POS_SIGMA, VEL_SIGMA),
    };

    let mut navigation = IndependentSwarm::<BaseEKF>::new(est_x_0, p_0, q);

    run(
        &mut world,
        &mut swarm,
        &mut navigation,
        &models,
        t_span[1],
        dt,
        save_every,
        full_covariance_flag,
    )
}
