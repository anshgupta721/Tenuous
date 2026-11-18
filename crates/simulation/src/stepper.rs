use anise::prelude::*;

use dynamics::gravity::spherical_harmonics::HarmonicCoeffs;

use onboard_software::config::{EstimatorMatrix, EstimatorVector};
use onboard_software::filters::ekf::BaseEKF;
use onboard_software::swarms::independent_swarm::IndependentSwarm;
use onboard_software::measurements::{LinearMeasurement, Models};
use onboard_software::swarms::swarm_navigation::SwarmNavigation;

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
    pub nees_sc: Vec<f64>,         // [step][craft]
    pub nees_grav_par: Vec<f64>,        // [step][craft]
    pub nees_full: Vec<f64>,
    pub snap_idx: Vec<usize>,           // step indices where full covariances are saved
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
            nees_par: Vec::with_capacity(steps * n_craft),
            nees_full: Vec::with_capacity(steps * n_craft),
            snap_idx: Vec::new(),
            cov_snap: Vec::new(),
            cov_full: Vec::new(),

        }
    }
}

pub fn run<SwrmNav: SwarmNavigation>(
    world: &mut World,
    sensors: &mut SwarmSensors,
    swarm_navigation: &mut SwrmNav,
    models: &Models,
    t_end: f64,
    dt: f64,
) -> History {
    let n = ((t_end - world.t) / dt).ceil() as usize + 1;
    let mut history = History::with_capacity(n, world.crafts.len());
    for _ in 0..n {
        // Propagate all spacecraft from n to n+1
        let epoch = world.step(dt);

        // Generate measurements for each spacecraft
        let meas = sensors.measurements(&world.crafts);
        // let y: Vec<SensorVector> = meas.iter.map(|per_craft| )

        // Propagate every spacecraft forward and add their value to 
        swarm_navigation.predict(epoch, dt, &world.almanac);
        swarm_navigation.correct(&models.pos_vel, &meas);
        history.var.extend(swarm_navigation.covariance().diag());

        // Save data from each spacecraft
        history.t.push(world.t);
        history
            .truth
            .extend(world.crafts.iter().map(|craft| craft.x));
        history
            .est
            .extend(filter.estimates().iter().map(|estimate| estimate));
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
    harmonics: HarmonicCoeffs,
    save_every: f64,
    seed: u64,
) -> History {
    let mut world = World::new(epoch_0, &x_0, t_span[0], seed, harmonics);
    let suites: Vec<SpacecraftSuite> = (0..x_0.len())
        .map(|c| {
            SpacecraftSuite::new(
                vec![Box::new(PosVelSensor::new(SensorID::new(0))) as Box<dyn Sensor>],
                sensor_rng_generator(seed, c, 0),
            )
        })
        .collect();

    let mut swarm = SwarmSensors::new(suites);

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
    )
}
