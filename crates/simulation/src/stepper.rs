use anise::prelude::*;

use dynamics::gravity::spherical_harmonics::HarmonicCoeffs;

use onboard_software::config::{EstimatorMatrix, EstimatorVector};
use onboard_software::filter::{Context, Filter};
use onboard_software::independent_swarm::IndependentSwarm;
use onboard_software::measurements::{Models, LinearMeasurement};
use onboard_software::swarm_navigation::SwarmNavigation;
use onboard_software::ekf::BaseEKF;


use crate::plant::StateVector;
use crate::sensors::{SensorID, Sensor, SpacecraftSuite, SwarmSensors, PosVelSensor, sensor_rng_generator, POS_SIGMA, VEL_SIGMA};
use crate::world::World;

pub struct History {
    pub n_craft: usize,
    pub t: Vec<f64>,
    pub truth: Vec<StateVector>, // [step][craft][NX]       Spacecraft1-tk(x, y, z, vx, vy, vz)----
    pub est: Vec<EstimatorVector>, // [step][craft][EST_STATES]
    pub cov: Vec<EstimatorMatrix>, // [step][craft][predict/update][row][col]
}

impl History {
    pub fn with_capacity(steps: usize, n_craft: usize) -> Self {
        Self {
            n_craft,
            t: Vec::with_capacity(steps),
            truth: Vec::with_capacity(steps * n_craft),
            est: Vec::with_capacity(steps * n_craft),
            cov: Vec::with_capacity(steps * n_craft),
        }
    }
}

pub fn run<F: Filter, SwrmNav: SwarmNavigation<F>>(
    world: &mut World,
    sensors: &mut SwarmSensors,
    filter: &mut SwrmNav,
    models: &Models,
    t_end: f64,
    dt: f64,
) -> History {
    let n = ((t_end - world.t) / dt).ceil() as usize + 1;
    let mut history = History::with_capacity(n, world.crafts.len());
    for _ in 0..n {
        // Propagate all spacecraft from n to n+1
        let epoch = world.step(dt);
        let ctx: Context = Context {
            almanac: &world.almanac,
            harmonics: &world.harmonics,
        };

        // Generate measurements for each spacecraft
        let meas = sensors.measurements(&world.crafts);
        // let y: Vec<SensorVector> = meas.iter.map(|per_craft| )


        // Each spacecraft should estimate
        filter.predict(epoch, dt, ctx);
        history.cov.extend(filter.covariance());
        filter.correct(&models.pos_vel, &meas);
        history.cov.extend(filter.covariance());

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

    run(&mut world, &mut swarm, &mut navigation, &models, t_span[1], dt)
}
