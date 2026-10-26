use anise::prelude::*;
use hifitime::{Epoch, Unit};

use nalgebra::{SMatrix, SVector};

use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::config::{
    BENNU_BSP_FILE, BENNU_MU, BENNU_PCA_FILE, BSP_FILE, FIXED_FRAME, INER_FRAME, PCA_FILE,
    REF_RADIUS_SHM, SHM_DEGREE, fix_bennu_parent, load,
};
use crate::plant::{Plant, StateVector};
use dynamics::gravity::gravity::SphericalHarmonics;
use dynamics::integrators::rk4::rk4;
use onboard_software::config::{EST_STATES, EstimatorVector, SensorVector};
use onboard_software::estimator::Estimator;
use onboard_software::measurements::LinearMeasurement;
use sensors::faults::apply_gaussian;

const BASE_SEED: u64 = 42;
const POS_SIGMA: f64 = 2.0; // m
const VEL_SIGMA: f64 = 0.0003; // k/s (3 mm/s)
pub fn sensor_dynamics(x: EstimatorVector, rng: &mut ChaCha12Rng) -> EstimatorVector {
    let mut y = x;
    for i in 0..3 {
        y[i] = apply_gaussian(x[i], 0.0, POS_SIGMA, rng);
        y[i + 3] = apply_gaussian(x[i + 3], 0.0, VEL_SIGMA, rng);
    }
    y
}

pub fn sim_stepper(
    epoch_0: Epoch,
    x_0: StateVector,
    // u_0: ControlVector,
    t_span: [f64; 2],
    dt: f64,
    mut navigation: Estimator,
) -> (
    Vec<f64>,
    Vec<StateVector>,
    // Vec<ControlVector>,
    Vec<EstimatorVector>,
) {
    // Load ephemeris files
    let almanac = Almanac::new(BSP_FILE)
        .unwrap()
        .load(PCA_FILE)
        .unwrap()
        .load(BENNU_BSP_FILE)
        .unwrap()
        .load(BENNU_PCA_FILE)
        .unwrap();
    let almanac = fix_bennu_parent(almanac);

    let harmonics = load();

    let plant = Plant::new(
        SphericalHarmonics::new(BENNU_MU, REF_RADIUS_SHM, SHM_DEGREE),
        INER_FRAME,
        FIXED_FRAME,
        harmonics.clone(),
    );
    let mut t = t_span[0];
    // let mut epoch: Epoch = epoch_0;
    let mut x = x_0;

    // Defined rng seed
    let mut rng = ChaCha12Rng::seed_from_u64(BASE_SEED);

    // Define the linear measurement
    let model = LinearMeasurement::position_velocity(2.0, 0.0003);

    let covariance0 =
        SMatrix::<f64, EST_STATES, EST_STATES>::from_diagonal(
            &SVector::<f64, EST_STATES>::from_column_slice(&[
                2.0_f64.powi(2),
                2.0_f64.powi(2),
                2.0_f64.powi(2), // position, km^2
                0.0003_f64.powi(2),
                0.0003_f64.powi(2),
                0.0003_f64.powi(2), // velocity, (km/s)^2
            ]),
        );
    let sensor_0 = sensor_dynamics(x, &mut rng);
    navigation.initialize(sensor_0, covariance0);

    let n_steps: usize = ((t_span[1] - t_span[0]) / dt).ceil() as usize + 1;
    let mut t_hist = Vec::with_capacity(n_steps);
    let mut x_hist = Vec::with_capacity(n_steps);
    // let mut u_hist = Vec::with_capacity(n_steps);
    let mut est_hist = Vec::with_capacity(n_steps);
    while t < t_span[1] {
        // Simulation loop!
        // Dynamics update -> Sensor update -> estimator update -> controller update ->

        let epoch = epoch_0 + Unit::Second * t;
        x = rk4(
            &(|t_local: f64, x: StateVector| {
                let e = epoch_0 + Unit::Second * t_local;
                plant.derivative(x, &almanac, e)
            }),
            t,
            dt,
            x,
        );
        // println!("{x}");
        // Sensor update based on state
        let sensor_n = sensor_dynamics(x, &mut rng);

        // Kalman Update/Predict
        navigation.predict(epoch, dt, &almanac, harmonics.clone());
        navigation.update(&model, &sensor_n);

        // Feeding estimated state into controller for controller update
        // let estimator_n = gnc.estimator.get_estimate();
        // u = gnc.controller.control(estimator_n);

        // Step sim forward in time
        t += dt;
        // epoch += dt * Unit::Second;
        t_hist.push(t);
        x_hist.push(x);
        // u_hist.push(u);
        est_hist.push(navigation.get_estimate());
    }
    (t_hist, x_hist, est_hist)
}
