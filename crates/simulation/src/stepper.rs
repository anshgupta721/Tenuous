use anise::prelude::*;
use hifitime::{Epoch, Unit};

use crate::config::{
    BSP_FILE, CEL_FIXED_FRAME, COEFFS, INER_FRAME, MOON_MU, MOON_PA_FILE, Plant, REF_RADIUS_SHM,
    StateVector, load,
};
use dynamics::gravity::gravity::SphericalHarmonics;
use dynamics::integrators::rk4::rk4;
// use dynamics::models::state_space_model::StateSpace;
// use onboard_software::config::NU;
use onboard_software::config::{EstimatorVector, SensorVector};
use onboard_software::control_stack::ControlStack;

pub fn sensor_dynamics(x: StateVector) -> SensorVector {
    x
}

pub fn sim_stepper(
    epoch_0: Epoch,
    x_0: StateVector,
    // u_0: ControlVector,
    t_span: [f64; 2],
    dt: f64,
    mut gnc: ControlStack,
) -> (
    Vec<f64>,
    Vec<StateVector>,
    // Vec<ControlVector>,
    Vec<EstimatorVector>,
) {
    // Load ephemeris files
    let almanac = Almanac::new(BSP_FILE).unwrap().load(MOON_PA_FILE).unwrap();
    let harmonics = load();

    let mut plant = Plant::new(
        Box::new(SphericalHarmonics::new(MOON_MU, REF_RADIUS_SHM, harmonics)),
        INER_FRAME,
        CEL_FIXED_FRAME,
    );
    let mut t = t_span[0];
    // let mut epoch: Epoch = epoch_0;
    let mut x = x_0;
    // let mut u = u_0;
    let sensor_0 = sensor_dynamics(x);
    gnc.estimator.initialize(sensor_0);

    let n_steps: usize = ((t_span[1] - t_span[0]) / dt).ceil() as usize + 1;
    let mut t_hist = Vec::with_capacity(n_steps);
    let mut x_hist = Vec::with_capacity(n_steps);
    // let mut u_hist = Vec::with_capacity(n_steps);
    let mut est_hist = Vec::with_capacity(n_steps);
    while t < t_span[1] {
        // Simulation loop!
        // Dynamics update -> Sensor update -> estimator update -> controller update ->
        x = rk4(
            &(|t_local: f64, x: StateVector| {
                let e = epoch_0 + Unit::Second * t_local;
                plant.derivative(t_local, x, &almanac, e)
            }),
            t,
            dt,
            x,
        );

        // Sensor update based on state
        let sensor_n = sensor_dynamics(x);

        // Kalman Update/Predict
        gnc.estimator.predict();
        gnc.estimator.update(sensor_n);

        // Feeding estimated state into controller for controller update
        // let estimator_n = gnc.estimator.get_estimate();
        // u = gnc.controller.control(estimator_n);

        // Step sim forward in time
        t += dt;
        // epoch += dt * Unit::Second;
        t_hist.push(t);
        x_hist.push(x);
        // u_hist.push(u);
        est_hist.push(gnc.estimator.get_estimate());
    }
    (t_hist, x_hist, est_hist)
}
