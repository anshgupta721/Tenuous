use nalgebra::SVector;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use sensors::faults::apply_gaussian;

use crate::plant::StateVector;
use crate::world::Spacecraft;

pub const MEASUREMENTS: usize = 6;
pub type SensorVector = SVector<f64, MEASUREMENTS>;

pub const POS_SIGMA: f64 = 2.0; // m
pub const VEL_SIGMA: f64 = 0.0003; // k/s (3 mm/s)

pub struct SensorID {
    pub index: u8,
}

impl SensorID {
    pub const fn new(index: u8) -> Self {
        Self { index }
    }
}

pub trait Sensor {
    fn new(id: SensorID) -> Self
    where
        Self: Sized;
    fn id(&self) -> &SensorID;
    fn measurement(&self, truth: StateVector, rng: &mut ChaCha12Rng) -> Vec<f64>;
}

pub struct PosVelSensor {
    id: SensorID,
}

impl Sensor for PosVelSensor {
    fn new(id: SensorID) -> Self {
        Self { id }
    }

    fn id(&self) -> &SensorID {
        &self.id
    }

    fn measurement(&self, truth: StateVector, rng: &mut ChaCha12Rng) -> Vec<f64> {
        let mut y = truth;
        for i in 0..3 {
            y[i] = apply_gaussian(truth[i], 0.0, POS_SIGMA, rng);
            y[i + 3] = apply_gaussian(truth[i + 3], 0.0, VEL_SIGMA, rng);
        }
        y.as_slice().to_vec()
    }
}

pub fn sensor_rng_generator(seed: u64, craft: usize, sensor_idx: usize) -> ChaCha12Rng {
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    rng.set_stream(craft as u64 * 256 + sensor_idx as u64);
    rng
}

pub struct SpacecraftSuite {
    spacecraft_sensors: Vec<Box<dyn Sensor>>,
    rng: ChaCha12Rng,
}

impl SpacecraftSuite {
    pub fn new(spacecraft_sensors: Vec<Box<dyn Sensor>>, rng: ChaCha12Rng) -> Self {
        Self {
            spacecraft_sensors: spacecraft_sensors,
            rng,
        }
    }

    pub fn measurements(&mut self, craft: &Spacecraft) -> SensorVector {
        let mut flat = Vec::new();
        for sensor in &self.spacecraft_sensors {
            // Compute measurements for one spacecraft
            flat.extend(sensor.measurement(craft.x, &mut self.rng));
        }
        SensorVector::from_column_slice(&flat)
    }
}

pub struct SwarmSensors {
    // Sensors
    swarm: Vec<SpacecraftSuite>,
}

impl SwarmSensors {
    pub fn new(swarm: Vec<SpacecraftSuite>) -> Self {
        Self { swarm }
    }

    pub fn measurements(&mut self, crafts: &[Spacecraft]) -> Vec<SensorVector> {
        assert_eq!(
            crafts.len(),
            self.swarm.len(),
            "number of spacecraft must match number of sensor suites"
        );
        self.swarm
            .iter_mut()
            .zip(crafts.iter())
            .map(|(suite, craft)| suite.measurements(craft))
            .collect()
    }
}
