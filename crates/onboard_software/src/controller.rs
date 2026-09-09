use crate::config::{ControlVector, EstimatorVector};

pub struct Controller {
    prev_integral: f64,
}

impl Controller {
    pub fn new() -> Controller {
        Controller { prev_integral: 0.0 }
    }

    pub fn control(&self, estimate: EstimatorVector) -> ControlVector {
        ControlVector::zeros()
    }
}
