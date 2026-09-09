use crate::controller::Controller;
use crate::estimator::Estimator;

pub struct ControlStack {
    pub estimator: Estimator,
    pub controller: Controller,
}

impl ControlStack {
    pub fn new() -> ControlStack {
        ControlStack {
            estimator: Estimator::new(),
            controller: Controller::new(),
        }
    }
}
