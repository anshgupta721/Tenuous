
// This simulations statevector:

//[x, y, z, x_dot, y_dot, z_dot]
pub type StateVector = SVector<f64, NX>;



pub struct Plant {
    gravity: Box<dyn GravityField>,
    inertial_frame: Frame,
    body_fixed_frame: Frame,
}

impl Plant {
    pub fn new(
        gravity: Box<dyn GravityField>,
        inertial_frame: Frame,
        body_fixed_frame: Frame,
    ) -> Plant {
        Plant {
            gravity,
            inertial_frame,
            body_fixed_frame,
        }
    }
    pub fn derivative(&self, x: StateVector, almanac: &Almanac, epoch: Epoch) -> StateVector {
        let pos_inertial = SVector::<f64, 3>::from_row_slice(&[x[0], x[1], x[2]]);

        let dcm_i2b = almanac
            .rotate(self.inertial_frame, self.body_fixed_frame, epoch)
            .unwrap();
        let dcm_b2i = almanac
            .rotate(self.body_fixed_frame, self.inertial_frame, epoch)
            .unwrap();
        let pos_body_fixed = dcm_i2b.rot_mat * pos_inertial;
        // For SRP, need to pass in sun's position wrt to Bennu
        let sun_pos_wrt_bennu = almanac
            .translate(SUN_J2000, INER_FRAME, epoch, None)
            .unwrap()
            .radius_km;
        let accel = self.gravity.acceleration(pos_body_fixed);
        // Acceleration in the Bennu J2000 frame,
        let accel_inertial = dcm_b2i.rot_mat * accel; //+ srp_acceleration(sun_pos_wrt_bennu, pos_inertial, SC_CS_AREA / SC_MASS,SC_REFLECTIVITY)
        StateVector::from_row_slice(&[
            x[3],
            x[4],
            x[5],
            accel_inertial[0],
            accel_inertial[1],
            accel_inertial[2],
        ])
    }
}