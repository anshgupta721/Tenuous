use std::range;

use nalgebra::{Isometry3, Rotation3, SVector, Translation, UnitQuaternion};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use anise::prelude::*;
use hifitime::Epoch;

use parry3d_f64::math::Vec3;
use parry3d_f64::shape::TriMesh;
use parry3d_f64::query::{Ray, RayCast};
use tobj;

use dynamics::gravity::gravity::SphericalHarmonics;
use dynamics::gravity::spherical_harmonics::HarmonicCoeffs;
use dynamics::integrators::rk4::rk4;

use crate::graph::Graph;
use crate::plant::{Plant, StateVector};

use crate::config::{
    BENNU_BSP_FILE, BENNU_PCA_FILE, BSP_FILE, FIXED_FRAME, INER_FRAME, PCA_FILE, REF_RADIUS_SHM,
    SHAPE_MODEL, SHAPE_SCALE, SHM_DEGREE, fix_bennu_parent, load_tobj_options,
};

pub struct Observables {
    specific_force: SVector<f64, 3>,
}

impl Observables {
    pub fn new() -> Self {
        Self {
            specific_force: SVector::<f64, 3>::zeros(),
        }
    }
}

pub struct Spacecraft {
    pub x: StateVector,
    pub y: Observables,
}

pub struct World {
    pub almanac: Almanac,
    pub harmonics: HarmonicCoeffs,
    pub mu: f64,
    plant: Plant,
    epoch_0: Epoch,
    pub t: f64,
    pub crafts: Vec<Spacecraft>,
    shape: TriMesh,
    pub shape_pose: Isometry3<f64>, // Pose in body frame
}

impl World {
    pub fn new(
        epoch_0: Epoch,
        x_0: &[StateVector],
        t_0: f64,
        seed: u64,
        mu: f64,
        harmonics: HarmonicCoeffs,
    ) -> Self {
        // Loading in SPICE kernels with Almanac
        let almanac = Almanac::new(BSP_FILE)
            .unwrap()
            .load(PCA_FILE)
            .unwrap()
            .load(BENNU_BSP_FILE)
            .unwrap()
            .load(BENNU_PCA_FILE)
            .unwrap();
        let almanac = fix_bennu_parent(almanac);

        // Instantiate the plant with spherical harmonics gravity and coordinate frames
        let plant = Plant::new(
            SphericalHarmonics::new(REF_RADIUS_SHM, SHM_DEGREE),
            INER_FRAME,
            FIXED_FRAME,
            mu,
            harmonics.clone(),
        );

        // Instantiate each craft with a different set of RNG's that the sensors module can use
        let crafts: Vec<Spacecraft> = x_0
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let mut rng = ChaCha12Rng::seed_from_u64(seed);
                rng.set_stream(i as u64);
                Spacecraft {
                    x,
                    y: Observables::new(),
                }
            })
            .collect();

        // Link the asteroid shape model into the spin given by the ANISE Almanac

        // Read OBJ
        let (models, _material) = tobj::load_obj(SHAPE_MODEL, &load_tobj_options()).unwrap();
        let mesh = &models[0].mesh;

        // Convert the tobj loaded mesh into a parry3d TriMesh object
        let vertices: Vec<_> = mesh
            .positions
            .chunks_exact(3)
            .map(|p| Vec3::new(p[0] as f64, p[1] as f64, p[2] as f64) * SHAPE_SCALE)
            .collect();
        let indices = mesh
            .indices
            .chunks_exact(3)
            .map(|f| [f[0], f[1], f[2]])
            .collect();
        // Conduct BVH on the asteroid
        let trimesh: TriMesh = TriMesh::new(vertices, indices).unwrap();

        Self {
            almanac,
            harmonics,
            mu,
            plant,
            epoch_0,
            t: t_0,
            crafts,
            shape: trimesh,
            shape_pose: Isometry3::identity(),
        }
    }

    pub fn step(&mut self, dt: f64) -> Epoch {
        let epoch = self.epoch();
        for i in &mut self.crafts {
            i.x = rk4(
                &(|t_local: f64, x: StateVector| {
                    let e = self.epoch_0 + Unit::Second * t_local;
                    self.plant.derivative(x, &self.almanac, e)
                }),
                self.t,
                dt,
                i.x,
            );
        }
        self.t += dt;
        epoch
    }

    pub fn epoch(&self) -> Epoch {
        self.epoch_0 + Unit::Second * self.t
    }

    pub fn update_shape_pose(&mut self) {
        // extract what the pose of the body is wrt inertial
        let pose = self
            .almanac
            .rotate(
                self.plant.inertial_frame,
                self.plant.body_fixed_frame,
                self.epoch(),
            )
            .unwrap();
        // Set the pose of the shape model
        let quaternion =
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(pose.rot_mat));
        // Translation is identity since we propagate all from a bennu centered coordinate frame
        self.shape_pose = Isometry3::from_parts(Translation::identity(), quaternion);
    }

    pub fn line_of_sight(&mut self, n_craft: usize, comm_graph: &mut Graph) {
        // Check to see if there are obstructions between spacecraft

        let body_pos: Vec<SVector<f64,3>> = self.crafts.iter().map(|craft| { let vec =  self.shape_pose.transform_vector(&craft.x.fixed_rows::<3>(0).into_owned()); vec } ).collect();


        for i in 0..n_craft {
            for j in i + 1..n_craft {
                // calculate position vector between spacecraft
                let range_vector: SVector<f64, 3> =
                    body_pos[i]
                        - body_pos[j];
                // Shoot a ray along the vector and see if it hits the object
                let origin = body_pos[i];
                
                let ray = Ray::new(Vec3::new(origin.x, origin.y, origin.z), Vec3::new(range_vector.x, range_vector.y, range_vector.z));
                let blocked = self.shape.cast_local_ray(&ray, 1.0, true).is_some();
                
                comm_graph.set_visibility(i, j, blocked);
            }
        }
    }
}
