//! Planned tests for the Bennu spherical-harmonics estimation stack (units: metres).
//!
//! Put this file at `<crate>/tests/gravity_estimation.rs` (next to `src/`).
//!
//! BEFORE RUNNING: fix the `use YOUR_CRATE::...` lines below (crate name and module
//! paths), make sure those modules are `pub`, and check the `MeasurementModel`
//! impl at the bottom matches your trait.
//!
//! Run:
//!   cargo test --test gravity_estimation                         # fast, no kernels needed
//!   ALMANAC_KERNELS=/path/a.bsp:/path/b.bpc \
//!     cargo test --test gravity_estimation -- --nocapture        # + STM tests
//!   ALMANAC_KERNELS=... cargo test --release --test gravity_estimation \
//!     -- --ignored --nocapture                                   # + long-arc / EKF tests
//!   (optional) TEST_EPOCH="2019-01-01T00:00:00 UTC" to pick an epoch your kernels cover.
//!   The kernel list uses your OS path separator (':' on Linux/macOS).
//!
//! TEST PLAN                                   a failure means ...
//! ---------------------------------------------------------------------------------
//! A. Pure gravity (no kernels)
//!  A1 partials_match_fd_of_acceleration       acceleration_partials disagrees with
//!                                             acceleration_and_gradient (units, scaling,
//!                                             normalization, index mapping, frames)
//!  A2 acceleration_is_linear_in_mu            da/dmu = a/mu is invalid
//!  A3 gradient_matches_fd_of_acceleration     dv/dr block of the Jacobian is wrong
//!  A4 coefficient_partial_has_physical_scale  partials are off by an (R/r)^n-type factor
//!  A5 zonal_partials_are_axisymmetric         longitude handling wrong for m = 0
//!  A6 index_mapping_is_a_bijection            idx_c / idx_s collide or leave gaps
//!  A7 parameter_roundtrip                     from/to_parameters disagree with idx_*
//!  A8 reference_radius_is_in_metres           unit mix-up (km vs m) in config
//! B. Dynamics / STM (needs kernels)
//!  B1 stm_matches_fd_of_propagation           STM columns (esp. coefficient columns) wrong
//!  B2 stm_theta_rows_are_identity             parameters leak into dynamics / STM
//!  B3 coefficient_columns_are_nonzero         coefficient sensitivity missing from STM
//! C. Information content / EKF (needs kernels, slow, #[ignore])
//!  C1 observability_report                    prints the best-case posterior sigma per
//!                                             parameter -> tells you if "flat" is physics
//!  C2 ekf_covariance_matches_batch_bound      EKF covariance update loses information
//!                                             (conditioning / cancellation / bug)
//!  C3 ekf_moves_observable_params_to_truth    estimates do not move although observable

use std::str::FromStr;

use anise::prelude::*;
use nalgebra::{SMatrix, SVector};

use dynamics::gravity::gravity::SphericalHarmonics;
use dynamics::gravity::spherical_harmonics::{
    degree_from_length, idx_c, idx_s, length_from_degree, HarmonicCoeffs,
};

// >>> ADJUST THESE PATHS <<<
use onboard_software::config::{
    EstimatorMatrix, EstimatorVector, EST_STATES, REF_RADIUS_SHM, SC_STATES, SHM_DEGREE, SHM_PARAMS,
};
use onboard_software::filters::ekf::BaseEKF;
use onboard_software::filters::filter::Filter;
use onboard_software::measurements::MeasurementModel;
use onboard_software::propagator::GravityDynamics;

// ------------------------------------------------------------------------------------
// Shared helpers
// ------------------------------------------------------------------------------------

/// Bennu GM, m^3/s^2 (approximate; only used to build test scenarios).
const MU: f64 = 4.892;
/// Degree used by the pure-gravity tests (higher than the estimator's to catch
/// indexing errors beyond n = 2).
const TEST_DEGREE: usize = 4;
const TEST_P: usize = length_from_degree(TEST_DEGREE);

/// (radius [m], latitude [deg], longitude [deg]); radii are outside the ~246 m body.
/// Includes near-polar points (the 1/cos(phi) terms).
const SAMPLE_POINTS: [(f64, f64, f64); 8] = [
    (400.0, 0.0, 0.0),
    (600.0, 30.0, 45.0),
    (1000.0, -45.0, 120.0),
    (800.0, 80.0, -70.0),
    (500.0, 10.0, 200.0),
    (1500.0, -20.0, -150.0),
    (600.0, 89.9, 33.0),
    (600.0, -89.9, 200.0),
];

fn pos(r: f64, lat_deg: f64, lon_deg: f64) -> SVector<f64, 3> {
    let (la, lo) = (lat_deg.to_radians(), lon_deg.to_radians());
    SVector::<f64, 3>::new(
        r * la.cos() * lo.cos(),
        r * la.cos() * lo.sin(),
        r * la.sin(),
    )
}

/// Deterministic, non-trivial coefficient set (every entry non-zero, mixed signs).
fn test_params(degree: usize) -> Vec<f64> {
    (0..length_from_degree(degree))
        .map(|k| {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            sign * 1.0e-3 / (1.0 + k as f64)
        })
        .collect()
}

/// "C20", "S21", ... for coefficient index `k`.
fn coeff_name(k: usize, degree: usize) -> String {
    for n in 2..=degree {
        for m in 0..=n {
            if idx_c(n, m) == k {
                return format!("C{n}{m}");
            }
            if m > 0 && idx_s(n, m) == k {
                return format!("S{n}{m}");
            }
        }
    }
    format!("theta[{k}]")
}

/// Name of entry `i` of the estimator state.
fn state_name(i: usize) -> String {
    match i {
        0 => "x".into(),
        1 => "y".into(),
        2 => "z".into(),
        3 => "vx".into(),
        4 => "vy".into(),
        5 => "vz".into(),
        6 => "mu".into(),
        _ => coeff_name(i - SC_STATES - 1, SHM_DEGREE),
    }
}

fn rot_z(a: f64) -> SMatrix<f64, 3, 3> {
    let (s, c) = a.sin_cos();
    SMatrix::<f64, 3, 3>::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
}

/// Builds the Almanac your simulation uses. Replace with your own builder if you
/// have one. Returns None (-> test is skipped) if ALMANAC_KERNELS is not set.
fn almanac() -> Option<Almanac> {
    let kernels = std::env::var_os("ALMANAC_KERNELS")?;
    let mut paths = std::env::split_paths(&kernels);
    let first = paths.next()?;
    let mut almanac = Almanac::new(first.to_str()?).ok()?;
    for p in paths {
        almanac = almanac.load(p.to_str()?).ok()?;
    }
    Some(almanac)
}

macro_rules! require_almanac {
    () => {
        match almanac() {
            Some(a) => a,
            None => {
                eprintln!("SKIPPED: set ALMANAC_KERNELS=<kernel1>:<kernel2>:...");
                return;
            }
        }
    };
}

fn test_epoch() -> Epoch {
    std::env::var("TEST_EPOCH")
        .ok()
        .and_then(|s| Epoch::from_str(&s).ok())
        .unwrap_or_else(|| Epoch::from_gregorian_utc_at_midnight(2019, 1, 1))
}

/// Inclined (45 deg) near-circular orbit at r = 1000 m with the given parameters.
fn nominal_state(params: &[f64]) -> EstimatorVector {
    assert_eq!(params.len(), SHM_PARAMS, "params must match SHM_PARAMS");
    let mut x = EstimatorVector::zeros();
    let r0 = 1000.0;
    let inc = 45.0_f64.to_radians();
    let v = (MU / r0).sqrt();
    x[0] = r0;
    x[4] = v * inc.cos();
    x[5] = v * inc.sin();
    x[6] = MU;
    for (i, &p) in params.iter().enumerate() {
        x[SC_STATES + 1 + i] = p;
    }
    x
}

// ------------------------------------------------------------------------------------
// A. Pure gravity tests (no kernels)
// ------------------------------------------------------------------------------------

fn assert_partials_match_fd(sh: &SphericalHarmonics, params: &[f64], p: SVector<f64, 3>) {
    let analytic = sh.acceleration_partials::<TEST_P>(p, MU);
    assert!(
        analytic.iter().all(|v| v.is_finite()),
        "non-finite partials at {p:?}"
    );

    // Acceleration is linear in each coefficient, so a central difference is exact
    // up to round-off for any step.
    let d = 1.0e-4;
    for k in 0..TEST_P {
        let accel_with = |delta: f64| -> SVector<f64, 3> {
            let mut th = params.to_vec();
            th[k] += delta;
            let c = HarmonicCoeffs::from_parameters(TEST_DEGREE, &th);
            sh.acceleration_and_gradient(p, MU, &c).0
        };
        let fd = (accel_with(d) - accel_with(-d)) / (2.0 * d);
        let col: SVector<f64, 3> = analytic.column(k).into_owned();
        let err = (fd - col).norm();
        let scale = fd.norm().max(col.norm());
        assert!(
            err <= 1.0e-6 * scale + 1.0e-18,
            "{}: analytic {col:?} vs fd {fd:?} (err {err:e}) at p = {p:?}",
            coeff_name(k, TEST_DEGREE)
        );
    }
}

#[test]
fn a1_partials_match_fd_of_acceleration() {
    let sh = SphericalHarmonics::new(REF_RADIUS_SHM, TEST_DEGREE);
    let params = test_params(TEST_DEGREE);
    for &(r, lat, lon) in &SAMPLE_POINTS {
        assert_partials_match_fd(&sh, &params, pos(r, lat, lon));
    }
}

#[test]
fn a2_acceleration_is_linear_in_mu() {
    let sh = SphericalHarmonics::new(REF_RADIUS_SHM, TEST_DEGREE);
    let c = HarmonicCoeffs::from_parameters(TEST_DEGREE, &test_params(TEST_DEGREE));
    for &(r, lat, lon) in &SAMPLE_POINTS[..6] {
        let p = pos(r, lat, lon);
        let a1 = sh.acceleration_and_gradient(p, MU, &c).0;
        let a2 = sh.acceleration_and_gradient(p, 2.0 * MU, &c).0;
        assert!(
            (a2 - 2.0 * a1).norm() <= 1.0e-12 * a1.norm(),
            "a(2 mu) != 2 a(mu) at {p:?}  (so da_dmu = a/mu is wrong)"
        );
    }
}

#[test]
fn a3_gradient_matches_fd_of_acceleration() {
    let sh = SphericalHarmonics::new(REF_RADIUS_SHM, TEST_DEGREE);
    let c = HarmonicCoeffs::from_parameters(TEST_DEGREE, &test_params(TEST_DEGREE));
    let h = 1.0e-2; // m
    for &(r, lat, lon) in &SAMPLE_POINTS[..6] {
        let p = pos(r, lat, lon);
        let (_, g) = sh.acceleration_and_gradient(p, MU, &c);
        for axis in 0..3 {
            let (mut pp, mut pm) = (p, p);
            pp[axis] += h;
            pm[axis] -= h;
            let fd = (sh.acceleration_and_gradient(pp, MU, &c).0
                - sh.acceleration_and_gradient(pm, MU, &c).0)
                / (2.0 * h);
            let col: SVector<f64, 3> = g.column(axis).into_owned();
            let err = (fd - col).norm();
            assert!(
                err <= 1.0e-6 * col.norm().max(fd.norm()),
                "gradient column {axis} at {p:?}: analytic {col:?} vs fd {fd:?}"
            );
        }
    }
}

#[test]
fn a4_coefficient_partial_has_physical_scale() {
    // With fully normalized P, |da/dC20| / |a_central| ~ 3.35 (R/r)^2 on the equator.
    // Accept a generous band: this catches factors like 1e-3, 1e-6, 1e6.
    const P2: usize = length_from_degree(2);
    let sh = SphericalHarmonics::new(REF_RADIUS_SHM, 2);
    let zero = HarmonicCoeffs::from_parameters(2, &vec![0.0; P2]);
    let p = pos(500.0, 0.0, 0.0);
    let a_central = sh.acceleration_and_gradient(p, MU, &zero).0.norm();
    let col = sh
        .acceleration_partials::<P2>(p, MU)
        .column(idx_c(2, 0))
        .norm();
    let ratio = col / a_central;
    let expected = (REF_RADIUS_SHM / 500.0).powi(2);
    assert!(
        ratio > 0.5 * expected && ratio < 10.0 * expected,
        "|da/dC20|/|a| = {ratio:e}, expected about {:e} ((R/r)^2 * O(1..3)); \
         check units of r, REF_RADIUS_SHM and mu in acceleration_partials",
        3.35 * expected
    );
    assert!(
        (a_central - MU / 500.0_f64.powi(2)).abs() < 1e-9 * a_central,
        "central term differs from mu/r^2 (is C00 = 1 set?)"
    );
}

#[test]
fn a5_zonal_partials_are_axisymmetric() {
    let sh = SphericalHarmonics::new(REF_RADIUS_SHM, TEST_DEGREE);
    let p = pos(700.0, 35.0, 20.0);
    let alpha = 1.1;
    let a = sh.acceleration_partials::<TEST_P>(p, MU);
    let b = sh.acceleration_partials::<TEST_P>(rot_z(alpha) * p, MU);
    for n in 2..=TEST_DEGREE {
        let k = idx_c(n, 0);
        let expect = rot_z(alpha) * a.column(k).into_owned();
        let got: SVector<f64, 3> = b.column(k).into_owned();
        assert!(
            (got - expect).norm() <= 1.0e-9 * expect.norm(),
            "C{n}0 partial is not axisymmetric: {got:?} vs {expect:?}"
        );
    }
}

#[test]
fn a6_index_mapping_is_a_bijection() {
    for degree in 2..=8 {
        let len = length_from_degree(degree);
        assert_eq!(degree_from_length(len), degree);
        let mut seen = vec![false; len];
        for n in 2..=degree {
            for m in 0..=n {
                let mut idxs = vec![idx_c(n, m)];
                if m > 0 {
                    idxs.push(idx_s(n, m));
                }
                for i in idxs {
                    assert!(i < len, "index {i} out of range (n={n}, m={m})");
                    assert!(!seen[i], "index {i} used twice (n={n}, m={m})");
                    seen[i] = true;
                }
            }
        }
        assert!(
            seen.iter().all(|&s| s),
            "gaps in index map for degree {degree}"
        );
    }
}

#[test]
fn a7_parameter_roundtrip() {
    let th = test_params(TEST_DEGREE);
    let c = HarmonicCoeffs::from_parameters(TEST_DEGREE, &th);
    assert_eq!(c.to_parameters(), th);
    assert_eq!(c.get_c(0, 0), 1.0);
    assert_eq!(c.get_c(2, 0), th[idx_c(2, 0)]);
    assert_eq!(c.get_s(2, 1), th[idx_s(2, 1)]);
    assert_eq!(c.get_c(3, 3), th[idx_c(3, 3)]);
    assert_eq!(c.get_s(3, 3), th[idx_s(3, 3)]);
}

#[test]
fn a8_reference_radius_is_in_metres() {
    // Bennu is ~246 m in radius. 0.246 (km) or 246000 (mm) would mean a unit mix-up.
    assert!(
        (100.0..500.0).contains(&REF_RADIUS_SHM),
        "REF_RADIUS_SHM = {REF_RADIUS_SHM}: expected ~246 (m)"
    );
}

// ------------------------------------------------------------------------------------
// B. Dynamics / STM tests (need kernels)
// ------------------------------------------------------------------------------------

/// FD step for entry `k` of the estimator state.
fn fd_step(k: usize) -> f64 {
    match k {
        0..=2 => 1.0e-2, // m
        3..=5 => 1.0e-6, // m/s
        6 => 1.0e-3,     // m^3/s^2
        _ => 1.0e-3,     // coefficients (dynamics are ~linear in them)
    }
}

#[test]
fn b1_stm_matches_fd_of_propagation() {
    let almanac = require_almanac!();
    let epoch = test_epoch();
    let gd = GravityDynamics::new();
    let dt = 120.0;
    let x0 = nominal_state(&test_params(SHM_DEGREE));
    let (_, phi) = gd.propagate(&x0, epoch, dt, &almanac);

    for k in 0..EST_STATES {
        let d = fd_step(k);
        let (mut xp, mut xm) = (x0, x0);
        xp[k] += d;
        xm[k] -= d;
        let (yp, _) = gd.propagate(&xp, epoch, dt, &almanac);
        let (ym, _) = gd.propagate(&xm, epoch, dt, &almanac);
        let fd = (yp - ym) / (2.0 * d);

        let fd6: SVector<f64, 6> = fd.fixed_rows::<6>(0).into_owned();
        let col6: SVector<f64, 6> = phi.fixed_view::<6, 1>(0, k).into_owned();
        let err = (fd6 - col6).norm();
        let scale = fd6.norm().max(col6.norm());
        assert!(
            err <= 1.0e-4 * scale + 1.0e-12,
            "STM column {} ({k}): |fd| = {:e}, |phi| = {:e}, err = {err:e}",
            state_name(k),
            fd6.norm(),
            col6.norm()
        );
    }
}

#[test]
fn b2_stm_theta_rows_are_identity() {
    let almanac = require_almanac!();
    let gd = GravityDynamics::new();
    let x0 = nominal_state(&test_params(SHM_DEGREE));
    let (x1, phi) = gd.propagate(&x0, test_epoch(), 120.0, &almanac);
    for i in SC_STATES..EST_STATES {
        assert!(
            (x1[i] - x0[i]).abs() < 1e-15,
            "{} changed in propagate",
            state_name(i)
        );
        for j in 0..EST_STATES {
            let expect = if i == j { 1.0 } else { 0.0 };
            assert!(
                (phi[(i, j)] - expect).abs() < 1e-15,
                "phi[{}, {}] = {} (expected {expect})",
                state_name(i),
                state_name(j),
                phi[(i, j)]
            );
        }
    }
}

#[test]
fn b3_coefficient_columns_are_nonzero() {
    let almanac = require_almanac!();
    let gd = GravityDynamics::new();
    let x0 = nominal_state(&test_params(SHM_DEGREE));
    let (_, phi) = gd.propagate(&x0, test_epoch(), 600.0, &almanac);
    println!("{:<6} {:>14} {:>14}", "param", "|dx/dp|", "|dv/dp|");
    for i in SC_STATES..EST_STATES {
        let dx = phi.fixed_view::<3, 1>(0, i).norm();
        let dv = phi.fixed_view::<3, 1>(3, i).norm();
        println!("{:<6} {:>14.4e} {:>14.4e}", state_name(i), dx, dv);
        assert!(
            dv > 1e-14,
            "velocity sensitivity to {} is ~0: it is missing from the STM",
            state_name(i)
        );
    }
}

// ------------------------------------------------------------------------------------
// C. Information content and EKF behaviour (slow)
// ------------------------------------------------------------------------------------

const ARC_S: f64 = 4.0e5;
const STEP_S: f64 = 120.0;
const SIGMA_MEAS: f64 = 1.0; // m, position measurement sigma (ADJUST to your sensor)

/// Prior 1-sigma for every estimator entry. Coefficient/mu values are read off your
/// plot (3-sigma bands); position/velocity are guesses -> ADJUST.
fn prior_sigmas() -> EstimatorVector {
    let mut s = EstimatorVector::zeros();
    for i in 0..3 {
        s[i] = 1.0; // m
    }
    for i in 3..6 {
        s[i] = 1.0e-3; // m/s
    }
    s[6] = 9.0e-4; // mu
    let theta = [5.2e-5, 7.7e-10, 7.7e-10, 5.2e-5, 1.05e-4]; // C20 C21 S21 C22 S22
    for i in 0..SHM_PARAMS {
        s[SC_STATES + 1 + i] = theta.get(i).copied().unwrap_or(1.0e-6);
    }
    s
}

/// Cumulative STMs Phi(t_k, t_0) for k = 1..N along the nominal trajectory.
fn cumulative_stms(
    gd: &GravityDynamics,
    x0: &EstimatorVector,
    t0: Epoch,
    almanac: &Almanac,
) -> Vec<EstimatorMatrix> {
    let n = (ARC_S / STEP_S) as usize;
    let (mut x, mut epoch) = (*x0, t0);
    let mut phi_tot = EstimatorMatrix::identity();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let (xn, phi) = gd.propagate(&x, epoch, STEP_S, almanac);
        phi_tot = phi * phi_tot;
        x = xn;
        epoch = epoch + STEP_S * Unit::Second;
        out.push(phi_tot);
    }
    out
}

/// Batch (information-form) posterior of the *normalized* parameters y = theta / sigma0
/// from position measurements at every step. Normalizing keeps the 1e0..1e-20 dynamic
/// range of the raw covariance out of the inversion. Returns inv(I + sum J^T J / R).
fn batch_posterior_normalized(phis: &[EstimatorMatrix], sig: &EstimatorVector) -> EstimatorMatrix {
    let s = EstimatorMatrix::from_diagonal(sig);
    let mut info = EstimatorMatrix::identity();
    for phi in phis {
        let j: SMatrix<f64, 3, EST_STATES> = phi.fixed_rows::<3>(0) * s;
        info += j.transpose() * j / (SIGMA_MEAS * SIGMA_MEAS);
    }
    info.try_inverse().expect("information matrix singular")
}

#[test]
#[ignore = "slow: run with --release --ignored --nocapture"]
fn c1_observability_report() {
    let almanac = require_almanac!();
    let gd = GravityDynamics::new();
    let sig = prior_sigmas();
    let x0 = nominal_state(&test_params(SHM_DEGREE));
    let phis = cumulative_stms(&gd, &x0, test_epoch(), &almanac);
    let post = batch_posterior_normalized(&phis, &sig);
    let last = phis.last().unwrap();

    println!(
        "\nBest-case posterior after {ARC_S:.0} s of {SIGMA_MEAS} m position fixes every {STEP_S} s"
    );
    println!(
        "{:<6} {:>12} {:>12} {:>10} {:>16}",
        "param", "prior sigma", "post sigma", "post/prior", "1-sigma drift [m]"
    );
    for i in 0..EST_STATES {
        let ratio = post[(i, i)].sqrt();
        let drift = last.fixed_view::<3, 1>(0, i).norm() * sig[i];
        println!(
            "{:<6} {:>12.3e} {:>12.3e} {:>10.4} {:>16.4e}",
            state_name(i),
            sig[i],
            ratio * sig[i],
            ratio,
            drift
        );
        assert!(post[(i, i)].is_finite() && post[(i, i)] > 0.0);
        assert!(
            post[(i, i)] <= 1.0 + 1e-9,
            "information cannot increase variance"
        );
    }
    if EST_STATES > 7 {
        let c = post[(6, 7)] / (post[(6, 6)] * post[(7, 7)]).sqrt();
        println!("posterior correlation mu <-> {}: {c:+.4}", state_name(7));
    }
    println!(
        "post/prior ~ 1 means the parameter is NOT observable with these settings \
         (flat plot = physics). Much below 1 means the EKF should be shrinking it."
    );
}

/// Position measurement. ASSUMPTION: `MeasurementModel<M>` has exactly these three
/// methods (as called by BaseEKF::update). Adjust to your trait if it differs.
struct PositionMeasurement {
    sigma: f64,
}

impl MeasurementModel<3> for PositionMeasurement {
    fn predict(&self, x: &EstimatorVector) -> SVector<f64, 3> {
        x.fixed_rows::<3>(0).into_owned()
    }
    fn jacobian(&self, _x: &EstimatorVector) -> SMatrix<f64, 3, EST_STATES> {
        let mut h = SMatrix::<f64, 3, EST_STATES>::zeros();
        h.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&SMatrix::<f64, 3, 3>::identity());
        h
    }
    fn noise(&self, _x: &EstimatorVector) -> SMatrix<f64, 3, 3> {
        SMatrix::<f64, 3, 3>::identity() * self.sigma.powi(2)
    }
}

/// Runs the EKF over the arc against noiseless position fixes generated from
/// `truth0` (propagated with the same dynamics). Returns (ekf, final truth).
fn run_ekf(
    almanac: &Almanac,
    x_guess: EstimatorVector,
    truth0: EstimatorVector,
    sig: &EstimatorVector,
) -> (BaseEKF, EstimatorVector) {
    let gd = GravityDynamics::new();
    let p0 = EstimatorMatrix::from_diagonal(&sig.component_mul(sig));
    // Q = 0: no process noise, so the EKF must equal the batch solution.
    let mut ekf = BaseEKF::new(x_guess, p0, EstimatorMatrix::zeros());
    let meas = PositionMeasurement { sigma: SIGMA_MEAS };

    let n = (ARC_S / STEP_S) as usize;
    let (mut truth, mut epoch) = (truth0, test_epoch());
    for _ in 0..n {
        ekf.predict(epoch, STEP_S, almanac);
        truth = gd.propagate(&truth, epoch, STEP_S, almanac).0;
        let z: SVector<f64, 3> = truth.fixed_rows::<3>(0).into_owned();
        ekf.update(&meas, &z);
        epoch = epoch + STEP_S * Unit::Second;
    }
    (ekf, truth)
}

#[test]
#[ignore = "slow: run with --release --ignored --nocapture"]
fn c2_ekf_covariance_matches_batch_bound() {
    let almanac = require_almanac!();
    let gd = GravityDynamics::new();
    let sig = prior_sigmas();
    let x0 = nominal_state(&test_params(SHM_DEGREE));

    // truth == guess -> innovations are exactly zero -> the linearization point never
    // moves, so EKF covariance must equal the batch information-form posterior.
    let (ekf, _) = run_ekf(&almanac, x0, x0, &sig);
    let p = ekf.covariance();

    let phis = cumulative_stms(&gd, &x0, test_epoch(), &almanac);
    let batch = batch_posterior_normalized(&phis, &sig);
    let s_inv = EstimatorMatrix::from_diagonal(&sig.map(|v| 1.0 / v));
    let p_norm = s_inv * p * s_inv;

    println!(
        "{:<6} {:>14} {:>14}",
        "param", "EKF post/prior", "batch post/prior"
    );
    for i in 0..EST_STATES {
        println!(
            "{:<6} {:>14.5} {:>14.5}",
            state_name(i),
            p_norm[(i, i)].sqrt(),
            batch[(i, i)].sqrt()
        );
    }

    assert!(
        (0..EST_STATES).all(|i| p[(i, i)] > 0.0),
        "EKF covariance has non-positive diagonal entries (numerical breakdown)"
    );
    assert!(
        (p - p.transpose()).norm() <= 1.0e-9 * p.norm(),
        "EKF covariance lost symmetry"
    );
    let max_diff = (p_norm - batch).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        max_diff < 1.0e-6,
        "normalized EKF covariance differs from batch bound by {max_diff:e}: the EKF \
         update is losing information (use Joseph form / scale theta) or Phi is inconsistent"
    );
}

#[test]
#[ignore = "slow: run with --release --ignored --nocapture"]
fn c3_ekf_moves_observable_params_to_truth() {
    let almanac = require_almanac!();
    let gd = GravityDynamics::new();
    let sig = prior_sigmas();
    let x_guess = nominal_state(&test_params(SHM_DEGREE));

    // Truth differs from the guess by +1 prior sigma in mu and every coefficient.
    let mut truth0 = x_guess;
    for i in SC_STATES..EST_STATES {
        truth0[i] += sig[i];
    }

    let phis = cumulative_stms(&gd, &x_guess, test_epoch(), &almanac);
    let batch = batch_posterior_normalized(&phis, &sig);
    let (ekf, truth) = run_ekf(&almanac, x_guess, truth0, &sig);
    let est = ekf.estimate();

    println!(
        "{:<6} {:>14} {:>14} {:>12}",
        "param", "init err/sig", "final err/sig", "post/prior"
    );
    let mut checked = 0;
    for i in SC_STATES..EST_STATES {
        let init = 1.0; // by construction
        let fin = ((est[i] - truth[i]) / sig[i]).abs();
        let ratio = batch[(i, i)].sqrt();
        println!(
            "{:<6} {:>14.4} {:>14.4} {:>12.4}",
            state_name(i),
            init,
            fin,
            ratio
        );
        if ratio < 0.3 {
            checked += 1;
            assert!(
                fin < 0.9 * init,
                "{} is observable (post/prior = {ratio:.3}) but the EKF error did not shrink",
                state_name(i)
            );
        }
    }
    if checked == 0 {
        println!("No parameter is observable enough to check; see c1_observability_report.");
    }
}