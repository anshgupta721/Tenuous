use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

pub fn jacobian<F>(f: F, x: SVector<f64, 3>, h: f64) -> Matrix3<f64>
where
    F: Fn(&Vector3<f64>) -> (SVector<f64, 3>, SMatrix<f64, 3, 3>),
{
    let mut j = Matrix3::zeros();
    for k in 0..3 {
        let mut xp = x;
        let mut xm = x;
        xp[k] += h;
        xm[k] -= h;
        let dx = xp[k] - xm[k]; // actual step after rounding
        j.set_column(k, &((f(&xp).0 - f(&xm).0) / dx));
    }
    j
}
