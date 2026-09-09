use pyo3::prelude::*;

mod python_bindings;

#[pymodule]
fn pyfrontend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python_bindings::pysim_runner, m)?)?;
    Ok(())
}
