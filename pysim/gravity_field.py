import numpy as np
import pandas as pd


def load_gravity_field(filename: str, degree: int) -> np.ndarray:
    """Loads gravitational field from CSV"""
    data = np.loadtxt(filename, delimiter=",", skiprows=1)  # columns: n, m, c, s
    rows = data[(data[:, 0] >= 2) & (data[:, 0] <= degree)]
    return np.array([x for n, m, c, s in rows for x in ((c,) if m == 0 else (c, s))])


def load_gravity_field_covariance(filename: str, degree: int) -> np.ndarray:
    """Loads gravitational covariance matrix"""
    cov = pd.read_csv(filename, index_col=0, float_precision="round_trip").to_numpy()
    n_terms = (degree + 1) ** 2 - 3
    return cov[:n_terms, :n_terms]


def length_from_degree(degree: int):
    """Outputs length from a spherical harmonic degree"""
    return (degree + 1) * (degree + 1) - 4
