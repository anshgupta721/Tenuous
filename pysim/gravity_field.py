import numpy as np

def load_gravity_field(filename: str, degree: int):
    # Load the gravitational field to some degree

    data = np.loadtxt(filename, delimiter=",", skiprows=1)  # columns: n, m, c, s
    rows = data[data[:, 0] <= degree]
    return rows[:, 2:4].ravel()


