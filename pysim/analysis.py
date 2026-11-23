import numpy as np
from scipy.linalg import cho_solve, cho_factor


# Function for calculating either the NEES or NIS for consistency
def calculate_error_statistic(err, P):
    return float(err.T @ cho_solve(cho_factor(P), err))
