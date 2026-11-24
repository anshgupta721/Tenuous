from scipy.linalg import cho_solve, cho_factor


def calculate_error_statistic(err, P):
    """Function for calculating either the NEES or NIS for consistency"""
    return float(err.T @ cho_solve(cho_factor(P), err))
