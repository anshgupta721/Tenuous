import numpy as np
import matplotlib.pyplot as plt


from config import PlotConfig
from .common import _grid

def _to_sigma(covariances):
    """(N, n, n) covariances (m^2) or (N, n) sigmas (m) -> (N, n) sigmas."""
    covariances = np.asarray(covariances)
    if covariances.ndim == 3:
        return np.sqrt(np.diagonal(covariances, axis1=1, axis2=2))
    if covariances.ndim == 2:
        return covariances
    raise ValueError(f"covariances must be (N, n, n) or (N, n); got {covariances.shape}")


def plot_error_sigma(t, error, sigma, labels, units, fig_size,
                     n_sigma=3.0, ncols=2, title=""):
    """error: (T, k); sigma: (2T, k) interleaved [predict, update, ...]."""
    t = np.asarray(t)
    if len(sigma) != 2 * len(t):
        raise ValueError(f"expected {2 * len(t)} sigma rows, got {len(sigma)}")
    interval = n_sigma * sigma
    update_interval = interval[1::2]
    t_both = np.repeat(t, 2)

    fig, axes = _grid(error.shape[1], ncols, fig_size)
    for k, ax in zip(range(error.shape[1]), axes.flat):
        inside = np.mean(np.abs(error[:, k]) <= update_interval[:, k]) * 100.0
        ax.plot(t, error[:, k], label="Error (post-update)")
        ax.plot(t_both, interval[:, k], "C3--", label=f"±{n_sigma:g}σ (predict/update)")
        ax.plot(t_both, -interval[:, k], "C3--")
        ax.fill_between(t_both, -interval[:, k], interval[:, k], color="C3", alpha=0.1)
        ax.set_title(f"{labels[k]} ({inside:.1f}% within update bounds)")
        ax.set_ylabel(f"Error ({units[k]})")
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig

def plot_state_sigma(t, states, est, cov, cfg, fig_size, craft=0, **kw):
    err = cfg.convert(est[:, craft, :6] - states[:, craft])
    sig = cfg.convert(_to_sigma(cov[:, craft, :6, :6]))
    return plot_error_sigma(t, err, sig, cfg.state_names, cfg.state_units,
                            fig_size, title=f"State error (craft {craft})", **kw)

def plot_param_sigma(t, est, cov, truth_params, labels, units, fig_size,
                     indices=None, craft=0, **kw):
    """truth_params: (p,) true [mu, coeffs...], aligned with est[..., 6:]."""
    idx = np.arange(len(truth_params)) if indices is None else np.asarray(indices)
    err = est[:, craft, 6:][:, idx] - np.asarray(truth_params)[idx]
    cov_p = cov[:, craft, 6:, 6:]
    sig = np.sqrt(np.diagonal(cov_p, axis1=1, axis2=2))[:, idx]
    return plot_error_sigma(t, err, sig, [labels[i] for i in idx],
                            [units[i] for i in idx], fig_size,
                            title=f"Parameter error (craft {craft})", **kw)