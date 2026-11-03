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


def plot_three_sigma_sawtooth(t, states, estimated_states, covariances, cfg: PlotConfig, fig_size: tuple, n_sigma=3.0, indices=None, ncols=2, craft=0):
    indices = list(range(len(cfg.state_names))) if indices is None else list(indices)

    t = np.asarray(t)

    # states / estimates are (T, n_craft, n): pick one craft -> (T, n)
    states = states[:, craft]
    estimated_states = estimated_states[:, craft]

    # covariances arrive interleaved [predict, update, ...]: (2T, n_craft, n, n) -> (2T, n, n)
    covariances = np.asarray(covariances)
    if covariances.ndim == 4:
        covariances = covariances[:, craft]

    interval = n_sigma * cfg.convert(_to_sigma(covariances))
    if len(interval) != 2 * len(t):
        raise ValueError(
            f"covariance has {len(interval)} entries; expected 2 * len(t) = {2 * len(t)}"
        )
    error = cfg.convert(estimated_states - states)
    t_both = np.repeat(t, 2)
    update_interval = interval[1::2]
    fig, axes = _grid(len(indices), ncols, fig_size)
    label = f"±{n_sigma:g}σ (predict/update)"
    for ax, i in zip(axes.flat, indices):
        inside = np.mean(np.abs(error[:, i]) <= update_interval[:, i]) * 100.0
        ax.plot(t, error[:, i], label="Error (post-update)")
        ax.plot(t_both, interval[:, i], "C3--", label=label)
        ax.plot(t_both, -interval[:, i], "C3--")
        ax.fill_between(t_both, -interval[:, i], interval[:, i], color="C3", alpha=0.1)
        ax.set_title(f"{cfg.state_names[i]} ({inside:.1f}% within update bounds)")
        ax.set_ylabel(f"Error ({cfg.state_units[i]})")
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend()
    fig.suptitle(f"Estimation Error with {n_sigma:g}σ Sawtooth Bounds (craft {craft})")
    fig.tight_layout()
    return fig