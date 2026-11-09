import matplotlib.pyplot as plt
import numpy as np


def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    middles = limits.mean(axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([middles[0] - radius, middles[0] + radius])
    ax.set_ylim3d([middles[1] - radius, middles[1] + radius])
    ax.set_zlim3d([middles[2] - radius, middles[2] + radius])

def _grid(n_plots, ncols, figsize):
    """Shared-x grid with enough rows for n_plots; extra panels are hidden."""
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False,
                             figsize=figsize)
    for ax in axes.flat[n_plots:]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    return fig, axes

LENGTH_UNITS = {
    "m": 1.0,
    "km": 1e3,
    "cm": 1e-2,
    "mm": 1e-3,
    "ft": 0.3048,
    "mi": 1609.344,
}

LENGTH = "length"
VELOCITY = "velocity"
# Default layout of the 6-state vector [x, y, z, vx, vy, vz].
DEFAULT_STATE_KINDS = (LENGTH,) * 3 + (VELOCITY,) * 3


def length_scale(unit: str) -> float:
    """Meters per `unit`. Raises a helpful error for unknown units."""
    try:
        return LENGTH_UNITS[unit]
    except KeyError:
        raise ValueError(
            f"Unknown length unit {unit!r}. Choose from {sorted(LENGTH_UNITS)}"
        ) from None

def from_meters(values, unit: str):
    """Convert meters (or m/s) to `unit` (or unit/s)."""
    return np.asarray(values) / length_scale(unit)



