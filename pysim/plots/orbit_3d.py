import matplotlib.pyplot as plt
from config import PlotConfig
from .common import set_axes_equal


def plot_orbit_3d(states, fig_size: tuple, cfg: PlotConfig):
    # states has shape (T, n_craft, n_states); every spacecraft goes on the same axes.
    n_craft = states.shape[1]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection="3d")
    for c in range(n_craft):
        pos = states[:, c, :3]
        ax.plot(*pos.T, label=f"Spacecraft {c}")
        # Label Start/End once so the legend doesn't repeat them for every craft.
        ax.scatter(*pos[0], color="green", s=50, label="Start" if c == 0 else None)
        ax.scatter(*pos[-1], color="red", s=50, label="End" if c == 0 else None)
    ax.set_title("3D Orbit")
    ax.set_xlabel(f"x ({cfg.length_unit})")
    ax.set_ylabel(f"y ({cfg.length_unit})")
    ax.set_zlabel(f"z ({cfg.length_unit})")
    ax.set_box_aspect((1, 1, 1))
    ax.legend()
    set_axes_equal(ax)

    def lock_roll(_event):
        """Keep the 3D plot's roll at zero while rotating."""
        ax.view_init(elev=ax.elev, azim=ax.azim, roll=0.0)

    fig.tight_layout()
    fig.canvas.mpl_connect("motion_notify_event", lock_roll)
    return fig