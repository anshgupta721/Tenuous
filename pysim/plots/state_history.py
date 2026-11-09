from config import PlotConfig
from .common import _grid
import matplotlib.pyplot as plt



def plot_state_history(t, states, estimated_states, cfg: PlotConfig, figsize=(12, 8), ncols=2, craft=0):
    # states and estimated_states have shape (T, n_craft, n_states).
    # Pick one craft first so cfg.convert still receives (T, n_states).
    states = cfg.convert(states[:, craft])
    estimated_states = cfg.convert(estimated_states[:, craft])
 
    fig, axes = _grid(len(cfg.state_names), ncols, figsize)
    for i, ax in enumerate(axes.flat[:len(cfg.state_names)]):
        ax.plot(t, states[:, i], label="True")
        ax.plot(t, estimated_states[:, i], label="Estimated")
        ax.set_title(cfg.state_names[i])
        ax.set_ylabel(cfg.state_units[i])
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend()
    fig.suptitle(f"Simulation State History (craft {craft})")
    fig.tight_layout()
    return fig

