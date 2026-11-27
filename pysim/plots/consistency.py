import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2


def plot_chi2(t, data, dims, title):
    """data: {name: (N_runs, T, n_craft)}, dims: {name: expected value}."""
    n_runs = next(iter(data.values())).shape[0]
    fig, axes = plt.subplots(1, len(data), figsize=(5 * len(data), 4), squeeze=False)
    for ax, (name, arr) in zip(axes.flat, data.items()):
        d = dims[name]
        ax.plot(t, np.nanmean(arr, axis=0))                    # one line per craft
        ax.axhline(d, color="k", lw=0.8)
        ax.axhspan(chi2.ppf(0.025, n_runs * d) / n_runs,
                   chi2.ppf(0.975, n_runs * d) / n_runs, color="gray", alpha=0.3)
        ax.set_yscale("log")
        ax.set_title(f"{title}: {name} (expected {d})")
        ax.set_xlabel("Time (s)")
    fig.tight_layout()