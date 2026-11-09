"""Configuration for both the simulation and analysis"""

from dataclasses import dataclass, field
from pathlib import Path
from plots.common import from_meters, DEFAULT_STATE_KINDS


@dataclass(frozen=True)
class SimConfig:
    """Run parameters"""
    epoch_0: str = "2019-07-15T11:23:27.30 UTC"
    x0: tuple = (1282.37, 0.0, 0.0, 0.0, 0.0617543738092, 0.0)
    dt: float = 10.0

    


@dataclass(frozen=True)
class PlotConfig:
    length_unit: str = "m"
    state_names: tuple = ("x", "y", "z", "x_dot", "y_dot", "z_dot")
    state_units = DEFAULT_STATE_KINDS
    time_label: str = "Time (s)"
    save_dir: Path | None = None

    def convert(self, values):
        """Convert meter-based data to this config's display unit."""
        return from_meters(values, self.length_unit)


