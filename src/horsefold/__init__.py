"""Simulate how much grouped data inflates accuracy when it is split or preprocessed the wrong way."""

from horsefold.data import simulate_features
from horsefold.experiments import Design, run
from horsefold.splits import exact_leak_rate, simulated_leak_rate, split_groups, split_units
from horsefold.structure import Structure, basran2022

__version__ = "0.1.0"

__all__ = [
    "Design",
    "Structure",
    "basran2022",
    "exact_leak_rate",
    "run",
    "simulate_features",
    "simulated_leak_rate",
    "split_groups",
    "split_units",
]
