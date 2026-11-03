"""Spin engine: exact simulation and total-lineshape fitting of coupled spin systems."""

from nmrsuite.spin.fit import SpinFit, SpinModel, fit_spin_region
from nmrsuite.spin.hamiltonian import SpinSystem, block_sizes, build_blocks, estimated_memory_mb
from nmrsuite.spin.simulate import LineList, lorentzian, simulate, simulate_like, transitions

__all__ = [
    "LineList",
    "SpinFit",
    "SpinModel",
    "SpinSystem",
    "block_sizes",
    "build_blocks",
    "estimated_memory_mb",
    "fit_spin_region",
    "lorentzian",
    "simulate",
    "simulate_like",
    "transitions",
]
