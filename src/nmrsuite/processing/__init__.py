"""Processing steps. Each takes an NMRData and returns a new one (never in place),
recording itself in the result's ``history`` (see :mod:`nmrsuite.history`)."""

from nmrsuite.processing.apodize import apodize, window
from nmrsuite.processing.baseline import baseline
from nmrsuite.processing.digital_filter import remove_group_delay
from nmrsuite.processing.fft import fft
from nmrsuite.processing.phase import autophase, phase
from nmrsuite.processing.pipeline import default_recipe_1d, process_1d
from nmrsuite.processing.reference import reference
from nmrsuite.processing.time_domain import dc_offset, linear_prediction, solvent_suppression
from nmrsuite.processing.utils import crop, magnitude, power, reverse
from nmrsuite.processing.zerofill import zerofill

__all__ = [
    "apodize",
    "autophase",
    "baseline",
    "crop",
    "dc_offset",
    "default_recipe_1d",
    "fft",
    "linear_prediction",
    "magnitude",
    "phase",
    "power",
    "process_1d",
    "reference",
    "remove_group_delay",
    "reverse",
    "solvent_suppression",
    "window",
    "zerofill",
]
