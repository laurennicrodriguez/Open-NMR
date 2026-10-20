"""Core data model: an N-dimensional NMR dataset and its axes.

Conventions
-----------
* Axis order follows the data array: for 2D data ``dims[0]`` is F1 (indirect) and
  ``dims[1]`` is F2 (direct). The last axis is always the directly detected one.
* Frequency-domain data are stored in display order: index 0 is the left edge of the
  spectrum (highest ppm), so ppm decreases with index.
* Shifts are in ppm, frequencies and couplings in Hz, spectrometer frequencies in MHz.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

Domain = Literal["time", "frequency"]


@dataclass
class Dimension:
    """One axis of a dataset.

    Attributes
    ----------
    nucleus:
        Observed nucleus on this axis, e.g. ``"1H"``, ``"13C"``.
    size:
        Number of points along the axis (complex points if ``is_complex``).
    sw_hz:
        Spectral width in Hz (``SW_h`` for time-domain data).
    obs_mhz:
        Spectrometer frequency of this axis in MHz. Time domain: the carrier frequency
        (Bruker ``SFO1``). Frequency domain: the reference frequency (Bruker ``SF``).
    domain:
        ``"time"`` or ``"frequency"``.
    is_complex:
        Whether points along this axis are complex (quadrature) values.
    bf_mhz:
        Basic frequency (Bruker ``BF1``); ppm values are relative to it before referencing.
    carrier_ppm:
        Carrier position (``O1/BF1``), if known.
    offset_ppm:
        Frequency domain only: ppm of the first point (left edge).
    acq_mode:
        Quadrature scheme for indirect dimensions (``"States"``, ``"echo-antiecho"``,
        ...) or the direct-dimension mode (``"DQD"``, ``"qsim"``, ``"qseq"``, ``"qf"``).

    The ppm scale of a frequency axis is ``offset_ppm - i · sw_hz / obs_mhz / size``.
    """

    nucleus: str
    size: int
    sw_hz: float
    obs_mhz: float
    domain: Domain = "time"
    is_complex: bool = True
    bf_mhz: float | None = None
    carrier_ppm: float | None = None
    offset_ppm: float | None = None
    acq_mode: str | None = None
    acquired: int | None = None
    """Number of acquired points when the axis has been zero filled (else ``None``)."""

    @property
    def dwell_s(self) -> float:
        """Dwell time between complex points, in seconds."""
        return 1.0 / self.sw_hz

    @property
    def sw_ppm(self) -> float:
        return self.sw_hz / self.obs_mhz

    def time_axis(self) -> np.ndarray:
        """Acquisition time of each point in seconds (time domain)."""
        return np.arange(self.size) * self.dwell_s

    def ppm_axis(self) -> np.ndarray:
        """Chemical shift of each point (frequency domain), decreasing with index."""
        if self.domain != "frequency":
            raise ValueError("ppm_axis() needs a frequency-domain dimension")
        if self.offset_ppm is None:
            raise ValueError("offset_ppm is not set for this dimension")
        step = self.sw_ppm / self.size
        return self.offset_ppm - step * np.arange(self.size)

    def hz_axis(self) -> np.ndarray:
        """Frequency of each point in Hz on the referenced scale (ppm × obs_mhz)."""
        return self.ppm_axis() * self.obs_mhz


@dataclass
class NMRData:
    """A dataset: numeric array plus axes, parameters and provenance.

    ``params`` keeps the vendor parameter files verbatim, keyed by file name
    (e.g. ``params["acqus"]["SW_h"]``), so nothing read from disk is lost.
    """

    data: np.ndarray
    dims: list[Dimension]
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    source: Path | None = None
    experiment: str = "unknown"
    group_delay: float = 0.0
    """Digital-filter group delay still present in the data, in complex points
    (0 once removed or for analog-filtered data)."""
    vendor: str = ""
    """Origin of the raw data (``"bruker"``, ...); selects vendor conventions such as
    the quadrature sign used by the Fourier transform."""
    history: list[dict[str, Any]] = field(default_factory=list)
    """Processing steps applied to the raw data (see :mod:`nmrsuite.history`)."""

    def __post_init__(self) -> None:
        if self.data.ndim != len(self.dims):
            raise ValueError(f"data has {self.data.ndim} axes but {len(self.dims)} dims given")

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def direct(self) -> Dimension:
        """The directly detected dimension (last axis)."""
        return self.dims[-1]

    def copy_with(self, data: np.ndarray | None = None, **changes: Any) -> NMRData:
        """Return a new dataset; the original (and its array) is left untouched."""
        new = replace(
            self,
            data=self.data.copy() if data is None else data,
            dims=[copy.copy(d) for d in changes.pop("dims", self.dims)],
            params=copy.deepcopy(self.params),
        )
        for key, value in changes.items():
            setattr(new, key, value)
        return new

    def summary(self) -> dict[str, Any]:
        """Key acquisition parameters for display (SW, SFO1, O1P, TD, NS, D1, ...)."""
        acq = self.params.get("acqus", {})
        d = acq.get("D")
        out: dict[str, Any] = {
            "experiment": self.experiment,
            "pulse_program": acq.get("PULPROG"),
            "nucleus": self.direct.nucleus,
            "solvent": acq.get("SOLVENT"),
            "temperature_K": acq.get("TE"),
            "SFO1_MHz": acq.get("SFO1", self.direct.obs_mhz),
            "SW_ppm": acq.get("SW"),
            "SW_Hz": acq.get("SW_h", self.direct.sw_hz),
            "O1P_ppm": self.direct.carrier_ppm,
            "TD": acq.get("TD"),
            "NS": acq.get("NS"),
            "DS": acq.get("DS"),
            "D1_s": d[1] if isinstance(d, list) and len(d) > 1 else None,
            "RG": acq.get("RG"),
            "shape": tuple(self.data.shape),
        }
        if self.ndim > 1:
            out["TD_F1"] = self.params.get("acqu2s", {}).get("TD")
            out["nucleus_F1"] = self.dims[0].nucleus
        return {k: v for k, v in out.items() if v is not None}
