"""Shared helpers and simple spectral utilities (crop, magnitude, power, reverse)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.signal import find_peaks

from nmrsuite.data import Dimension, NMRData
from nmrsuite.history import processing_step


def noise_sigma(y: np.ndarray) -> float:
    """Robust noise estimate: MAD of first differences, scaled to a Gaussian σ.

    Differencing removes baseline and broad signal, so this works on whole spectra.
    """
    d = np.diff(np.asarray(y, float))
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / np.sqrt(2))


def find_peak_indices(
    y: np.ndarray,
    threshold: float,
    max_peaks: int | None = None,
    prominence: float | None = None,
) -> np.ndarray:
    """Local maxima above ``threshold`` (and ``prominence``, if given), strongest first."""
    idx, props = find_peaks(y, height=threshold, prominence=prominence)
    order = np.argsort(props["peak_heights"])[::-1]
    idx = idx[order]
    return idx[:max_peaks] if max_peaks else idx


def ppm_to_index(dim: Dimension, ppm: float) -> float:
    """Fractional point index of a chemical shift on a frequency axis."""
    if dim.offset_ppm is None:
        raise ValueError("axis has no ppm scale")
    return (dim.offset_ppm - ppm) / (dim.sw_ppm / dim.size)


def refine_peak(y: np.ndarray, i: int) -> float:
    """Sub-point peak position by parabolic interpolation around maximum ``i``."""
    if i <= 0 or i >= y.size - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    denom = a - 2 * b + c
    return float(i) if denom == 0 else i + 0.5 * (a - c) / denom


def _require_frequency(nmr: NMRData) -> None:
    if nmr.direct.domain != "frequency":
        raise ValueError("needs frequency-domain data")


@processing_step("crop")
def crop(nmr: NMRData, ppm_from: float, ppm_to: float) -> NMRData:
    """Keep the region between two shifts (inclusive) of the direct dimension."""
    _require_frequency(nmr)
    d = nmr.direct
    hi, lo = max(ppm_from, ppm_to), min(ppm_from, ppm_to)
    i0 = max(0, int(np.ceil(ppm_to_index(d, hi))))
    i1 = min(d.size, int(np.floor(ppm_to_index(d, lo))) + 1)
    if i1 <= i0:
        raise ValueError("crop region is empty")
    step = d.sw_ppm / d.size
    new = replace(
        d, size=i1 - i0, sw_hz=(i1 - i0) * step * d.obs_mhz, offset_ppm=d.offset_ppm - i0 * step
    )
    return nmr.copy_with(data=nmr.data[..., i0:i1].copy(), dims=[*nmr.dims[:-1], new])


@processing_step("magnitude")
def magnitude(nmr: NMRData) -> NMRData:
    """Magnitude mode |S| (phase-insensitive)."""
    _require_frequency(nmr)
    return nmr.copy_with(data=np.abs(nmr.data))


@processing_step("power")
def power(nmr: NMRData) -> NMRData:
    """Power mode |S|²."""
    _require_frequency(nmr)
    return nmr.copy_with(data=np.abs(nmr.data) ** 2)


@processing_step("reverse")
def reverse(nmr: NMRData) -> NMRData:
    """Mirror the spectrum left-right (the ppm axis is unchanged)."""
    _require_frequency(nmr)
    return nmr.copy_with(data=nmr.data[..., ::-1].copy())
