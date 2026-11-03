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


#: ¹J(C,H) range (Hz) for ¹³C satellites; satellites sit at ±J/2 from the parent.
C13_SATELLITE_J_HZ = (110.0, 220.0)
#: Satellite height relative to the parent (0.55 % each for natural abundance).
C13_SATELLITE_RATIO = (0.0015, 0.015)


def satellite_parents(
    hz: np.ndarray, amp: np.ndarray, max_parents: int = 30
) -> dict[int, tuple[int, float]]:
    """¹³C satellites among lines at ``hz`` with heights ``amp``: {line: (parent, ¹J in Hz)}.

    A satellite pair lies symmetrically about a much taller parent (isotope shift ≲ 1 Hz,
    so within 3 Hz), each 0.15–1.5 % of its height. The two may differ several-fold in
    height (isotope shift, different widths, overlap).
    """
    hz, amp = np.asarray(hz, float), np.asarray(amp, float)
    lo, hi = (j / 2 for j in C13_SATELLITE_J_HZ)
    found: dict[int, tuple[int, float]] = {}
    for k in np.argsort(-amp)[:max_parents]:
        if amp[k] <= 0:
            continue
        ratio = amp / amp[k]
        cand = (ratio >= C13_SATELLITE_RATIO[0]) & (ratio <= C13_SATELLITE_RATIO[1])
        above = np.nonzero(cand & (hz > hz[k] + lo) & (hz < hz[k] + hi))[0]
        below = np.nonzero(cand & (hz < hz[k] - lo) & (hz > hz[k] - hi))[0]
        for a in above:
            for b in below:
                da, db = hz[a] - hz[k], hz[k] - hz[b]
                if abs(da - db) < 3.0 and 0.25 < amp[a] / amp[b] < 4.0:
                    found.setdefault(int(a), (int(k), float(da + db)))
                    found.setdefault(int(b), (int(k), float(da + db)))
    return found


def half_height_points(y: np.ndarray, i: int) -> int:
    """Full width of the maximum at ``i`` at half its height, in points (integer)."""
    h = y[i] / 2
    left = i
    while left > 0 and y[left] > h:
        left -= 1
    right = i
    while right < y.size - 1 and y[right] > h:
        right += 1
    return right - left


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
