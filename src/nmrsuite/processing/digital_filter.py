"""Removal of the Bruker digital-filter group delay.

Oversampling digital filters delay the FID by ``g`` (generally fractional) complex
points: the first ~``g`` points hold the filter's rise and the signal proper starts
after them. In the frequency domain this is a linear phase ramp of ``2π·g`` across
the spectral width. We remove it exactly by applying the opposite ramp and returning
to the time domain, which circularly advances the FID by ``g`` points. The filter
rise wraps around to the end of the FID; those last ``ceil(g)`` points are then
dropped (``truncate=True``, the TopSpin-equivalent default) or zeroed.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from nmrsuite.data import NMRData


def shift_fid(fid: np.ndarray, points: float) -> np.ndarray:
    """Circularly advance a complex FID by a (fractional) number of points along the last axis."""
    n = fid.shape[-1]
    k = np.fft.fftfreq(n) * n  # signed frequency index
    ramp = np.exp(2j * np.pi * points * k / n)
    return np.fft.ifft(np.fft.fft(fid, axis=-1) * ramp, axis=-1)


def remove_group_delay(nmr: NMRData, truncate: bool = True) -> NMRData:
    """Return a copy of time-domain ``nmr`` with its group delay removed.

    The original dataset is not modified. Applies along the direct (last) dimension.
    """
    g = nmr.group_delay
    if g == 0:
        return nmr.copy_with()
    direct = nmr.direct
    if direct.domain != "time":
        raise ValueError("group-delay removal needs time-domain data")
    if not direct.is_complex:
        raise ValueError("group-delay removal needs complex (quadrature) data")
    shifted = shift_fid(nmr.data, g)
    n_bad = math.ceil(g)
    if truncate:
        shifted = shifted[..., : shifted.shape[-1] - n_bad]
    else:
        shifted[..., -n_bad:] = 0
    dims = [*nmr.dims[:-1], replace(direct, size=shifted.shape[-1])]
    return nmr.copy_with(data=shifted, dims=dims, group_delay=0.0)
