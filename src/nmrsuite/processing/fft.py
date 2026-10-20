"""Fourier transform of the direct dimension into display order.

Convention (checked against TopSpin on Bruker DQD data): positive offsets from the
carrier are recorded as exp(+iωt), and the spectrum in display order (index 0 = left
edge = highest frequency) is ``fftshift(fft(conj(fid)))``. Point ``i`` of ``N`` then
lies ``sw/2 - i·sw/N`` Hz from the carrier, so before referencing

    offset_ppm = carrier_ppm + sw / 2 / reference_MHz,

with ppm relative to the basic frequency BF1 (TopSpin's SR = 0).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from nmrsuite.data import NMRData
from nmrsuite.history import processing_step


@processing_step("fft")
def fft(nmr: NMRData, first_point: float = 0.5, conjugate: bool = True) -> NMRData:
    """Complex FT of the last axis. ``first_point`` scales point 0 (0.5 avoids a DC offset)."""
    d = nmr.direct
    if d.domain != "time":
        raise ValueError("fft needs time-domain data")
    if not d.is_complex:
        raise NotImplementedError("real (qf/qseq) FIDs are not supported yet")
    fid = nmr.data.astype(complex, copy=True)
    fid[..., 0] *= first_point
    if conjugate:
        fid = np.conj(fid)
    spec = np.fft.fftshift(np.fft.fft(fid, axis=-1), axes=-1)
    ref_mhz = d.bf_mhz or d.obs_mhz
    carrier = d.carrier_ppm or 0.0
    freq_dim = replace(
        d,
        domain="frequency",
        obs_mhz=ref_mhz,
        offset_ppm=carrier + d.sw_hz / 2 / ref_mhz,
    )
    return nmr.copy_with(data=spec, dims=[*nmr.dims[:-1], freq_dim])
