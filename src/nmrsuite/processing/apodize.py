"""Apodization (window functions) along the direct dimension.

Window definitions follow TopSpin, with ``t`` the acquisition time of each point and
``AQ`` the acquisition time of the acquired (not zero-filled) FID:

=========  ==================================================================
``em``     exp(-π·lb·t)                                       (TopSpin EM, LB)
``gm``     exp(-a·t - b·t²), a = π·lb, b = -a / (2·gb·AQ)     (GM, LB < 0, GB)
``sine``   sin((π - φ)·t/AQ + φ), φ = π/ssb (φ = 0 if ssb < 2) (SINE, SSB)
``qsine``  the sine window squared                              (QSINE, SSB)
``traf``   E·(E + e) / (E² + e²), E = exp(-π·lb·t), e = exp(-π·lb·(AQ - t))
``trap``   rises linearly to 1 at tm1, flat, falls to 0 from tm2 to AQ (s)
``none``   1
=========  ==================================================================
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from nmrsuite.data import NMRData
from nmrsuite.history import processing_step

WindowKind = Literal["none", "em", "gm", "sine", "qsine", "traf", "trap"]

#: Default exponential line broadening (Hz) per nucleus.
DEFAULT_LB: dict[str, float] = {"1H": 0.3, "19F": 0.5, "13C": 1.0, "31P": 1.0, "15N": 2.0}


def default_lb(nucleus: str) -> float:
    return DEFAULT_LB.get(nucleus, 1.0)


def window(
    kind: WindowKind,
    n: int,
    sw_hz: float,
    *,
    lb: float = 0.0,
    gb: float = 0.0,
    ssb: float = 0.0,
    tm1: float = 0.0,
    tm2: float | None = None,
) -> np.ndarray:
    """Window values for ``n`` points at spectral width ``sw_hz`` (for display, too)."""
    t = np.arange(n) / sw_hz
    aq = n / sw_hz
    if kind == "none":
        return np.ones(n)
    if kind == "em":
        return np.exp(-np.pi * lb * t)
    if kind == "gm":
        if gb <= 0 or gb > 1:
            raise ValueError("gm needs 0 < gb <= 1")
        a = np.pi * lb
        b = -a / (2 * gb * aq)
        return np.exp(-a * t - b * t**2)
    if kind in ("sine", "qsine"):
        phi = np.pi / ssb if ssb >= 2 else 0.0
        w = np.sin((np.pi - phi) * t / aq + phi)
        return w**2 if kind == "qsine" else w
    if kind == "traf":
        e_start = np.exp(-np.pi * lb * t)
        e_end = np.exp(-np.pi * lb * (aq - t))
        return e_start * (e_start + e_end) / (e_start**2 + e_end**2)
    if kind == "trap":
        tm2 = aq if tm2 is None else tm2
        w = np.ones(n)
        if tm1 > 0:
            w = np.minimum(w, t / tm1)
        if tm2 < aq:
            w = np.minimum(w, np.clip((aq - t) / (aq - tm2), 0, 1))
        return w
    raise ValueError(f"unknown window {kind!r}")


@processing_step("apodize")
def apodize(
    nmr: NMRData,
    kind: WindowKind = "em",
    lb: float | None = None,
    gb: float = 0.0,
    ssb: float = 0.0,
    tm1: float = 0.0,
    tm2: float | None = None,
) -> NMRData:
    """Multiply the FID by a window. ``lb=None`` uses the per-nucleus default for ``em``.

    The window spans the acquired points only (``Dimension.acquired`` after zero filling).
    """
    d = nmr.direct
    if d.domain != "time":
        raise ValueError("apodize needs time-domain data")
    if lb is None:
        lb = default_lb(d.nucleus) if kind in ("em", "traf") else 0.0
    n = d.acquired or d.size
    w = window(kind, n, d.sw_hz, lb=lb, gb=gb, ssb=ssb, tm1=tm1, tm2=tm2)
    data = nmr.data.copy()
    data[..., :n] *= w
    out = nmr.copy_with(data=data)
    return (out, {"lb": lb}) if kind in ("em", "traf") else out
