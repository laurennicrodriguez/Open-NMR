"""Time-domain corrections: DC offset, linear prediction, digital solvent suppression."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from nmrsuite.data import NMRData
from nmrsuite.history import processing_step


def _require_time(nmr: NMRData) -> None:
    if nmr.direct.domain != "time":
        raise ValueError("needs time-domain data")


@processing_step("dc_offset")
def dc_offset(nmr: NMRData, tail_fraction: float = 0.2) -> NMRData:
    """Subtract the mean of the last ``tail_fraction`` of the acquired FID (both channels)."""
    _require_time(nmr)
    n = nmr.direct.acquired or nmr.direct.size
    tail = nmr.data[..., int(n * (1 - tail_fraction)) : n]
    data = nmr.data.copy()
    data[..., :n] -= tail.mean(axis=-1, keepdims=True)
    return nmr.copy_with(data=data)


# ---------------------------------------------------------------------------
# Linear prediction
# ---------------------------------------------------------------------------


def lp_coefficients(x: np.ndarray, order: int) -> np.ndarray:
    """Forward LP coefficients a with x[n] ≈ Σ_k a[k]·x[n-1-k], by least squares (SVD)."""
    if x.size <= 2 * order:
        raise ValueError(f"need more than {2 * order} points for LP order {order}")
    rows = x.size - order
    # Toeplitz-structured data matrix: row n holds x[n+order-1], ..., x[n]
    a_mat = np.lib.stride_tricks.sliding_window_view(x, order)[:rows, ::-1]
    b = x[order:]
    coef, *_ = np.linalg.lstsq(a_mat, b, rcond=None)
    return coef


def _stabilize(coef: np.ndarray) -> np.ndarray:
    """Reflect characteristic roots outside the unit circle inside (decaying signals)."""
    roots = np.roots(np.concatenate([[1.0], -coef]))
    big = np.abs(roots) > 1
    roots[big] = 1 / np.conj(roots[big])
    return -np.poly(roots)[1:]


def lp_extend(x: np.ndarray, n_new: int, order: int, stabilize: bool = True) -> np.ndarray:
    """Predict ``n_new`` points after the end of ``x``."""
    coef = lp_coefficients(x, order)
    if stabilize:
        coef = _stabilize(coef)
    out = np.concatenate([x, np.zeros(n_new, x.dtype)])
    for n in range(x.size, out.size):
        out[n] = np.dot(coef, out[n - order : n][::-1])
    return out


@processing_step("linear_prediction")
def linear_prediction(
    nmr: NMRData,
    mode: Literal["backward", "forward"] = "backward",
    points: int = 2,
    order: int = 16,
    basis: int | None = None,
) -> NMRData:
    """Repair the first ``points`` (backward) or append ``points`` predicted points (forward).

    ``basis`` limits how many points are used to fit the coefficients (default: up to
    1024 points next to the predicted region).
    """
    _require_time(nmr)
    if nmr.ndim != 1:
        raise NotImplementedError("linear prediction handles 1D data")
    d = nmr.direct
    n = d.acquired or d.size
    fid = nmr.data[:n].astype(complex)
    m = min(basis or 1024, n - points if mode == "backward" else n)
    if mode == "backward":
        # predict the reversed series forward: the reversed FID grows, so no stabilisation
        seg = fid[points : points + m][::-1]
        pred = lp_extend(seg, points, order, stabilize=False)[-points:][::-1]
        data = nmr.data.copy()
        data[:points] = pred
        return nmr.copy_with(data=data)
    if mode == "forward":
        seg = fid[n - m : n]
        ext = lp_extend(seg, points, order)[m:]
        data = np.concatenate([fid, ext])
        return nmr.copy_with(data=data, dims=[replace(d, size=data.size, acquired=None)])
    raise ValueError(f"unknown LP mode {mode!r}")


# ---------------------------------------------------------------------------
# Digital solvent suppression (convolution difference)
# ---------------------------------------------------------------------------


@processing_step("solvent_suppression")
def solvent_suppression(
    nmr: NMRData,
    ppm: float | None = None,
    width: int = 32,
) -> NMRData:
    """Remove a signal near ``ppm`` (default: carrier) by convolution difference.

    The FID is mixed down so the solvent sits at 0 Hz, low-pass filtered with a
    Gaussian kernel of ``width`` points (Marion, Ikura & Bax, J. Magn. Reson. 84, 425,
    1989), and the filtered (solvent) part is subtracted. Signals within roughly
    ``sw / width`` Hz of the solvent are attenuated as well.
    """
    _require_time(nmr)
    d = nmr.direct
    n = d.acquired or d.size
    carrier = d.carrier_ppm or 0.0
    target = carrier if ppm is None else ppm
    offset_hz = (target - carrier) * (d.bf_mhz or d.obs_mhz)
    t = np.arange(n) / d.sw_hz
    # the stored FID uses exp(+iωt) for positive offsets (see processing.fft)
    mix = np.exp(-2j * np.pi * offset_hz * t)
    fid = nmr.data[..., :n] * mix
    half = 2 * width
    k = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (k / (width / 2.355)) ** 2)  # FWHM = width points
    kernel /= kernel.sum()
    padded = np.pad(fid, [(0, 0)] * (nmr.ndim - 1) + [(half, half)], mode="reflect")
    low = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), -1, padded)
    data = nmr.data.copy()
    data[..., :n] = (fid - low) / mix
    return nmr.copy_with(data=data)
