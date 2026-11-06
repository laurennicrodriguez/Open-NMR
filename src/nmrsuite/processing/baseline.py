"""Baseline correction of the real part of a spectrum.

Methods
-------
``poly`` (default)
    Chebyshev polynomial of order ``order`` (default 5, as TopSpin's ABSG) fitted
    iteratively to points classified as baseline (residual within 3σ of noise). Handles
    negative peaks (DEPT, APT) and matched TopSpin's ``abs`` best on our data.
``signalfree``
    A Whittaker smoother through the points detected as signal-free, after Cobas et al.,
    J. Magn. Reson. 183, 145 (2006): a point is signal if the spectrum's slope at any of
    several scales stands out from the typical slope, plus a margin for line tails. It
    does not assume a sign, and crowded regions are bridged rather than fitted.
``airpls``
    Adaptive iteratively reweighted penalized least squares (Zhang, Chen & Liang,
    Analyst 135, 1138, 2010): a Whittaker smoother whose weights drop to zero on
    points above the current baseline. Assumes peaks are positive.
``whittaker``
    Asymmetric least squares (Eilers & Boelens 2005): Whittaker smoother with weight
    ``p`` above and ``1 - p`` below the baseline. Assumes peaks are positive.
``points``
    Manual multipoint: the baseline at each given shift is the local median of the
    spectrum; points are joined by a cubic spline (``order=3``) or linearly.

All methods accept ``exclude_ppm=[(ppm_a, ppm_b), ...]``: regions forced to be signal
(never used to define the baseline). Only the real part is corrected.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded
from scipy.ndimage import binary_dilation, uniform_filter1d

from nmrsuite.data import NMRData
from nmrsuite.history import processing_step
from nmrsuite.processing.utils import noise_sigma, ppm_to_index

Method = Literal["signalfree", "airpls", "whittaker", "poly", "points"]

#: Scales (Hz) of the slope test: sharp lines, multiplets, broad humps.
SLOPE_SCALES_HZ = (1.0, 4.0, 16.0)
#: A point is signal when its slope exceeds this many robust σ of the typical slope.
SLOPE_THRESHOLD = 4.0
#: Margin (Hz) kept clear around detected signal, for line tails.
SIGNAL_MARGIN_HZ = 10.0


def default_lambda(n: int) -> float:
    """Smoothness giving a baseline stiffness of roughly 1/40 of the spectrum.

    For a second-difference penalty the effective smoothing width is about λ^(1/4)
    points, so λ = (n/40)^4 scales the result consistently with the number of points.
    """
    return float((n / 40.0) ** 4)


def _whittaker(y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """Solve (W + λ DᵀD) z = W y with D the second-difference operator (banded).

    Uses banded LU with pivoting: with very uneven weights (airPLS) a Cholesky
    factorisation can fail numerically even though the matrix is positive definite.
    """
    n = y.size
    main = np.full(n, 6.0)
    main[[0, -1]] = 1.0
    main[[1, -2]] = 5.0
    off1 = np.full(n - 1, -4.0)
    off1[[0, -1]] = -2.0
    ab = np.zeros((5, n))  # rows: +2, +1, 0, -1, -2 diagonals
    ab[0, 2:] = lam
    ab[1, 1:] = lam * off1
    ab[2] = lam * main + w
    ab[3, :-1] = lam * off1
    ab[4, :-2] = lam
    return solve_banded((2, 2), ab, w * y, check_finite=False)


def _mask_from_regions(nmr: NMRData, regions: Sequence[Sequence[float]] | None) -> np.ndarray:
    keep = np.ones(nmr.direct.size, bool)
    for a, b in regions or []:
        i0, i1 = sorted((ppm_to_index(nmr.direct, a), ppm_to_index(nmr.direct, b)))
        keep[max(0, int(np.floor(i0))) : int(np.ceil(i1)) + 1] = False
    return keep


def airpls_baseline(
    y: np.ndarray, lam: float, keep: np.ndarray, max_iter: int = 30, tol: float = 1e-3
) -> np.ndarray:
    w = keep.astype(float)
    z = y
    for it in range(1, max_iter + 1):
        z = _whittaker(y, w, lam)
        d = y - z
        neg = d[d < 0]
        dn = np.abs(neg).sum()
        if dn < tol * np.abs(y).sum() or neg.size == 0:
            break
        w = np.where(d >= 0, 0.0, np.exp(np.clip(it * np.abs(d) / dn, None, 50)))
        w[0] = w[-1] = np.exp(np.clip(it * np.abs(neg).max() / dn, None, 50))
        w *= keep
    return z


def signal_free_points(y: np.ndarray, hz_per_pt: float) -> np.ndarray:
    """True at points whose slope, at every scale in SLOPE_SCALES_HZ, looks like noise."""
    free = np.ones(y.size, bool)
    for hz in SLOPE_SCALES_HZ:
        w = max(1, round(hz / hz_per_pt))
        smooth = uniform_filter1d(y, w)
        slope = np.pad(np.abs(smooth[2 * w :] - smooth[: -2 * w]), w, mode="edge")
        free &= slope < SLOPE_THRESHOLD * 1.4826 * np.median(slope)
    margin = max(1, round(SIGNAL_MARGIN_HZ / hz_per_pt))
    return ~binary_dilation(~free, iterations=margin)


def als_baseline(
    y: np.ndarray, lam: float, keep: np.ndarray, p: float = 0.01, n_iter: int = 15
) -> np.ndarray:
    w = keep.astype(float)
    z = y
    for _ in range(n_iter):
        z = _whittaker(y, w, lam)
        w = np.where(y > z, p, 1 - p) * keep
    return z


def poly_baseline(y: np.ndarray, order: int, keep: np.ndarray, n_iter: int = 20) -> np.ndarray:
    x = np.linspace(-1, 1, y.size)
    sigma = noise_sigma(y)
    base = keep.copy()
    z = np.zeros_like(y)
    for _ in range(n_iter):
        if base.sum() <= order + 1:
            break
        cheb = np.polynomial.Chebyshev.fit(x[base], y[base], order)
        z = cheb(x)
        new = keep & (np.abs(y - z) < 3 * sigma)
        if np.array_equal(new, base):
            break
        base = new
    return z


def points_baseline(
    nmr: NMRData, y: np.ndarray, points_ppm: Sequence[float], order: int, half_width: int
) -> np.ndarray:
    idx = np.sort([ppm_to_index(nmr.direct, p) for p in points_ppm])
    if idx.size < 2:
        raise ValueError("points baseline needs at least two points")
    if idx[0] < 0 or idx[-1] > y.size - 1:
        raise ValueError("baseline points must lie within the spectrum")
    vals = []
    for i in idx:
        c = int(np.rint(i))
        vals.append(np.median(y[max(0, c - half_width) : c + half_width + 1]))
    grid = np.arange(y.size)
    if order >= 3 and idx.size >= 4:
        return CubicSpline(idx, vals, extrapolate=True)(grid)
    return np.interp(grid, idx, vals)


@processing_step("baseline")
def baseline(
    nmr: NMRData,
    method: Method = "poly",
    lam: float | None = None,
    order: int = 5,
    p: float = 0.01,
    points_ppm: list[float] | None = None,
    half_width: int = 16,
    exclude_ppm: list[list[float]] | None = None,
) -> NMRData:
    """Subtract a baseline from the real part of a 1D spectrum."""
    d = nmr.direct
    if d.domain != "frequency":
        raise ValueError("baseline correction needs frequency-domain data")
    if nmr.ndim != 1:
        raise NotImplementedError("baseline correction handles 1D spectra")
    y = np.real(nmr.data).astype(float)
    keep = _mask_from_regions(nmr, exclude_ppm)
    lam = default_lambda(y.size) if lam is None else lam
    if method == "signalfree":
        z = _whittaker(y, (keep & signal_free_points(y, d.sw_hz / d.size)).astype(float), lam)
    elif method == "airpls":
        z = airpls_baseline(y, lam, keep)
    elif method == "whittaker":
        z = als_baseline(y, lam, keep, p=p)
    elif method == "poly":
        z = poly_baseline(y, order, keep)
    elif method == "points":
        if not points_ppm:
            raise ValueError("method='points' needs points_ppm")
        z = points_baseline(nmr, y, points_ppm, order, half_width)
    else:
        raise ValueError(f"unknown baseline method {method!r}")
    return nmr.copy_with(data=nmr.data - z)
