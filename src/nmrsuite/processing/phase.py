"""Zero- and first-order phase correction, manual and automatic.

Phase convention (TopSpin-compatible, validated on 1H/13C data): point ``i`` of ``N``
is multiplied by ``exp(+i·φ_i)`` with

    φ_i = p0 + p1 · i / N        (degrees; i = 0 is the left edge).

With a pivot, ``p0`` is the phase *at the pivot* and ``φ_i = p0 + p1·(i - i_pivot)/N``.

Automatic methods
-----------------
``peaks`` (default)
    Peak-symmetry fitting. For each strong, isolated peak the angle that makes it purely
    absorptive is ``-arg Σ S`` over a window symmetric about the peak (the dispersive
    part integrates to zero). ``p0`` and ``p1`` are then fitted to these angles,
    weighted by peak height, with a grid search over ``p1`` that is immune to 2π wraps.
``entropy``
    ACME entropy minimization (Chen et al., J. Magn. Reson. 158, 164, 2002): minimizes the
    entropy of the derivative of the real spectrum plus a penalty on negative intensity.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import minimize

from nmrsuite.data import NMRData
from nmrsuite.history import processing_step
from nmrsuite.processing.utils import find_peak_indices, noise_sigma, ppm_to_index


def apply_phase(spec: np.ndarray, p0: float, p1: float = 0.0, pivot: float = 0.0) -> np.ndarray:
    """Phase a spectrum along the last axis; ``pivot`` is a (fractional) point index."""
    n = spec.shape[-1]
    phi = np.deg2rad(p0 + p1 * (np.arange(n) - pivot) / n)
    return spec * np.exp(1j * phi)


def _phase_from_record(params: dict, result: dict) -> tuple[str, dict]:
    return "phase", {"p0": result["p0"], "p1": result["p1"], "pivot_ppm": None}


def _require_complex_spectrum(nmr: NMRData) -> None:
    if nmr.direct.domain != "frequency":
        raise ValueError("phase correction needs frequency-domain data")
    if not np.iscomplexobj(nmr.data):
        raise ValueError("phase correction needs complex data (real and imaginary parts)")


@processing_step("phase")
def phase(
    nmr: NMRData, p0: float = 0.0, p1: float = 0.0, pivot_ppm: float | None = None
) -> NMRData:
    """Apply zero-order ``p0`` and first-order ``p1`` phase (degrees)."""
    _require_complex_spectrum(nmr)
    pivot = 0.0 if pivot_ppm is None else ppm_to_index(nmr.direct, pivot_ppm)
    return nmr.copy_with(data=apply_phase(nmr.data, p0, p1, pivot))


# ---------------------------------------------------------------------------
# Automatic phasing
# ---------------------------------------------------------------------------


def _peak_angles(spec: np.ndarray, max_peaks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions, absorptive phase angles (rad) and weights of strong isolated peaks."""
    mag = np.abs(spec)
    sigma = noise_sigma(mag)
    level = max(20 * sigma, 1e-3 * mag.max())
    # prominence rejects truncation wiggles on the skirts of strong peaks
    idx = find_peak_indices(mag, threshold=level, prominence=level, max_peaks=4 * max_peaks)
    pos, ang, wts = [], [], []
    taken: list[tuple[int, int]] = []  # windows of accepted (stronger) peaks
    for i in idx:
        if len(pos) >= max_peaks:
            break
        if any(lo <= i < hi for lo, hi in taken):
            continue
        h = mag[i]
        # half-width at half height, then integrate ±3 HWHM symmetric about the peak
        left = i
        while left > 0 and mag[left] > h / 2:
            left -= 1
        right = i
        while right < mag.size - 1 and mag[right] > h / 2:
            right += 1
        hw = max(2, 3 * max(i - left, right - i))
        lo, hi = i - hw, i + hw + 1
        if lo < 0 or hi > mag.size:
            continue
        window = mag[lo:hi]
        # Skip peaks with a comparably large neighbour in the outer half of the window.
        # (An isolated Lorentzian's magnitude there is at most ~0.36 of the maximum.)
        taken.append((lo, hi))
        if np.max(np.concatenate([window[: hw // 2], window[-(hw // 2) :]])) > 0.5 * h:
            continue
        pos.append(i)
        ang.append(-np.angle(spec[lo:hi].sum()))
        wts.append(h)
    return np.asarray(pos, float), np.asarray(ang), np.asarray(wts)


#: Experiments whose spectra legitimately contain negative peaks.
NEGATIVE_PEAK_EXPERIMENTS = {"DEPT-135", "DEPTQ", "APT", "HSQC-edited"}

#: First-order phase search range (degrees) and the score fraction within which the
#: smallest |p1| is preferred: when peaks cluster in a narrow region, p1 is poorly
#: determined and large values only fit noise.
P1_RANGE = 720.0
P1_TOLERANCE = 0.999


def _fit_linear_phase(
    x: np.ndarray, ang: np.ndarray, w: np.ndarray, fit_p1: bool, axial: bool = False
) -> tuple[float, float]:
    """Fit ang ≈ p0 + p1·x (rad, x in [0, 1)) on the circle; returns degrees.

    ``axial=True`` fits angles modulo π (peaks may be positive or negative); the
    returned p0 is then defined modulo 180°.
    """
    k = 2.0 if axial else 1.0

    def best_p0(p1: float) -> tuple[float, float]:
        c = np.sum(w * np.exp(1j * k * (ang - p1 * x)))
        return float(np.angle(c) / k), float(np.abs(c))

    if not fit_p1 or x.size < 3:
        p0, _ = best_p0(0.0)
        return float(np.rad2deg(p0)), 0.0
    grid = np.deg2rad(np.arange(-P1_RANGE, P1_RANGE + 0.5, 1.0))
    scores = np.array([best_p0(p1)[1] for p1 in grid])
    # candidate solutions are the local maxima of the score; among (near-)equal ones —
    # aliases when the peaks cluster — take the smallest |p1|
    inner = np.nonzero((scores[1:-1] >= scores[:-2]) & (scores[1:-1] >= scores[2:]))[0] + 1
    peaks = np.concatenate([inner, [int(np.argmax(scores))]])
    ok = peaks[scores[peaks] >= P1_TOLERANCE * scores.max()]
    p1 = grid[ok][np.argmin(np.abs(grid[ok]))]
    fine = p1 + np.deg2rad(np.linspace(-1, 1, 201))
    p1 = fine[int(np.argmax([best_p0(v)[1] for v in fine]))]
    p0, _ = best_p0(p1)
    return float(np.rad2deg(p0)), float(np.rad2deg(p1))


def _signal_mask(spec: np.ndarray, grow: int = 64) -> np.ndarray:
    """Points within ``grow`` of a peak (where the phase matters).

    The magnitude baseline sits well above the noise σ (broad offset and rolls), so the
    threshold is relative to the median level and to the tallest peak.
    """
    mag = np.abs(spec)
    hot = mag > max(np.median(mag) + 10 * noise_sigma(mag), 2e-3 * mag.max())
    if not hot.any():
        return np.ones(spec.size, bool)
    kernel = np.ones(2 * grow + 1)
    return np.convolve(hot.astype(float), kernel, mode="same") > 0


def _acme_cost(re: np.ndarray, gamma: float) -> float:
    d = np.abs(np.diff(re))
    total = d.sum()
    if total == 0:
        return 0.0
    h = d / total
    h = h[h > 0]
    entropy = -np.sum(h * np.log(h))
    if gamma:
        neg = re[re < 0]
        entropy += gamma * np.sum(neg**2) / np.sum(re**2)
    return float(entropy)


def _autophase_entropy(
    spec: np.ndarray, fit_p1: bool, start: tuple[float, float], allow_negative: bool
) -> tuple[float, float]:
    """ACME refinement from a starting guess, evaluated on signal regions only."""
    mask = _signal_mask(spec)
    s = spec / np.abs(spec).max()
    idx = np.nonzero(mask)[0]
    seg = s[idx]
    xs = idx / s.size
    gamma = 0.0 if allow_negative else 1000.0

    def cost(v: np.ndarray) -> float:
        p1 = v[1] if fit_p1 else 0.0
        re = (seg * np.exp(1j * np.deg2rad(v[0] + p1 * xs))).real
        return _acme_cost(re, gamma)

    best: tuple[float, float, float] | None = None
    for dp0 in (0.0, 180.0) if not allow_negative else (0.0,):
        x0 = [start[0] + dp0] + ([start[1]] if fit_p1 else [])
        res = minimize(cost, x0, method="Nelder-Mead",
                       options={"xatol": 0.05, "fatol": 1e-9, "maxiter": 600})  # fmt: skip
        if best is None or res.fun < best[0]:
            best = (float(res.fun), float(res.x[0]), float(res.x[1]) if fit_p1 else 0.0)
    assert best is not None
    return best[1], best[2]


@processing_step("autophase", replay=_phase_from_record)
def autophase(
    nmr: NMRData,
    method: Literal["peaks", "entropy"] = "entropy",
    first_order: bool = True,
    max_peaks: int = 40,
    allow_negative: bool | None = None,
) -> tuple[NMRData, dict[str, float]]:
    """Automatic phase correction; records the angles found as ``result``.

    ``entropy`` (default) refines the ``peaks`` estimate by ACME minimization.
    ``allow_negative=None`` decides from the experiment type (DEPT-135, APT, edited
    HSQC have real negative peaks); then peak angles are fitted modulo 180°, the
    entropy step runs without its negativity penalty, and the overall sign is chosen so
    the spectrum is mostly positive.
    """
    _require_complex_spectrum(nmr)
    if nmr.ndim != 1:
        raise NotImplementedError("autophase handles 1D spectra")
    if allow_negative is None:
        allow_negative = nmr.experiment in NEGATIVE_PEAK_EXPERIMENTS
    spec = nmr.data
    n = spec.size
    pos, ang, wts = _peak_angles(spec, max_peaks)
    if pos.size:
        p0, p1 = _fit_linear_phase(pos / n, ang, wts, first_order, axial=allow_negative)
    else:
        p0, p1 = 0.0, 0.0
    if method == "entropy":
        p0, p1 = _autophase_entropy(spec, first_order, (p0, p1), allow_negative)
    elif method != "peaks":
        raise ValueError(f"unknown autophase method {method!r}")
    phased = apply_phase(spec, p0, p1)
    if allow_negative and phased.real.sum() < 0:
        p0 += 180.0
        phased = -phased
    p0 = (p0 + 180.0) % 360.0 - 180.0
    return nmr.copy_with(data=phased), {"p0": p0, "p1": p1}
