"""Line fitting / deconvolution (global spectral deconvolution equivalent).

Lines are pseudo-Voigt: ``h · [(1-η)·L(x) + η·G(x)]`` with a common FWHM ``w`` per
line (η = 0 Lorentzian, 1 Gaussian). A region is fitted with one line per seed plus
a linear local baseline, by bounded least squares in Hz. Areas are analytic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import least_squares

from nmrsuite.analysis.peaks import PeakList, estimate_noise, pick_peaks
from nmrsuite.data import NMRData
from nmrsuite.processing.utils import ppm_to_index

Shape = Literal["lorentz", "gauss", "pvoigt"]

_G = 4 * np.log(2)
_GAUSS_AREA = np.sqrt(np.pi / _G)  # area of unit-height Gaussian per unit FWHM


def pseudo_voigt(x: np.ndarray, x0: float, h: float, w: float, eta: float) -> np.ndarray:
    dx = (x - x0) / w
    lor = 1.0 / (1.0 + 4.0 * dx**2)
    gau = np.exp(-_G * dx**2)
    return h * ((1 - eta) * lor + eta * gau)


def line_area(h: float, w: float, eta: float) -> float:
    return float(h * w * ((1 - eta) * np.pi / 2 + eta * _GAUSS_AREA))


@dataclass
class Line:
    ppm: float
    hz: float
    height: float
    fwhm_hz: float
    eta: float
    area: float


@dataclass
class FitResult:
    lines: list[Line]
    ppm_from: float
    ppm_to: float
    rms_residual: float
    noise: float
    baseline: tuple[float, float]
    ppm: np.ndarray = field(repr=False)
    data: np.ndarray = field(repr=False)
    model: np.ndarray = field(repr=False)

    @property
    def residual(self) -> np.ndarray:
        return self.data - self.model

    @property
    def total_area(self) -> float:
        return float(sum(line.area for line in self.lines))


def fit_region(
    nmr: NMRData,
    ppm_from: float,
    ppm_to: float,
    seeds_ppm: Sequence[float] | None = None,
    shape: Shape = "pvoigt",
    width_hz: float | None = None,
) -> FitResult:
    """Fit lines to one region. Seeds default to the peaks picked in the region."""
    d = nmr.direct
    if d.domain != "frequency" or nmr.ndim != 1:
        raise ValueError("deconvolution needs a 1D frequency-domain spectrum")
    hi, lo = max(ppm_from, ppm_to), min(ppm_from, ppm_to)
    i0 = max(0, int(np.floor(ppm_to_index(d, hi))))
    i1 = min(d.size, int(np.ceil(ppm_to_index(d, lo))) + 1)
    y_all = np.real(nmr.data)
    y = y_all[i0:i1].astype(float)
    ppm = d.ppm_axis()[i0:i1]
    hz = ppm * d.obs_mhz
    sigma = estimate_noise(y_all)
    if seeds_ppm is None:
        picked = pick_peaks(nmr, ppm_range=(lo, hi), classify=False)
        seeds = [(p.ppm, p.intensity, p.width_hz) for p in picked]
    else:
        seeds = []
        for s in seeds_ppm:
            j = int(np.clip(round(ppm_to_index(d, s)) - i0, 0, y.size - 1))
            seeds.append((s, float(y[j]), width_hz or 1.0))
    if not seeds:
        raise ValueError("no lines to fit in this region")
    hz_per_pt = d.sw_hz / d.size
    w0 = width_hz or float(np.median([max(s[2], 2 * hz_per_pt) for s in seeds]))
    eta0 = {"lorentz": 0.0, "gauss": 1.0, "pvoigt": 0.3}[shape]
    fit_eta = shape == "pvoigt"
    span = hz.max() - hz.min()

    # parameter vector: per line (x0 Hz, h, w[, eta]) + baseline (a, b)
    per = 4 if fit_eta else 3
    p0, lb, ub = [], [], []
    ymax = max(abs(y).max(), sigma)
    for s_ppm, h, _w in seeds:
        x0 = s_ppm * d.obs_mhz
        p0 += [x0, max(h, sigma), w0]
        lb += [x0 - 0.5 * w0 - 2 * hz_per_pt, 0.0, hz_per_pt]
        ub += [x0 + 0.5 * w0 + 2 * hz_per_pt, 3 * ymax, max(span, 3 * w0)]
        if fit_eta:
            p0.append(eta0)
            lb.append(0.0)
            ub.append(1.0)
    p0 += [0.0, 0.0]
    lb += [-ymax, -ymax]
    ub += [ymax, ymax]
    xc = (hz - hz.mean()) / max(span, 1e-9)

    def model(p: np.ndarray) -> np.ndarray:
        out = p[-2] + p[-1] * xc
        for k in range(len(seeds)):
            q = p[k * per : (k + 1) * per]
            eta = q[3] if fit_eta else eta0
            out = out + pseudo_voigt(hz, q[0], q[1], q[2], eta)
        return out

    res = least_squares(lambda p: (model(p) - y) / ymax, p0, bounds=(lb, ub), x_scale="jac")
    p = res.x
    lines = []
    for k in range(len(seeds)):
        q = p[k * per : (k + 1) * per]
        eta = float(q[3]) if fit_eta else eta0
        lines.append(Line(
            ppm=float(q[0] / d.obs_mhz), hz=float(q[0]), height=float(q[1]),
            fwhm_hz=float(q[2]), eta=eta, area=line_area(q[1], q[2], eta),
        ))  # fmt: skip
    m = model(p)
    lines.sort(key=lambda line: -line.ppm)
    return FitResult(
        lines=lines,
        ppm_from=hi,
        ppm_to=lo,
        rms_residual=float(np.sqrt(np.mean((y - m) ** 2))),
        noise=sigma,
        baseline=(float(p[-2]), float(p[-1])),
        ppm=ppm,
        data=y,
        model=m,
    )


def deconvolve(
    nmr: NMRData,
    regions: Sequence[tuple[float, float]] | None = None,
    peaks: PeakList | None = None,
    shape: Shape = "pvoigt",
) -> list[FitResult]:
    """Fit every region (default: automatic integral regions) independently."""
    from nmrsuite.analysis.integrate import auto_regions

    if regions is None:
        regions = auto_regions(nmr, peaks)
    out = []
    for a, b in regions:
        try:
            out.append(fit_region(nmr, a, b, shape=shape))
        except ValueError:
            continue
    return out
