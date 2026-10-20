"""Total-lineshape fitting of coupled spin systems to experimental spectra.

A :class:`SpinModel` is a spin system whose shifts, couplings and line widths are
numbers (fixed) or names (fitted parameters). Using one name in several places ties
them, which is how symmetry is expressed — an AA′BB′C pyridinium ring::

    model = SpinModel(
        shifts=["d2", "d3", "d4", "d3", "d2"],        # H2, H3, H4, H5, H6
        couplings={(0, 1): "J23", (4, 3): "J23", (1, 2): "J34", (3, 2): "J34",
                   (0, 2): "J24", (4, 2): "J24", (1, 3): "J35", (0, 4): "J26",
                   (0, 3): "J25", (4, 1): "J25"},
        linewidths=["w2", "w3", "w4", "w3", "w2"],
    )

:func:`fit_spin_region` fits the nonlinear parameters (δ in ppm, J and widths in Hz,
a residual phase in degrees) by bounded least squares. For each trial the intensity
scale and a linear local baseline are solved exactly (variable projection), so they
never need starting values. The fit is repeated from several starting points —
shifts and J values perturbed, small couplings sign-flipped — because J-sign
permutations and near-degenerate shifts create local minima.

Uncertainties come from the covariance s²(JᵀJ)⁻¹ at the minimum. Zero filling makes
neighbouring points correlated, so a k-fold zero-filled spectrum is fitted on every
k-th point (independent residuals, half the cost for the usual ×2).
They describe the fit to this spectrum (noise and model misfit), not systematic
errors such as an inappropriate spin system.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from nmrsuite.data import NMRData
from nmrsuite.processing.utils import ppm_to_index
from nmrsuite.spin.hamiltonian import SpinSystem
from nmrsuite.spin.simulate import (
    LineList,
    kernel_for,
    lorentzian,
    lorentzian_jacobian,
    transitions_from,
)

Value = float | str

#: Default bounds by parameter kind.
SHIFT_BOUND_PPM = 0.2  # each shift may move this far from its start
J_BOUND_HZ = 25.0
WIDTH_BOUNDS_HZ = (0.05, 20.0)
PHASE_BOUND_DEG = 30.0
#: Multi-start stops early once this many starts reach the same lowest minimum
#: (cost within AGREE_RTOL); ``starts`` is the maximum.
AGREE_STARTS = 2
AGREE_RTOL = 1e-3
#: Evaluations allowed per start: good starts converge in < 100; a start stuck far
#: from any good minimum is abandoned rather than polished for minutes.
MAX_NFEV_PER_START = 200
#: Lines weaker than this (relative) are left out while fitting.
FIT_MIN_RELATIVE = 1e-4


@dataclass
class SpinModel:
    """A spin system with named (fitted, possibly shared) parameters."""

    shifts: Sequence[Value]
    couplings: Mapping[tuple[int, int], Value] = field(default_factory=dict)
    linewidths: Sequence[Value] | Value = 1.0
    nuclei: Sequence[str] | None = None
    labels: Sequence[str] | None = None

    def __post_init__(self) -> None:
        n = len(self.shifts)
        if isinstance(self.linewidths, str | int | float):
            self.linewidths = [self.linewidths] * n
        if len(self.linewidths) != n:
            raise ValueError("linewidths must have one entry per spin")
        self.kinds: dict[str, str] = {}
        for v in self.shifts:
            self._register(v, "shift")
        for (a, b), v in self.couplings.items():
            if not (0 <= a < n and 0 <= b < n) or a == b:
                raise ValueError(f"bad coupling indices {(a, b)}")
            self._register(v, "j")
        for v in self.linewidths:
            self._register(v, "width")

    def _register(self, v: Value, kind: str) -> None:
        if isinstance(v, str) and self.kinds.setdefault(v, kind) != kind:
            raise ValueError(f"parameter {v!r} used as both {self.kinds[v]} and {kind}")

    @property
    def names(self) -> list[str]:
        return list(self.kinds)

    def system(self, values: Mapping[str, float]) -> SpinSystem:
        def get(v: Value) -> float:
            return float(values[v]) if isinstance(v, str) else float(v)

        return SpinSystem(
            [get(v) for v in self.shifts],
            {k: get(v) for k, v in self.couplings.items()},
            nuclei=self.nuclei,
            labels=self.labels,
            linewidths_hz=[get(v) for v in self.linewidths],
        )


@dataclass
class SpinFit:
    values: dict[str, float]
    stderr: dict[str, float]
    phase_deg: float
    scale: float
    baseline: tuple[float, float]
    rms: float
    noise: float
    starts: int
    converged_starts: int
    ppm: np.ndarray = field(repr=False)
    data: np.ndarray = field(repr=False)
    model: np.ndarray = field(repr=False)
    lines: LineList | None = field(default=None, repr=False)
    spin_model: SpinModel | None = field(default=None, repr=False)

    @property
    def residual(self) -> np.ndarray:
        return self.data - self.model

    @property
    def rms_over_noise(self) -> float:
        return self.rms / self.noise if self.noise else float("inf")

    def system(self) -> SpinSystem:
        assert self.spin_model is not None
        return self.spin_model.system(self.values)

    def table(self, obs_mhz: float | None = None) -> str:
        """Parameters with standard errors, one per line (δ ppm, J and widths Hz)."""
        rows = []
        kinds = self.spin_model.kinds if self.spin_model else {}
        for name, v in self.values.items():
            e = self.stderr.get(name, float("nan"))
            if kinds.get(name) == "shift":
                hz = f"  (± {e * obs_mhz:.3f} Hz)" if obs_mhz else ""
                rows.append(f"{name:>8} = {v:9.4f} ± {e:.4f} ppm{hz}")
            else:
                rows.append(f"{name:>8} = {v:9.3f} ± {e:.3f} Hz")
        return "\n".join(rows)


def _region(
    nmr: NMRData, ppm_from: float, ppm_to: float, step: int = 1
) -> tuple[slice, np.ndarray]:
    d = nmr.direct
    hi, lo = max(ppm_from, ppm_to), min(ppm_from, ppm_to)
    i0 = max(0, int(np.floor(ppm_to_index(d, hi))))
    i1 = min(d.size, int(np.ceil(ppm_to_index(d, lo))) + 1)
    return slice(i0, i1, step), d.ppm_axis()[i0:i1:step]


def _value_getter(v: Value):
    return (lambda vals: vals[v]) if isinstance(v, str) else (lambda vals, c=float(v): c)


def _bounds(kind: str, start: float) -> tuple[float, float]:
    if kind == "shift":
        return start - SHIFT_BOUND_PPM, start + SHIFT_BOUND_PPM
    if kind == "j":
        return -J_BOUND_HZ, J_BOUND_HZ
    return WIDTH_BOUNDS_HZ


def _starts(
    names: list[str], kinds: dict[str, str], x0: np.ndarray, n: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Starting points: the given one, then perturbed copies (J ±30 %, small J
    sign-flipped, shifts ± a couple of Hz-equivalent, widths ×0.7–1.4)."""
    out = [x0.copy()]
    for _ in range(max(0, n - 1)):
        x = x0.copy()
        for i, name in enumerate(names):
            k = kinds[name]
            if k == "j":
                x[i] *= rng.uniform(0.7, 1.3)
                if abs(x0[i]) < 3.0 and rng.random() < 0.5:
                    x[i] = -x[i]
            elif k == "shift":
                x[i] += rng.normal(0, 0.004)
            elif k == "width":
                x[i] *= rng.uniform(0.7, 1.4)
        out.append(x)
    return out


def fit_spin_region(
    nmr: NMRData,
    ppm_from: float,
    ppm_to: float,
    model: SpinModel,
    initial: Mapping[str, float],
    *,
    fixed: Sequence[str] = (),
    fit_phase: bool = True,
    starts: int = 8,
    seed: int = 0,
    observed: str | None = None,
) -> SpinFit:
    """Fit ``model`` to the real part of ``nmr`` between two shifts."""
    d = nmr.direct
    if d.domain != "frequency" or nmr.ndim != 1:
        raise ValueError("spin fitting needs a processed 1D spectrum")
    missing = [n for n in model.names if n not in initial]
    if missing:
        raise ValueError(f"no starting value for {missing}")
    # A spectrum zero filled k-fold carries one independent value per k points: fit every
    # k-th point (half the cost for the usual ×2) and the residuals are uncorrelated.
    zf = d.size / d.acquired if d.acquired else 1.0
    step = max(1, round(zf))
    sl, ppm = _region(nmr, ppm_from, ppm_to, step)
    y = np.real(nmr.data[sl]).astype(float)
    hz = ppm * d.obs_mhz
    obs = observed or d.nucleus
    from nmrsuite.analysis.peaks import estimate_noise

    noise = estimate_noise(np.real(nmr.data))
    ymax = float(np.abs(y).max()) or 1.0
    xb = np.linspace(-1.0, 1.0, y.size)

    free = [n for n in model.names if n not in fixed]
    kinds = model.kinds
    values = {k: float(v) for k, v in initial.items()}
    x0 = np.array([values[n] for n in free] + ([0.0] if fit_phase else []))
    lb = [_bounds(kinds[n], values[n])[0] for n in free]
    ub = [_bounds(kinds[n], values[n])[1] for n in free]
    if fit_phase:
        lb.append(-PHASE_BOUND_DEG)
        ub.append(PHASE_BOUND_DEG)
    lb_a, ub_a = np.array(lb), np.array(ub)

    def unpack(x: np.ndarray) -> tuple[dict[str, float], float]:
        vals = dict(values)
        vals.update(zip(free, x[: len(free)], strict=True))
        return vals, (float(x[-1]) if fit_phase else 0.0)

    # topology fixed by the model (a coupling that starts at 0 Hz stays in the kernel)
    kernel = kernel_for(model.system(values), obs, pairs=list(model.couplings))
    shift_of = [_value_getter(v) for v in model.shifts]
    width_of = [_value_getter(v) for v in model.linewidths]
    jmap = {tuple(sorted(k)): v for k, v in model.couplings.items()}
    j_of = [_value_getter(jmap[p]) for p in kernel.pairs]

    # chain rule from model parameters to kernel quantities (ν_k = δ_k·SF, J_p, w_k)
    n_free = len(free)
    col = {name: i for i, name in enumerate(free)}
    m_f = np.zeros((len(kernel.obs) + len(kernel.pairs), n_free))
    m_w = np.zeros((len(kernel.obs), n_free))
    for o, k in enumerate(kernel.obs):
        if model.shifts[k] in col:
            m_f[o, col[model.shifts[k]]] = d.obs_mhz
        if model.linewidths[k] in col:
            m_w[o, col[model.linewidths[k]]] = 1.0
    for q, pair in enumerate(kernel.pairs):
        if jmap[pair] in col:
            m_f[len(kernel.obs) + q, col[jmap[pair]]] = 1.0

    def lines_at(x: np.ndarray, gradients: bool = False) -> tuple[LineList, float]:
        vals, phase = unpack(x)
        nu = np.array([g(vals) for g in shift_of]) * d.obs_mhz
        jv = np.array([g(vals) for g in j_of])
        widths = np.array([g(vals) for g in width_of])
        lines = transitions_from(
            kernel, nu, jv, widths, d.obs_mhz, FIT_MIN_RELATIVE, gradients=gradients
        )
        return lines, phase

    def basis(x: np.ndarray) -> tuple[np.ndarray, LineList]:
        lines, phase = lines_at(x)
        return lorentzian(lines, hz, phase), lines

    def linear(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.column_stack([sim, np.ones_like(sim), xb])
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        return coef, a @ coef

    cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Residual and analytic Jacobian at x (computed together, cached for x)."""
        key = x.tobytes()
        if key not in cache:
            cache.clear()
            lines, phase = lines_at(x, gradients=True)
            inten = lines.intensity[:, None]
            coef_f = inten * (lines.dfreq @ m_f)
            coef_w = inten * (lines.wweights @ m_w)
            spec, dspec, dphi = lorentzian_jacobian(lines, hz, phase, coef_f, coef_w)
            basis_m = np.column_stack([spec, np.ones_like(spec), xb])
            coef, *_ = np.linalg.lstsq(basis_m, y, rcond=None)
            res = (basis_m @ coef - y) / ymax
            jr = coef[0] * dspec
            if fit_phase:
                jr = np.column_stack([jr, coef[0] * dphi * np.pi / 180.0])
            # variable projection (Kaufman): remove what the linear terms absorb
            proj, *_ = np.linalg.lstsq(basis_m, jr, rcond=None)
            cache[key] = (res, (jr - basis_m @ proj) / ymax)
        return cache[key]

    def residual(x: np.ndarray) -> np.ndarray:
        return evaluate(x)[0]

    def jacobian(x: np.ndarray) -> np.ndarray:
        return evaluate(x)[1]

    def numeric_jacobian(x: np.ndarray) -> np.ndarray:
        """Exact (finite-difference) Jacobian, including intensity changes: for errors."""
        r0 = residual(x)
        jac = np.empty((r0.size, x.size))
        for i in range(x.size):
            h = 1e-6 * max(1.0, abs(x[i]))
            xp = x.copy()
            xp[i] += h
            jac[:, i] = (residual(xp) - r0) / h
        return jac

    rng = np.random.default_rng(seed)
    best = None
    converged = 0
    agree = 0  # how many starts reached (within AGREE_RTOL) the best minimum so far
    used = 0
    for start in _starts(free, kinds, x0[: len(free)], starts, rng):
        used += 1
        xs = np.concatenate([np.clip(start, lb_a[: len(free)], ub_a[: len(free)]),
                             x0[len(free):]])  # fmt: skip
        res = least_squares(
            residual, xs, jac=jacobian, bounds=(lb_a, ub_a), x_scale="jac",
            max_nfev=MAX_NFEV_PER_START,
        )  # fmt: skip
        converged += int(res.success)
        if best is None or res.cost < best.cost * (1 - AGREE_RTOL):
            best, agree = res, 1
        elif res.cost <= best.cost * (1 + AGREE_RTOL):
            agree += 1
            if res.cost < best.cost:
                best = res
        if agree >= AGREE_STARTS:
            break  # two independent starts found the same (lowest) minimum
    assert best is not None

    vals, phase = unpack(best.x)
    sim, lines = basis(best.x)
    coef, fitted = linear(sim)
    m, k = y.size, best.x.size
    s2 = 2 * best.cost / max(1, m - k - 3)
    jac = numeric_jacobian(best.x)
    try:
        cov = np.linalg.pinv(jac.T @ jac) * s2
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
    corr = np.sqrt(zf / step)  # remaining correlation if zero filling is not integral
    err = np.sqrt(np.clip(np.diag(cov), 0, None)) * corr
    stderr = {n: float(e) for n, e in zip(free, err[: len(free)], strict=True)}
    for n in fixed:
        stderr[n] = 0.0
    return SpinFit(
        values={n: vals[n] for n in model.names},
        stderr=stderr,
        phase_deg=phase,
        scale=float(coef[0]),
        baseline=(float(coef[1]), float(coef[2])),
        rms=float(np.sqrt(np.mean((fitted - y) ** 2))),
        noise=noise,
        starts=used,
        converged_starts=converged,
        ppm=ppm,
        data=y,
        model=fitted,
        lines=lines,
        spin_model=model,
    )
