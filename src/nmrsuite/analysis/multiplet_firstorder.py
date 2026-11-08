"""First-order multiplet analysis.

Lines (picked peaks) are clustered into multiplets. For each cluster we search for the
set of couplings — each a J value with ``n`` equivalent partners — whose first-order
stick pattern (binomial splittings, lines closer than the resolution merged) matches
the observed line positions *and* relative intensities. The span of the multiplet
must equal Σ n·J, which keeps the search small. J values are then refined by linear
least squares on all line positions. Clusters that cannot be explained are reported
as ``m`` with their range.

Pattern names follow the usual convention: letters in order of decreasing J
(``dt``: the larger coupling is a doublet), with d, t, q, p (pentet), sext, sept.

This is the parity baseline. Second-order systems (Δν/J < 10) are handled by the
spin-simulation engine in a later build step; until then such multiplets may be
reported with apparent (first-order) splittings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from math import comb

import numpy as np

from nmrsuite.analysis.integrate import auto_proton_counts, auto_regions, integrate_region
from nmrsuite.analysis.peaks import Peak, PeakList, pick_peaks
from nmrsuite.data import NMRData
from nmrsuite.processing.utils import ppm_to_index

LETTERS = {1: "d", 2: "t", 3: "q", 4: "p", 5: "sext", 6: "sept", 7: "oct"}

#: Lines further apart than this (Hz) start a new multiplet.
CLUSTER_GAP_HZ = 20.0
#: Couplings smaller than this (Hz) are not resolved into the pattern.
MIN_J_HZ = 0.6
#: Largest ¹H–¹H coupling considered (geminal and trans-vinyl couplings reach ~18 Hz).
MAX_J_HZ = 20.0
MAX_COUPLINGS = 4
#: In a pattern with several couplings, each has at most this many equal partners
#: (qd, tt are common; "dsext" is not a credible first-order reading).
MAX_PARTNERS_MIXED = 3
#: Multiplets smaller than this many protons (after scaling) are flagged as minor
#: and left out of the report.
MINOR_H = 0.5
#: Integration window beyond the outer lines, in median line widths.
INTEGRAL_PAD_FWHM = 12.0
#: A split between multiplets above this fraction of the region's tallest point is not baseline.
BASELINE_FRACTION = 0.01
MAX_LINES = 32
#: Acceptance limits for a first-order explanation.
MAX_POSITION_ERROR = 1.5  # × tolerance
MAX_INTENSITY_ERROR = 0.35  # half the L1 distance of normalized intensities
#: A singlet wider than this many times the median line width is "br s".
BROAD_FACTOR = 2.5


@dataclass
class Coupling:
    j_hz: float
    n: int = 1  # number of equivalent partners

    @property
    def letter(self) -> str:
        return LETTERS.get(self.n, f"{self.n + 1}-plet")


@dataclass
class Multiplet:
    ppm: float
    ppm_from: float  # high-shift edge of the multiplet
    ppm_to: float  # low-shift edge
    lines_hz: list[float]
    lines_intensity: list[float]
    pattern: str = "m"
    couplings: list[Coupling] = field(default_factory=list)
    integral: float = 0.0
    n_h: int | None = None
    fit_error: float | None = None
    flags: list[str] = field(default_factory=list)
    manual: bool = False
    obs_mhz: float = 0.0

    @property
    def js(self) -> list[float]:
        return [c.j_hz for c in self.couplings]

    @property
    def is_first_order(self) -> bool:
        return self.pattern not in ("m",)


# ---------------------------------------------------------------------------
# Pattern search
# ---------------------------------------------------------------------------


def stick_pattern(couplings: Sequence[Coupling]) -> tuple[np.ndarray, np.ndarray]:
    """Positions (relative to the lowest line) and intensities of a first-order pattern."""
    pos = np.array([0.0])
    amp = np.array([1.0])
    for c in couplings:
        offs = np.arange(c.n + 1) * c.j_hz
        w = np.array([comb(c.n, k) for k in range(c.n + 1)], float)
        pos = (pos[:, None] + offs[None, :]).ravel()
        amp = (amp[:, None] * w[None, :]).ravel()
    order = np.argsort(pos)
    return pos[order], amp[order] / amp.sum()


def _merge(pos: np.ndarray, amp: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray, list]:
    groups: list[list[int]] = []
    for i in range(pos.size):
        if groups and pos[i] - pos[groups[-1][0]] <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    mp = np.array([np.average(pos[g], weights=amp[g]) for g in groups])
    ma = np.array([amp[g].sum() for g in groups])
    return mp, ma, groups


def _coefficients(couplings: Sequence[Coupling]) -> np.ndarray:
    """Unmerged sticks as integer multiples of each J (rows: sticks, cols: couplings)."""
    rows = [np.zeros(len(couplings))]
    for ci, c in enumerate(couplings):
        new = []
        for r in rows:
            for k in range(c.n + 1):
                rr = r.copy()
                rr[ci] = k
                new.append(rr)
        rows = new
    coef = np.array(rows)
    jv = np.array([c.j_hz for c in couplings])
    return coef[np.argsort(coef @ jv, kind="stable")]


def _evaluate(
    couplings: list[Coupling], hz: np.ndarray, amp: np.ndarray, tol: float
) -> tuple[float, list[Coupling]] | None:
    pos, pamp = stick_pattern(couplings)
    mp, ma, groups = _merge(pos, pamp, tol)
    if mp.size != hz.size:
        return None
    centre_obs = (hz[0] + hz[-1]) / 2
    pred = mp - (mp[0] + mp[-1]) / 2 + centre_obs
    dev = np.abs(pred - hz)
    if dev.max() > MAX_POSITION_ERROR * tol:
        return None
    int_err = 0.5 * np.abs(ma - amp).sum()
    if int_err > MAX_INTENSITY_ERROR:
        return None
    # refine J by least squares: every unmerged stick sits on its matched observed line
    coef = _coefficients(couplings)
    jv = np.array([c.j_hz for c in couplings])
    stick_pos = coef @ jv
    order = np.argsort(stick_pos, kind="stable")
    owner = np.empty(stick_pos.size, int)
    for gi, g in enumerate(groups):
        owner[order[g]] = gi
    a = np.column_stack([coef, np.ones(len(coef))])
    sol, *_ = np.linalg.lstsq(a, hz[owner], rcond=None)
    refined = [Coupling(float(j), c.n) for j, c in zip(sol[:-1], couplings, strict=True)]
    rms = float(np.sqrt(np.mean((a @ sol - hz[owner]) ** 2)))
    score = rms / tol + 2.0 * int_err + 0.05 * sum(c.n for c in couplings)
    return score, refined


def analyse_lines(
    hz: Sequence[float], intensity: Sequence[float], tol: float
) -> tuple[str, list[Coupling], float | None]:
    """Best first-order explanation of a set of lines: (pattern, couplings, score)."""
    order = np.argsort(hz)
    hz = np.asarray(hz, float)[order]
    amp = np.clip(np.asarray(intensity, float)[order], 1e-12, None)
    amp = amp / amp.sum()
    n = hz.size
    if n == 1:
        return "s", [], 0.0
    if n > MAX_LINES:
        return "m", [], None
    span = hz[-1] - hz[0]
    raw = np.concatenate([hz[1:] - hz[0], hz[-1] - hz[:-1]])
    raw = np.sort(raw[(raw >= MIN_J_HZ) & (raw <= min(span, MAX_J_HZ) + tol)])[::-1]
    cand: list[float] = []
    for v in raw:
        if not cand or abs(cand[-1] - v) > tol / 2:
            cand.append(float(v))
    best: tuple[float, list[Coupling]] | None = None

    def search(start: int, remaining: float, chosen: list[Coupling]) -> None:
        nonlocal best
        mixed_ok = len(chosen) < 2 or max(c.n for c in chosen) <= MAX_PARTNERS_MIXED
        if abs(remaining) <= tol and chosen and mixed_ok:
            res = _evaluate(chosen, hz, amp, tol)
            if res and (best is None or res[0] < best[0]):
                best = res
        if len(chosen) >= MAX_COUPLINGS or remaining < MIN_J_HZ - tol:
            return
        for k in range(start, len(cand)):
            j = cand[k]
            for mult in range(1, 7):
                if mult * j > remaining + tol:
                    break
                search(k + 1, remaining - mult * j, [*chosen, Coupling(j, mult)])

    search(0, span, [])
    if best is None:
        return "m", [], None
    couplings = sorted(best[1], key=lambda c: -c.j_hz)
    return "".join(c.letter for c in couplings), couplings, best[0]


# ---------------------------------------------------------------------------
# Multiplet detection
# ---------------------------------------------------------------------------


def _cluster(
    peaks: list[Peak], gap_hz: float, regions: Sequence[tuple[float, float]] = ()
) -> list[list[Peak]]:
    """Group lines: neighbours within ``gap_hz`` and in the same integral region
    (the baseline returning between two signals separates them)."""

    def region_of(p: Peak) -> int:
        for k, (a, b) in enumerate(regions):
            if min(a, b) <= p.ppm <= max(a, b):
                return k
        return -1

    peaks = sorted(peaks, key=lambda p: p.hz)
    clusters: list[list[Peak]] = []
    last_region = None
    for p in peaks:
        r = region_of(p)
        if clusters and p.hz - clusters[-1][-1].hz <= gap_hz and r == last_region:
            clusters[-1].append(p)
        else:
            clusters.append([p])
        last_region = r
    return clusters


def _build(lines: list[Peak], tol: float, typical_width: float, obs_mhz: float) -> Multiplet:
    hz = np.array([p.hz for p in lines])
    amp = np.array([p.intensity for p in lines])
    pattern, couplings, score = analyse_lines(hz, amp, tol)
    # median: a small line on the side of a taller one measures far too wide
    half = float(np.median([p.width_hz for p in lines]))
    hi = (hz.max() + half) / obs_mhz
    lo = (hz.min() - half) / obs_mhz
    if pattern == "s":
        centre = lines[0].ppm
        if lines[0].width_hz > BROAD_FACTOR * typical_width:
            pattern = "br s"
    elif pattern == "m":
        centre = float(np.average(hz, weights=np.clip(amp, 1e-12, None)) / obs_mhz)
    else:
        centre = float((hz.min() + hz.max()) / 2 / obs_mhz)
    return Multiplet(
        ppm=centre,
        ppm_from=float(hi),
        ppm_to=float(lo),
        lines_hz=[float(v) for v in hz],
        lines_intensity=[float(v) for v in amp],
        pattern=pattern,
        couplings=couplings,
        fit_error=score,
        obs_mhz=obs_mhz,
    )


def detect_multiplets(
    nmr: NMRData,
    peaks: PeakList | None = None,
    gap_hz: float = CLUSTER_GAP_HZ,
    total_h: int | None = None,
) -> list[Multiplet]:
    """Find and analyse the compound multiplets of a processed ¹H spectrum.

    Integrals are taken over each multiplet's range (bounded halfway to its
    neighbours) and converted to whole-proton counts (``total_h`` if the formula is
    known, else the most consistent integer scale).
    """
    if peaks is None:
        peaks = pick_peaks(nmr)
    compound = peaks.of_kind("compound")
    if not compound:
        return []
    typical = float(np.median([p.width_hz for p in compound if p.width_hz > 0] or [1.0]))
    tol = max(0.3, 0.6 * typical)
    obs = nmr.direct.obs_mhz
    regions = auto_regions(nmr, peaks)
    out: list[Multiplet] = []
    for cl in _cluster(compound, gap_hz, regions):
        m = _build(cl, tol, typical, obs)
        if m.pattern == "m" and len(cl) >= 4:
            # maybe two multiplets close together: try the largest gap
            hz = np.array([p.hz for p in cl])
            k = int(np.argmax(np.diff(hz))) + 1
            a, b = _build(cl[:k], tol, typical, obs), _build(cl[k:], tol, typical, obs)
            if a.pattern != "m" and b.pattern != "m":
                out += [a, b]
                continue
        out.append(m)
    out.sort(key=lambda m: -m.ppm)
    _integrate(nmr, out, typical)
    _assign_protons(out, total_h)
    return out


def _assign_protons(
    mults: list[Multiplet], total_h: int | None, scale: float | None = None
) -> None:
    """Whole-proton counts; tiny multiplets (impurities, satellites) become 'minor'.

    Without an explicit ``scale`` (area per proton) it is chosen from the substantial
    multiplets only (≥ 20 % of the median area), so a small unlabelled impurity cannot
    be taken as the 1H reference. Automatic counts are suggestions: with no formula
    an incomplete multiplet can still halve the scale. Pass ``total_h`` or use
    :func:`normalize_multiplets` when it matters.
    """
    areas = np.array([max(m.integral, 0.0) for m in mults])
    if not areas.size or areas.max() <= 0:
        return
    if scale is None:
        big = areas >= 0.2 * np.median(areas)
        scale, _ = auto_proton_counts(areas[big], total=total_h)
    for m, a in zip(mults, areas, strict=True):
        m.flags = [f for f in m.flags if f != "minor"]
        v = a / scale
        if v < MINOR_H:
            m.flags.append("minor")
            m.n_h = None
        else:
            m.n_h = max(1, round(v))


def normalize_multiplets(mults: list[Multiplet], ppm: float, n_h: int) -> list[Multiplet]:
    """Set the multiplet nearest ``ppm`` to ``n_h`` protons and rescale the others."""
    ref = min(mults, key=lambda m: abs(m.ppm - ppm))
    if ref.integral <= 0:
        raise ValueError("reference multiplet has no positive integral")
    _assign_protons(mults, None, scale=ref.integral / n_h)
    return mults


def _boundary_ppm(nmr: NMRData, upper: Multiplet, lower: Multiplet) -> float:
    """Where to split two neighbouring multiplets: halfway between their ranges or, when
    the ranges overlap, at the lowest point between their facing lines."""
    middle = (upper.ppm_to + lower.ppm_from) / 2
    if upper.ppm_to > lower.ppm_from:
        return middle
    d = nmr.direct
    i0 = int(np.ceil(ppm_to_index(d, min(upper.lines_hz) / d.obs_mhz)))
    i1 = int(np.floor(ppm_to_index(d, max(lower.lines_hz) / d.obs_mhz)))
    if i1 <= i0:  # interleaved lines: no valley to find
        return middle
    k = i0 + int(np.argmin(np.real(nmr.data)[i0 : i1 + 1]))
    return d.offset_ppm - k * d.sw_ppm / d.size


def _boundary_on_signal(nmr: NMRData, ppm: float, hi: float, lo: float) -> bool:
    """Whether a boundary inside hi–lo is above BASELINE_FRACTION of the tallest point."""
    d = nmr.direct
    y = np.abs(np.real(nmr.data))
    i0, i1 = sorted(round(ppm_to_index(d, v)) for v in (hi, lo))
    return bool(y[round(ppm_to_index(d, ppm))] > BASELINE_FRACTION * y[i0 : i1 + 1].max())


def _integrate(nmr: NMRData, mults: list[Multiplet], typical: float) -> None:
    """Integrate each multiplet ±INTEGRAL_PAD_FWHM line widths beyond its outer lines
    (Lorentzian tails hold real area), bounded by the splits to its neighbours. A split
    still on signal is not baseline, so that region gets no slope/bias correction."""
    obs = nmr.direct.obs_mhz
    pad = max(INTEGRAL_PAD_FWHM * typical, 3.0) / obs
    splits = [_boundary_ppm(nmr, a, b) for a, b in pairwise(mults)]
    for i, m in enumerate(mults):  # sorted high → low ppm
        hi = m.ppm_from + pad
        lo = m.ppm_to - pad
        used = []
        if i > 0 and splits[i - 1] < hi:
            hi = splits[i - 1]
            used.append(hi)
        if i + 1 < len(mults) and splits[i] > lo:
            lo = splits[i]
            used.append(lo)
        on_signal = any(_boundary_on_signal(nmr, b, hi, lo) for b in used)
        m.integral = integrate_region(nmr, hi, lo, slope_bias=not on_signal)


# ---------------------------------------------------------------------------
# Coupling matching and manual editing
# ---------------------------------------------------------------------------


@dataclass
class JMatch:
    a: int
    b: int
    j_hz: float


def match_couplings(
    mults: Sequence[Multiplet], tol: float = 0.3
) -> tuple[list[JMatch], list[tuple[int, float]]]:
    """Pair equal J values between multiplets; return pairs and unmatched (index, J).

    In a consistent first-order analysis every coupling appears in both partners, so
    an unmatched J points at a misread multiplet (or a partner outside the spectrum,
    e.g. ¹⁹F/³¹P, or an exchangeable proton).
    """
    pairs: list[JMatch] = []
    used: set[tuple[int, int]] = set()
    for i, mi in enumerate(mults):
        for ci, c in enumerate(mi.couplings):
            for j, mj in enumerate(mults):
                if j <= i:
                    continue
                for cj, d in enumerate(mj.couplings):
                    if (j, cj) in used or (i, ci) in used:
                        continue
                    if abs(c.j_hz - d.j_hz) <= tol:
                        pairs.append(JMatch(i, j, (c.j_hz + d.j_hz) / 2))
                        used |= {(i, ci), (j, cj)}
                        break
    unmatched = [
        (i, c.j_hz)
        for i, m in enumerate(mults)
        for ci, c in enumerate(m.couplings)
        if (i, ci) not in used
    ]
    return pairs, unmatched


def merge_multiplets(a: Multiplet, b: Multiplet, tol: float = 0.6) -> Multiplet:
    """Combine two multiplets into one and re-analyse it."""
    hz = [*a.lines_hz, *b.lines_hz]
    amp = [*a.lines_intensity, *b.lines_intensity]
    pattern, couplings, score = analyse_lines(hz, amp, tol)
    obs = a.obs_mhz or b.obs_mhz
    centre = (min(hz) + max(hz)) / 2 / obs if pattern != "m" and obs else (a.ppm + b.ppm) / 2
    return Multiplet(
        ppm=centre,
        ppm_from=max(a.ppm_from, b.ppm_from),
        ppm_to=min(a.ppm_to, b.ppm_to),
        lines_hz=sorted(hz),
        lines_intensity=[x for _, x in sorted(zip(hz, amp, strict=True))],
        pattern=pattern,
        couplings=couplings,
        integral=a.integral + b.integral,
        n_h=(a.n_h or 0) + (b.n_h or 0) or None,
        fit_error=score,
        manual=True,
        obs_mhz=obs,
    )


def set_pattern(m: Multiplet, pattern: str, js: Sequence[float] = ()) -> Multiplet:
    """Override the analysis, e.g. ``set_pattern(m, "dt", [10.2, 6.5])`` or ``"m"``."""
    letters = {v: k for k, v in LETTERS.items()}
    parts: list[int] = []
    rest = pattern
    while rest and rest not in ("s", "m", "br s"):
        for code in sorted(letters, key=len, reverse=True):
            if rest.startswith(code):
                parts.append(letters[code])
                rest = rest[len(code) :]
                break
        else:
            raise ValueError(f"cannot parse multiplicity {pattern!r}")
    if len(parts) != len(js):
        raise ValueError(f"{pattern!r} needs {len(parts)} J values, got {len(js)}")
    m.pattern = pattern
    m.couplings = [Coupling(float(j), n) for j, n in zip(js, parts, strict=True)]
    m.manual = True
    return m
