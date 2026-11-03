"""Integration: automatic regions, slope/bias correction, normalization.

Areas are in intensity × Hz. ``value`` is the normalized integral (e.g. protons).
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from nmrsuite.analysis.peaks import PeakList, estimate_noise
from nmrsuite.data import NMRData
from nmrsuite.processing.utils import ppm_to_index

#: A region edge above this fraction of the tallest point is on signal, not baseline.
EDGE_SIGNAL_FRACTION = 0.05


@dataclass
class Integral:
    ppm_from: float  # high-shift edge
    ppm_to: float  # low-shift edge
    area: float
    value: float | None = None
    kind: str = "compound"

    @property
    def center(self) -> float:
        return (self.ppm_from + self.ppm_to) / 2


@dataclass
class IntegralList:
    integrals: list[Integral]
    meta: dict = field(default_factory=dict)

    def __iter__(self):
        return iter(self.integrals)

    def __len__(self) -> int:
        return len(self.integrals)

    def __getitem__(self, i: int) -> Integral:
        return self.integrals[i]

    def normalize(self, index: int, value: float) -> IntegralList:
        """Scale so integral ``index`` equals ``value`` (e.g. its proton count)."""
        scale = value / self.integrals[index].area
        for it in self.integrals:
            it.value = it.area * scale
        self.meta["scale"] = scale
        return self

    def total(self, kinds: Sequence[str] = ("compound",)) -> float:
        return float(sum(it.value or 0.0 for it in self.integrals if it.kind in kinds))

    def to_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["ppm_from", "ppm_to", "area", "value", "kind"])
            w.writeheader()
            for it in self.integrals:
                w.writerow(asdict(it))


def integrate_region(
    nmr: NMRData, ppm_from: float, ppm_to: float, slope_bias: bool = True, edge: int = 4
) -> float:
    """Area between two shifts; optional linear slope/bias correction from the edges.

    The correction is skipped when an edge is above EDGE_SIGNAL_FRACTION of the region's
    tallest point: that edge is on the side of a peak, not on baseline.
    """
    d = nmr.direct
    y = np.real(nmr.data)
    i0 = int(np.floor(ppm_to_index(d, max(ppm_from, ppm_to))))
    i1 = int(np.ceil(ppm_to_index(d, min(ppm_from, ppm_to)))) + 1
    i0, i1 = max(0, i0), min(y.size, i1)
    seg = y[i0:i1].astype(float)
    if seg.size == 0:
        return 0.0
    if slope_bias and seg.size > 2 * edge:
        a, b = seg[:edge].mean(), seg[-edge:].mean()
        if max(abs(a), abs(b)) <= EDGE_SIGNAL_FRACTION * np.abs(seg).max():
            seg = seg - np.linspace(a, b, seg.size)
    return float(seg.sum() * d.sw_hz / d.size)


def auto_regions(
    nmr: NMRData,
    peaks: PeakList | None = None,
    min_snr: float = 3.0,
    margin_hz: float | None = None,
    merge_gap_hz: float = 4.0,
) -> list[tuple[float, float]]:
    """Integral regions around signals, as (ppm_from, ppm_to) high → low.

    Points above ``min_snr`` σ are signal; regions are widened by ``margin_hz``
    (default: 4 × the median line width, at least 2 Hz), merged when closer than
    ``merge_gap_hz``, and kept only if they contain a picked peak.
    """
    d = nmr.direct
    y = np.real(nmr.data)
    sigma = estimate_noise(y)
    hz_per_pt = d.sw_hz / d.size
    if peaks is None:
        from nmrsuite.analysis.peaks import pick_peaks

        peaks = pick_peaks(nmr)
    widths = [p.width_hz for p in peaks if p.width_hz > 0]
    if margin_hz is None:
        margin_hz = max(2.0, 4 * float(np.median(widths))) if widths else 3.0
    hot = y > min_snr * sigma
    grow = max(1, round(margin_hz / hz_per_pt))
    hot = np.convolve(hot.astype(float), np.ones(2 * grow + 1), mode="same") > 0
    # contiguous runs
    edges = np.diff(np.concatenate([[0], hot.astype(int), [0]]))
    starts, stops = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    runs: list[list[int]] = []
    gap = merge_gap_hz / hz_per_pt
    for s, e in zip(starts, stops, strict=True):
        if runs and s - runs[-1][1] <= gap:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    peak_idx = np.array([p.index for p in peaks])
    step = d.sw_ppm / d.size
    regions = []
    for s, e in runs:
        if peak_idx.size and np.any((peak_idx >= s) & (peak_idx < e)):
            regions.append((d.offset_ppm - s * step, d.offset_ppm - (e - 1) * step))
    return regions


def integrate(
    nmr: NMRData,
    regions: Sequence[tuple[float, float]] | None = None,
    peaks: PeakList | None = None,
    slope_bias: bool = True,
    method: Literal["sum", "deconvolution"] = "sum",
) -> IntegralList:
    """Integrate given regions, or automatic ones. Regions are labelled by the peaks
    inside them (a region containing only solvent/impurity peaks takes that kind).

    ``method="deconvolution"`` uses the summed areas of fitted lines instead of the
    running sum: better for overlapped regions and for lines whose tails leave the
    region (analytic areas include the full Lorentzian tails).
    """
    if peaks is None:
        from nmrsuite.analysis.peaks import pick_peaks

        peaks = pick_peaks(nmr)
    if regions is None:
        regions = auto_regions(nmr, peaks)
    out = []
    for a, b in regions:
        hi, lo = max(a, b), min(a, b)
        inside = [p for p in peaks if lo <= p.ppm <= hi]
        kinds = {p.kind for p in inside}
        kind = "compound" if "compound" in kinds or not kinds else sorted(kinds)[0]
        if method == "deconvolution" and inside:
            from nmrsuite.analysis.deconvolve import fit_region

            area = fit_region(nmr, hi, lo, seeds_ppm=[p.ppm for p in inside]).total_area
        elif method in ("sum", "deconvolution"):
            area = integrate_region(nmr, hi, lo, slope_bias)
        else:
            raise ValueError(f"unknown integration method {method!r}")
        out.append(Integral(hi, lo, area, kind=kind))
    return IntegralList(out, meta={"method": method})


def auto_proton_counts(
    areas: Sequence[float], total: int | None = None, max_per_region: int = 12
) -> tuple[float, list[int]]:
    """Scale (area per proton) making the integrals closest to whole numbers.

    With ``total`` (from a known formula) the scale is simply Σ area / total.
    Otherwise each region is tried as 1–4 H. The rounding error is weighted by area
    (a small fragment or impurity barely counts), and among scales within
    ``SCALE_TOLERANCE`` of the best error the one giving the fewest protons wins —
    doubling or tripling every count always fits integers at least as well.
    """
    a = np.asarray(areas, float)
    if a.size == 0:
        return 1.0, []
    if total:
        scale = float(a.sum() / total)
        return scale, [max(1, round(v)) for v in a / scale]
    trials: list[tuple[float, int, float, list[int]]] = []
    for ak in a:
        for n in (1, 2, 3, 4):
            scale = ak / n
            v = a / scale
            if v.max() > max_per_region + 0.5:
                continue
            # Every region must be worth at least ~1 H (no rounding 0.5 H up to 1 H),
            # except a few very small fragments, which then do not count at all.
            small = v < 0.75
            shares = a[small] / a.sum()
            if shares.size and (shares.max() > FRAGMENT_SHARE or shares.sum() > 2 * FRAGMENT_SHARE):
                continue
            kept = ~small
            r = np.maximum(1, np.round(v))
            # area-weighted mean relative deviation: |Δ|/n, so shrinking the scale
            # (which shrinks absolute deviations) earns nothing
            err = float(np.sum((a * np.abs(v - r) / r)[kept]) / a[kept].sum())
            trials.append((err, int(r[kept].sum()), float(scale), [int(x) for x in r]))
    if not trials:
        scale = float(a.min())
        return scale, [max(1, round(v)) for v in a / scale]
    best_err = min(t[0] for t in trials)
    ok = [t for t in trials if t[0] <= best_err + SCALE_TOLERANCE]
    err, _, scale, counts = min(ok, key=lambda t: (t[1], t[0]))
    return scale, counts


#: Weighted mean relative rounding error within which a smaller total count is preferred.
SCALE_TOLERANCE = 0.03
#: A region may fall below 1 H, as a fragment, if it holds at most this share of the
#: total area (and all such regions at most twice this).
FRAGMENT_SHARE = 0.03
