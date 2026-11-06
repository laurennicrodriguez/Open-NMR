"""Peak picking and peak classification.

Noise is estimated from the quietest parts of the spectrum (signal-free regions), and
peaks are local maxima above ``min_snr`` × noise with at least that prominence.
Positions are refined to sub-point precision; widths are full widths at half height.

Classification labels each peak ``compound``, ``solvent``, ``impurity`` (common lab
solvents and grease, after Fulmer et al.), ``c13_satellite`` or ``artifact``.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.signal import find_peaks, peak_widths

from nmrsuite.analysis.impurities import IMPURITIES_1H, IMPURITY_TOLERANCE_PPM
from nmrsuite.data import NMRData
from nmrsuite.processing.reference import (
    RESIDUAL_PATTERN_HZ,
    SOLVENT_13C_PATTERN_HZ,
    SOLVENT_SHIFTS,
    canonical_solvent,
    solvent_of,
)
from nmrsuite.processing.utils import refine_peak

PeakKind = Literal["compound", "solvent", "impurity", "c13_satellite", "artifact"]

#: ¹J(C,H) range (Hz) for ¹³C satellites; satellites sit at ±J/2 from the parent.
C13_SATELLITE_J_HZ = (110.0, 220.0)
#: Satellite height relative to the parent (0.55 % each for natural abundance).
C13_SATELLITE_RATIO = (0.0015, 0.015)
#: Smallest noise level assumed, relative to the tallest point.
NOISE_FLOOR = 1e-6
#: Water's shift depends on concentration and temperature.
WATER_TOLERANCE_PPM = 0.06
WATER_PEAK_HZ = 5.0
#: Reach (Hz) within which taller compound lines on both sides overrule an impurity shift.
IMPURITY_BRACKET_HZ = 10.0
#: Artefact thresholds: spikes narrower than this many points; broad features wider
#: than both BROAD_HZ and BROAD_FACTOR × the median line width.
SPIKE_POINTS = 2.5
SOLVENT_SKIRT_FRACTION = 0.05  # of the residual's tallest line
BROAD_HZ = 20.0
BROAD_FACTOR = 10.0


@dataclass
class Peak:
    ppm: float
    hz: float
    index: float
    intensity: float
    width_hz: float
    snr: float
    kind: PeakKind = "compound"
    note: str = ""

    def as_row(self) -> dict:
        return asdict(self)


@dataclass
class PeakList:
    peaks: list[Peak]
    noise: float
    nucleus: str
    obs_mhz: float
    solvent: str | None = None
    meta: dict = field(default_factory=dict)

    def __iter__(self):
        return iter(self.peaks)

    def __len__(self) -> int:
        return len(self.peaks)

    def of_kind(self, *kinds: str) -> list[Peak]:
        return [p for p in self.peaks if p.kind in kinds]

    def to_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(Peak.__dataclass_fields__))
            w.writeheader()
            for p in self.peaks:
                w.writerow(p.as_row())


def estimate_noise(y: np.ndarray, segments: int = 64) -> float:
    """σ of the noise from signal-free regions.

    The spectrum is cut into ``segments`` pieces; each is linearly detrended and its
    standard deviation taken. The median of the quietest quarter is the estimate, so
    peaks and baseline features do not inflate it.
    """
    y = np.asarray(y, float)
    n = y.size // segments
    if n < 8:
        return float(np.std(y)) or 1.0
    x = np.arange(n)
    sds = []
    for k in range(segments):
        seg = y[k * n : (k + 1) * n]
        coef = np.polyfit(x, seg, 1)
        sds.append(np.std(seg - np.polyval(coef, x)))
    sds = np.sort(sds)
    return float(np.median(sds[: max(1, segments // 4)])) or 1.0


def pick_peaks(
    nmr: NMRData,
    min_snr: float = 8.0,
    ppm_range: tuple[float, float] | None = None,
    negative: bool = False,
    classify: bool = True,
) -> PeakList:
    """Pick peaks in the real part of a processed 1D spectrum."""
    d = nmr.direct
    if d.domain != "frequency" or nmr.ndim != 1:
        raise ValueError("peak picking needs a 1D frequency-domain spectrum")
    y = np.real(nmr.data).astype(float)
    # floor: no real spectrum has more than ~10⁶ dynamic range; without it a noise-free
    # (simulated) spectrum would turn floating-point ripple into thousands of peaks
    sigma = max(estimate_noise(y), NOISE_FLOOR * float(np.abs(y).max()))
    thr = min_snr * sigma
    step_ppm = d.sw_ppm / d.size
    hz_per_pt = d.sw_hz / d.size
    signs = (1, -1) if negative else (1,)
    peaks: list[Peak] = []
    for sign in signs:
        s = sign * y
        idx, _ = find_peaks(s, height=thr, prominence=thr)
        # at half prominence: a small line on a hump or a taller line's shoulder never
        # falls to half its height, so that would measure the hump instead of the line
        widths = peak_widths(s, idx, rel_height=0.5)[0] * hz_per_pt
        for i, width in zip(idx, widths, strict=True):
            pos = refine_peak(s, int(i))
            ppm = d.offset_ppm - pos * step_ppm
            if ppm_range and not (min(ppm_range) <= ppm <= max(ppm_range)):
                continue
            peaks.append(Peak(
                ppm=float(ppm),
                hz=float(ppm * d.obs_mhz),
                index=float(pos),
                intensity=float(y[i]),
                width_hz=float(width),
                snr=float(s[i] / sigma),
            ))  # fmt: skip
    peaks.sort(key=lambda p: -p.ppm)
    solvent = solvent_of(nmr)
    try:
        solvent = canonical_solvent(solvent) if solvent else None
    except KeyError:
        solvent = None
    out = PeakList(
        peaks=peaks, noise=sigma, nucleus=d.nucleus, obs_mhz=d.obs_mhz, solvent=solvent,
        meta={"hz_per_pt": hz_per_pt},
    )  # fmt: skip
    if classify:
        classify_peaks(out)
    return out


def _mark_solvent(pl: PeakList) -> None:
    if not pl.solvent:
        return
    shifts = SOLVENT_SHIFTS[pl.solvent].get(pl.nucleus, [])
    if pl.nucleus == "1H":
        half = RESIDUAL_PATTERN_HZ.get(pl.solvent, 0.0) + 2.0
    else:
        half = SOLVENT_13C_PATTERN_HZ.get(pl.solvent, 30.0) + 6.0
    for p in pl.peaks:
        # a broad foot under the residual (shimming, exchange) also belongs to it
        reach = half + max(0.0, p.width_hz - BROAD_HZ) / 2
        if any(abs(p.ppm - s) * pl.obs_mhz <= reach for s in shifts):
            p.kind, p.note = "solvent", pl.solvent
    # small lines on the shoulders of a strong residual are its skirt, not the compound's
    own = [p.intensity for p in pl.peaks if p.kind == "solvent"]
    if pl.nucleus == "1H" and own:
        for p in pl.peaks:
            if (p.kind == "compound" and p.intensity < SOLVENT_SKIRT_FRACTION * max(own)
                    and any(abs(p.ppm - s) * pl.obs_mhz <= 2 * half for s in shifts)):  # fmt: skip
                p.kind, p.note = "solvent", pl.solvent


def _mark_satellites(pl: PeakList) -> None:
    peaks = pl.peaks
    if pl.nucleus != "1H" or len(peaks) < 3:
        return
    hz = np.array([p.hz for p in peaks])
    amp = np.array([p.intensity for p in peaks])
    order = np.argsort(-amp)
    lo, hi = (j / 2 for j in C13_SATELLITE_J_HZ)
    for k in order[: min(len(order), 30)]:
        parent = peaks[k]
        if parent.intensity <= 0:
            continue
        ratio = amp / parent.intensity
        cand = (ratio >= C13_SATELLITE_RATIO[0]) & (ratio <= C13_SATELLITE_RATIO[1])
        left = np.nonzero(cand & (hz > parent.hz + lo) & (hz < parent.hz + hi))[0]
        right = np.nonzero(cand & (hz < parent.hz - lo) & (hz > parent.hz - hi))[0]
        for a in left:
            for b in right:
                da, db = hz[a] - parent.hz, parent.hz - hz[b]
                # symmetric about the parent (isotope shift ≲ 1 Hz); heights may differ
                # several-fold (isotope shift, different widths, overlap)
                if abs(da - db) < 3.0 and 0.25 < amp[a] / amp[b] < 4.0:
                    for j in (a, b):
                        if peaks[j].kind == "compound":
                            peaks[j].kind = "c13_satellite"
                            peaks[j].note = f"{parent.ppm:.3f}, 1J(CH) ≈ {da + db:.0f} Hz"


def _mark_impurities(pl: PeakList) -> None:
    if pl.nucleus != "1H" or not pl.solvent or pl.solvent not in IMPURITIES_1H:
        return
    compound = [p.intensity for p in pl.peaks if p.kind == "compound"]
    if not compound:
        return
    table = IMPURITIES_1H[pl.solvent]
    # Only small peaks are labelled (a big peak at 1.26 ppm is the compound's own),
    # except water, whose shift and size vary: the line nearest the water shift, and
    # any line within WATER_PEAK_HZ of it (a lumpy water peak), is water whatever its
    # size; others nearby are water only if small (a methyl singlet beside water, e.g.
    # at 3.34 in DMSO or 2.08 in CD3CN, is the compound's)
    water = [
        k for k, p in enumerate(pl.peaks)
        if p.kind == "compound"
        and any(abs(p.ppm - s) <= WATER_TOLERANCE_PPM for s in table.get("water", []))
    ]  # fmt: skip
    nearest = min(
        water,
        key=lambda k: min(abs(pl.peaks[k].ppm - s) for s in table["water"]),
        default=None,
    )
    # sized against the compound alone: water can be the tallest line by far
    limit = 0.2 * max(
        (p.intensity for k, p in enumerate(pl.peaks) if p.kind == "compound" and k != nearest),
        default=max(compound),
    )
    candidates: dict[int, list[str]] = {}
    for k, p in enumerate(pl.peaks):
        if p.kind != "compound":
            continue
        own = nearest is not None and abs(p.hz - pl.peaks[nearest].hz) <= WATER_PEAK_HZ
        if k in water and (own or p.intensity <= limit):
            p.kind, p.note = "impurity", "water"
            continue
        if p.intensity > limit:
            continue
        hits = [
            name
            for name, shifts in table.items()
            if name != "water" and any(abs(p.ppm - s) <= IMPURITY_TOLERANCE_PPM for s in shifts)
        ]
        if hits:
            candidates[k] = hits
    # A line with taller compound lines close on both sides belongs to their multiplet.
    others = [q for k, q in enumerate(pl.peaks) if q.kind == "compound" and k not in candidates]
    for k, hits in candidates.items():
        p = pl.peaks[k]
        near = [q for q in others if q.intensity > p.intensity
                and abs(q.hz - p.hz) <= IMPURITY_BRACKET_HZ]  # fmt: skip
        if not (any(q.hz > p.hz for q in near) and any(q.hz < p.hz for q in near)):
            p.kind, p.note = "impurity", " or ".join(hits)


def _mark_artifacts(pl: PeakList, hz_per_pt: float) -> None:
    """Spikes (a couple of points wide) and very broad features.

    Broad features — baseline humps at the spectrum edges, the feet of very strong
    lines — are wider than BROAD_HZ and BROAD_FACTOR × the median line width. An
    exchange-broadened OH/NH can also land here; it is kept in the peak list, marked,
    and left out of automatic multiplets.
    """
    widths = [p.width_hz for p in pl.peaks if p.kind == "compound" and p.width_hz > 0]
    typical = float(np.median(widths)) if widths else 1.0
    for p in pl.peaks:
        if p.kind != "compound":
            continue
        if 0 < p.width_hz < SPIKE_POINTS * hz_per_pt:
            p.kind, p.note = "artifact", "spike (narrower than the digital resolution allows)"
        elif p.width_hz > max(BROAD_HZ, BROAD_FACTOR * typical):
            p.kind, p.note = "artifact", "broad: baseline hump or exchange-broadened signal"


def classify_peaks(pl: PeakList) -> PeakList:
    """Label peaks in place (solvent, ¹³C satellites, impurities, artefacts)."""
    _mark_solvent(pl)
    _mark_artifacts(pl, pl.meta.get("hz_per_pt", 0.0))
    _mark_satellites(pl)
    _mark_impurities(pl)
    return pl
