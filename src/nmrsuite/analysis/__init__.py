"""Analysis: peaks, integrals, deconvolution, multiplets."""

from __future__ import annotations

from dataclasses import dataclass, field

from nmrsuite.analysis.deconvolve import FitResult, deconvolve, fit_region
from nmrsuite.analysis.integrate import (
    Integral,
    IntegralList,
    auto_proton_counts,
    auto_regions,
    integrate,
    integrate_region,
)
from nmrsuite.analysis.multiplet_firstorder import (
    Coupling,
    Multiplet,
    analyse_lines,
    detect_multiplets,
    match_couplings,
    merge_multiplets,
    normalize_multiplets,
    set_pattern,
)
from nmrsuite.analysis.peaks import Peak, PeakList, classify_peaks, estimate_noise, pick_peaks
from nmrsuite.data import NMRData


@dataclass
class Analysis1D:
    peaks: PeakList
    integrals: IntegralList
    multiplets: list[Multiplet] = field(default_factory=list)
    report: str = ""
    unmatched_j: list[tuple[int, float]] = field(default_factory=list)


def analyse_1d(
    nmr: NMRData,
    style: str = "ACS",
    total_h: int | None = None,
    normalize: tuple[float, int] | None = None,
    min_snr: float = 8.0,
) -> Analysis1D:
    """Peaks, integrals and (for ¹H) multiplets plus the journal report line.

    Proton counts: ``total_h`` (from the formula) or ``normalize=(ppm, n)`` (that
    multiplet has n H); otherwise an automatic, suggested scale.
    """
    from nmrsuite.report import carbon_report, proton_report

    peaks = pick_peaks(nmr, min_snr=min_snr, negative=nmr.experiment.startswith("DEPT"))
    integrals = integrate(nmr, peaks=peaks)
    if nmr.direct.nucleus == "1H":
        mults = detect_multiplets(nmr, peaks, total_h=total_h)
        if normalize:
            normalize_multiplets(mults, *normalize)
        _, unmatched = match_couplings(mults)
        return Analysis1D(peaks, integrals, mults, proton_report(nmr, mults, style), unmatched)
    return Analysis1D(peaks, integrals, report=carbon_report(nmr, peaks, style))


__all__ = [
    "Analysis1D",
    "Coupling",
    "FitResult",
    "Integral",
    "IntegralList",
    "Multiplet",
    "Peak",
    "PeakList",
    "analyse_1d",
    "analyse_lines",
    "auto_proton_counts",
    "auto_regions",
    "classify_peaks",
    "deconvolve",
    "detect_multiplets",
    "estimate_noise",
    "fit_region",
    "integrate",
    "integrate_region",
    "match_couplings",
    "merge_multiplets",
    "normalize_multiplets",
    "pick_peaks",
    "set_pattern",
]
