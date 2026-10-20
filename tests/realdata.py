"""Committed real datasets (tests/data) and the summary that is frozen for regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nmrsuite.data import NMRData
from nmrsuite.processing.utils import find_peak_indices, noise_sigma, refine_peak

DATA = Path(__file__).parent / "data" / "chem2081"
DATASETS: dict[str, Path] = {p.parent.name: p for p in sorted(DATA.glob("*/1")) if p.is_dir()}

N_PEAKS = 12


def summarize(nmr: NMRData) -> dict[str, Any]:
    """Reference shift and the positions of the tallest peaks (ppm, sub-point)."""
    y = nmr.data.real
    d = nmr.direct
    idx = find_peak_indices(y, 50 * noise_sigma(y), max_peaks=N_PEAKS)
    step = d.sw_ppm / d.size
    peaks = sorted((d.offset_ppm - refine_peak(y, int(i)) * step for i in idx), reverse=True)
    ref = next((h for h in nmr.history if h["step"] == "reference"), None)
    return {
        "reference_shift_ppm": round(ref["result"]["shift_ppm"], 5) if ref else None,
        "peaks_ppm": [round(float(p), 5) for p in peaks],
        "size": int(d.size),
        "max_over_noise": float(np.round(y.max() / noise_sigma(y), -1)),
    }
