"""Chemical-shift referencing.

Methods
-------
``solvent`` (default)
    Find the residual solvent signal near its literature shift and move it there.
    Shifts from Fulmer et al., Organometallics 29, 2176 (2010).
``tms``
    The same search for TMS at 0 ppm.
``peak``
    Move the peak nearest ``ppm_from`` to ``ppm_to``.
``offset``
    Shift the whole scale by ``delta`` ppm.
``stored``
    Use the reference TopSpin stored for this experiment (``SF`` in pdata/1/procs).
``xi``
    Indirect referencing via IUPAC Ξ ratios (Harris et al., Pure Appl. Chem. 73, 1795,
    2001) from ``h_ref_mhz``, the absolute frequency of TMS ¹H (0 ppm) on the same magnet.

The data are never resampled; referencing only changes ``offset_ppm``. Each method
records the applied shift (``result.shift_ppm``) so exact replay is a plain offset.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import replace
from typing import Literal

import numpy as np

from nmrsuite.data import NMRData
from nmrsuite.history import processing_step
from nmrsuite.io._paramfile import read_param_file
from nmrsuite.processing.utils import noise_sigma, ppm_to_index, refine_peak

#: Residual solvent signals (ppm) used for referencing: the first entry per nucleus.
SOLVENT_SHIFTS: dict[str, dict[str, list[float]]] = {
    "CDCl3": {"1H": [7.26], "13C": [77.16]},
    "DMSO-d6": {"1H": [2.50], "13C": [39.52]},
    "CD3OD": {"1H": [3.31], "13C": [49.00]},
    "D2O": {"1H": [4.79]},
    "acetone-d6": {"1H": [2.05], "13C": [29.84, 206.26]},
    "C6D6": {"1H": [7.16], "13C": [128.06]},
    "CD3CN": {"1H": [1.94], "13C": [1.32, 118.26]},
    "CD2Cl2": {"1H": [5.32], "13C": [53.84]},
    "THF-d8": {"1H": [3.58, 1.72], "13C": [67.21, 25.31]},
    "toluene-d8": {"1H": [2.08], "13C": [20.43]},
    "pyridine-d5": {"1H": [8.74], "13C": [150.35]},
}

_ALIASES = {
    "cdcl3": "CDCl3", "chloroform": "CDCl3",
    "dmso": "DMSO-d6", "dmsod6": "DMSO-d6",
    "meod": "CD3OD", "cd3od": "CD3OD", "methanol": "CD3OD",
    "d2o": "D2O",
    "acetone": "acetone-d6", "acetoned6": "acetone-d6",
    "c6d6": "C6D6", "benzene": "C6D6",
    "cd3cn": "CD3CN", "acn": "CD3CN", "acetonitrile": "CD3CN",
    "cd2cl2": "CD2Cl2",
    "thf": "THF-d8", "thfd8": "THF-d8",
    "tol": "toluene-d8", "toluene": "toluene-d8", "toluened8": "toluene-d8",
    "pyr": "pyridine-d5", "pyridine": "pyridine-d5", "pyridined5": "pyridine-d5",
}  # fmt: skip

#: IUPAC Ξ ratios (%), relative to ¹H of TMS = 100 %.
XI_RATIOS: dict[str, float] = {
    "1H": 100.0,
    "2H": 15.350609,
    "11B": 32.083974,
    "13C": 25.145020,
    "15N": 10.136767,
    "19F": 94.094011,
    "29Si": 19.867187,
    "31P": 40.480742,
}

#: Half-width (ppm) of the search for a reference signal around its expected shift.
SEARCH_WINDOW_PPM = 0.15


def canonical_solvent(name: str) -> str:
    key = re.sub(r"[\s\-_]", "", name).lower()
    if key in _ALIASES:
        return _ALIASES[key]
    for canon in SOLVENT_SHIFTS:
        if re.sub(r"[\s\-_]", "", canon).lower() == key:
            return canon
    raise KeyError(f"unknown solvent {name!r}")


def solvent_of(nmr: NMRData) -> str | None:
    for source in ("acqus", "jcamp"):
        p = nmr.params.get(source, {})
        name = p.get("SOLVENT") or p.get(".SOLVENTNAME")
        if name:
            return str(name)
    return None


#: Half-width (Hz) of each ¹H residual's own multiplet: ²J(H,D) quintets for CHD₂
#: groups (outer lines at ±2J), the CHDCl₂ 1:1:1 triplet, broad multiplets for THF and
#: pyridine. CHCl₃, C₆D₅H, HOD and TMS are singlets (0).
RESIDUAL_PATTERN_HZ: dict[str, float] = {
    "DMSO-d6": 3.8, "CD3OD": 3.4, "acetone-d6": 4.4, "CD3CN": 5.0, "toluene-d8": 4.6,
    "CD2Cl2": 1.0, "THF-d8": 5.0, "pyridine-d5": 5.0,
}  # fmt: skip

#: A reference candidate is "isolated" if no line ≥ this fraction of its height lies
#: between 1.5 and 15 Hz beyond its own pattern (so it is not a line of a compound
#: multiplet that happens to sit near the expected shift).
ISOLATION_FRACTION = 0.25
ISOLATION_HZ = (1.5, 15.0)


def _local_maxima(y: np.ndarray, threshold: float) -> np.ndarray:
    inner = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > threshold)
    return np.nonzero(inner)[0] + 1


def _is_isolated(
    y: np.ndarray, maxima: np.ndarray, i: int, pts_per_hz: float, pattern_hz: float
) -> bool:
    near = np.abs(maxima - i)
    lo = (pattern_hz + ISOLATION_HZ[0]) * pts_per_hz
    hi = (pattern_hz + ISOLATION_HZ[1]) * pts_per_hz
    neighbours = maxima[(near >= lo) & (near <= hi)]
    return not np.any(y[neighbours] >= ISOLATION_FRACTION * y[i])


def find_reference_peak(
    nmr: NMRData,
    expected_ppm: float,
    window_ppm: float,
    pattern_hz: float = 0.0,
    prefer_isolated: bool = True,
) -> float:
    """Shift of the reference signal near ``expected_ppm``.

    Among significant peaks within ``±window_ppm``, isolated ones (no strong lines
    beyond the reference's own ``±pattern_hz`` multiplet) are preferred, so a line of a
    nearby compound multiplet is not mistaken for the solvent residual; the closest one
    to the expected shift wins, then snaps to the tallest line of its own pattern (the
    centre of a 1:2:3:2:1 quintet).
    """
    y = np.real(nmr.data)
    d = nmr.direct
    lo = int(np.floor(ppm_to_index(d, expected_ppm + window_ppm)))
    hi = int(np.ceil(ppm_to_index(d, expected_ppm - window_ppm))) + 1
    lo, hi = max(lo, 1), min(hi, y.size - 1)
    if hi - lo < 3:
        raise ValueError("reference window lies outside the spectrum")
    sigma = noise_sigma(y)
    maxima = _local_maxima(y, 10 * sigma)
    # No height filter relative to compound peaks: a dry sample's CHCl3 residual can
    # be < 0.1 % of the tallest line and still be the right reference.
    cand = maxima[(maxima >= lo) & (maxima < hi) & (y[maxima] > 20 * sigma)]
    if cand.size == 0:
        raise ValueError(f"no significant peak within ±{window_ppm} ppm of {expected_ppm}")
    step = d.sw_ppm / d.size
    pts_per_hz = d.size / d.sw_hz
    pool = cand
    if prefer_isolated:
        iso = np.array(
            [i for i in cand if _is_isolated(y, maxima, int(i), pts_per_hz, pattern_hz)], int
        )
        if iso.size:
            pool = iso
    ppms = d.offset_ppm - pool * step
    best = int(pool[np.argmin(np.abs(ppms - expected_ppm))])
    if pattern_hz:
        own = maxima[np.abs(maxima - best) <= (pattern_hz + 0.5) * pts_per_hz]
        best = int(own[np.argmax(y[own])])
    return float(d.offset_ppm - refine_peak(y, best) * step)


#: Solvent and TMS references further apart than this (ppm) raise a warning.
REFERENCE_CONFLICT_PPM = 0.02


def _offset_from_record(params: dict, result: dict) -> tuple[str, dict]:
    return "reference", {**params, "method": "offset", "delta": result["shift_ppm"]}


@processing_step("reference", replay=_offset_from_record)
def reference(
    nmr: NMRData,
    method: Literal["solvent", "tms", "peak", "offset", "stored", "xi"] = "solvent",
    solvent: str | None = None,
    ppm_from: float | None = None,
    ppm_to: float | None = None,
    delta: float = 0.0,
    h_ref_mhz: float | None = None,
    window_ppm: float = SEARCH_WINDOW_PPM,
) -> tuple[NMRData, dict[str, float | str]]:
    """Reference the direct dimension; returns the dataset and the applied shift."""
    d = nmr.direct
    if d.domain != "frequency" or d.offset_ppm is None:
        raise ValueError("referencing needs a frequency axis")
    info: dict[str, float | str] = {}
    if method == "solvent":
        name = canonical_solvent(solvent or solvent_of(nmr) or "")
        shifts = SOLVENT_SHIFTS[name].get(d.nucleus)
        if not shifts:
            raise ValueError(f"no {d.nucleus} reference shift for {name}")
        target = shifts[0]
        is_h = d.nucleus == "1H"
        shift = target - find_reference_peak(
            nmr,
            target,
            window_ppm,
            pattern_hz=RESIDUAL_PATTERN_HZ.get(name, 0.0) if is_h else 0.0,
            prefer_isolated=is_h,
        )
        info["solvent"] = name
        if d.nucleus == "1H":
            try:
                tms_shift = -find_reference_peak(nmr, 0.0, window_ppm)
            except ValueError:
                tms_shift = None
            if tms_shift is not None and abs(tms_shift - shift) > REFERENCE_CONFLICT_PPM:
                info["warning"] = (
                    f"solvent ({name}) and a singlet near 0 ppm (TMS or grease?) disagree "
                    f"by {abs(tms_shift - shift):.3f} ppm; solvent used"
                )
                warnings.warn(info["warning"], stacklevel=3)
    elif method == "tms":
        shift = 0.0 - find_reference_peak(nmr, 0.0, window_ppm, prefer_isolated=d.nucleus == "1H")
    elif method == "peak":
        if ppm_from is None or ppm_to is None:
            raise ValueError("method='peak' needs ppm_from and ppm_to")
        shift = ppm_to - find_reference_peak(nmr, ppm_from, window_ppm, prefer_isolated=False)
    elif method == "offset":
        shift = delta
    elif method == "stored":
        if nmr.source is None or not (nmr.source / "pdata" / "1" / "procs").is_file():
            raise ValueError("no TopSpin pdata/1/procs next to the raw data")
        procs = read_param_file(nmr.source / "pdata" / "1" / "procs")
        bf = d.bf_mhz or d.obs_mhz
        # TopSpin: ppm = (ν - SF)/SF; ours before referencing: (ν - BF1)/BF1
        shift = -(float(procs["SF"]) - bf) / bf * 1e6
    elif method == "xi":
        if h_ref_mhz is None:
            raise ValueError("method='xi' needs h_ref_mhz")
        if d.nucleus not in XI_RATIOS:
            raise ValueError(f"no Ξ ratio for {d.nucleus}")
        ref = h_ref_mhz * XI_RATIOS[d.nucleus] / 100.0
        bf = d.bf_mhz or d.obs_mhz
        # current scale is relative to bf (assumes no earlier referencing step)
        shift = (bf - ref) / ref * 1e6
    else:
        raise ValueError(f"unknown reference method {method!r}")
    new = replace(d, offset_ppm=d.offset_ppm + shift)
    return nmr.copy_with(dims=[*nmr.dims[:-1], new]), {"shift_ppm": float(shift), **info}
