"""Default processing recipes: what runs automatically on import."""

from __future__ import annotations

from typing import Any

from nmrsuite.data import NMRData
from nmrsuite.history import replay
from nmrsuite.processing.reference import SOLVENT_SHIFTS, canonical_solvent, solvent_of

SOLVENT_INVISIBLE = {"DEPT-135", "DEPT-90", "DEPT-45"}


def default_recipe_1d(nmr: NMRData) -> list[dict[str, Any]]:
    """The automatic 1D chain for a raw FID, as recipe records.

    group delay → exponential window (per-nucleus LB) → zero fill ×2 → FT →
    automatic phase → baseline → solvent referencing (when the solvent is known).
    """
    if nmr.ndim != 1 or nmr.direct.domain != "time":
        raise ValueError("the default 1D recipe needs a raw 1D FID")
    steps: list[dict[str, Any]] = []
    if nmr.group_delay:
        steps.append({"step": "remove_group_delay"})
    steps += [
        {"step": "apodize", "params": {"kind": "em"}},
        {"step": "zerofill"},
        {"step": "fft"},
        {"step": "autophase"},
        {"step": "baseline"},
    ]
    solvent = solvent_of(nmr)
    # DEPT-135/90/45 cannot show the deuterated solvent's carbon (no attached H)
    if solvent and nmr.experiment not in SOLVENT_INVISIBLE:
        try:
            name = canonical_solvent(solvent)
        except KeyError:
            name = None
        if name and nmr.direct.nucleus in SOLVENT_SHIFTS[name]:
            steps.append({"step": "reference", "params": {"method": "solvent", "solvent": name}})
    return steps


def process_1d(nmr: NMRData) -> NMRData:
    """Run the default 1D recipe; the result's ``history`` is the full resolved recipe."""
    return replay(nmr, default_recipe_1d(nmr), exact=False)
