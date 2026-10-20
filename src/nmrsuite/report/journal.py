"""Journal-format NMR data reports (ACS default; RSC, Wiley and custom styles).

Example (ACS)::

    ¹H NMR (500 MHz, CDCl₃) δ 7.81 (d, J = 7.6 Hz, 2H), 7.60–7.55 (m, 1H), 3.89 (s, 3H).

A :class:`Style` is a set of format strings, so a group can define its own template.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass, replace

from nmrsuite.analysis.multiplet_firstorder import Multiplet
from nmrsuite.analysis.peaks import PeakList
from nmrsuite.data import NMRData
from nmrsuite.processing.reference import canonical_solvent, solvent_of

_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


@dataclass(frozen=True)
class Style:
    """Format strings. Fields available: {nucleus} {mhz} {solvent} {shift} {range}
    {mult} {j} {nh}. ``j_list`` joins J values; ``j_part`` wraps them."""

    name: str
    header: str
    item: str  # first-order multiplet with couplings
    item_no_j: str  # singlet / broad singlet
    item_range: str  # unresolved multiplet
    j_part: str
    j_sep: str = ", "
    sep: str = ", "
    end: str = "."
    range_dash: str = "–"
    carbon_header: str = "{nucleus} NMR ({mhz} MHz, {solvent}) δ "


ACS = Style(
    name="ACS",
    header="{nucleus} NMR ({mhz} MHz, {solvent}) δ ",
    item="{shift} ({mult}, {j}, {nh}H)",
    item_no_j="{shift} ({mult}, {nh}H)",
    item_range="{range} ({mult}, {nh}H)",
    j_part="J = {js} Hz",
)
RSC = Style(
    name="RSC",
    header="δH ({mhz} MHz, {solvent}) ",
    item="{shift} ({nh}H, {mult}, {j})",
    item_no_j="{shift} ({nh}H, {mult})",
    item_range="{range} ({nh}H, {mult})",
    j_part="J {js}",
    carbon_header="δC ({mhz} MHz, {solvent}) ",
)
WILEY = Style(
    name="Wiley",
    header="{nucleus} NMR ({mhz} MHz, {solvent}): δ = ",
    item="{shift} ({mult}, {j}, {nh}H)",
    item_no_j="{shift} ({mult}, {nh}H)",
    item_range="{range} ({mult}, {nh}H)",
    j_part="J = {js} Hz",
    carbon_header="{nucleus} NMR ({mhz} MHz, {solvent}): δ = ",
)
STYLES = {s.name.lower(): s for s in (ACS, RSC, WILEY)}


def get_style(style: str | Style) -> Style:
    if isinstance(style, Style):
        return style
    try:
        return STYLES[style.lower()]
    except KeyError:
        raise KeyError(f"unknown report style {style!r} ({', '.join(STYLES)})") from None


def nucleus_label(nucleus: str, unicode: bool = True) -> str:
    mass = "".join(ch for ch in nucleus if ch.isdigit())
    sym = nucleus[len(mass) :]
    return (mass.translate(_SUP) if unicode else mass) + sym


def solvent_label(name: str | None, unicode: bool = True) -> str:
    if not name:
        return "?"
    with contextlib.suppress(KeyError):
        name = canonical_solvent(name)
    return name.translate(_SUB) if unicode else name


def _fmt(v: float, decimals: int) -> str:
    return f"{v:.{decimals}f}"


def _header_fields(nmr: NMRData, unicode: bool) -> dict[str, str]:
    d = nmr.direct
    return {
        "nucleus": nucleus_label(d.nucleus, unicode),
        "mhz": str(round(d.obs_mhz)),
        "solvent": solvent_label(solvent_of(nmr), unicode),
    }


def format_multiplet(
    m: Multiplet, style: str | Style = ACS, ppm_decimals: int = 2, j_decimals: int = 1
) -> str:
    st = get_style(style)
    nh = m.n_h if m.n_h is not None else "?"
    if m.pattern == "m" or (not m.couplings and m.pattern not in ("s", "br s")):
        rng = st.range_dash.join(_fmt(v, ppm_decimals) for v in (m.ppm_from, m.ppm_to))
        if _fmt(m.ppm_from, ppm_decimals) == _fmt(m.ppm_to, ppm_decimals):
            return st.item_no_j.format(shift=_fmt(m.ppm, ppm_decimals), mult="m", nh=nh)
        return st.item_range.format(range=rng, mult="m", nh=nh)
    if not m.couplings:
        return st.item_no_j.format(shift=_fmt(m.ppm, ppm_decimals), mult=m.pattern, nh=nh)
    js = st.j_sep.join(_fmt(c.j_hz, j_decimals) for c in m.couplings)
    return st.item.format(
        shift=_fmt(m.ppm, ppm_decimals), mult=m.pattern, j=st.j_part.format(js=js), nh=nh
    )


def proton_report(
    nmr: NMRData,
    multiplets: Sequence[Multiplet],
    style: str | Style = ACS,
    ppm_decimals: int = 2,
    j_decimals: int = 1,
    unicode: bool = True,
) -> str:
    """The ¹H data line for an SI, multiplets in decreasing shift (minor ones omitted)."""
    st = get_style(style)
    items = [
        format_multiplet(m, st, ppm_decimals, j_decimals)
        for m in sorted(multiplets, key=lambda m: -m.ppm)
        if "minor" not in m.flags
    ]
    return st.header.format(**_header_fields(nmr, unicode)) + st.sep.join(items) + st.end


def carbon_report(
    nmr: NMRData,
    peaks: PeakList,
    style: str | Style = ACS,
    decimals: int = 1,
    unicode: bool = True,
) -> str:
    """The ¹³C data line: compound peaks (not solvent, impurities or artefacts)."""
    st = get_style(style)
    shifts = sorted((p.ppm for p in peaks if p.kind == "compound"), reverse=True)
    texts: list[str] = []
    for s in shifts:
        t = _fmt(s, decimals)
        if not texts or texts[-1] != t:
            texts.append(t)
    return st.carbon_header.format(**_header_fields(nmr, unicode)) + st.sep.join(texts) + st.end


def custom_style(base: str | Style = ACS, **changes: str) -> Style:
    """A new style from a built-in one, e.g. ``custom_style(ACS, sep="; ")``."""
    return replace(get_style(base), name=changes.pop("name", "custom"), **changes)
