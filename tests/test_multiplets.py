"""First-order multiplet analysis and journal reports, on synthetic spectra."""

from __future__ import annotations

import pytest

from conftest import multiplet_lines, synth_spectrum
from nmrsuite.analysis import (
    analyse_1d,
    analyse_lines,
    detect_multiplets,
    match_couplings,
    merge_multiplets,
    set_pattern,
)
from nmrsuite.analysis.integrate import integrate_region
from nmrsuite.analysis.multiplet_firstorder import (
    Coupling,
    Multiplet,
    _build,
    _integrate,
    stick_pattern,
)
from nmrsuite.analysis.peaks import Peak
from nmrsuite.report import ACS, RSC, WILEY, custom_style, format_multiplet, proton_report

CASES = [
    ("d", [(7.5, 1)]),
    ("t", [(7.2, 2)]),
    ("q", [(7.1, 3)]),
    ("p", [(6.8, 4)]),
    ("sept", [(6.9, 6)]),
    ("dd", [(10.2, 1), (4.1, 1)]),
    ("dt", [(15.6, 1), (6.8, 2)]),
    ("td", [(7.8, 2), (1.5, 1)]),
    ("ddd", [(12.0, 1), (8.0, 1), (3.0, 1)]),
    ("tt", [(7.5, 2), (2.0, 2)]),
    ("qd", [(7.0, 3), (2.0, 1)]),
    ("dq", [(6.5, 1), (1.6, 3)]),
    ("ddt", [(10.5, 1), (6.2, 1), (1.4, 2)]),
]


def test_stick_pattern_binomial() -> None:
    pos, amp = stick_pattern([Coupling(7.0, 3)])
    assert list(pos) == [0.0, 7.0, 14.0, 21.0]
    assert list(amp * 8) == [1, 3, 3, 1]


@pytest.mark.parametrize(("pattern", "couplings"), CASES)
def test_analyse_lines_exact_sticks(pattern, couplings) -> None:
    pos, amp = stick_pattern([Coupling(j, n) for j, n in couplings])
    # merge coincident sticks the way a spectrum would
    merged: dict[float, float] = {}
    for p, a in zip(pos, amp, strict=True):
        merged[round(p, 6)] = merged.get(round(p, 6), 0.0) + a
    got, cs, _ = analyse_lines(list(merged), list(merged.values()), tol=0.3)
    assert got == pattern
    assert [c.j_hz for c in cs] == pytest.approx([j for j, _ in couplings], abs=1e-6)


@pytest.mark.parametrize(("pattern", "couplings"), CASES)
def test_detect_multiplet_in_spectrum(pattern, couplings) -> None:
    lines = [*multiplet_lines(4.0, couplings, n_h=2.0), (7.5, 3.0)]
    s = synth_spectrum(lines, fwhm_hz=0.5, noise=2e-5)
    mults = detect_multiplets(s)
    m = min(mults, key=lambda m: abs(m.ppm - 4.0))
    assert m.pattern == pattern
    assert m.ppm == pytest.approx(4.0, abs=2e-4)
    assert m.js == pytest.approx([j for j, _ in couplings], abs=0.1)  # spec: ±0.1 Hz
    # the singlet at 7.5 ppm (3 H) and this multiplet (2 H)
    assert sorted(x.n_h for x in mults) == [2, 3]


def test_small_line_on_a_shoulder_does_not_widen_the_range() -> None:
    # LR-7-82-2: the small 7.7556 line measures 16.9 Hz wide on the side of a taller line
    widths = {7.7556: 16.9, 7.7503: 1.8, 7.7371: 6.1, 7.7324: 1.6}
    lines = [Peak(ppm, ppm * 500.0, 0.0, 1.0, w, 100.0) for ppm, w in widths.items()]
    m = _build(lines, 0.5, 1.8, 500.0)
    assert m.ppm_to > 7.7324 - 5.0 / 500.0
    assert m.ppm_from < 7.7556 + 5.0 / 500.0


def test_neighbouring_multiplets_split_at_the_valley() -> None:
    # LR-7-82-2: ranges that overlap must not put the boundary on the side of a peak
    upper = [(7.7556, 0.05), (7.7503, 0.43), (7.7371, 0.245), (7.7324, 1.0)]
    lower = [(7.7094, 0.152), (7.7044, 0.933), (7.7001, 0.253), (7.6908, 0.158), (7.6865, 0.403)]
    s = synth_spectrum([*upper, *lower], fwhm_hz=1.6, noise=1e-5)
    a = Multiplet(7.744, 7.7894, 7.6986, [p * 500.0 for p, _ in upper], [x for _, x in upper])
    b = Multiplet(7.698, 7.7235, 7.6724, [p * 500.0 for p, _ in lower], [x for _, x in lower])
    _integrate(s, [a, b], 1.8)
    full = integrate_region(s, 7.85, 7.60)
    assert a.integral + b.integral == pytest.approx(full, rel=0.05)
    assert a.integral / b.integral == pytest.approx(1.725 / 1.891, rel=0.1)


def test_unresolvable_pattern_is_m() -> None:
    # second-order-like cluster: lines without a first-order explanation
    lines = [(3.0, 1.0), (3.0 + 3.0 / 500, 0.2), (3.0 + 4.1 / 500, 0.9), (3.0 + 9.5 / 500, 0.4)]
    m = detect_multiplets(synth_spectrum(lines, fwhm_hz=0.5, noise=2e-5))[0]
    assert m.pattern == "m" and m.couplings == []
    assert "m" in format_multiplet(m)


def test_close_multiplets_are_split() -> None:
    # two doublets 14 Hz apart (centres): lines within the 20 Hz cluster gap
    a = multiplet_lines(5.0, [(4.0, 1)], 1.0)
    b = multiplet_lines(5.0 + 14.0 / 500, [(6.0, 1)], 1.0)
    mults = detect_multiplets(synth_spectrum(a + b, fwhm_hz=0.5, noise=2e-5))
    assert [m.pattern for m in mults] == ["d", "d"]


def test_ethyl_group_couplings_match() -> None:
    lines = multiplet_lines(4.12, [(7.1, 3)], 2.0) + multiplet_lines(1.26, [(7.1, 2)], 3.0)
    s = synth_spectrum(lines, fwhm_hz=0.6, noise=2e-5, solvent="CDCl3")
    mults = detect_multiplets(s)
    assert [(m.pattern, m.n_h) for m in mults] == [("q", 2), ("t", 3)]
    pairs, unmatched = match_couplings(mults)
    assert len(pairs) == 1 and pairs[0].j_hz == pytest.approx(7.1, abs=0.05)
    assert unmatched == []


def test_unmatched_coupling_is_reported() -> None:
    lines = multiplet_lines(6.0, [(8.0, 1)], 1.0) + multiplet_lines(3.0, [(5.0, 1)], 1.0)
    mults = detect_multiplets(synth_spectrum(lines, fwhm_hz=0.5, noise=2e-5))
    _, unmatched = match_couplings(mults)
    assert sorted(round(j, 1) for _, j in unmatched) == [5.0, 8.0]


def test_manual_edits() -> None:
    lines = multiplet_lines(5.0, [(7.0, 1)], 1.0) + multiplet_lines(4.0, [(7.0, 1)], 1.0)
    mults = detect_multiplets(synth_spectrum(lines, fwhm_hz=0.5, noise=2e-5))
    m = set_pattern(mults[0], "dt", [10.2, 6.5])
    assert m.pattern == "dt" and [c.n for c in m.couplings] == [1, 2] and m.manual
    with pytest.raises(ValueError):
        set_pattern(mults[1], "dd", [7.0])
    set_pattern(mults[1], "m")
    assert mults[1].couplings == []
    merged = merge_multiplets(*detect_multiplets(synth_spectrum(lines, fwhm_hz=0.5)))
    assert merged.n_h == 2 and merged.manual


# --- reports -------------------------------------------------------------------------


@pytest.fixture
def ethyl_spectrum():
    lines = (
        multiplet_lines(7.30, [], 5.0)
        + multiplet_lines(4.30, [(7.1, 3)], 2.0)
        + multiplet_lines(1.35, [(7.1, 2)], 3.0)
    )
    return synth_spectrum(lines, fwhm_hz=0.6, noise=2e-5, solvent="CDCl3")


def test_acs_report(ethyl_spectrum) -> None:
    text = proton_report(ethyl_spectrum, detect_multiplets(ethyl_spectrum))
    assert text == (
        "¹H NMR (500 MHz, CDCl₃) δ 7.30 (s, 5H), 4.30 (q, J = 7.1 Hz, 2H), "
        "1.35 (t, J = 7.1 Hz, 3H)."
    )


def test_rsc_and_wiley_and_plain(ethyl_spectrum) -> None:
    mults = detect_multiplets(ethyl_spectrum)
    assert proton_report(ethyl_spectrum, mults, RSC) == (
        "δH (500 MHz, CDCl₃) 7.30 (5H, s), 4.30 (2H, q, J 7.1), 1.35 (3H, t, J 7.1)."
    )
    assert proton_report(ethyl_spectrum, mults, WILEY).startswith(
        "¹H NMR (500 MHz, CDCl₃): δ = 7.30 (s, 5H)"
    )
    plain = proton_report(ethyl_spectrum, mults, ACS, unicode=False, ppm_decimals=3)
    assert plain.startswith("1H NMR (500 MHz, CDCl3) δ 7.300 (s, 5H), 4.300 (q,")


def test_custom_style(ethyl_spectrum) -> None:
    mults = detect_multiplets(ethyl_spectrum)
    style = custom_style(ACS, sep="; ", end="")
    assert proton_report(ethyl_spectrum, mults, style).endswith("(t, J = 7.1 Hz, 3H)")
    assert "; " in proton_report(ethyl_spectrum, mults, style)
    with pytest.raises(KeyError):
        proton_report(ethyl_spectrum, mults, "Nature")


def test_analyse_1d_end_to_end(ethyl_spectrum) -> None:
    a = analyse_1d(ethyl_spectrum)
    assert a.report.startswith("¹H NMR (500 MHz, CDCl₃) δ 7.30 (s, 5H)")
    assert a.unmatched_j == []
    assert len(a.integrals) >= 3


def test_carbon_report() -> None:
    from nmrsuite.analysis import pick_peaks
    from nmrsuite.report import carbon_report

    lines = [(170.1, 1.0), (77.16 + 0.253, 1.0), (77.16, 1.0), (77.16 - 0.253, 1.0), (60.4, 2.0),
             (14.2, 2.0)]  # fmt: skip
    s = synth_spectrum(lines, bf=125.0, sw=30000.0, carrier_ppm=100.0, nucleus="13C",
                       solvent="CDCl3", fwhm_hz=3.0, noise=1e-4)  # fmt: skip
    text = carbon_report(s, pick_peaks(s))
    assert text == "¹³C NMR (125 MHz, CDCl₃) δ 170.1, 60.4, 14.2."
