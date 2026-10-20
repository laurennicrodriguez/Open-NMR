"""Peak picking, classification, integration and deconvolution on synthetic spectra."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import synth_spectrum
from nmrsuite.analysis import (
    auto_proton_counts,
    auto_regions,
    deconvolve,
    estimate_noise,
    fit_region,
    integrate,
    integrate_region,
    pick_peaks,
)
from nmrsuite.analysis.deconvolve import line_area, pseudo_voigt

# --- noise and peak picking ----------------------------------------------------------


def test_noise_estimate_ignores_peaks() -> None:
    rng = np.random.default_rng(3)
    y = rng.normal(0, 2.0, 65536)
    y[1000:1100] += 1e4  # a big "peak" must not inflate the estimate
    y += np.linspace(0, 50, y.size)  # nor a sloping baseline
    assert estimate_noise(y) == pytest.approx(2.0, rel=0.1)


def test_peaks_positions_widths_snr() -> None:
    s = synth_spectrum([(7.0, 1.0), (3.2, 0.5), (1.1, 0.25)], fwhm_hz=1.0, noise=0.05)
    pl = pick_peaks(s, classify=False)
    assert [round(p.ppm, 3) for p in pl] == [7.0, 3.2, 1.1]
    for p in pl:
        assert p.width_hz == pytest.approx(1.0, abs=0.15)
        assert p.snr > 30
    assert pl.peaks[0].intensity > pl.peaks[1].intensity > pl.peaks[2].intensity


def test_min_snr_and_range() -> None:
    s = synth_spectrum([(7.0, 1.0), (3.0, 0.002)], noise=0.02)
    assert len(pick_peaks(s, min_snr=8)) == 1
    assert [round(p.ppm, 2) for p in pick_peaks(s, min_snr=8, ppm_range=(6.0, 8.0))] == [7.0]


def test_negative_peaks_on_request() -> None:
    s = synth_spectrum([(5.0, 1.0), (2.0, -0.7)])
    assert len(pick_peaks(s)) == 1
    neg = pick_peaks(s, negative=True)
    assert sorted(round(p.ppm, 2) for p in neg) == [2.0, 5.0]
    assert min(p.intensity for p in neg) < 0


# --- classification ----------------------------------------------------------------


def test_classification_solvent_water_impurity_satellites() -> None:
    lines = [
        (7.26, 0.3),  # CHCl3 residual
        (1.56, 0.4),  # water in CDCl3: always labelled, whatever its size
        (0.88, 0.03),  # small hexane-like peak
        (1.26, 3.0),  # big peak at an impurity shift: the compound's own
        (3.80, 3.0),  # compound singlet with 13C satellites (1J = 145 Hz)
        (3.80 + 72.5 / 500, 3.0 * 0.0055),
        (3.80 - 72.5 / 500, 3.0 * 0.0055),
    ]
    pl = pick_peaks(synth_spectrum(lines, solvent="CDCl3", noise=1e-4))

    def at(ppm: float) -> tuple[str, str]:
        p = min(pl, key=lambda p: abs(p.ppm - ppm))
        assert abs(p.ppm - ppm) < 0.002, f"no peak at {ppm}"
        return p.kind, p.note

    assert at(7.26)[0] == "solvent"
    assert at(1.56) == ("impurity", "water")
    assert at(0.88)[0] == "impurity" and "hexane" in at(0.88)[1]
    assert at(1.26)[0] == "compound"
    assert at(3.80)[0] == "compound"
    assert at(3.945)[0] == at(3.655)[0] == "c13_satellite"


def test_classification_artifacts() -> None:
    s = synth_spectrum([(6.0, 1.0), (5.0, 1.0), (4.0, 1.0), (3.0, 1.0), (2.0, 1.0)], noise=1e-4)
    y = s.data.copy()
    hz_axis = s.direct.hz_axis()
    y += 0.3 * np.max(y.real) * np.exp(-((hz_axis - 9.0 * 500) ** 2) / (2 * 150.0**2))  # hump
    i = int(np.argmin(np.abs(s.direct.ppm_axis() - 8.0)))
    y[i] += 0.5 * np.max(y.real)  # one-point spike
    pl = pick_peaks(s.copy_with(data=y))
    kinds = {round(p.ppm, 1): p.kind for p in pl}
    assert kinds[9.0] == "artifact" and kinds[8.0] == "artifact"
    assert [kinds[k] for k in (6.0, 5.0, 4.0, 3.0, 2.0)] == ["compound"] * 5


def test_peaklist_csv(tmp_path) -> None:
    pl = pick_peaks(synth_spectrum([(7.0, 1.0)]))
    pl.to_csv(tmp_path / "p.csv")
    text = (tmp_path / "p.csv").read_text()
    assert text.startswith("ppm,hz,index,intensity,width_hz,snr,kind,note")


# --- integration -------------------------------------------------------------------


def test_integrals_ratio_and_normalize() -> None:
    s = synth_spectrum([(7.2, 3.0), (4.1, 2.0), (1.3, 1.0)], fwhm_hz=1.0, noise=1e-4)
    il = integrate(s)
    assert len(il) == 3
    areas = [it.area for it in il]
    np.testing.assert_allclose(np.array(areas) / areas[-1], [3, 2, 1], rtol=0.02)
    il.normalize(2, 1.0)
    assert [round(it.value, 2) for it in il] == pytest.approx([3.0, 2.0, 1.0], abs=0.05)
    assert il.total() == pytest.approx(6.0, abs=0.1)


def test_deconvolution_integration_includes_tails() -> None:
    # broad and narrow line, equal areas: the running sum over a tight region loses
    # more of the broad line's tails; fitted (analytic) areas do not
    s = synth_spectrum([(6.0, 1.0), (3.0, 1.0)], fwhm_hz=0.5, noise=1e-5)
    broad = synth_spectrum([(3.0, 1.0)], fwhm_hz=4.0, noise=1e-5)
    s = s.copy_with(data=s.data - synth_spectrum([(3.0, 1.0)], fwhm_hz=0.5).data + broad.data)
    regions = [(6.03, 5.97), (3.03, 2.97)]
    summed = integrate(s, regions=regions)
    fitted = integrate(s, regions=regions, method="deconvolution")
    assert summed[1].area / summed[0].area < 0.95
    assert fitted[1].area / fitted[0].area == pytest.approx(1.0, rel=0.03)
    assert fitted.meta["method"] == "deconvolution"


def test_slope_bias_correction() -> None:
    s = synth_spectrum([(5.0, 1.0)], fwhm_hz=1.0)
    tilted = s.copy_with(data=s.data + np.linspace(0, 0.05, s.data.size) * s.data.real.max())
    plain = integrate_region(s, 5.2, 4.8)
    assert integrate_region(tilted, 5.2, 4.8) == pytest.approx(plain, rel=0.05)
    assert integrate_region(tilted, 5.2, 4.8, slope_bias=False) > 1.2 * plain


def test_auto_regions_separate_signals() -> None:
    s = synth_spectrum([(7.0, 1.0), (7.01, 1.0), (3.0, 1.0)], noise=1e-4)
    regions = auto_regions(s)
    assert len(regions) == 2  # 7.00/7.01 (5 Hz apart) share a region
    hi, lo = regions[0]
    assert hi > 7.01 and lo < 7.0


@pytest.mark.parametrize(
    ("areas", "total", "expected"),
    [
        ([3.0, 2.0, 1.0], None, [3, 2, 1]),
        ([6.1, 3.9, 2.05, 1.0], None, [6, 4, 2, 1]),
        ([2.0, 2.0, 3.0], None, [2, 2, 3]),  # not [1, 1, 1.5]
        ([5.0, 2.0, 3.0], None, [5, 2, 3]),  # not [2, 1, 1]
        ([1.0, 1.0, 1.0], 9, [3, 3, 3]),
    ],
)
def test_auto_proton_counts(areas, total, expected) -> None:
    _, counts = auto_proton_counts(areas, total=total)
    assert counts == expected


# --- deconvolution ----------------------------------------------------------------


def test_pseudo_voigt_area_formula() -> None:
    x = np.linspace(-500, 500, 400001)
    for eta in (0.0, 0.4, 1.0):
        y = pseudo_voigt(x, 0.0, 2.0, 3.0, eta)
        numeric = np.trapezoid(y, x)
        # Lorentzian tails beyond ±500 Hz hold ~0.4 % of the area for w = 3 Hz
        assert line_area(2.0, 3.0, eta) == pytest.approx(numeric, rel=5e-3)


def test_fit_region_resolves_overlap() -> None:
    # two Lorentzians 1.5 Hz apart (FWHM 1.2 Hz), areas 2:1
    s = synth_spectrum([(5.0, 2.0), (5.0 + 1.5 / 500, 1.0)], fwhm_hz=1.2, noise=1e-4)
    fit = fit_region(s, 5.03, 4.97, seeds_ppm=[5.0, 5.003], shape="lorentz", width_hz=1.0)
    hz = sorted(line.hz for line in fit.lines)
    assert hz[1] - hz[0] == pytest.approx(1.5, abs=0.05)
    areas = sorted((line.area for line in fit.lines), reverse=True)
    assert areas[0] / areas[1] == pytest.approx(2.0, rel=0.03)
    for line in fit.lines:
        assert line.fwhm_hz == pytest.approx(1.2, abs=0.05)
    assert fit.rms_residual < 0.01 * s.data.real.max()


def test_deconvolve_all_regions_matches_integrals() -> None:
    s = synth_spectrum([(7.2, 3.0), (4.1, 2.0), (1.3, 1.0)], fwhm_hz=1.0, noise=1e-4)
    fits = deconvolve(s, shape="pvoigt")
    assert len(fits) == 3
    areas = np.array([f.total_area for f in fits])
    np.testing.assert_allclose(areas / areas[-1], [3, 2, 1], rtol=0.02)
    for f in fits:
        assert all(line.eta < 0.1 for line in f.lines)  # Lorentzian data → η ≈ 0
