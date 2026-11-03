"""Unit tests for the 1D processing steps, on synthetic FIDs with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import lorentzian_fid
from nmrsuite.data import Dimension, NMRData
from nmrsuite.processing import (
    apodize,
    autophase,
    baseline,
    crop,
    dc_offset,
    fft,
    linear_prediction,
    magnitude,
    phase,
    reference,
    reverse,
    solvent_suppression,
    window,
    zerofill,
)
from nmrsuite.processing.apodize import default_lb
from nmrsuite.processing.phase import apply_phase
from nmrsuite.processing.utils import ppm_to_index

SW = 5000.0
BF = 500.0
CARRIER_PPM = 5.0


def fid_data(peaks_ppm, n=8192, r2=3.0, amps=None, nucleus="1H", **kw) -> NMRData:
    """FID with peaks at given shifts (exp(+iωt) convention, unreferenced scale)."""
    amps = amps or [1.0] * len(peaks_ppm)
    offsets = [(p - CARRIER_PPM) * BF for p in peaks_ppm]
    fid = lorentzian_fid(n, SW, list(zip(offsets, amps, strict=True)), r2=r2)
    dim = Dimension(nucleus, n, SW, BF, bf_mhz=BF, carrier_ppm=CARRIER_PPM)
    return NMRData(fid, [dim], vendor="bruker", **kw)


def spectrum(peaks_ppm, **kw) -> NMRData:
    return fft(zerofill(fid_data(peaks_ppm, **kw)))


def peak_ppm(nmr: NMRData) -> float:
    i = int(np.argmax(nmr.data.real))
    return float(nmr.direct.ppm_axis()[i])


# --- windows -------------------------------------------------------------------------


def test_window_shapes() -> None:
    n, sw = 1000, 1000.0
    t = np.arange(n) / sw
    np.testing.assert_allclose(window("em", n, sw, lb=2.0), np.exp(-np.pi * 2.0 * t))
    s = window("sine", n, sw, ssb=0)
    assert s[0] == 0 and s.max() == pytest.approx(1, abs=1e-5)
    c = window("sine", n, sw, ssb=2)
    assert c[0] == pytest.approx(1) and abs(c[-1]) < 1e-2
    np.testing.assert_allclose(window("qsine", n, sw, ssb=2), c**2)
    g = window("gm", n, sw, lb=-1.0, gb=0.3)
    assert np.argmax(g) == pytest.approx(0.3 * n, abs=2)
    tr = window("trap", n, sw, tm1=0.1, tm2=0.8)
    assert tr[50] == pytest.approx(0.5) and tr[500] == 1 and tr[-1] < 0.01
    traf = window("traf", n, sw, lb=1.0)
    assert np.all(np.isfinite(traf)) and traf[0] == pytest.approx(1.0, abs=0.05)
    assert np.all(window("none", n, sw) == 1)


def test_apodize_records_resolved_lb_and_respects_acquired_points() -> None:
    nmr = zerofill(fid_data([7.0]))
    out = apodize(nmr)
    rec = out.history[-1]
    assert rec["step"] == "apodize" and rec["result"]["lb"] == default_lb("1H") == 0.3
    n = nmr.direct.acquired
    w = window("em", n, SW, lb=0.3)
    np.testing.assert_allclose(out.data[:n], nmr.data[:n] * w)
    assert np.all(out.data[n:] == 0)
    np.testing.assert_array_equal(
        nmr.data[:n], zerofill(fid_data([7.0])).data[:n]
    )  # input untouched


def test_em_broadening_matches_lb() -> None:
    """Exponential window adds lb Hz to the full width at half height."""
    base = fid_data([7.0], n=32768, r2=np.pi * 1.0)  # natural FWHM 1 Hz
    widths = []
    for lb in (0.0, 2.0):
        s = fft(zerofill(apodize(base, kind="em", lb=lb), factor=4)).data.real
        half = s.max() / 2
        widths.append(np.sum(s > half) * SW / s.size)
    assert widths[1] - widths[0] == pytest.approx(2.0, abs=0.1)


# --- zero fill / FFT -----------------------------------------------------------------


def test_zerofill_defaults_to_next_power_of_two() -> None:
    nmr = NMRData(np.ones(3000, complex), [Dimension("1H", 3000, SW, BF)])
    out = zerofill(nmr)
    assert out.data.size == 8192 and out.direct.acquired == 3000
    assert zerofill(nmr, size=4096).data.size == 4096
    with pytest.raises(ValueError):
        zerofill(nmr, size=1024)


def test_fft_axis_puts_peaks_at_their_shifts() -> None:
    s = spectrum([7.26, 1.5, 9.0], amps=[1.0, 0.5, 0.25])
    assert s.direct.domain == "frequency"
    assert s.direct.offset_ppm == pytest.approx(CARRIER_PPM + SW / 2 / BF)
    ppm = s.direct.ppm_axis()
    assert ppm[0] > ppm[-1]
    assert peak_ppm(s) == pytest.approx(7.26, abs=SW / BF / s.data.size)
    # without phase errors the spectrum is absorptive: real part positive at peaks
    assert s.data.real.max() > 10 * abs(s.data.real.min())


def test_first_point_scaling_removes_dc_offset() -> None:
    nmr = zerofill(apodize(fid_data([7.0], r2=30.0), lb=0.0))
    edge = slice(0, 200)  # far from the peak
    half = fft(nmr, first_point=0.5).data.real[edge].mean()
    full = fft(nmr, first_point=1.0).data.real[edge].mean()
    assert abs(half) < 0.05 * abs(full)


# --- phase -----------------------------------------------------------------------------


def test_manual_phase_and_pivot() -> None:
    s = spectrum([7.0])
    rot = phase(s, p0=90.0)
    np.testing.assert_allclose(rot.data, s.data * 1j, atol=1e-9)
    pivot = 7.0
    piv = phase(s, p0=30.0, p1=120.0, pivot_ppm=pivot)
    i = round(ppm_to_index(s.direct, pivot))
    ratio = piv.data[i] / s.data[i]
    assert np.rad2deg(np.angle(ratio)) == pytest.approx(30.0, abs=0.1)


@pytest.mark.parametrize("method", ["peaks", "entropy"])
@pytest.mark.parametrize(("p0", "p1"), [(0.0, 0.0), (75.0, 0.0), (-140.0, 60.0), (30.0, -200.0)])
def test_autophase_recovers_known_phase(method, p0, p1) -> None:
    peaks = [9.5, 8.1, 7.3, 6.0, 4.2, 3.3, 2.1, 1.2, 0.9]
    s = spectrum(peaks, amps=[0.3, 0.8, 1.0, 0.5, 0.7, 0.4, 0.9, 0.6, 0.35])
    rng = np.random.default_rng(1)
    noisy = s.copy_with(
        data=s.data
        + 2e-3
        * np.abs(s.data).max()
        * (rng.normal(size=s.data.size) + 1j * rng.normal(size=s.data.size))
    )
    scrambled = noisy.copy_with(data=apply_phase(noisy.data, -p0, -p1))
    out = autophase(scrambled, method=method)
    res = out.history[-1]["result"]
    # compare the applied phase with the true one across the spectral width
    x = np.linspace(0, 1, 11)
    err = (res["p0"] + res["p1"] * x) - (p0 + p1 * x)
    err = (err + 180) % 360 - 180
    assert np.max(np.abs(err)) < 3.0, res
    np.testing.assert_allclose(out.data.real, noisy.data.real, atol=0.05 * s.data.real.max())


@pytest.mark.parametrize(
    "method",
    [
        "peaks",
        pytest.param(
            "entropy",
            marks=pytest.mark.xfail(
                reason="ACME without its negativity penalty is poorly conditioned; "
                "known limitation for DEPT/APT (see README)",
                strict=False,
            ),
        ),
    ],
)
def test_autophase_handles_negative_peaks(method) -> None:
    raw = fid_data([7.3, 5.0, 3.0, 1.0, 8.6], amps=[1.0, -0.8, 0.6, -0.4, 0.5])
    s = fft(zerofill(apodize(raw, kind="em", lb=1.0)))  # apodized, like real data
    s.experiment = "DEPT-135"
    scrambled = s.copy_with(data=apply_phase(s.data, -50.0, 40.0))
    out = autophase(scrambled, method=method)
    assert out.history[-1]["params"]["allow_negative"] is None
    np.testing.assert_allclose(out.data.real, s.data.real, atol=0.03 * s.data.real.max())


def test_autophase_replays_exactly_from_result() -> None:
    from nmrsuite.history import replay

    raw = fid_data([7.3, 2.0])
    scrambled = raw.copy_with(data=raw.data * np.exp(1j * 1.0))
    done = autophase(fft(zerofill(scrambled)))
    again = replay(scrambled, done.history)
    assert again.history[-1]["step"] == "phase"
    np.testing.assert_allclose(again.data, done.data)


# --- baseline ----------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["poly", "airpls", "whittaker"])
def test_baseline_removes_smooth_offset(method) -> None:
    s = phase(spectrum([8.0, 6.5, 3.0, 1.0]), p0=0.0)
    n = s.data.size
    x = np.linspace(-1, 1, n)
    roll = 0.02 * s.data.real.max() * (1 + 0.5 * x - 0.8 * x**2 + 0.3 * x**3)
    out = baseline(s.copy_with(data=s.data + roll), method=method)
    resid = out.data.real - s.data.real
    away = np.abs(s.data.real) < 1e-3 * s.data.real.max()  # signal-free points
    assert np.abs(resid[away]).max() < 0.15 * np.abs(roll).max()
    np.testing.assert_allclose(out.data.imag, s.data.imag)  # only the real part changes


def test_baseline_poly_keeps_negative_peaks() -> None:
    s = spectrum([7.0, 4.0, 2.0], amps=[1.0, -0.7, 0.5])
    out = baseline(s.copy_with(data=s.data + 0.01 * s.data.real.max()), method="poly")
    assert out.data.real.min() == pytest.approx(s.data.real.min(), rel=0.02)


def test_baseline_points_and_exclusion() -> None:
    s = spectrum([7.0])
    offset = 0.01 * s.data.real.max()
    lifted = s.copy_with(data=s.data + offset)
    out = baseline(lifted, method="points", points_ppm=[9.8, 9.0, 4.0, 0.2], order=1)
    away = np.abs(s.data.real) < 1e-3 * s.data.real.max()
    assert np.abs((out.data.real - s.data.real)[away]).max() < 0.05 * offset
    with pytest.raises(ValueError, match="within the spectrum"):
        baseline(lifted, method="points", points_ppm=[14.0, 4.0])
    ex = baseline(lifted, method="poly", exclude_ppm=[[7.5, 6.5]])
    assert ex.history[-1]["params"]["exclude_ppm"] == [[7.5, 6.5]]


# --- referencing -----------------------------------------------------------------------------


def test_reference_solvent_tms_peak_offset() -> None:
    # CDCl3 residual appears at 7.30 on the unreferenced scale, TMS at 0.04
    s = spectrum([7.30, 3.0, 0.04], amps=[0.3, 1.0, 0.5])
    s.params["acqus"] = {"SOLVENT": "CDCl3"}
    out = reference(s)
    assert out.history[-1]["result"]["shift_ppm"] == pytest.approx(-0.04, abs=2e-3)
    assert peak_ppm(out) == pytest.approx(2.96, abs=2e-3)
    tms = reference(s, method="tms")
    assert tms.history[-1]["result"]["shift_ppm"] == pytest.approx(-0.04, abs=2e-3)
    pk = reference(s, method="peak", ppm_from=3.0, ppm_to=3.5)
    assert peak_ppm(pk) == pytest.approx(3.5, abs=2e-3)
    off = reference(s, method="offset", delta=0.1)
    assert off.direct.offset_ppm == pytest.approx(s.direct.offset_ppm + 0.1)


def test_reference_prefers_singlet_over_nearby_multiplet() -> None:
    # a strong doublet (J = 8 Hz) at 7.24/7.256 and a weaker CHCl3 singlet at 7.33
    J = 8.0 / BF
    s = spectrum([7.248 - J / 2, 7.248 + J / 2, 7.33], amps=[1.0, 1.0, 0.3])
    s.params["acqus"] = {"SOLVENT": "CDCl3"}
    out = reference(s)
    assert out.history[-1]["result"]["shift_ppm"] == pytest.approx(7.26 - 7.33, abs=2e-3)


def test_reference_dmso_quintet_centre() -> None:
    # DMSO-d5 residual: 1:2:3:2:1 quintet, J(H,D) = 1.9 Hz, centred at 2.53 here
    j = 1.9 / BF
    lines = [2.53 + k * j for k in (-2, -1, 0, 1, 2)]
    s = fft(zerofill(apodize(
        fid_data([*lines, 2.40, 3.33], amps=[0.1, 0.2, 0.3, 0.2, 0.1, 1.0, 0.8], n=32768),
        kind="em", lb=0.2,
    )))  # fmt: skip
    s.params["acqus"] = {"SOLVENT": "DMSO"}
    out = reference(s)
    assert out.history[-1]["result"]["solvent"] == "DMSO-d6"
    assert out.history[-1]["result"]["shift_ppm"] == pytest.approx(2.50 - 2.53, abs=1e-3)


def test_reference_warns_on_solvent_tms_conflict() -> None:
    s = spectrum([7.26, 0.08], amps=[0.4, 0.2])
    s.params["acqus"] = {"SOLVENT": "CDCl3"}
    with pytest.warns(UserWarning, match="disagree"):
        out = reference(s)
    assert "warning" in out.history[-1]["result"]


def test_reference_xi_for_carbon() -> None:
    h_ref = 500.13
    c_bf = 125.76
    s = spectrum([77.0], nucleus="13C")
    s.direct.bf_mhz = c_bf
    out = reference(s, method="xi", h_ref_mhz=h_ref)
    expected = (c_bf - h_ref * 0.25145020) / (h_ref * 0.25145020) * 1e6
    assert out.history[-1]["result"]["shift_ppm"] == pytest.approx(expected)


def test_reference_13c_cd3cn_septet_far_off() -> None:
    """Real case (LR-7-82-*): ¹³C in CD₃CN unreferenced by ~0.9 ppm. The CD₃ septet
    (1:6:15:20:15:6:1, J(C,D) = 20.8 Hz) is found and its centre put at 1.32."""
    from conftest import synth_spectrum
    from nmrsuite.analysis.multiplet_firstorder import Coupling, stick_pattern

    bf, off = 125.8, -0.95  # solvent appears 0.95 ppm too low
    pos, amp = stick_pattern([Coupling(20.8, 6)])
    pos -= pos.mean()
    sept = [(1.32 + off + p / bf, 3.0 * a) for p, a in zip(pos, amp, strict=True)]
    lines = [*sept, (118.26 + off, 0.5), (148.1 + off, 1.0), (127.3 + off, 1.0)]
    s = synth_spectrum(lines, bf=bf, sw=30000.0, carrier_ppm=100.0, nucleus="13C",
                       solvent="CD3CN", fwhm_hz=2.0, n=65536)  # fmt: skip
    out = reference(s)
    assert out.history[-1]["result"]["shift_ppm"] == pytest.approx(-off, abs=0.005)


def test_reference_skips_c13_satellites_and_humps() -> None:
    """Real case (2-ethylpyridine): a satellite of a compound doublet lies nearer 7.26
    than the weak CHCl3 residual, which is the line to reference to."""
    j = 7.85 / BF
    half = 79.0 / BF  # ¹J(C,H) = 158 Hz
    doublet = [(7.11 - j / 2, 1.0), (7.11 + j / 2, 1.0)]
    sats = [(p + s, 0.0055) for p, _ in doublet for s in (half, -half)]
    s = spectrum(
        [p for p, _ in doublet + sats] + [7.35], amps=[a for _, a in doublet + sats] + [0.008]
    )
    s.params["acqus"] = {"SOLVENT": "CDCl3"}
    out = reference(s)
    assert out.history[-1]["result"]["shift_ppm"] == pytest.approx(7.26 - 7.35, abs=2e-3)


def test_reference_failure_is_optional() -> None:
    s = spectrum([3.0])  # no CHCl3 residual anywhere near 7.26
    s.params["acqus"] = {"SOLVENT": "CDCl3"}
    with pytest.raises(ValueError):
        reference(s)
    with pytest.warns(UserWarning, match="not referenced"):
        out = reference(s, required=False)
    assert out.history[-1]["result"]["shift_ppm"] == 0.0
    assert out.direct.offset_ppm == s.direct.offset_ppm


def test_unknown_solvent_raises() -> None:
    s = spectrum([7.0])
    with pytest.raises(KeyError):
        reference(s, solvent="unobtainium")


# --- time domain -----------------------------------------------------------------------------


def test_dc_offset() -> None:
    nmr = fid_data([7.0], r2=50.0)
    out = dc_offset(nmr.copy_with(data=nmr.data + (3 + 2j)))
    assert abs(out.data[-500:].mean()) < 1e-3


def test_backward_lp_repairs_first_points() -> None:
    clean = fid_data([7.3, 6.1, 2.2], n=4096, r2=10.0)
    broken = clean.data.copy()
    broken[:3] = 0
    out = linear_prediction(clean.copy_with(data=broken), mode="backward", points=3, order=8)
    np.testing.assert_allclose(out.data[:3], clean.data[:3], rtol=1e-6, atol=1e-8)


def test_forward_lp_extends_truncated_fid() -> None:
    full = fid_data([7.3, 6.1, 2.2], n=2048, r2=10.0)
    head = full.copy_with(data=full.data[:1024].copy(), dims=[Dimension("1H", 1024, SW, BF)])
    out = linear_prediction(head, mode="forward", points=1024, order=8)
    assert out.data.size == 2048 and out.direct.size == 2048
    np.testing.assert_allclose(out.data, full.data, atol=1e-6)


def test_solvent_suppression_on_carrier() -> None:
    nmr = fid_data([CARRIER_PPM, 8.0], amps=[100.0, 1.0], n=16384)
    out = solvent_suppression(nmr, width=32)
    before = fft(zerofill(nmr)).data
    after = fft(zerofill(out)).data
    i_solv = round(ppm_to_index(fft(zerofill(nmr)).direct, CARRIER_PPM))
    i_peak = round(ppm_to_index(fft(zerofill(nmr)).direct, 8.0))
    assert abs(after[i_solv]) < 0.01 * abs(before[i_solv])
    assert abs(after[i_peak]) == pytest.approx(abs(before[i_peak]), rel=0.05)


# --- utilities -------------------------------------------------------------------------------


def test_crop_magnitude_reverse() -> None:
    s = spectrum([7.0, 2.0])
    c = crop(s, ppm_from=8.0, ppm_to=1.0)
    ppm = c.direct.ppm_axis()
    assert ppm[0] <= 8.0 and ppm[-1] >= 1.0 and ppm[0] > 7.99 and ppm[-1] < 1.01
    np.testing.assert_allclose(
        c.direct.ppm_axis(),
        s.direct.ppm_axis()[(s.direct.ppm_axis() <= 8.0) & (s.direct.ppm_axis() >= 1.0)],
    )
    assert np.all(magnitude(s).data >= 0) and not np.iscomplexobj(magnitude(s).data)
    np.testing.assert_array_equal(reverse(s).data, s.data[::-1])


def test_steps_check_domain() -> None:
    f = fid_data([7.0])
    with pytest.raises(ValueError):
        phase(f, p0=10)
    with pytest.raises(ValueError):
        apodize(fft(f))
