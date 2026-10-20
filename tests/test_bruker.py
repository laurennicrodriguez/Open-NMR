from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import lorentzian_fid
from nmrsuite.data import Dimension, NMRData
from nmrsuite.io.bruker import (
    DSP_GROUP_DELAY,
    BrukerFormatError,
    detect_experiment,
    find_experiments,
    group_delay,
    read_bruker,
    read_bruker_processed,
    tile_2d,
    untile_2d,
)
from nmrsuite.processing.digital_filter import remove_group_delay, shift_fid

SW = 10000.0


@pytest.fixture
def fid() -> np.ndarray:
    return lorentzian_fid(1024, SW, [(1200.0, 1.0), (-2500.0, 0.5)]) * 1e6


# --- raw 1D ------------------------------------------------------------------


@pytest.mark.parametrize("dtypa", [0, 2])
@pytest.mark.parametrize("bytorda", [0, 1])
def test_read_1d_dtypes_and_byte_order(make_bruker_1d, fid, dtypa, bytorda) -> None:
    folder = make_bruker_1d(fid, DTYPA=dtypa, BYTORDA=bytorda)
    nmr = read_bruker(folder)
    assert nmr.data.shape == (1024,)
    assert np.iscomplexobj(nmr.data)
    tol = 1.0 if dtypa == 0 else 1e-9  # int32 storage rounds to integers
    np.testing.assert_allclose(nmr.data, fid, atol=tol)


def test_int_data_scaled_by_nc(make_bruker_1d, fid) -> None:
    nmr = read_bruker(make_bruker_1d(fid, DTYPA=0, NC=3))
    np.testing.assert_allclose(nmr.data, fid, atol=2.0**3)


def test_padded_fid_is_trimmed_to_td(make_bruker_1d, fid) -> None:
    nmr = read_bruker(make_bruker_1d(fid, DTYPA=0, pad_to=4096))
    assert nmr.data.size == fid.size


def test_dimension_and_metadata(make_bruker_1d, fid) -> None:
    nmr = read_bruker(make_bruker_1d(fid))
    d = nmr.direct
    assert (d.nucleus, d.size, d.sw_hz, d.domain, d.acq_mode) == ("1H", 1024, SW, "time", "DQD")
    assert d.obs_mhz == pytest.approx(500.323089476)
    assert d.carrier_ppm == pytest.approx(3089.476 / 500.32)
    assert nmr.group_delay == 76.0
    assert nmr.experiment == "1H"
    s = nmr.summary()
    assert s["solvent"] == "CDCl3" and s["NS"] == 16 and s["D1_s"] == 1.0 and s["TD"] == 2048


def test_qf_data_are_real(make_bruker_1d) -> None:
    real = np.arange(64.0)
    folder = make_bruker_1d(np.zeros(32, complex))
    # overwrite with real data and AQ_mod=0
    (folder / "acqus").write_text(
        (folder / "acqus").read_text().replace("##$AQ_mod= 3", "##$AQ_mod= 0")
    )
    real.astype("<f8").tofile(folder / "fid")
    nmr = read_bruker(folder)
    assert not np.iscomplexobj(nmr.data)
    np.testing.assert_array_equal(nmr.data, real)


def test_missing_files_raise(tmp_path) -> None:
    with pytest.raises(BrukerFormatError):
        read_bruker(tmp_path)


# --- raw 2D ------------------------------------------------------------------


@pytest.mark.parametrize("pad_rows", [True, False])
@pytest.mark.parametrize("dtypa", [0, 2])
def test_read_2d_ser(make_bruker_2d, pad_rows, dtypa) -> None:
    rng = np.random.default_rng(0)
    # 100 complex points per row: int32 rows are 800 B, padded to 1024 B on disk
    ser = (rng.normal(size=(8, 100)) + 1j * rng.normal(size=(8, 100))) * 1e5
    nmr = read_bruker(make_bruker_2d(ser, pad_rows=pad_rows, DTYPA=dtypa))
    assert nmr.data.shape == (8, 100)
    np.testing.assert_allclose(nmr.data, ser, atol=1.0)
    f1, f2 = nmr.dims
    assert (f1.nucleus, f1.acq_mode, f1.is_complex, f1.size) == ("13C", "echo-antiecho", True, 8)
    assert f2.nucleus == "1H"
    assert nmr.experiment == "HSQC-edited"


def test_2d_partial_acquisition_warns(make_bruker_2d) -> None:
    ser = np.ones((4, 128), complex)
    with pytest.warns(UserWarning, match="acquisition stopped"):
        nmr = read_bruker(make_bruker_2d(ser, acqu2s={"TD": 6}))
    assert nmr.data.shape == (4, 128)


# --- processed ---------------------------------------------------------------


def test_read_1r_1i(make_processed_1d) -> None:
    real = np.linspace(-1000, 1000, 256)
    exp = make_processed_1d(real, -real, DTYPP=0, NC_proc=2)
    nmr = read_bruker_processed(exp)
    np.testing.assert_allclose(nmr.data.real, real, atol=4)
    np.testing.assert_allclose(nmr.data.imag, -real, atol=4)
    ppm = nmr.direct.ppm_axis()
    assert ppm[0] == 16.0
    assert ppm[1] - ppm[0] == pytest.approx(-10000.0 / 500.32 / 256)
    assert read_bruker_processed(exp / "pdata" / "1").data.shape == (256,)


@pytest.mark.parametrize(("xdim1", "xdim2"), [(16, 32), (64, 128), (0, 0)])
def test_read_2rr_untiles(make_processed_2d, xdim1, xdim2) -> None:
    m = np.arange(64 * 128).reshape(64, 128)
    nmr = read_bruker_processed(make_processed_2d(m, xdim1 or 64, xdim2 or 128))
    np.testing.assert_array_equal(nmr.data, m)
    assert nmr.dims[0].nucleus == "13C"
    assert nmr.dims[0].ppm_axis()[0] == 160.0


def test_tile_round_trip() -> None:
    m = np.arange(32 * 48).reshape(32, 48)
    np.testing.assert_array_equal(untile_2d(tile_2d(m, 8, 16), 32, 48, 8, 16), m)


# --- group delay -------------------------------------------------------------


def test_group_delay_sources() -> None:
    assert group_delay({"GRPDLY": 67.98, "DSPFVS": 20}) == 67.98
    assert group_delay({"GRPDLY": -1, "DSPFVS": 12, "DECIM": 24}) == pytest.approx(70.1666667)
    assert group_delay({"DSPFVS": 10, "DECIM": 2048.0}) == pytest.approx(70.492431640625)
    assert group_delay({"DIGMOD": 0, "GRPDLY": 76}) == 0.0
    assert group_delay({"DSPFVS": 0}) == 0.0
    with pytest.raises(BrukerFormatError):
        group_delay({"DSPFVS": 23, "GRPDLY": -1})


def test_dsp_table_matches_nmrglue() -> None:
    bruker_ng = pytest.importorskip("nmrglue.fileio.bruker")
    for fvs, table in DSP_GROUP_DELAY.items():
        for decim, value in table.items():
            assert bruker_ng.bruker_dsp_table[fvs][decim] == pytest.approx(value), (fvs, decim)


def test_remove_group_delay_is_exact_inverse_of_delay() -> None:
    ideal = lorentzian_fid(2048, SW, [(800.0, 1.0), (-3100.0, 0.3)], r2=30.0)
    for g in (76.0, 67.98, 70.492431640625):
        delayed = shift_fid(ideal, -g)
        nmr = NMRData(delayed, [Dimension("1H", 2048, SW, 500.0)], group_delay=g)
        out = remove_group_delay(nmr)
        keep = 2048 - math.ceil(g)
        assert out.data.shape == (keep,) and out.direct.size == keep
        assert out.group_delay == 0.0
        np.testing.assert_allclose(out.data, ideal[:keep], atol=1e-10)
        np.testing.assert_array_equal(nmr.data, delayed)  # input untouched

        zeroed = remove_group_delay(nmr, truncate=False)
        assert zeroed.data.shape == (2048,)
        assert np.all(zeroed.data[keep:] == 0)


def test_remove_group_delay_flattens_spectral_phase() -> None:
    """Delayed peaks show a 2π·g·k/N phase ramp; after removal all are in phase."""
    n = 4096
    bins = (-1600, -400, 800, 1800)  # on-grid frequencies, so each peak's phase is exact
    ideal = lorentzian_fid(n, SW, [(k * SW / n, 1.0) for k in bins], r2=20.0)
    nmr = NMRData(shift_fid(ideal, -76.0), [Dimension("1H", n, SW, 500.0)], group_delay=76.0)
    idx = [k % n for k in bins]
    before = np.angle(np.fft.fft(nmr.data)[idx])
    assert np.ptp(np.unwrap(before)) > 1.0
    after = np.angle(np.fft.fft(remove_group_delay(nmr, truncate=False).data)[idx])
    np.testing.assert_allclose(after, 0.0, atol=0.02)


# --- experiment detection ----------------------------------------------------


@pytest.mark.parametrize(
    ("pulprog", "nuc", "two_d", "expected"),
    [
        ("zg30", "1H", None, "1H"),
        ("zgpg30", "13C", None, "13C{1H}"),
        ("zgig", "13C", None, "13C{1H}"),
        ("zg", "19F", None, "19F"),
        ("dept135", "13C", None, "DEPT-135"),
        ("deptsp90", "13C", None, "DEPT-90"),
        ("deptqgpsp", "13C", None, "DEPTQ"),
        ("jmod", "13C", None, "APT"),
        ("selnogp", "1H", None, "1D NOESY"),
        ("cosygpppqf", "1H", "1H", "COSY"),
        ("cosygpmfqf", "1H", "1H", "COSY"),
        ("dqfcosygpph", "1H", "1H", "DQF-COSY"),
        ("hsqcedetgpsisp2.3", "1H", "13C", "HSQC-edited"),
        ("hsqcetgpsisp2", "1H", "13C", "HSQC"),
        ("hmbcgplpndqf", "1H", "13C", "HMBC"),
        ("noesygpphpp", "1H", "1H", "NOESY"),
        ("roesyph.2", "1H", "1H", "ROESY"),
        ("mlevphpp", "1H", "1H", "TOCSY"),
        ("t1ir", "1H", "1H", "pseudo-2D"),
        ("ledbpgp2s", "1H", "off", "pseudo-2D"),
    ],
)
def test_detect_experiment(pulprog, nuc, two_d, expected) -> None:
    acqu2s = None if two_d is None else {"NUC1": two_d}
    assert detect_experiment({"PULPROG": pulprog, "NUC1": nuc}, acqu2s) == expected


def test_find_experiments(make_bruker_1d, fid) -> None:
    for name in ("10", "2", "1"):
        folder = make_bruker_1d(fid, name=name)
    (folder.parent / "notes").mkdir()
    found = find_experiments(folder.parent)
    assert [p.name for p in found] == ["1", "2", "10"]
