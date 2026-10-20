from __future__ import annotations

import numpy as np
import pytest

from conftest import lorentzian_fid
from nmrsuite.data import Dimension, NMRData
from nmrsuite.io import jcamp
from nmrsuite.io.jcamp import (
    JcampError,
    decode_xydata,
    dif,
    dup,
    encode_difdup,
    format_jcamp,
    parse_jcamp,
    read_jcamp,
    sqz,
    write_jcamp,
)

# --- ASDF tokens ---------------------------------------------------------------


def test_token_encoding() -> None:
    assert [sqz(v) for v in (0, 1, 9, 123, -5, -1234)] == ["@", "A", "I", "A23", "e", "a234"]
    assert [dif(v) for v in (0, 7, -12, 305)] == ["%", "P", "j2", "L05"]
    assert [dup(v) for v in (2, 9, 12, 100)] == ["T", "s", "S2", "S00"]


@pytest.mark.parametrize(
    "lines",
    [
        ["1 0 10 20 30"],  # AFFN
        ["1,0,10,20,30"],  # AFFN, comma separated
        ["1@A0B0C0"],  # SQZ
        ["1@J0J0J0"],  # DIF
        ["1@J0U"],  # DIFDUP: +10 three times
        ["1@J0J0", "3B0J0"],  # DIF with Y-check (B0 = 20 repeats the previous last value)
        ["1@J0J0", "3B0J0", "4C0"],  # ... plus a final check line
        ["1 @ J0 J0 $$ comment", "3 B0 J0"],
    ],
)
def test_decode_forms(lines) -> None:
    assert decode_xydata(lines) == [0, 10, 20, 30]


def test_dup_after_absolute_value_repeats_value() -> None:
    assert decode_xydata(["1A0U"]) == [10, 10, 10]


def test_sqz_e_directly_after_x_is_not_an_exponent() -> None:
    assert decode_xydata(["1234E5e2"]) == [55, -52]


def test_signed_exponent_x_and_question_mark() -> None:
    assert decode_xydata(["1.5E+03 1 ? 3"])[0] == 1
    assert np.isnan(decode_xydata(["1.5E+03 1 ? 3"])[1])


def test_y_check_mismatch_raises() -> None:
    with pytest.raises(JcampError, match="Y-check"):
        decode_xydata(["1@J0J0", "3B5J0"])


@pytest.mark.parametrize("seed", range(5))
def test_difdup_round_trip_and_line_width(seed) -> None:
    rng = np.random.default_rng(seed)
    y = np.concatenate(
        [
            rng.integers(-(10**9), 10**9, 300),
            np.full(40, 7),  # runs exercise DUP
            np.arange(0, 400, 10),
            rng.integers(-5, 5, 200),
        ]
    )
    lines = encode_difdup(y.tolist(), np.arange(y.size, dtype=float))
    assert max(len(line) for line in lines) <= jcamp.LINE_WIDTH
    assert decode_xydata(lines) == y.tolist()


def test_difdup_single_point() -> None:
    assert decode_xydata(encode_difdup([42], [0.0])) == [42]


# --- whole files -------------------------------------------------------------------


def _spectrum(n: int = 2048, complex_: bool = False) -> NMRData:
    x = np.linspace(-1, 1, n)
    y = 1e6 / (1 + (x / 0.01) ** 2) - 2e5 / (1 + ((x - 0.3) / 0.02) ** 2)
    data = y + 1j * np.gradient(y) if complex_ else y
    dim = Dimension(
        "13C", n, 30000.0, 125.77, domain="frequency", is_complex=complex_, offset_ppm=230.0
    )
    return NMRData(data, [dim], params={"acqus": {"SOLVENT": "CDCl3", "PULPROG": "zgpg30"}})


@pytest.mark.parametrize("compression", ["DIFDUP", "AFFN"])
@pytest.mark.parametrize("complex_", [False, True])
def test_spectrum_round_trip(tmp_path, compression, complex_) -> None:
    src = _spectrum(complex_=complex_)
    path = tmp_path / "spec.jdx"
    write_jcamp(src, path, compression=compression)
    back = read_jcamp(path)
    assert back.direct.domain == "frequency"
    assert back.direct.nucleus == "13C"
    assert back.direct.size == src.direct.size
    assert back.direct.sw_hz == pytest.approx(src.direct.sw_hz)
    assert back.direct.offset_ppm == pytest.approx(230.0)
    np.testing.assert_allclose(back.direct.ppm_axis(), src.direct.ppm_axis(), atol=1e-9)
    tol = 2e-9 * np.abs(src.data).max()
    np.testing.assert_allclose(back.data, src.data, atol=tol)
    assert back.experiment == "13C{1H}"
    assert back.params["jcamp"][".SOLVENTNAME"] == "CDCl3"


def test_fid_round_trip_keeps_group_delay(tmp_path) -> None:
    sw = 10000.0
    fid = lorentzian_fid(4096, sw, [(1500.0, 1e7), (-200.0, 3e6)])
    dim = Dimension("1H", 4096, sw, 500.323, bf_mhz=500.32, carrier_ppm=6.175)
    src = NMRData(fid, [dim], group_delay=76.0)
    path = tmp_path / "fid.jdx"
    write_jcamp(src, path)
    text = path.read_text()
    assert "##NTUPLES= NMR FID" in text and "##PAGE= N=2" in text
    back = read_jcamp(path)
    assert back.direct.domain == "time"
    assert back.direct.sw_hz == pytest.approx(sw)
    assert back.direct.carrier_ppm == pytest.approx(6.175)
    assert back.group_delay == 76.0
    np.testing.assert_allclose(back.data, fid, atol=2e-9 * np.abs(fid).max())


def test_ascending_ppm_spectrum_is_stored_in_display_order() -> None:
    text = "\n".join(
        [
            "##TITLE= ascending",
            "##JCAMP-DX= 4.24",
            "##DATA TYPE= NMR SPECTRUM",
            "##.OBSERVE FREQUENCY= 100",
            "##.OBSERVE NUCLEUS= ^1H",
            "##XUNITS= PPM",
            "##YUNITS= ARBITRARY UNITS",
            "##FIRSTX= 0",
            "##LASTX= 3",
            "##NPOINTS= 4",
            "##XFACTOR= 1",
            "##YFACTOR= 0.5",
            "##XYDATA= (X++(Y..Y))",
            "0 2 4 6 8",
            "##END=",
        ]
    )
    nmr = jcamp.jcamp_to_nmrdata(parse_jcamp(text))
    np.testing.assert_allclose(nmr.data, [4, 3, 2, 1])
    np.testing.assert_allclose(nmr.direct.ppm_axis(), [3, 2, 1, 0])


def test_xypoints() -> None:
    text = "\n".join(
        [
            "##TITLE= pts",
            "##JCAMP-DX= 5.01",
            "##DATA TYPE= NMR SPECTRUM",
            "##.OBSERVE FREQUENCY= 400",
            "##.OBSERVE NUCLEUS= ^1H",
            "##XUNITS= HZ",
            "##XFACTOR= 1",
            "##YFACTOR= 1",
            "##XYPOINTS= (XY..XY)",
            "4000, 1; 3600, 2",
            "3200, 3",
            "##END=",
        ]
    )
    nmr = jcamp.jcamp_to_nmrdata(parse_jcamp(text))
    np.testing.assert_allclose(nmr.data, [1, 2, 3])
    np.testing.assert_allclose(nmr.direct.ppm_axis(), [10, 9, 8])


def test_npoints_mismatch_raises() -> None:
    text = format_jcamp(_spectrum(16)).replace("##NPOINTS= 16", "##NPOINTS= 17")
    with pytest.raises(JcampError, match="NPOINTS"):
        jcamp.jcamp_to_nmrdata(parse_jcamp(text))


def test_rejects_2d() -> None:
    nmr = NMRData(np.zeros((2, 2)), [Dimension("1H", 2, 1.0, 1.0)] * 2)
    with pytest.raises(JcampError):
        format_jcamp(nmr)


def test_nmrglue_reads_our_files(tmp_path) -> None:
    """Interoperability: an independent reader decodes our DIFDUP output identically."""
    ng = pytest.importorskip("nmrglue")
    src = _spectrum(1024)
    path = tmp_path / "spec.jdx"
    write_jcamp(src, path)
    _dic, data = ng.jcampdx.read(str(path))
    np.testing.assert_allclose(np.asarray(data), src.data, atol=2e-9 * np.abs(src.data).max())
