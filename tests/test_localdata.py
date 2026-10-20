"""Regression tests on real Bruker data kept outside the repository.

Set ``NMRSUITE_LOCAL_DATA`` to one or more folders (separated by ``os.pathsep``)
containing TopSpin experiment folders; the tests are skipped otherwise.

For 1D data processed by TopSpin with exponential (or no) apodization, the raw FID is
processed here with TopSpin's own parameters and compared against TopSpin's ``1r``.
This checks the reader, the group-delay removal and our FFT/phase conventions:

* spectrum = fftshift(fft(conj(fid))), index 0 at the left edge (OFFSET),
* phase(i) = PHC0 + PHC1 · i/SI degrees, applied as ``spec · exp(+i·phase)``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest

from nmrsuite.io import find_experiments, read_bruker, read_bruker_processed
from nmrsuite.io.jcamp import read_jcamp, write_jcamp
from nmrsuite.processing import remove_group_delay

pytestmark = pytest.mark.localdata

_ROOTS = [Path(p) for p in os.environ.get("NMRSUITE_LOCAL_DATA", "").split(os.pathsep) if p]
EXPERIMENTS = [e for root in _ROOTS for e in find_experiments(root)]

if not EXPERIMENTS:
    pytest.skip("NMRSUITE_LOCAL_DATA not set or holds no Bruker data", allow_module_level=True)


def _ids(paths: list[Path]) -> list[str]:
    return ["/".join(p.parts[-2:]) for p in paths]


def _audit_commands(pdir: Path) -> list[str]:
    audit = pdir / "auditp.txt"
    if not audit.is_file():
        return []
    text = audit.read_text(encoding="latin-1")
    return re.findall(r"^\s*<(\w+)", text, flags=re.MULTILINE)


def _reproducible_1d(exp: Path) -> str | None:
    """Why TopSpin's 1r cannot be reproduced from its stored parameters, or None."""
    pdir = exp / "pdata" / "1"
    if (exp / "ser").exists() or not (pdir / "1r").is_file():
        return "not a processed 1D dataset"
    procs = read_bruker_processed(exp).params["procs"]
    if procs.get("WDW", 1) not in (0, 1):
        return f"window function WDW={procs.get('WDW')} not implemented yet"
    if procs.get("ME_mod", 0) != 0:
        return "linear prediction used"
    if procs.get("PH_mod", 1) != 1 or str(procs.get("REVERSE", "no")) != "no":
        return "phase mode / reverse"
    if procs.get("STSI", 0) not in (0, procs["SI"]):
        return "strip transform"
    tdeff = procs.get("TDeff", 0)
    if tdeff and tdeff != read_bruker(exp).params["acqus"]["TD"]:
        return "TDeff truncation"
    cmds = _audit_commands(pdir)
    if "abs" in cmds:
        last_abs = len(cmds) - 1 - cmds[::-1].index("abs")
        if any(c.startswith("apk") or c == "pk" for c in cmds[last_abs + 1 :]):
            return "phased after baseline correction; stored PHC0/PHC1 do not reproduce 1r"
    return None


def topspin_like_1d(exp: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = read_bruker(exp)
    proc = read_bruker_processed(exp)
    p = proc.params["procs"]
    fid = remove_group_delay(raw).data.copy()
    si = int(p["SI"])
    if p.get("WDW", 1) == 1:
        fid *= np.exp(-np.pi * float(p.get("LB", 0.0)) * np.arange(fid.size) / raw.direct.sw_hz)
    fid[0] *= 0.5
    spec = np.fft.fftshift(np.fft.fft(np.conj(fid), si))
    phase = np.deg2rad(p.get("PHC0", 0.0) + p.get("PHC1", 0.0) * np.arange(si) / si)
    return (spec * np.exp(1j * phase)).real, proc.data.real


@pytest.mark.parametrize("exp", EXPERIMENTS, ids=_ids(EXPERIMENTS))
def test_every_experiment_loads(exp: Path) -> None:
    nmr = read_bruker(exp)
    assert nmr.data.size > 0
    assert np.isfinite(nmr.data).all()
    assert nmr.direct.sw_hz > 0 and nmr.direct.obs_mhz > 0
    assert nmr.group_delay >= 0
    if (exp / "pdata" / "1" / "procs").is_file():
        proc = read_bruker_processed(exp)
        assert proc.data.shape[-1] == proc.params["procs"]["SI"]


ONE_D = [e for e in EXPERIMENTS if not (e / "ser").exists() and (e / "pdata/1/1r").is_file()]


@pytest.mark.parametrize("exp", ONE_D, ids=_ids(ONE_D))
def test_reproduces_topspin_1r(exp: Path) -> None:
    reason = _reproducible_1d(exp)
    if reason:
        pytest.skip(reason)
    ours, topspin = topspin_like_1d(exp)
    corr = np.corrcoef(ours, topspin)[0, 1]
    assert corr > 0.995, f"correlation with TopSpin 1r = {corr:.5f}"
    # same peak in the same place
    assert abs(int(np.argmax(ours)) - int(np.argmax(topspin))) <= 1


@pytest.mark.parametrize("exp", ONE_D[:3], ids=_ids(ONE_D[:3]))
def test_jcamp_round_trip_of_real_fid(exp: Path, tmp_path: Path) -> None:
    raw = read_bruker(exp)
    write_jcamp(raw, tmp_path / "fid.jdx")
    back = read_jcamp(tmp_path / "fid.jdx")
    assert back.group_delay == raw.group_delay
    np.testing.assert_allclose(back.data, raw.data, atol=2e-9 * np.abs(raw.data).max())
