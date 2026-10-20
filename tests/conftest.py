"""Shared fixtures: builders for small synthetic Bruker datasets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nmrsuite.io._paramfile import format_param_text
from nmrsuite.io.bruker import tile_2d

BASE_ACQUS: dict[str, Any] = {
    "AQ_mod": 3,
    "BF1": 500.32,
    "BYTORDA": 0,
    "DECIM": 2000,
    "DIGMOD": 3,
    "DSPFVS": 21,
    "DTYPA": 2,
    "GRPDLY": 76.0,
    "NC": 0,
    "NS": 16,
    "NUC1": "1H",
    "O1": 3089.476,
    "PULPROG": "zg30",
    "SFO1": 500.323089476,
    "SOLVENT": "CDCl3",
    "SW": 19.987084766512,
    "SW_h": 10000.0,
    "TD": 0,
    "TE": 298.15,
    "D": [0.0, 1.0, 0.0],
}


def lorentzian_fid(
    n: int, sw: float, peaks: list[tuple[float, float]], r2: float = 5.0
) -> np.ndarray:
    """Sum of decaying complex exponentials; ``peaks`` = [(offset_hz, amplitude), ...]."""
    t = np.arange(n) / sw
    fid = np.zeros(n, complex)
    for freq, amp in peaks:
        fid += amp * np.exp(2j * np.pi * freq * t - r2 * t)
    return fid


def synth_spectrum(
    lines: list[tuple[float, float]],
    *,
    bf: float = 500.0,
    sw: float = 6000.0,
    carrier_ppm: float = 5.0,
    n: int = 32768,
    fwhm_hz: float = 0.6,
    noise: float = 0.0,
    solvent: str | None = None,
    nucleus: str = "1H",
    seed: int = 0,
):
    """A processed, perfectly phased 1D spectrum with Lorentzian lines [(ppm, area), ...].

    Built in the time domain and transformed with the real pipeline (zero fill, FT).
    """
    from nmrsuite.data import Dimension, NMRData
    from nmrsuite.processing import fft, zerofill

    r2 = np.pi * fwhm_hz
    peaks = [((ppm - carrier_ppm) * bf, amp) for ppm, amp in lines]
    fid = lorentzian_fid(n, sw, peaks, r2=r2)
    if noise:
        rng = np.random.default_rng(seed)
        fid = fid + noise * (rng.normal(size=n) + 1j * rng.normal(size=n))
    dim = Dimension(nucleus, n, sw, bf, bf_mhz=bf, carrier_ppm=carrier_ppm)
    params = {"acqus": {"SOLVENT": solvent}} if solvent else {}
    nmr = NMRData(fid, [dim], params=params, vendor="bruker")
    return fft(zerofill(nmr))


def multiplet_lines(
    centre_ppm: float, couplings: list[tuple[float, int]], n_h: float = 1.0, bf: float = 500.0
) -> list[tuple[float, float]]:
    """First-order stick lines (ppm, area) for a multiplet; total area = n_h."""
    from nmrsuite.analysis.multiplet_firstorder import Coupling, stick_pattern

    pos, amp = stick_pattern([Coupling(j, k) for j, k in couplings])
    pos = pos - (pos[0] + pos[-1]) / 2
    return [(centre_ppm + p / bf, n_h * a) for p, a in zip(pos, amp, strict=True)]


def interleave(z: np.ndarray) -> np.ndarray:
    out = np.empty((*z.shape[:-1], 2 * z.shape[-1]))
    out[..., 0::2] = z.real
    out[..., 1::2] = z.imag
    return out


def _dtype(acqus: dict[str, Any], kind: str = "A") -> np.dtype:
    endian = ">" if acqus.get(f"BYTORD{kind}", 0) == 1 else "<"
    return np.dtype(endian + {0: "i4", 1: "f4", 2: "f8"}[acqus.get(f"DTYP{kind}", 0)])


@pytest.fixture
def make_bruker_1d(tmp_path: Path):
    def build(fid: np.ndarray, name: str = "1", pad_to: int = 0, **acqus_changes: Any) -> Path:
        acqus = {**BASE_ACQUS, **acqus_changes, "TD": 2 * fid.size}
        folder = tmp_path / "data" / name
        folder.mkdir(parents=True)
        (folder / "acqus").write_text(format_param_text(acqus))
        values = interleave(fid)
        dt = _dtype(acqus)
        if dt.kind == "i":
            values = np.rint(values / 2.0 ** acqus.get("NC", 0))
        if pad_to:
            values = np.concatenate([values, np.zeros(pad_to - values.size)])
        values.astype(dt).tofile(folder / "fid")
        return folder

    return build


@pytest.fixture
def make_bruker_2d(tmp_path: Path):
    def build(
        ser: np.ndarray,
        acqu2s: dict[str, Any] | None = None,
        pad_rows: bool = True,
        **acqus_changes: Any,
    ) -> Path:
        """``ser``: complex array (n_fids, td2/2)."""
        acqus = {
            **BASE_ACQUS,
            "PULPROG": "hsqcedetgpsisp2.3",
            **acqus_changes,
            "TD": 2 * ser.shape[1],
        }
        a2 = {
            "NUC1": "13C",
            "TD": ser.shape[0],
            "SW_h": 20000.0,
            "SFO1": 125.8,
            "BF1": 125.77,
            "O1": 9000.0,
            "FnMODE": 6,
            **(acqu2s or {}),
        }
        folder = tmp_path / "data2d" / "2"
        folder.mkdir(parents=True)
        (folder / "acqus").write_text(format_param_text(acqus))
        (folder / "acqu2s").write_text(format_param_text(a2))
        dt = _dtype(acqus)
        rows = interleave(ser)
        if dt.kind == "i":
            rows = np.rint(rows)
        if pad_rows:
            row_len = math.ceil(rows.shape[1] * dt.itemsize / 1024) * 1024 // dt.itemsize
            rows = np.pad(rows, ((0, 0), (0, row_len - rows.shape[1])))
        rows.astype(dt).tofile(folder / "ser")
        return folder

    return build


def _write_processed(values: np.ndarray, path: Path, procs: dict[str, Any]) -> None:
    dt = _dtype(procs, "P")
    if dt.kind == "i":
        values = np.rint(values / 2.0 ** procs.get("NC_proc", 0))
    values.astype(dt).tofile(path)


@pytest.fixture
def make_processed_1d(tmp_path: Path):
    def build(real: np.ndarray, imag: np.ndarray | None = None, **procs_changes: Any) -> Path:
        exp = tmp_path / "proc" / "1"
        pdir = exp / "pdata" / "1"
        pdir.mkdir(parents=True)
        (exp / "acqus").write_text(format_param_text(BASE_ACQUS))
        procs = {
            "SI": real.size,
            "SW_p": 10000.0,
            "SF": 500.32,
            "OFFSET": 16.0,
            "DTYPP": 0,
            "BYTORDP": 0,
            "NC_proc": 0,
            **procs_changes,
        }
        (pdir / "procs").write_text(format_param_text(procs))
        _write_processed(real, pdir / "1r", procs)
        if imag is not None:
            _write_processed(imag, pdir / "1i", procs)
        return exp

    return build


@pytest.fixture
def make_processed_2d(tmp_path: Path):
    def build(matrix: np.ndarray, xdim1: int, xdim2: int) -> Path:
        exp = tmp_path / "proc2d" / "3"
        pdir = exp / "pdata" / "1"
        pdir.mkdir(parents=True)
        (exp / "acqus").write_text(format_param_text(BASE_ACQUS))
        (exp / "acqu2s").write_text(format_param_text({"NUC1": "13C", "TD": 128}))
        si1, si2 = matrix.shape
        (pdir / "procs").write_text(
            format_param_text(
                {
                    "SI": si2,
                    "XDIM": xdim2,
                    "SW_p": 6000.0,
                    "SF": 500.32,
                    "OFFSET": 11.0,
                    "DTYPP": 0,
                    "BYTORDP": 0,
                    "NC_proc": 0,
                }
            )
        )
        (pdir / "proc2s").write_text(
            format_param_text(
                {"SI": si1, "XDIM": xdim1, "SW_p": 20000.0, "SF": 125.8, "OFFSET": 160.0}
            )
        )
        tile_2d(matrix, xdim1, xdim2).astype("<i4").tofile(pdir / "2rr")
        return exp

    return build
