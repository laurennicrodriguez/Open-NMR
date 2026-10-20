"""Bruker TopSpin reader: raw FID/SER (1D, 2D) and processed 1r/1i/2rr spectra.

Raw data are returned exactly as acquired (non-destructive); the digital-filter group
delay is reported in ``NMRData.group_delay`` and removed by
:func:`nmrsuite.processing.digital_filter.remove_group_delay`.

Layout of an experiment folder (``<name>/<expno>/``)::

    acqus, acqu2s       acquisition parameters (direct, indirect)
    fid | ser           raw 1D | nD time-domain data
    pdata/<procno>/     procs, proc2s, 1r, 1i, 2rr, ... (TopSpin-processed data)
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from nmrsuite.data import Dimension, NMRData
from nmrsuite.io._paramfile import read_param_file

# Group delay (in complex points) of Bruker digital filters for firmware versions
# DSPFVS 10-13, indexed by decimation factor DECIM. From W. M. Westler and
# F. Abildgaard, "DMX digital filters and non-Bruker offline processing III" (1996),
# as tabulated in nmrglue. Data from DSPFVS >= 20 record GRPDLY directly.
DSP_GROUP_DELAY: dict[int, dict[int, float]] = {
    10: {
        2: 44.75, 3: 33.5, 4: 66.625, 6: 59.083333333333333, 8: 68.5625,
        12: 60.375, 16: 69.53125, 24: 61.020833333333333, 32: 70.015625,
        48: 61.34375, 64: 70.2578125, 96: 61.505208333333333, 128: 70.37890625,
        192: 61.5859375, 256: 70.439453125, 384: 61.626302083333333,
        512: 70.4697265625, 768: 61.646484375, 1024: 70.48486328125,
        1536: 61.656575520833333, 2048: 70.492431640625,
    },
    11: {
        2: 46.0, 3: 36.5, 4: 48.0, 6: 50.166666666666667, 8: 53.25, 12: 69.5,
        16: 72.25, 24: 70.166666666666667, 32: 72.75, 48: 70.5, 64: 73.0,
        96: 70.666666666666667, 128: 72.5, 192: 71.333333333333333, 256: 72.25,
        384: 71.666666666666667, 512: 72.125, 768: 71.833333333333333,
        1024: 72.0625, 1536: 71.916666666666667, 2048: 72.03125,
    },
    12: {
        2: 46.0, 3: 36.5, 4: 48.0, 6: 50.166666666666667, 8: 53.25, 12: 69.5,
        16: 71.625, 24: 70.166666666666667, 32: 72.125, 48: 70.5, 64: 72.375,
        96: 70.666666666666667, 128: 72.5, 192: 71.333333333333333, 256: 72.25,
        384: 71.666666666666667, 512: 72.125, 768: 71.833333333333333,
        1024: 72.0625, 1536: 71.916666666666667, 2048: 72.03125,
    },
    13: {
        2: 2.75, 3: 2.8333333333333333, 4: 2.875, 6: 2.9166666666666667,
        8: 2.9375, 12: 2.9583333333333333, 16: 2.96875, 24: 2.9791666666666667,
        32: 2.984375, 48: 2.9895833333333333, 64: 2.9921875, 96: 2.9947916666666667,
    },
}  # fmt: skip

_AQ_MOD = {0: "qf", 1: "qsim", 2: "qseq", 3: "DQD"}
_FN_MODE = {
    0: None,
    1: "QF",
    2: "QSEQ",
    3: "TPPI",
    4: "States",
    5: "States-TPPI",
    6: "echo-antiecho",
}
_COMPLEX_FN_MODES = {"States", "States-TPPI", "echo-antiecho"}
_ROW_BLOCK_BYTES = 1024  # each SER row is padded to a whole number of 1 KiB blocks


class BrukerFormatError(ValueError):
    """The folder is not a readable Bruker dataset."""


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def group_delay(acqus: dict[str, Any]) -> float:
    """Digital-filter group delay in complex points (0 for analog or unfiltered data)."""
    if acqus.get("DIGMOD", 1) == 0:
        return 0.0
    grpdly = acqus.get("GRPDLY", -1)
    if isinstance(grpdly, int | float) and grpdly > 0:
        return float(grpdly)
    dspfvs = int(acqus.get("DSPFVS", 0))
    decim = acqus.get("DECIM", 1)
    if dspfvs in DSP_GROUP_DELAY:
        table = DSP_GROUP_DELAY[dspfvs]
        key = round(decim)
        if key not in table:
            raise BrukerFormatError(f"no group-delay entry for DSPFVS={dspfvs}, DECIM={decim}")
        return table[key]
    if dspfvs >= 20:
        raise BrukerFormatError(f"DSPFVS={dspfvs} but GRPDLY is missing")
    return 0.0


def _nucleus(params: dict[str, Any]) -> str:
    nuc = str(params.get("NUC1", "")).strip()
    return nuc if nuc and nuc.lower() != "off" else "unknown"


def detect_experiment(acqus: dict[str, Any], acqu2s: dict[str, Any] | None = None) -> str:
    """Classify an experiment from its pulse program, nuclei and dimensionality.

    Returns labels such as ``"1H"``, ``"13C{1H}"``, ``"DEPT-135"``, ``"COSY"``,
    ``"HSQC-edited"``, ``"HMBC"``, ``"NOESY"``, ``"pseudo-2D"`` or ``"unknown"``.
    """
    pp = str(acqus.get("PULPROG", "")).lower()
    nuc = _nucleus(acqus)
    if acqu2s is not None:
        nuc1 = _nucleus(acqu2s)
        rules_2d = [
            ("hsqc", "HSQC"), ("hmqc", "HMQC"), ("hmbc", "HMBC"), ("hoesy", "HOESY"),
            ("noesy", "NOESY"), ("roesy", "ROESY"), ("tocsy", "TOCSY"), ("mlev", "TOCSY"),
            ("dipsi2", "TOCSY"), ("jres", "J-resolved"), ("cosy", "COSY"),
            ("inadeq", "INADEQUATE"),
        ]  # fmt: skip
        for key, label in rules_2d:
            if key in pp:
                if label == "HSQC" and ("ed" in pp.replace("hsqc", "", 1)[:3]):
                    return "HSQC-edited"
                if label == "COSY" and "dqf" in pp:
                    return "DQF-COSY"
                return label
        pseudo = ("t1ir", "t1sr", "cpmg", "t2", "ledbp", "stebp", "dstebp", "diff", "dosy")
        if any(k in pp for k in pseudo) or (nuc1 in ("unknown", nuc) and "zg" in pp):
            return "pseudo-2D"
        return "2D"
    if "dept" in pp:
        for angle in ("135", "90", "45"):
            if angle in pp:
                return f"DEPT-{angle}"
        return "DEPTQ" if "deptq" in pp else "DEPT-135"
    if "jmod" in pp or "apt" in pp:
        return "APT"
    if "selno" in pp:
        return "1D NOESY"
    if "seltocsy" in pp or "selml" in pp:
        return "1D TOCSY"
    if nuc != "1H" and any(k in pp for k in ("pg", "dc", "ig")):
        return f"{nuc}{{1H}}"
    return nuc


# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------


def _raw_dtype(acqus: dict[str, Any], dtype_key: str, order_key: str) -> np.dtype:
    endian = ">" if acqus.get(order_key, 0) == 1 else "<"
    dtyp = acqus.get(dtype_key, 0)
    if dtyp == 0:
        return np.dtype(endian + "i4")
    if dtyp == 1:
        return np.dtype(endian + "f4")
    if dtyp == 2:
        return np.dtype(endian + "f8")
    raise BrukerFormatError(f"unsupported {dtype_key}={dtyp}")


def _direct_dimension(acqus: dict[str, Any], n_points: int, is_complex: bool) -> Dimension:
    bf = float(acqus.get("BF1", acqus.get("SFO1", 0.0)))
    return Dimension(
        nucleus=_nucleus(acqus),
        size=n_points,
        sw_hz=float(acqus["SW_h"]),
        obs_mhz=float(acqus["SFO1"]),
        domain="time",
        is_complex=is_complex,
        bf_mhz=bf,
        carrier_ppm=float(acqus.get("O1", 0.0)) / bf if bf else None,
        acq_mode=_AQ_MOD.get(int(acqus.get("AQ_mod", 3))),
    )


def _to_complex(values: np.ndarray) -> np.ndarray:
    return values[..., 0::2] + 1j * values[..., 1::2]


def find_experiments(root: str | Path) -> list[Path]:
    """All Bruker experiment folders (with ``acqus`` and ``fid``/``ser``) under ``root``."""
    root = Path(root)
    found = {
        p.parent
        for name in ("fid", "ser")
        for p in root.rglob(name)
        if (p.parent / "acqus").is_file()
    }

    def sort_key(p: Path) -> tuple[Any, ...]:
        return tuple((0, int(part)) if part.isdigit() else (1, part) for part in p.parts)

    return sorted(found, key=sort_key)


def read_bruker(path: str | Path) -> NMRData:
    """Read raw 1D (``fid``) or 2D (``ser``) Bruker data from an experiment folder.

    The returned array is complex along the direct dimension (for qsim/DQD data).
    For 2D data the rows are the FIDs in acquisition order, so for States,
    States-TPPI and echo-antiecho the cosine/sine (or echo/antiecho) pairs are
    consecutive rows; they are combined during 2D processing.
    """
    folder = Path(path)
    if folder.is_file():
        folder = folder.parent
    acqus_path = folder / "acqus"
    if not acqus_path.is_file():
        raise BrukerFormatError(f"no acqus file in {folder}")
    acqus = read_param_file(acqus_path)
    params: dict[str, dict[str, Any]] = {"acqus": acqus}

    acqu2s_path = folder / "acqu2s"
    is_2d = (folder / "ser").is_file() and acqu2s_path.is_file()
    if (folder / "acqu3s").is_file():
        raise BrukerFormatError("3D and higher data are not supported yet")

    dtype = _raw_dtype(acqus, "DTYPA", "BYTORDA")
    aq_mod = int(acqus.get("AQ_mod", 3))
    is_complex = aq_mod in (1, 3)
    td = int(acqus["TD"])
    if is_complex and td % 2:
        raise BrukerFormatError(f"odd TD={td} for complex data")
    scale = 2.0 ** int(acqus.get("NC", 0)) if dtype.kind == "i" else 1.0

    if not is_2d:
        fid_path = folder / "fid"
        if not fid_path.is_file():
            raise BrukerFormatError(f"no fid file in {folder}")
        raw = np.fromfile(fid_path, dtype=dtype)
        if raw.size < td:
            warnings.warn(
                f"fid holds {raw.size} values but TD={td}; using what is there", stacklevel=2
            )
            td = raw.size - raw.size % 2 if is_complex else raw.size
        values = raw[:td].astype(np.float64) * scale
        data = _to_complex(values) if is_complex else values
        dims = [_direct_dimension(acqus, data.shape[-1], is_complex)]
        acqu2s = None
    else:
        acqu2s = read_param_file(acqu2s_path)
        params["acqu2s"] = acqu2s
        td1 = int(acqu2s["TD"])
        row_bytes = td * dtype.itemsize
        padded = math.ceil(row_bytes / _ROW_BLOCK_BYTES) * _ROW_BLOCK_BYTES
        file_bytes = (folder / "ser").stat().st_size
        if file_bytes == td1 * row_bytes:
            row_len = td
        elif file_bytes % padded == 0 and file_bytes // padded <= td1:
            row_len = padded // dtype.itemsize
        else:
            raise BrukerFormatError(
                f"ser size {file_bytes} B does not match TD={td} x TD1={td1} ({dtype})"
            )
        raw = np.fromfile(folder / "ser", dtype=dtype)
        n_rows = raw.size // row_len
        if n_rows < td1:
            warnings.warn(
                f"ser holds {n_rows} of TD1={td1} FIDs (acquisition stopped?)", stacklevel=2
            )
        values = raw[: n_rows * row_len].reshape(n_rows, row_len)[:, :td].astype(np.float64)
        values *= scale
        data = _to_complex(values) if is_complex else values
        fn_mode = _FN_MODE.get(int(acqu2s.get("FnMODE", 0)))
        bf1 = float(acqu2s.get("BF1", acqu2s.get("SFO1", 0.0)))
        indirect = Dimension(
            nucleus=_nucleus(acqu2s),
            size=n_rows,
            sw_hz=float(acqu2s["SW_h"]),
            obs_mhz=float(acqu2s["SFO1"]),
            domain="time",
            is_complex=fn_mode in _COMPLEX_FN_MODES,
            bf_mhz=bf1,
            carrier_ppm=float(acqu2s.get("O1", 0.0)) / bf1 if bf1 else None,
            acq_mode=fn_mode,
        )
        dims = [indirect, _direct_dimension(acqus, data.shape[-1], is_complex)]

    return NMRData(
        data=data,
        dims=dims,
        params=params,
        source=folder,
        experiment=detect_experiment(acqus, acqu2s),
        group_delay=group_delay(acqus),
    )


# ---------------------------------------------------------------------------
# Processed data
# ---------------------------------------------------------------------------


def _freq_dimension(procs: dict[str, Any], acq: dict[str, Any], is_complex: bool) -> Dimension:
    axnuc = str(procs.get("AXNUC", "")).strip()
    return Dimension(
        nucleus=axnuc if axnuc and axnuc.lower() != "off" else _nucleus(acq),
        size=int(procs["SI"]),
        sw_hz=float(procs["SW_p"]),
        obs_mhz=float(procs["SF"]),
        domain="frequency",
        is_complex=is_complex,
        bf_mhz=float(acq["BF1"]) if "BF1" in acq else None,
        offset_ppm=float(procs["OFFSET"]),
    )


def _read_processed_file(path: Path, procs: dict[str, Any], count: int) -> np.ndarray:
    dtype = _raw_dtype(procs, "DTYPP", "BYTORDP")
    raw = np.fromfile(path, dtype=dtype)
    if raw.size < count:
        raise BrukerFormatError(f"{path.name} holds {raw.size} values, expected {count}")
    scale = 2.0 ** int(procs.get("NC_proc", 0)) if dtype.kind == "i" else 1.0
    return raw[:count].astype(np.float64) * scale


def untile_2d(flat: np.ndarray, si1: int, si2: int, xdim1: int, xdim2: int) -> np.ndarray:
    """Rebuild a 2D matrix from Bruker's submatrix (tiled) storage."""
    xdim1 = xdim1 or si1
    xdim2 = xdim2 or si2
    if si1 % xdim1 or si2 % xdim2:
        raise BrukerFormatError(f"SI {si1}x{si2} not divisible by XDIM {xdim1}x{xdim2}")
    tiles = flat.reshape(si1 // xdim1, si2 // xdim2, xdim1, xdim2)
    return tiles.transpose(0, 2, 1, 3).reshape(si1, si2)


def tile_2d(matrix: np.ndarray, xdim1: int, xdim2: int) -> np.ndarray:
    """Inverse of :func:`untile_2d` (used to build test datasets)."""
    si1, si2 = matrix.shape
    tiles = matrix.reshape(si1 // xdim1, xdim1, si2 // xdim2, xdim2).transpose(0, 2, 1, 3)
    return tiles.reshape(-1)


def read_bruker_processed(path: str | Path, procno: int = 1) -> NMRData:
    """Read TopSpin-processed data (``1r``/``1i`` or ``2rr``) from ``pdata/<procno>``.

    ``path`` may be the experiment folder or the procno folder itself. 1D data are
    complex (``1r + i·1i``) when ``1i`` exists. Index 0 is the left edge (``OFFSET``).
    """
    folder = Path(path)
    if (folder / "procs").is_file():
        pdir, expdir = folder, folder.parent.parent
    else:
        pdir, expdir = folder / "pdata" / str(procno), folder
    if not (pdir / "procs").is_file():
        raise BrukerFormatError(f"no procs file in {pdir}")
    procs = read_param_file(pdir / "procs")
    params: dict[str, dict[str, Any]] = {"procs": procs}
    acqus: dict[str, Any] = {}
    if (expdir / "acqus").is_file():
        acqus = read_param_file(expdir / "acqus")
        params["acqus"] = acqus
    acqu2s = None

    if (pdir / "2rr").is_file():
        proc2s = read_param_file(pdir / "proc2s")
        params["proc2s"] = proc2s
        if (expdir / "acqu2s").is_file():
            acqu2s = read_param_file(expdir / "acqu2s")
            params["acqu2s"] = acqu2s
        si1, si2 = int(proc2s["SI"]), int(procs["SI"])
        flat = _read_processed_file(pdir / "2rr", procs, si1 * si2)
        data = untile_2d(flat, si1, si2, int(proc2s.get("XDIM", 0)), int(procs.get("XDIM", 0)))
        dims = [
            _freq_dimension(proc2s, acqu2s or {}, is_complex=False),
            _freq_dimension(procs, acqus, is_complex=False),
        ]
    elif (pdir / "1r").is_file():
        si = int(procs["SI"])
        data = _read_processed_file(pdir / "1r", procs, si)
        has_imag = (pdir / "1i").is_file()
        if has_imag:
            data = data + 1j * _read_processed_file(pdir / "1i", procs, si)
        dims = [_freq_dimension(procs, acqus, is_complex=has_imag)]
    else:
        raise BrukerFormatError(f"no 1r or 2rr in {pdir}")

    return NMRData(
        data=data,
        dims=dims,
        params=params,
        source=pdir,
        experiment=detect_experiment(acqus, acqu2s) if acqus else "unknown",
    )
