"""JCAMP-DX import and export for 1D NMR spectra and FIDs.

Supports JCAMP-DX 4.24/5.x as used for NMR (McDonald & Wilks 1988; Davies & Lampen
1993):

* ``##XYDATA= (X++(Y..Y))`` in AFFN or ASDF form (SQZ, DIF, DUP, with Y-checks),
* ``##XYPOINTS= (XY..XY)``,
* ``##NTUPLES= NMR SPECTRUM`` / ``NMR FID`` with real and imaginary pages.

Writing produces ``XYDATA`` for real spectra and ``NTUPLES`` for complex data, in
DIFDUP (default) or AFFN form. Frequency axes are written in Hz on the referenced
scale (``ppm × .OBSERVE FREQUENCY``), so ppm = x / observe frequency on reading.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nmrsuite.data import Dimension, NMRData
from nmrsuite.io.bruker import detect_experiment
from nmrsuite.io.bruker import group_delay as _bruker_group_delay

LINE_WIDTH = 80

_SQZ_POS = "@ABCDEFGHI"
_SQZ_NEG = " abcdefghi"
_DIF_POS = "%JKLMNOPQR"
_DIF_NEG = " jklmnopqr"
_DUP = " STUVWXYZs"

_SQZ = {c: i for i, c in enumerate(_SQZ_POS)} | {c: -i for i, c in enumerate(_SQZ_NEG) if i}
_DIF = {c: i for i, c in enumerate(_DIF_POS)} | {c: -i for i, c in enumerate(_DIF_NEG) if i}
_DUPV = {c: i for i, c in enumerate(_DUP) if i}

_ASDF_TOKEN = re.compile(r"([@A-Ia-i%J-Rj-rS-Zs])(\d*)|([+-]?(?:\d+\.?\d*|\.\d+))|\?")
_AFFN_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_AFFN_LINE = re.compile(rf"^(?:[\s,;]*{_AFFN_NUMBER}|[\s,;]*\?)*[\s,;]*$")
_AFFN_TOKEN = re.compile(_AFFN_NUMBER + r"|\?")
# Exponents in X are accepted only when signed ("1.5E+03"): an unsigned "E5" or "e5"
# straight after X is an SQZ ordinate (+5 / -5), not an exponent.
_FIRST_X = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]\d+)?)")


class JcampError(ValueError):
    """Malformed or unsupported JCAMP-DX content."""


def normalize_label(label: str) -> str:
    """JCAMP labels ignore case, spaces, hyphens, slashes and underscores."""
    return re.sub(r"[\s\-/_]", "", label).upper()


# ---------------------------------------------------------------------------
# ASDF decoding
# ---------------------------------------------------------------------------


def _decode_y_tokens(text: str) -> tuple[list[float], bool]:
    """Decode the ordinates on one data line (after the X value).

    Returns the values and whether the last ordinate was written in DIF form
    (which makes the first ordinate of the next line a Y-check value).
    """
    if _AFFN_LINE.match(text):
        vals = [np.nan if t == "?" else float(t) for t in _AFFN_TOKEN.findall(text)]
        return vals, False

    values: list[float] = []
    last_kind = None  # "abs" or "dif"
    last_dif = 0
    for m in _ASDF_TOKEN.finditer(text):
        char, digits, plain = m.group(1), m.group(2), m.group(3)
        if m.group(0) == "?":
            values.append(np.nan)
            last_kind = "abs"
        elif plain is not None:
            values.append(float(plain))
            last_kind = "abs"
        elif char in _SQZ:
            sign = -1 if char.islower() else 1
            values.append(float(sign * int(str(abs(_SQZ[char])) + digits)))
            last_kind = "abs"
        elif char in _DIF:
            if not values:
                raise JcampError(f"DIF value with nothing to difference from: {text!r}")
            sign = -1 if char.islower() else 1
            last_dif = sign * int(str(abs(_DIF[char])) + digits)
            values.append(values[-1] + last_dif)
            last_kind = "dif"
        elif char in _DUPV:
            count = int(str(_DUPV[char]) + digits)
            if not values:
                raise JcampError(f"DUP value with nothing to repeat: {text!r}")
            for _ in range(count - 1):
                values.append(values[-1] + last_dif if last_kind == "dif" else values[-1])
    return values, last_kind == "dif"


def decode_xydata(lines: Iterable[str]) -> list[float]:
    """Decode ``(X++(Y..Y))`` data lines into ordinates (unscaled), applying Y-checks."""
    out: list[float] = []
    prev_ended_dif = False
    for raw in lines:
        line = raw.split("$$", 1)[0].strip()
        if not line:
            continue
        m = _FIRST_X.match(line)
        if not m:
            raise JcampError(f"data line does not start with an X value: {raw!r}")
        values, ended_dif = _decode_y_tokens(line[m.end() :])
        if prev_ended_dif and values:
            check = values.pop(0)
            if out and not np.isclose(check, out[-1]):
                raise JcampError(f"Y-check failed: line starts {check}, previous ended {out[-1]}")
        out.extend(values)
        prev_ended_dif = ended_dif
    return out


def decode_xypoints(lines: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    """Decode ``(XY..XY)`` pairs."""
    nums = [
        float(t) if t != "?" else np.nan
        for raw in lines
        for t in _AFFN_TOKEN.findall(raw.split("$$", 1)[0])
    ]
    if len(nums) % 2:
        raise JcampError("odd number of values in (XY..XY) data")
    arr = np.asarray(nums).reshape(-1, 2)
    return arr[:, 0], arr[:, 1]


# ---------------------------------------------------------------------------
# ASDF encoding
# ---------------------------------------------------------------------------


def _digits_token(value: int, pos: str, neg: str) -> str:
    s = str(abs(value))
    first = int(s[0])
    return (neg[first] if value < 0 else pos[first]) + s[1:]


def sqz(value: int) -> str:
    return _digits_token(value, _SQZ_POS, _SQZ_NEG)


def dif(value: int) -> str:
    return _digits_token(value, _DIF_POS, _DIF_NEG)


def dup(count: int) -> str:
    s = str(count)
    return _DUP[int(s[0])] + s[1:]


def encode_difdup(y: Sequence[int], x: Sequence[float], x_format: str = "{:.10g} ") -> list[str]:
    """Encode integer ordinates as DIFDUP ``(X++(Y..Y))`` lines with Y-checks."""
    y = [int(v) for v in y]
    n = len(y)
    if n == 0:
        return []
    lines: list[str] = []
    start = 0
    while True:
        line = x_format.format(x[start]) + sqz(y[start])
        j = start + 1
        while j < n:
            d = y[j] - y[j - 1]
            run = 1
            while j + run < n and y[j + run] - y[j + run - 1] == d:
                run += 1
            tok = dif(d)
            group = ""
            while run >= 1:
                group = tok + (dup(run) if run > 1 else "")
                if len(line) + len(group) <= LINE_WIDTH:
                    break
                run -= 1
            if run == 0:
                break
            line += group
            j += run
        lines.append(line)
        if j >= n:
            if n > 1 and j - 1 > start:
                lines.append(x_format.format(x[n - 1]) + sqz(y[n - 1]))  # final Y-check
            return lines
        start = j - 1  # next line repeats the last ordinate as its Y-check


def encode_affn(y: Sequence[int], x: Sequence[float], x_format: str = "{:.10g}") -> list[str]:
    """Encode integer ordinates as plain AFFN ``(X++(Y..Y))`` lines."""
    lines: list[str] = []
    i = 0
    n = len(y)
    while i < n:
        line = x_format.format(x[i])
        while i < n and len(line) + len(str(int(y[i]))) + 1 <= LINE_WIDTH:
            line += f" {int(y[i])}"
            i += 1
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Block parsing
# ---------------------------------------------------------------------------

_DATA_LABELS = {"XYDATA", "XYPOINTS", "DATATABLE", "PEAKTABLE"}


def parse_jcamp(text: str) -> dict[str, Any]:
    """Parse JCAMP-DX text into labels and data sections.

    Returns ``{"labels": {norm_label: value}, "raw_labels": {norm_label: original},
    "data": [(norm_label, header_value, lines, page_label_or_None), ...]}``.
    Only the first data-bearing block of a compound file is returned.
    """
    labels: dict[str, str] = {}
    raw_labels: dict[str, str] = {}
    data: list[tuple[str, str, list[str], str | None]] = []
    current_data: list[str] | None = None
    page: str | None = None
    current_label: str | None = None
    depth = 0

    for raw in text.splitlines():
        if raw.startswith("##"):
            label, _, value = raw[2:].partition("=")
            key = normalize_label(label)
            value = value.split("$$", 1)[0].strip() if not key.startswith("$") else value.strip()
            current_data = None
            current_label = key
            if key == "TITLE":
                depth += 1
                if depth > 1 and data:
                    break  # a later block in a compound file; keep the first one only
            if key == "END":
                depth -= 1
                if data:
                    break
                continue
            if key == "PAGE":
                page = value
                continue
            if key in _DATA_LABELS:
                current_data = []
                data.append((key, value, current_data, page))
                continue
            labels.setdefault(key, value)
            raw_labels.setdefault(key, label.strip())
        elif current_data is not None:
            current_data.append(raw)
        elif current_label is not None and raw.strip() and not raw.startswith("$$"):
            labels[current_label] = (labels.get(current_label, "") + "\n" + raw.strip()).strip()

    return {"labels": labels, "raw_labels": raw_labels, "data": data}


def _num(labels: dict[str, str], key: str, default: float | None = None) -> float:
    if key in labels and labels[key] != "":
        return float(labels[key].split()[0].rstrip(","))
    if default is None:
        raise JcampError(f"missing required label ##{key}")
    return default


def _split_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",")]


def _private_params(labels: dict[str, str]) -> dict[str, Any]:
    """Vendor-private ``##$KEY`` labels as typed values (keys upper-cased)."""
    out: dict[str, Any] = {}
    for k, v in labels.items():
        if not k.startswith("$"):
            continue
        v = v.strip()
        if v.startswith("<") and v.endswith(">"):
            out[k[1:]] = v[1:-1]
            continue
        try:
            out[k[1:]] = int(v)
        except ValueError:
            try:
                out[k[1:]] = float(v)
            except ValueError:
                out[k[1:]] = v
    return out


def _page_symbol(header: str) -> str:
    m = re.search(r"\(\s*X\+\+\(\s*(\w)\s*\.\.", header.upper())
    if not m:
        raise JcampError(f"unsupported DATA TABLE form: {header!r}")
    return m.group(1)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_jcamp(path: str | Path) -> NMRData:
    """Read a 1D NMR spectrum or FID from a JCAMP-DX file."""
    path = Path(path)
    return jcamp_to_nmrdata(parse_jcamp(path.read_text(encoding="latin-1")), source=path)


def jcamp_to_nmrdata(parsed: dict[str, Any], source: Path | None = None) -> NMRData:
    labels: dict[str, str] = parsed["labels"]
    sections = parsed["data"]
    if not sections:
        raise JcampError("no spectral data found")

    datatype = (labels.get("DATATYPE", "") + " " + labels.get("NTUPLES", "")).upper()
    is_fid = "FID" in datatype
    obs = _num(labels, ".OBSERVEFREQUENCY", 0.0)
    nucleus = labels.get(".OBSERVENUCLEUS", "").replace("^", "").strip() or "unknown"

    if "NTUPLES" in labels:
        symbols = [s.upper() for s in _split_list(labels["SYMBOL"])]
        first = _split_list(labels.get("FIRST", ""))
        last = _split_list(labels.get("LAST", ""))
        factors = _split_list(labels.get("FACTOR", ""))
        units = _split_list(labels.get("UNITS", ""))
        dims_n = _split_list(labels.get("VARDIM", ""))
        ix = symbols.index("X")

        def var(lst: list[str], i: int, default: float) -> float:
            try:
                return float(lst[i])
            except (IndexError, ValueError):
                return default

        pages: dict[str, np.ndarray] = {}
        for key, header, lines, _page in sections:
            if key != "DATATABLE":
                continue
            sym = _page_symbol(header)
            pages[sym] = np.asarray(decode_xydata(lines)) * var(factors, symbols.index(sym), 1.0)
        if "R" not in pages:
            raise JcampError("NTUPLES without a real (R) page")
        n = int(var(dims_n, ix, len(pages["R"])))
        y: np.ndarray = pages["R"][:n]
        if "I" in pages:
            y = y + 1j * pages["I"][:n]
        x_first, x_last = var(first, ix, 0.0), var(last, ix, float(n - 1))
        x_units = units[ix].upper() if ix < len(units) else ""
    else:
        key, _header, lines, _page = sections[0]
        if key == "XYPOINTS":
            xs, y = decode_xypoints(lines)
            xs = xs * _num(labels, "XFACTOR", 1.0)
            y = y * _num(labels, "YFACTOR", 1.0)
            x_first, x_last, n = float(xs[0]), float(xs[-1]), len(xs)
        elif key == "XYDATA":
            y = np.asarray(decode_xydata(lines)) * _num(labels, "YFACTOR", 1.0)
            n = int(_num(labels, "NPOINTS", len(y)))
            if len(y) != n:
                raise JcampError(f"NPOINTS={n} but {len(y)} ordinates decoded")
            x_first, x_last = _num(labels, "FIRSTX"), _num(labels, "LASTX")
        else:
            raise JcampError(f"unsupported data section ##{key}")
        x_units = labels.get("XUNITS", "").upper()

    private = _private_params(labels)
    bf = private.get("BF1")
    o1 = private.get("O1")

    if is_fid:
        if "SECOND" not in x_units and x_units:
            raise JcampError(f"FID with X units {x_units!r}; expected SECONDS")
        sw = (n - 1) / (x_last - x_first) if n > 1 else float(private.get("SWH", 1.0))
        dim = Dimension(
            nucleus=nucleus,
            size=n,
            sw_hz=float(sw),
            obs_mhz=obs,
            domain="time",
            is_complex=np.iscomplexobj(y),
            bf_mhz=float(bf) if bf else None,
            carrier_ppm=float(o1) / float(bf) if bf and o1 is not None else None,
        )
        gd = _bruker_group_delay(private) if "GRPDLY" in private or "DSPFVS" in private else 0.0
        data = np.asarray(y)
    else:
        if "PPM" in x_units:
            first_ppm, last_ppm = x_first, x_last
        elif "HZ" in x_units or not x_units:
            if not obs:
                raise JcampError("spectrum in Hz needs ##.OBSERVE FREQUENCY")
            first_ppm, last_ppm = x_first / obs, x_last / obs
        else:
            raise JcampError(f"unsupported X units {x_units!r} for a spectrum")
        data = np.asarray(y)
        if first_ppm < last_ppm:
            data = data[::-1]
            first_ppm, last_ppm = last_ppm, first_ppm
        step_ppm = (first_ppm - last_ppm) / (n - 1) if n > 1 else 0.0
        dim = Dimension(
            nucleus=nucleus,
            size=n,
            sw_hz=step_ppm * n * obs,
            obs_mhz=obs,
            domain="frequency",
            is_complex=np.iscomplexobj(data),
            bf_mhz=float(bf) if bf else None,
            offset_ppm=first_ppm,
        )
        gd = 0.0

    pulprog = labels.get(".PULSESEQUENCE", "") or str(private.get("PULPROG", ""))
    vendor = str(private.get("VENDOR", "")).lower()
    if not vendor and "BRUKER" in labels.get("ORIGIN", "").upper():
        vendor = "bruker"
    return NMRData(
        data=data,
        dims=[dim],
        params={"jcamp": dict(labels), "jcamp_private": private},
        source=source,
        experiment=detect_experiment({"PULPROG": pulprog, "NUC1": nucleus}),
        group_delay=gd,
        vendor=vendor,
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _int_scale(values: np.ndarray, digits: float) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    factor = peak / digits if peak > 0 else 1.0
    return np.rint(values / factor).astype(np.int64), factor


def _fmt(v: float) -> str:
    return f"{v:.12g}"


def write_jcamp(
    nmr: NMRData,
    path: str | Path,
    *,
    title: str | None = None,
    compression: str = "DIFDUP",
    digits: float = 1e9,
) -> None:
    """Write a 1D dataset as JCAMP-DX 5.01.

    ``compression`` is ``"DIFDUP"`` (compact, default) or ``"AFFN"`` (plain numbers).
    Ordinates are stored as integers scaled so the largest is about ``digits``.
    """
    Path(path).write_text(
        format_jcamp(nmr, title=title, compression=compression, digits=digits),
        encoding="latin-1",
        newline="\n",
    )


def format_jcamp(
    nmr: NMRData,
    *,
    title: str | None = None,
    compression: str = "DIFDUP",
    digits: float = 1e9,
) -> str:
    if nmr.ndim != 1:
        raise JcampError("JCAMP-DX export supports 1D data only")
    encode = {"DIFDUP": encode_difdup, "AFFN": encode_affn}.get(compression.upper())
    if encode is None:
        raise JcampError(f"unknown compression {compression!r}")
    form = "ASDF" if encode is encode_difdup else "AFFN"
    dim = nmr.direct
    n = dim.size
    is_fid = dim.domain == "time"
    acq = nmr.params.get("acqus", {})

    if is_fid:
        x = dim.time_axis()
        x_units = "SECONDS"
        kind = "NMR FID"
        names = ("TIME", "FID/REAL", "FID/IMAG")
    else:
        x = dim.hz_axis()
        x_units = "HZ"
        kind = "NMR SPECTRUM"
        names = ("FREQUENCY", "SPECTRUM/REAL", "SPECTRUM/IMAG")

    head = [
        f"##TITLE= {title or (nmr.source.name if nmr.source else 'nmrsuite export')}",
        "##JCAMP-DX= 5.01",
        f"##DATA TYPE= {kind}",
        f"##DATA CLASS= {'NTUPLES' if np.iscomplexobj(nmr.data) else 'XYDATA'}",
        "##ORIGIN= nmrsuite",
        f"##OWNER= {acq.get('OWNER') or 'unknown'}",
        f"##LONGDATE= {_dt.datetime.now(_dt.UTC).strftime('%Y/%m/%d %H:%M:%S')}",
        f"##.OBSERVE FREQUENCY= {_fmt(dim.obs_mhz)}",
        f"##.OBSERVE NUCLEUS= ^{dim.nucleus}",
    ]
    if acq.get("SOLVENT"):
        head.append(f"##.SOLVENT NAME= {acq['SOLVENT']}")
    if acq.get("PULPROG"):
        head.append(f"##.PULSE SEQUENCE= {acq['PULPROG']}")
    if nmr.vendor:
        head.append(f"##$VENDOR= <{nmr.vendor}>")
    if dim.bf_mhz:
        head.append(f"##$BF1= {_fmt(dim.bf_mhz)}")
        if dim.carrier_ppm is not None:
            head.append(f"##$O1= {_fmt(dim.carrier_ppm * dim.bf_mhz)}")
    if is_fid:
        head.append(f"##$SW_h= {_fmt(dim.sw_hz)}")
        if nmr.group_delay:
            head.append(f"##$GRPDLY= {_fmt(nmr.group_delay)}")

    body: list[str] = []
    if not np.iscomplexobj(nmr.data):
        yi, f = _int_scale(nmr.data.astype(float), digits)
        body += [
            f"##XUNITS= {x_units}",
            "##YUNITS= ARBITRARY UNITS",
            "##XFACTOR= 1",
            f"##YFACTOR= {_fmt(f)}",
            f"##FIRSTX= {_fmt(x[0])}",
            f"##LASTX= {_fmt(x[-1])}",
            f"##DELTAX= {_fmt((x[-1] - x[0]) / (n - 1) if n > 1 else 0.0)}",
            f"##MAXY= {_fmt(yi.max() * f)}",
            f"##MINY= {_fmt(yi.min() * f)}",
            f"##NPOINTS= {n}",
            f"##FIRSTY= {_fmt(yi[0] * f)}",
            "##XYDATA= (X++(Y..Y))",
            *encode(yi, x),
        ]
    else:
        ri, fr = _int_scale(nmr.data.real, digits)
        ii, fi = _int_scale(nmr.data.imag, digits)
        body += [
            f"##NTUPLES= {kind}",
            f"##VAR_NAME= {names[0]}, {names[1]}, {names[2]}, PAGE NUMBER",
            "##SYMBOL= X, R, I, N",
            "##VAR_TYPE= INDEPENDENT, DEPENDENT, DEPENDENT, PAGE",
            f"##VAR_FORM= AFFN, {form}, {form}, AFFN",
            f"##VAR_DIM= {n}, {n}, {n}, 2",
            f"##UNITS= {x_units}, ARBITRARY UNITS, ARBITRARY UNITS, ",
            f"##FIRST= {_fmt(x[0])}, {_fmt(ri[0] * fr)}, {_fmt(ii[0] * fi)}, 1",
            f"##LAST= {_fmt(x[-1])}, {_fmt(ri[-1] * fr)}, {_fmt(ii[-1] * fi)}, 2",
            f"##MIN= {_fmt(x.min())}, {_fmt(ri.min() * fr)}, {_fmt(ii.min() * fi)}, 1",
            f"##MAX= {_fmt(x.max())}, {_fmt(ri.max() * fr)}, {_fmt(ii.max() * fi)}, 2",
            f"##FACTOR= 1, {_fmt(fr)}, {_fmt(fi)}, 1",
        ]
        for page, sym, vals in ((1, "R", ri), (2, "I", ii)):
            body += [
                f"##PAGE= N={page}",
                f"##NPOINTS= {n}",
                f"##DATA TABLE= (X++({sym}..{sym})), XYDATA",
                *encode(vals, x),
            ]
        body.append(f"##END NTUPLES= {kind}")

    return "\n".join([*head, *body, "##END="]) + "\n"
