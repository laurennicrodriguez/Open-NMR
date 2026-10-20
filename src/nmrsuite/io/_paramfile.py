"""Parser and writer for Bruker parameter files (acqus, acqu2s, procs, proc2s, ...).

These files use JCAMP-DX labelled-data syntax with Bruker private labels::

    ##$TD= 65536
    ##$NUC1= <1H>
    ##$D= (0..63)
    0 1 0 0 ...

Values are returned as ``int``, ``float``, ``str`` (angle brackets removed) or a
``list`` of those for ``(0..n)`` arrays. Keys keep Bruker's case (``SW_h``,
``AQ_mod``) with the leading ``##`` and ``$`` removed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_LABEL = re.compile(r"^##\$?([^=]+)=\s?(.*)$")
_ARRAY = re.compile(r"^\(\s*0\s*\.\.\s*(\d+)\s*\)$")
_TOKEN = re.compile(r"<[^>]*>|\S+")


def _strip_comment(line: str) -> str:
    """Remove a trailing ``$$`` comment that is not inside ``<...>``."""
    depth = 0
    for i, ch in enumerate(line):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif ch == "$" and depth == 0 and line.startswith("$$", i):
            return line[:i].rstrip()
    return line.rstrip()


def _convert(token: str) -> Any:
    if token.startswith("<") and token.endswith(">"):
        return token[1:-1]
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def parse_param_text(text: str) -> dict[str, Any]:
    """Parse the text of a Bruker parameter file into a dict."""
    entries: list[tuple[str, list[str]]] = []
    for raw in text.splitlines():
        if raw.startswith("$$"):
            continue
        line = _strip_comment(raw)
        m = _LABEL.match(line)
        if m:
            entries.append((m.group(1).strip(), [m.group(2)]))
        elif entries and line:
            entries[-1][1].append(line)

    out: dict[str, Any] = {}
    for key, chunks in entries:
        if key == "END":
            break
        first = chunks[0].strip()
        rest = " ".join(chunks[1:])
        arr = _ARRAY.match(first)
        if arr:
            values = [_convert(t) for t in _TOKEN.findall(rest)]
            out[key] = values
        elif first.startswith("<") and not first.endswith(">"):
            # string continued on the following lines
            out[key] = (first + "\n" + "\n".join(chunks[1:])).strip()[1:].rstrip(">")
        elif rest and first.startswith("("):
            # other structured forms, e.g. "(0..3)" variants or XY tables: keep raw text
            out[key] = first + "\n" + "\n".join(chunks[1:])
        else:
            out[key] = _convert(first) if first else ""
    return out


def read_param_file(path: str | Path) -> dict[str, Any]:
    return parse_param_text(Path(path).read_text(encoding="latin-1"))


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return f"<{value}>"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def format_param_text(params: dict[str, Any], title: str = "Parameter file") -> str:
    """Write a dict back in Bruker parameter-file syntax (used to build test datasets)."""
    lines = [f"##TITLE= {title}", "##JCAMPDX= 5.0", "##DATATYPE= Parameter Values"]
    for key, value in params.items():
        if isinstance(value, list):
            lines.append(f"##${key}= (0..{len(value) - 1})")
            lines.append(" ".join(_format_value(v) for v in value))
        else:
            lines.append(f"##${key}= {_format_value(value)}")
    lines.append("##END=")
    return "\n".join(lines) + "\n"
