"""Command-line interface: ``nmrsuite info | convert`` (``process``, ``multiplets`` and
``benchmark`` follow in later build steps)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from nmrsuite.io.bruker import BrukerFormatError, find_experiments, read_bruker
from nmrsuite.io.jcamp import write_jcamp


def _cmd_info(args: argparse.Namespace) -> int:
    root = Path(args.path)
    exps = [root] if (root / "acqus").is_file() else find_experiments(root)
    if not exps:
        print(f"no Bruker experiments found under {root}", file=sys.stderr)
        return 1
    for exp in exps:
        try:
            nmr = read_bruker(exp)
        except BrukerFormatError as err:
            print(f"{exp}: {err}", file=sys.stderr)
            continue
        print(f"{exp}  [{nmr.experiment}]")
        for key, value in nmr.summary().items():
            print(f"    {key:<14} {value}")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    nmr = read_bruker(args.source)
    if nmr.ndim != 1:
        print("JCAMP-DX export supports 1D data only", file=sys.stderr)
        return 1
    write_jcamp(nmr, args.output, compression=args.compression)
    print(f"wrote {args.output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nmrsuite", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="show acquisition parameters of Bruker experiments")
    p_info.add_argument("path", help="experiment folder, or a parent folder of many")
    p_info.set_defaults(func=_cmd_info)

    p_conv = sub.add_parser("convert", help="convert a raw Bruker 1D FID to JCAMP-DX")
    p_conv.add_argument("source", help="Bruker experiment folder")
    p_conv.add_argument("output", help="output .jdx file")
    p_conv.add_argument("--compression", choices=["DIFDUP", "AFFN"], default="DIFDUP")
    p_conv.set_defaults(func=_cmd_convert)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
