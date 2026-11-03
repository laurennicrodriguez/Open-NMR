"""Command-line interface: ``nmrsuite info | convert | process | multiplets``
(``benchmark`` follows in a later build step)."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections.abc import Sequence
from pathlib import Path

from nmrsuite.data import NMRData
from nmrsuite.history import load_recipe, replay, save_recipe
from nmrsuite.io.bruker import BrukerFormatError, find_experiments, read_bruker
from nmrsuite.io.csvio import write_spectrum_csv
from nmrsuite.io.jcamp import write_jcamp
from nmrsuite.processing import default_recipe_1d

_OUTPUT_FORMATS = {".jdx": "jcamp", ".dx": "jcamp", ".csv": "csv"}


def _write_spectrum(nmr: NMRData, path: Path) -> None:
    kind = _OUTPUT_FORMATS.get(path.suffix.lower())
    if kind == "jcamp":
        write_jcamp(nmr, path)
    elif kind == "csv":
        write_spectrum_csv(nmr, path)
    else:
        raise ValueError(f"unknown output format {path.suffix!r} (use .jdx or .csv)")


def _process_one(exp: Path, recipe: list[dict] | None, exact: bool) -> NMRData:
    raw = read_bruker(exp)
    if raw.ndim != 1:
        raise BrukerFormatError("2D processing is not implemented yet")
    return replay(raw, recipe if recipe is not None else default_recipe_1d(raw), exact=exact)


def _cmd_process(args: argparse.Namespace) -> int:
    source = Path(args.source)
    recipe = load_recipe(args.recipe) if args.recipe else None
    single = (source / "acqus").is_file()
    exps = [source] if single else [e for e in find_experiments(source) if not (e / "ser").exists()]
    if not exps:
        print(f"no 1D Bruker experiments found under {source}", file=sys.stderr)
        return 1
    if not single and args.output:
        print("for a folder of experiments use --out-dir, not --output", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    status = 0
    for exp in exps:
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                nmr = _process_one(exp, recipe, exact=not args.rerun)
            except (BrukerFormatError, ValueError, KeyError) as err:
                print(f"{exp}: FAILED: {err}", file=sys.stderr)
                status = 1
                continue
        elapsed = time.perf_counter() - t0
        steps = ", ".join(h["step"] for h in nmr.history)
        print(f"{exp}  [{nmr.experiment}]  {elapsed:.2f} s  ({steps})")
        for w in caught:
            print(f"    warning: {w.message}")
        name = exp.parent.name if exp.name.isdigit() else exp.name
        stem = f"{name}_{exp.name}" if exp.name.isdigit() else name
        target = (
            Path(args.output)
            if args.output
            else (out_dir / f"{stem}.{args.format}" if out_dir else None)
        )
        if target:
            _write_spectrum(nmr, target)
            print(f"    wrote {target}")
        recipe_path = (
            Path(args.save_recipe)
            if args.save_recipe and single
            else (out_dir / f"{stem}.recipe.json" if out_dir else None)
        )
        if recipe_path:
            save_recipe(nmr.history, recipe_path)
            print(f"    wrote {recipe_path}")
    return status


def _parse_normalize(text: str | None) -> tuple[float, int] | None:
    if not text:
        return None
    ppm, _, n = text.partition("=")
    try:
        return float(ppm), int(n)
    except ValueError:
        raise SystemExit(f"--normalize expects PPM=N, e.g. 3.89=3 (got {text!r})") from None


def _cmd_multiplets(args: argparse.Namespace) -> int:
    from nmrsuite.analysis import analyse_1d
    from nmrsuite.analysis.multiplet_firstorder import MINOR_H
    from nmrsuite.report import get_style, proton_report

    source = Path(args.source)
    recipe = load_recipe(args.recipe) if args.recipe else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nmr = _process_one(source, recipe, exact=True)
        a = analyse_1d(
            nmr, style=args.style, total_h=args.total_h, normalize=_parse_normalize(args.normalize)
        )
    report = a.report
    if args.plain and a.multiplets:
        report = proton_report(nmr, a.multiplets, get_style(args.style), unicode=False)
    print(report)
    if a.multiplets:
        print()
        print(f"{'shift':>8}  {'mult':<6} {'J (Hz)':<22} {'nH':>3}  {'integral':>12}  flags")
        for m in a.multiplets:
            js = ", ".join(f"{j:.2f}" for j in m.js)
            print(f"{m.ppm:8.3f}  {m.pattern:<6} {js:<22} {m.n_h or '-':>3}  "
                  f"{m.integral:12.4g}  {' '.join(m.flags)}")  # fmt: skip
        for i, j in a.unmatched_j:
            print(f"    note: J = {j:.1f} Hz at {a.multiplets[i].ppm:.2f} ppm has no partner")
        for m in a.multiplets:
            if "minor" in m.flags:
                print(f"    note: {m.ppm:.2f} ppm integrates below {MINOR_H} H; "
                      "left out of the report")  # fmt: skip
    others = [p for p in a.peaks if p.kind != "compound"]
    if others:
        print()
        for p in others:
            print(f"    {p.ppm:7.3f}  {p.kind:<14} {p.note}")
    if args.peaks_csv:
        a.peaks.to_csv(args.peaks_csv)
    if args.integrals_csv:
        a.integrals.to_csv(args.integrals_csv)
    return 0


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

    p_proc = sub.add_parser(
        "process",
        help="process raw 1D Bruker data (default chain, or a saved recipe)",
        description="Process one experiment, or every 1D experiment under a folder. "
        "Without --recipe the automatic chain runs (group delay, EM window, zero fill, "
        "FT, autophase, baseline, solvent referencing).",
    )
    p_proc.add_argument("source", help="experiment folder, or a parent folder of many")
    p_proc.add_argument("--recipe", help="recipe JSON to apply instead of the default chain")
    p_proc.add_argument("-o", "--output", help="output spectrum (.jdx or .csv); single experiment")
    p_proc.add_argument("--out-dir", help="write spectra and recipes for each experiment here")
    p_proc.add_argument("--format", choices=["jdx", "csv"], default="jdx",
                        help="file type written to --out-dir (default jdx)")  # fmt: skip
    p_proc.add_argument("--save-recipe", help="save the resolved recipe JSON (single experiment)")
    p_proc.add_argument("--rerun", action="store_true",
                        help="re-run automatic steps of a recipe instead of reusing "
                        "their recorded results")  # fmt: skip
    p_proc.set_defaults(func=_cmd_process)

    p_mult = sub.add_parser(
        "multiplets",
        help="peaks, integrals, multiplets and a journal-format report for a 1D spectrum",
    )
    p_mult.add_argument("source", help="Bruker experiment folder (raw data)")
    p_mult.add_argument("--recipe", help="processing recipe JSON (default: automatic chain)")
    p_mult.add_argument("--style", default="ACS", help="ACS (default), RSC or Wiley")
    p_mult.add_argument("--total-h", type=int, help="total protons (from the formula)")
    p_mult.add_argument("--normalize", metavar="PPM=N",
                        help="set the multiplet nearest PPM to N protons")  # fmt: skip
    p_mult.add_argument("--plain", action="store_true", help="no Unicode super/subscripts")
    p_mult.add_argument("--peaks-csv", help="write the peak table")
    p_mult.add_argument("--integrals-csv", help="write the integral table")
    p_mult.set_defaults(func=_cmd_multiplets)

    args = parser.parse_args(argv)
    _ensure_unicode_output()
    return int(args.func(args))


def _ensure_unicode_output() -> None:
    """Reports use δ, ¹H, CDCl₃. A redirected or piped stdout on Windows defaults to a
    legacy code page (cp1252) that cannot encode them; switch it to UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            "δ¹₃–".encode(stream.encoding or "ascii")
        except (UnicodeEncodeError, LookupError):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure:
                reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
