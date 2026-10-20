"""Command-line interface: ``nmrsuite info | convert | process`` (``multiplets`` and
``benchmark`` follow in later build steps)."""

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

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
