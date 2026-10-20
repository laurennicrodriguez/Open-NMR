import numpy as np

from conftest import lorentzian_fid
from nmrsuite.cli import main
from nmrsuite.io.jcamp import read_jcamp


def test_info_and_convert(make_bruker_1d, tmp_path, capsys) -> None:
    exp = make_bruker_1d(lorentzian_fid(256, 10000.0, [(500.0, 1e6)]))
    assert main(["info", str(exp.parent)]) == 0
    out = capsys.readouterr().out
    assert "[1H]" in out and "zg30" in out and "CDCl3" in out

    target = tmp_path / "out.jdx"
    assert main(["convert", str(exp), str(target)]) == 0
    assert read_jcamp(target).data.shape == (256,)
    assert np.iscomplexobj(read_jcamp(target).data)


def test_unicode_output_on_legacy_code_page(monkeypatch) -> None:
    """Redirected stdout on Windows is cp1252, which cannot print δ, ¹H, CDCl₃."""
    import io
    import sys

    from nmrsuite.cli import _ensure_unicode_output

    raw = io.BytesIO()
    fake = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", fake)
    _ensure_unicode_output()
    print("¹H NMR (500 MHz, CDCl₃) δ 7.26")
    sys.stdout.flush()
    assert "CDCl₃".encode() in raw.getvalue()


def test_info_empty_folder(tmp_path) -> None:
    assert main(["info", str(tmp_path)]) == 1


def test_process_single_with_recipe_round_trip(tmp_path, capsys) -> None:
    from realdata import DATASETS

    exp = next(iter(DATASETS.values()))
    spec, recipe = tmp_path / "out.jdx", tmp_path / "recipe.json"
    assert main(["process", str(exp), "-o", str(spec), "--save-recipe", str(recipe)]) == 0
    first = read_jcamp(spec)
    assert first.direct.domain == "frequency" and recipe.is_file()
    # replaying the saved recipe gives the same spectrum
    again = tmp_path / "again.csv"
    assert main(["process", str(exp), "--recipe", str(recipe), "-o", str(again)]) == 0
    table = np.loadtxt(again, delimiter=",", skiprows=1)
    np.testing.assert_allclose(table[:, 2], first.data.real, rtol=1e-6,
                               atol=1e-6 * np.abs(first.data.real).max())  # fmt: skip
    assert "autophase" in capsys.readouterr().out


def test_multiplets_command(tmp_path, capsys) -> None:
    from realdata import DATASETS

    exp = DATASETS["119-61-9"]  # benzophenone
    peaks, integrals = tmp_path / "peaks.csv", tmp_path / "integrals.csv"
    assert main(["multiplets", str(exp), "--peaks-csv", str(peaks),
                 "--integrals-csv", str(integrals)]) == 0  # fmt: skip
    out = capsys.readouterr().out
    first = out.splitlines()[0]
    assert first.startswith("¹H NMR (500 MHz, CDCl₃) δ 7.81 (")
    assert "2H), 7.60 (" in first and "solvent" in out
    assert peaks.read_text().startswith("ppm,") and integrals.read_text().startswith("ppm_from,")
    assert main(["multiplets", str(exp), "--normalize", "7.60=2", "--plain", "--style", "RSC"]) == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("δH (500 MHz, CDCl3) 7.81 (4H,")


def test_process_folder_to_out_dir(tmp_path) -> None:
    from realdata import DATA

    out = tmp_path / "out"
    assert main(["process", str(DATA), "--out-dir", str(out), "--format", "csv"]) == 0
    assert len(list(out.glob("*.csv"))) == 4
    assert len(list(out.glob("*.recipe.json"))) == 4
    assert main(["process", str(DATA), "-o", str(tmp_path / "x.jdx")]) == 2
