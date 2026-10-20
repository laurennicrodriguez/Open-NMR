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


def test_info_empty_folder(tmp_path) -> None:
    assert main(["info", str(tmp_path)]) == 1
