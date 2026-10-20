import json

import numpy as np
import pytest

from conftest import lorentzian_fid
from nmrsuite.data import Dimension, NMRData
from nmrsuite.history import (
    load_recipe,
    processing_step,
    recipe_from_json,
    recipe_to_json,
    registered_steps,
    replay,
    save_recipe,
)
from nmrsuite.processing import apodize, default_recipe_1d, fft, process_1d, zerofill


def raw_fid() -> NMRData:
    # CHCl3-like singlet at 7.30 ppm (carrier 4.7 ppm + 1300 Hz at 500 MHz) and two others
    fid = lorentzian_fid(4096, 5000.0, [(1300.0, 1.0), (-1250.0, 0.6), (300.0, 0.4)])
    fid = fid * np.exp(1j * np.deg2rad(40.0))
    dim = Dimension("1H", 4096, 5000.0, 500.0, bf_mhz=500.0, carrier_ppm=4.7)
    return NMRData(
        fid, [dim], params={"acqus": {"SOLVENT": "CDCl3"}}, vendor="bruker", group_delay=0.0
    )


def test_steps_record_all_params_including_defaults() -> None:
    out = fft(zerofill(apodize(raw_fid(), kind="em", lb=1.0)))
    assert [h["step"] for h in out.history] == ["apodize", "zerofill", "fft"]
    assert out.history[0]["params"] == {
        "kind": "em",
        "lb": 1.0,
        "gb": 0.0,
        "ssb": 0.0,
        "tm1": 0.0,
        "tm2": None,
    }
    assert out.history[1]["params"] == {"size": None, "factor": 2}
    json.dumps(out.history)  # serializable


def test_input_history_is_not_mutated() -> None:
    raw = raw_fid()
    a = apodize(raw)
    b = zerofill(a)
    assert raw.history == [] and len(a.history) == 1 and len(b.history) == 2


def test_replay_reproduces_processing_exactly() -> None:
    raw = raw_fid()
    done = process_1d(raw)
    again = replay(raw, done.history)
    np.testing.assert_allclose(again.data, done.data)
    assert again.direct.offset_ppm == pytest.approx(done.direct.offset_ppm)
    # exact replay turns data-dependent decisions into explicit values
    steps = [h["step"] for h in again.history]
    assert "autophase" not in steps and "phase" in steps
    assert again.history[steps.index("reference")]["params"]["method"] == "offset"


def test_recipe_json_round_trip(tmp_path) -> None:
    done = process_1d(raw_fid())
    path = tmp_path / "recipe.json"
    save_recipe(done.history, path)
    doc = json.loads(path.read_text())
    assert doc["nmrsuite_recipe"] == 1
    assert load_recipe(path) == done.history
    assert recipe_from_json(recipe_to_json(done.history)) == done.history
    with pytest.raises(ValueError):
        recipe_from_json('{"nmrsuite_recipe": 99, "steps": []}')


def test_default_recipe_contents() -> None:
    steps = [r["step"] for r in default_recipe_1d(raw_fid())]
    assert steps == ["apodize", "zerofill", "fft", "autophase", "baseline", "reference"]
    with_gd = raw_fid()
    with_gd.group_delay = 76.0
    assert default_recipe_1d(with_gd)[0]["step"] == "remove_group_delay"
    no_solvent = raw_fid()
    no_solvent.params = {}
    assert "reference" not in [r["step"] for r in default_recipe_1d(no_solvent)]


def test_registry_and_unknown_step() -> None:
    names = set(registered_steps())
    assert {"apodize", "fft", "phase", "autophase", "baseline", "reference"} <= names
    with pytest.raises(KeyError):
        replay(raw_fid(), [{"step": "no_such_step"}])
    with pytest.raises(ValueError):

        @processing_step("apodize")
        def again(nmr: NMRData) -> NMRData:
            return nmr
