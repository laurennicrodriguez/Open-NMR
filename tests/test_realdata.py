"""Regression on the committed Chem 2081 Bruker datasets (run in CI).

* The automatic chain must reproduce TopSpin's final spectrum (correlation > 0.99,
  compared on TopSpin's own ppm grid and reference).
* Outputs must match the frozen ``expected.json`` (reference shift, tallest-peak
  positions). Regenerate it only for intended changes: ``python tests/data/make_expected.py``.
* Performance budget from the spec: a 64k-point 1H opened and processed in under 1 s
  (2 s allowed here for slow CI machines).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from nmrsuite.io import read_bruker, read_bruker_processed
from nmrsuite.processing import process_1d, reference
from realdata import DATA, DATASETS, summarize

EXPECTED = (
    json.loads((DATA / "expected.json").read_text()) if (DATA / "expected.json").exists() else {}
)

pytestmark = pytest.mark.skipif(not DATASETS, reason="tests/data/chem2081 missing")


@pytest.fixture(scope="module")
def processed() -> dict:
    return {name: process_1d(read_bruker(exp)) for name, exp in DATASETS.items()}


@pytest.mark.parametrize("name", list(DATASETS))
def test_matches_topspin(processed, name) -> None:
    ours = processed[name]
    ts = read_bruker_processed(DATASETS[name])
    bf = ours.direct.bf_mhz
    ts_shift = -(ts.params["procs"]["SF"] - bf) / bf * 1e6
    ours_shift = next(h for h in ours.history if h["step"] == "reference")["result"]["shift_ppm"]
    aligned = reference(ours, method="offset", delta=ts_shift - ours_shift)
    y = np.interp(ts.direct.ppm_axis()[::-1], aligned.direct.ppm_axis()[::-1],
                  aligned.data.real[::-1])[::-1]  # fmt: skip
    assert np.corrcoef(y, ts.data.real)[0, 1] > 0.99


@pytest.mark.parametrize("name", list(DATASETS))
def test_frozen_expected_output(processed, name) -> None:
    if name not in EXPECTED:
        pytest.fail(f"no expected output for {name}; run tests/data/make_expected.py")
    got, exp = summarize(processed[name]), EXPECTED[name]
    assert got["size"] == exp["size"]
    assert got["reference_shift_ppm"] == pytest.approx(exp["reference_shift_ppm"], abs=5e-4)
    assert len(got["peaks_ppm"]) == len(exp["peaks_ppm"])
    np.testing.assert_allclose(got["peaks_ppm"], exp["peaks_ppm"], atol=5e-4)


@pytest.mark.parametrize("name", list(DATASETS))
def test_cdcl3_reference_is_consistent(processed, name) -> None:
    """All four were run on the same 500 MHz instrument: offsets agree within 0.01 ppm."""
    shift = next(h for h in processed[name].history if h["step"] == "reference")["result"]
    assert shift["solvent"] == "CDCl3"
    assert shift["shift_ppm"] == pytest.approx(-0.025, abs=0.01)


def test_performance_budget() -> None:
    exp = next(iter(DATASETS.values()))
    process_1d(read_bruker(exp))  # warm-up (imports, FFT plans)
    t0 = time.perf_counter()
    process_1d(read_bruker(exp))
    assert time.perf_counter() - t0 < 2.0
