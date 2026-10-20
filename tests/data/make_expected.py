"""Regenerate the frozen expected outputs for the committed real datasets.

Run only when a change to the expected output is intended, and review the diff:

    python tests/data/make_expected.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from nmrsuite.io import read_bruker  # noqa: E402
from nmrsuite.processing import process_1d  # noqa: E402
from realdata import DATASETS, summarize  # noqa: E402


def main() -> None:
    expected = {}
    for name, exp in DATASETS.items():
        expected[name] = summarize(process_1d(read_bruker(exp)))
    path = HERE / "chem2081" / "expected.json"
    path.write_text(json.dumps(expected, indent=2, default=np.ndarray.tolist) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
