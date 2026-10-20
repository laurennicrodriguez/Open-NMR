"""Zero filling of the direct dimension."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from nmrsuite.data import NMRData
from nmrsuite.history import processing_step


def next_pow2(n: int) -> int:
    return 1 << max(0, int(n - 1).bit_length())


@processing_step("zerofill")
def zerofill(nmr: NMRData, size: int | None = None, factor: int = 2) -> NMRData:
    """Pad the FID with zeros to ``size`` points (default: next power of two ≥ factor × TD).

    The number of acquired points is kept in ``Dimension.acquired`` so later windows
    and linear prediction still see the true acquisition time.
    """
    d = nmr.direct
    if d.domain != "time":
        raise ValueError("zerofill needs time-domain data")
    n = d.size
    target = size if size is not None else next_pow2(factor * n)
    if target < n:
        raise ValueError(f"zerofill size {target} is smaller than the data ({n})")
    pad = [(0, 0)] * (nmr.ndim - 1) + [(0, target - n)]
    data = np.pad(nmr.data, pad)
    dims = [*nmr.dims[:-1], replace(d, size=target, acquired=d.acquired or n)]
    return nmr.copy_with(data=data, dims=dims)
