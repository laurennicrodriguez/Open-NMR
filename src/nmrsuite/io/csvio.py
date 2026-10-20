"""Plain CSV export of 1D spectra (ppm, Hz, real[, imaginary])."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nmrsuite.data import NMRData


def write_spectrum_csv(nmr: NMRData, path: str | Path) -> None:
    if nmr.ndim != 1 or nmr.direct.domain != "frequency":
        raise ValueError("CSV export needs a 1D frequency-domain spectrum")
    d = nmr.direct
    cols = [d.ppm_axis(), d.hz_axis(), np.real(nmr.data)]
    header = "ppm,hz,real"
    if np.iscomplexobj(nmr.data):
        cols.append(np.imag(nmr.data))
        header += ",imag"
    np.savetxt(path, np.column_stack(cols), delimiter=",", header=header, comments="",
               fmt="%.8g")  # fmt: skip
