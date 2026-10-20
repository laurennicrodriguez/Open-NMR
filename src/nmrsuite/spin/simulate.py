"""Exact simulation of 1D spectra from a :class:`SpinSystem`.

Transitions are the non-zero elements of F₋ = Σ_k I₋k (observed spins k) between
eigenstates of neighbouring blocks (ΔM = −1, same heteronuclear states)::

    ν = E_upper − E_lower,     intensity = |⟨lower|F₋|upper⟩|² / 2^(N−1)

normalized so the intensities of a spectrum sum to the number of observed spins
(each proton contributes 1, as an integral would). Each line gets a width from the
spins it involves, weighted by |⟨lower|I₋k|upper⟩|².
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np

from nmrsuite.data import Dimension, NMRData
from nmrsuite.spin.hamiltonian import (
    SpinSystem,
    build_blocks,
    diagonalize,
    estimated_memory_mb,
)

#: Lines weaker than this fraction of the strongest are dropped.
MIN_RELATIVE_INTENSITY = 1e-6
#: Above this many (lines × points) the lineshape is evaluated by binning + FFT.
DIRECT_LIMIT = 2e7
#: Simulations estimated to need more working memory than this (MB) raise a warning.
MEMORY_WARNING_MB = 1500.0


@dataclass
class LineList:
    freq_hz: np.ndarray  # from 0 ppm (ν = δ·SF)
    intensity: np.ndarray
    width_hz: np.ndarray
    sf_mhz: float

    @property
    def ppm(self) -> np.ndarray:
        return self.freq_hz / self.sf_mhz

    @property
    def total(self) -> float:
        return float(self.intensity.sum())

    def __len__(self) -> int:
        return int(self.freq_hz.size)

    def sorted(self) -> LineList:
        o = np.argsort(self.freq_hz)
        return LineList(self.freq_hz[o], self.intensity[o], self.width_hz[o], self.sf_mhz)


def transitions(
    system: SpinSystem,
    sf_mhz: float,
    observed: str = "1H",
    min_relative: float = MIN_RELATIVE_INTENSITY,
) -> LineList:
    """All transitions of the observed nucleus at spectrometer frequency ``sf_mhz``."""
    mb = estimated_memory_mb(system, observed)
    if mb > MEMORY_WARNING_MB:
        warnings.warn(
            f"exact simulation of {system.n} spins needs ~{mb / 1000:.1f} GB and may take "
            "minutes; consider splitting into weakly coupled fragments",
            ResourceWarning,
            stacklevel=2,
        )
    blocks = build_blocks(system, sf_mhz, observed)
    diagonalize(blocks)
    obs = system.observed(observed)
    widths = np.array(system.linewidths_hz)
    norm = float(2 ** (system.n - 1))
    freqs, ints, wids = [], [], []
    for (m2, hb), up in blocks.items():
        low = blocks.get((m2 - 2, hb))
        if low is None:
            continue
        t_total = np.zeros((low.states.size, up.states.size))
        t_w = np.zeros_like(t_total)
        t_n = np.zeros_like(t_total)
        for k in obs:
            src = up.states[((up.states >> k) & 1) == 1]  # spin k is α in the upper state
            dst = src ^ (1 << k)
            iu = np.searchsorted(up.states, src)
            il = np.searchsorted(low.states, dst)
            tk = low.vectors[il].T @ up.vectors[iu]  # ⟨lower|I₋k|upper⟩
            t_total += tk
            tk2 = tk**2
            t_w += tk2 * widths[k]
            t_n += tk2
        inten = t_total**2 / norm
        freq = up.energies[None, :] - low.energies[:, None]
        width = np.where(t_n > 0, t_w / np.where(t_n > 0, t_n, 1.0), widths[obs].mean())
        freqs.append(freq.ravel())
        ints.append(inten.ravel())
        wids.append(width.ravel())
    f = np.concatenate(freqs)
    i = np.concatenate(ints)
    w = np.concatenate(wids)
    keep = i > min_relative * i.max()
    return LineList(f[keep], i[keep], w[keep], sf_mhz).sorted()


def lorentzian(
    lines: LineList, hz: np.ndarray, phase_deg: float = 0.0, method: str = "auto"
) -> np.ndarray:
    """Sum of area-normalized Lorentzians on the frequency axis ``hz`` (real part after
    an optional phase rotation, which mixes in dispersion)."""
    hz = np.asarray(hz, float)
    if method == "auto":
        method = "direct" if len(lines) * hz.size <= DIRECT_LIMIT else "binned"
    phi = np.deg2rad(phase_deg)
    if method == "direct":
        out = np.zeros(hz.size, complex)
        chunk = max(1, int(DIRECT_LIMIT // max(hz.size, 1)) // 4)
        for s in range(0, len(lines), chunk):
            f = lines.freq_hz[s : s + chunk, None]
            a = lines.intensity[s : s + chunk, None]
            hw = lines.width_hz[s : s + chunk, None] / 2
            x = hz[None, :] - f
            out += (a / np.pi * (hw + 1j * x) / (hw**2 + x**2)).sum(axis=0)
        return (out * np.exp(1j * phi)).real
    if method == "binned":
        return _binned(lines, hz, phi)
    raise ValueError(f"unknown method {method!r}")


def _binned(lines: LineList, hz: np.ndarray, phi: float) -> np.ndarray:
    """Sticks on the grid (linear interpolation), convolved with each width's Lorentzian
    by FFT. Accurate when the grid spacing is well below the line width."""
    n = hz.size
    step = (hz[-1] - hz[0]) / (n - 1)
    pad = n  # zero padding against wrap-around of tails
    out = np.zeros(n)
    for w in np.unique(np.round(lines.width_hz, 3)):
        sel = np.round(lines.width_hz, 3) == w
        pos = (lines.freq_hz[sel] - hz[0]) / step
        amp = lines.intensity[sel]
        sticks = np.zeros(n + pad)
        i0 = np.floor(pos).astype(int)
        frac = pos - i0
        ok = (i0 >= 0) & (i0 < n - 1)
        np.add.at(sticks, i0[ok], amp[ok] * (1 - frac[ok]))
        np.add.at(sticks, i0[ok] + 1, amp[ok] * frac[ok])
        k = np.fft.fftfreq(n + pad) * (n + pad) * step  # signed Hz offsets
        hw = w / 2
        kern = (hw + 1j * k) / (hw**2 + k**2) / np.pi
        kern = (kern * np.exp(1j * phi)).real
        out += np.fft.ifft(np.fft.fft(sticks) * np.fft.fft(kern)).real[:n]
    return out


def simulate(
    system: SpinSystem,
    sf_mhz: float,
    ppm_from: float,
    ppm_to: float,
    points: int = 16384,
    phase_deg: float = 0.0,
    observed: str = "1H",
) -> tuple[np.ndarray, np.ndarray]:
    """Spectrum between two shifts: (ppm axis high → low, intensity)."""
    ppm = np.linspace(max(ppm_from, ppm_to), min(ppm_from, ppm_to), points)
    lines = transitions(system, sf_mhz, observed)
    return ppm, lorentzian(lines, ppm * sf_mhz, phase_deg)


def simulate_like(
    system: SpinSystem, template: NMRData, observed: str | None = None, phase_deg: float = 0.0
) -> NMRData:
    """Simulated spectrum on the same axis as a processed experimental 1D spectrum."""
    d: Dimension = template.direct
    if d.domain != "frequency":
        raise ValueError("template must be a frequency-domain spectrum")
    lines = transitions(system, d.obs_mhz, observed or d.nucleus)
    y = lorentzian(lines, d.hz_axis(), phase_deg)
    return NMRData(data=y, dims=[replace(d, is_complex=False)], experiment="simulation")
