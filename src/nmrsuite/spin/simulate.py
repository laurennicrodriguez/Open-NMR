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
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from nmrsuite.data import Dimension, NMRData
from nmrsuite.spin.hamiltonian import SpinKernel, SpinSystem, estimated_memory_mb

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
    #: Optional (with ``gradients=True``): ∂ν_line/∂θ for θ = (ν of each observed spin in
    #: Hz, then each kernel coupling pair in Hz), exact by Hellmann–Feynman.
    dfreq: np.ndarray | None = None
    #: Optional: ∂w_line/∂w_k, the share of each observed spin in each line's width.
    wweights: np.ndarray | None = None

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
        return LineList(
            self.freq_hz[o], self.intensity[o], self.width_hz[o], self.sf_mhz,
            None if self.dfreq is None else self.dfreq[o],
            None if self.wweights is None else self.wweights[o],
        )  # fmt: skip


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
    kernel = kernel_for(system, observed)
    nu = np.array(system.shifts_ppm) * sf_mhz
    jv = np.array([system.j_matrix[a, b] for a, b in kernel.pairs])
    return transitions_from(kernel, nu, jv, np.array(system.linewidths_hz), sf_mhz, min_relative)


@lru_cache(maxsize=64)
def _kernel(n: int, nuclei: tuple[str, ...], pairs: tuple, observed: str) -> SpinKernel:
    return SpinKernel(n, nuclei, pairs, observed)


def kernel_for(
    system: SpinSystem, observed: str = "1H", pairs: Sequence[tuple[int, int]] | None = None
) -> SpinKernel:
    """Cached :class:`SpinKernel` for a system's topology (or an explicit pair list)."""
    if pairs is None:
        pairs = [
            (a, b) for a in range(system.n) for b in range(a + 1, system.n) if system.j_matrix[a, b]
        ]
    key = tuple(sorted(tuple(sorted(p)) for p in pairs))
    return _kernel(system.n, tuple(system.nuclei), key, observed)


def transitions_from(
    kernel: SpinKernel,
    nu_hz: np.ndarray,
    j_pairs: np.ndarray,
    widths_hz: np.ndarray,
    sf_mhz: float,
    min_relative: float = MIN_RELATIVE_INTENSITY,
    gradients: bool = False,
) -> LineList:
    """Transitions for given shifts (Hz, all spins), couplings (kernel pair order) and
    per-spin widths, reusing the kernel's precomputed structure.

    With ``gradients=True`` the line list also carries ∂ν/∂(ν_k, J_p) — by
    Hellmann–Feynman, ∂E_i/∂θ = ⟨i|∂H/∂θ|i⟩, exact and nearly free given the
    eigenvectors — and each line's width weights. Intensity derivatives are not
    computed (they need eigenvector derivatives).
    """
    eig = kernel.evaluate(nu_hz, j_pairs)
    widths = np.asarray(widths_hz, float)
    norm = float(2 ** (kernel.n - 1))
    obs_mean = widths[kernel.obs].mean()
    egrad = _energy_gradients(kernel, eig) if gradients else None
    freqs, ints, wids, dfs, wws = [], [], [], [], []
    for low_key, up_key, per_k in kernel.links:
        e_low, v_low = eig[low_key]
        e_up, v_up = eig[up_key]
        t_total = np.zeros((e_low.size, e_up.size))
        t_w = np.zeros_like(t_total)
        t_n = np.zeros_like(t_total)
        shares = []
        for k, il, iu in per_k:
            tk = v_low[il].T @ v_up[iu]  # ⟨lower|I₋k|upper⟩
            t_total += tk
            tk2 = tk * tk
            t_w += tk2 * widths[k]
            t_n += tk2
            shares.append(tk2)
        freqs.append((e_up[None, :] - e_low[:, None]).ravel())
        ints.append((t_total * t_total / norm).ravel())
        safe = np.where(t_n > 0, t_n, 1.0)
        wids.append(np.where(t_n > 0, t_w / safe, obs_mean).ravel())
        if gradients:
            g_low, g_up = egrad[low_key], egrad[up_key]  # (nθ, n_states)
            df = g_up[:, None, :] - g_low[:, :, None]  # (nθ, n_low, n_up)
            dfs.append(df.reshape(df.shape[0], -1).T)
            ww = np.stack([s / safe for s in shares])  # (n_obs, n_low, n_up)
            wws.append(ww.reshape(ww.shape[0], -1).T)
    f = np.concatenate(freqs)
    i = np.concatenate(ints)
    w = np.concatenate(wids)
    keep = i > min_relative * i.max()
    dfreq = np.concatenate(dfs)[keep] if gradients else None
    wweights = np.concatenate(wws)[keep] if gradients else None
    return LineList(f[keep], i[keep], w[keep], sf_mhz, dfreq, wweights).sorted()


def _energy_gradients(kernel: SpinKernel, eig: dict) -> dict:
    """∂E/∂θ for every eigenstate of every block: θ = (ν_k of observed spins, J_p)."""
    out = {}
    for key, b in kernel.blocks.items():
        _, v = eig[key]
        w = v * v  # ⟨i|s⟩² per basis state s and eigenstate i
        g_nu = b["mz_obs"] @ w  # (n_obs, n_eig)
        g_j = b["zz"] @ w if b["zz"].size else np.zeros((0, v.shape[1]))
        for p, rows, cols in b["flips"]:
            g_j[p] += 0.5 * np.einsum("ri,ri->i", v[rows], v[cols])
        out[key] = np.vstack([g_nu, g_j])
    return out


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
        # Re[(hw + i·x)/(hw² + x²) · e^{iφ}] in real arithmetic (≈3× faster than complex)
        c, s_ = np.cos(phi), np.sin(phi)
        out = np.zeros(hz.size)
        chunk = max(1, int(DIRECT_LIMIT // max(hz.size, 1)) // 4)
        for s in range(0, len(lines), chunk):
            f = lines.freq_hz[s : s + chunk, None]
            a = lines.intensity[s : s + chunk, None] / np.pi
            hw = lines.width_hz[s : s + chunk, None] / 2
            x = hz[None, :] - f
            num = hw * c - x * s_ if s_ else hw
            out += (a * num / (hw * hw + x * x)).sum(axis=0)
        return out
    if method == "binned":
        return _binned(lines, hz, phi)
    raise ValueError(f"unknown method {method!r}")


def lorentzian_jacobian(
    lines: LineList,
    hz: np.ndarray,
    phase_deg: float,
    coef_f: np.ndarray,
    coef_w: np.ndarray,
    chunk_elems: float = 4e6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spectrum S and its derivatives through line positions and widths.

    ``coef_f`` (lines × nθ) holds intensity·∂ν/∂θ and ``coef_w`` intensity·∂w/∂θ; the
    result is S (points), ∂S/∂θ (points × nθ) and ∂S/∂φ (points, per radian). Lines
    are processed in chunks of at most ``chunk_elems`` line-points, so memory stays
    small for large spin systems.
    """
    hz = np.asarray(hz, float)
    c, s_ = np.cos(np.deg2rad(phase_deg)), np.sin(np.deg2rad(phase_deg))
    n_t = coef_f.shape[1]
    spec = np.zeros(hz.size)
    jac = np.zeros((hz.size, n_t))
    dphi = np.zeros(hz.size)
    step = max(1, int(chunk_elems // max(hz.size, 1)))
    for a in range(0, len(lines), step):
        b = a + step
        f = lines.freq_hz[a:b, None]
        inten = lines.intensity[a:b]
        hw = lines.width_hz[a:b, None] / 2
        x = hz[None, :] - f
        den = hw * hw + x * x
        num = hw * c - x * s_
        inv = 1.0 / (np.pi * den)
        lin = num * inv  # unit-area Lorentzian (phased)
        spec += inten @ lin
        d_f = (s_ * den + 2 * x * num) * inv / den  # ∂L/∂ν
        d_w = 0.5 * (c * den - 2 * hw * num) * inv / den  # ∂L/∂w
        jac += d_f.T @ coef_f[a:b] + d_w.T @ coef_w[a:b]
        dphi += inten @ ((-hw * s_ - x * c) * inv)
    return spec, jac, dphi


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
