"""Spin systems and their Hamiltonians, block-diagonalized.

Units: shifts in ppm, couplings in Hz; energies are in Hz (H/h).

For spins of one observed nucleus (e.g. all ¹H) the Hamiltonian in the rotating frame is

    H = Σ_i ν_i I_zi + Σ_{i<j} J_ij (I_xi I_xj + I_yi I_yj + I_zi I_zj),
    ν_i = δ_i · SF  (Hz, relative to 0 ppm; SF = spectrometer frequency in MHz).

Couplings to *other* nuclei (¹⁹F, ³¹P, …) are always weak at any field, so only their
J·I_z I_z term is kept (exact, not an approximation, in the sense of first-order
heteronuclear coupling).

The Hamiltonian commutes with the total z-magnetization of the observed spins, F_z, and
with I_z of every heteronucleus, so it is block-diagonal: one block per value of
(M_obs, heteronuclear states). Each block is diagonalized separately, which is what
makes 12–14 coupled protons tractable (largest block C(14, 7) = 3432 for 14 ¹H).

Basis states are integers: bit k set ⇔ spin k is α (m = +½).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

MAX_SPINS = 16


@dataclass
class SpinSystem:
    """Coupled spin-½ nuclei.

    Attributes
    ----------
    shifts_ppm:
        Chemical shift of each spin. Heteronuclear shifts are ignored (only the observed
        nucleus is simulated) but must be given (any value).
    couplings:
        ``{(i, j): J_Hz}`` (any order of i, j) or an N×N symmetric matrix.
    nuclei:
        Nucleus of each spin, default all ``"1H"``.
    labels:
        Names for reporting (default ``"A"``, ``"B"``, …).
    linewidths_hz:
        Lorentzian full width at half height per spin (used by the lineshape).
    """

    shifts_ppm: Sequence[float]
    couplings: Mapping[tuple[int, int], float] | np.ndarray = field(default_factory=dict)
    nuclei: Sequence[str] | None = None
    labels: Sequence[str] | None = None
    linewidths_hz: Sequence[float] | float = 1.0

    def __post_init__(self) -> None:
        n = len(self.shifts_ppm)
        if n == 0:
            raise ValueError("a spin system needs at least one spin")
        if n > MAX_SPINS:
            raise ValueError(f"{n} spins: exact simulation is limited to {MAX_SPINS}")
        self.shifts_ppm = [float(s) for s in self.shifts_ppm]
        self.nuclei = list(self.nuclei) if self.nuclei is not None else ["1H"] * n
        if len(self.nuclei) != n:
            raise ValueError("nuclei must have one entry per spin")
        self.labels = (
            list(self.labels) if self.labels is not None else [chr(65 + k) for k in range(n)]
        )
        if isinstance(self.linewidths_hz, int | float):
            self.linewidths_hz = [float(self.linewidths_hz)] * n
        self.linewidths_hz = [float(w) for w in self.linewidths_hz]
        j = np.zeros((n, n))
        if isinstance(self.couplings, np.ndarray):
            if self.couplings.shape != (n, n):
                raise ValueError(f"coupling matrix must be {n}×{n}")
            j = 0.5 * (self.couplings + self.couplings.T)
        else:
            for (a, b), v in self.couplings.items():
                if a == b:
                    raise ValueError("a spin cannot couple to itself")
                j[a, b] = j[b, a] = float(v)
        np.fill_diagonal(j, 0.0)
        self.j_matrix = j

    @property
    def n(self) -> int:
        return len(self.shifts_ppm)

    def observed(self, nucleus: str = "1H") -> list[int]:
        return [k for k, nuc in enumerate(self.nuclei) if nuc == nucleus]

    def coupling(self, a: int, b: int) -> float:
        return float(self.j_matrix[a, b])


@dataclass
class Block:
    """One diagonal block of H: basis states, matrix, eigen-decomposition."""

    key: tuple[int, int]  # (2·M_obs, hetero-state bits)
    states: np.ndarray  # basis integers, sorted
    h: np.ndarray
    energies: np.ndarray | None = None
    vectors: np.ndarray | None = None


def _bits(states: np.ndarray, k: int) -> np.ndarray:
    return (states >> k) & 1


def build_blocks(
    system: SpinSystem, sf_mhz: float, observed: str = "1H"
) -> dict[tuple[int, int], Block]:
    """Hamiltonian blocks for the observed nucleus at spectrometer frequency ``sf_mhz``."""
    n = system.n
    obs = system.observed(observed)
    if not obs:
        raise ValueError(f"no {observed} spins in the system")
    het = [k for k in range(n) if k not in obs]
    all_states = np.arange(1 << n, dtype=np.int64)
    m2 = sum(2 * _bits(all_states, k) - 1 for k in obs)  # 2·M_obs
    hbits = np.zeros_like(all_states)
    for pos, k in enumerate(het):
        hbits |= _bits(all_states, k) << pos

    nu = np.array([s * sf_mhz for s in system.shifts_ppm])  # Hz from 0 ppm
    jm = system.j_matrix
    blocks: dict[tuple[int, int], Block] = {}
    keys = np.unique(np.stack([m2, hbits], axis=1), axis=0)
    index = np.full(1 << n, -1, dtype=np.int64)  # basis integer -> row within its block
    for key2, hb in keys:
        states = all_states[(m2 == key2) & (hbits == hb)]
        index[states] = np.arange(states.size)
        mz = [_bits(states, k) - 0.5 for k in range(n)]
        diag = np.zeros(states.size)
        for k in obs:
            diag += nu[k] * mz[k]
        for a, b in combinations(range(n), 2):
            if jm[a, b]:
                diag += jm[a, b] * mz[a] * mz[b]
        h = np.diag(diag)
        # flip-flop terms ½J(I+I- + I-I+) between observed spins only
        for a, b in combinations(obs, 2):
            jab = jm[a, b]
            if not jab:
                continue
            differ = _bits(states, a) != _bits(states, b)
            src = states[differ]
            dst = src ^ ((1 << a) | (1 << b))
            h[index[src], index[dst]] += 0.5 * jab
        blocks[(int(key2), int(hb))] = Block((int(key2), int(hb)), states, h)
    return blocks


def diagonalize(blocks: Mapping[tuple[int, int], Block]) -> None:
    for b in blocks.values():
        if b.energies is None:
            b.energies, b.vectors = np.linalg.eigh(b.h)


def block_sizes(system: SpinSystem, observed: str = "1H") -> list[int]:
    """Sizes of the H blocks (for estimating cost before simulating)."""
    from math import comb

    n_obs = len(system.observed(observed))
    n_het = system.n - n_obs
    return [comb(n_obs, k) for k in range(n_obs + 1) for _ in range(1 << n_het)]


def spins_from(items: Iterable[tuple[float, str]]) -> SpinSystem:
    """Convenience: ``spins_from([(7.26, "1H"), (−110.0, "19F")])`` without couplings."""
    shifts, nuclei = zip(*items, strict=True)
    return SpinSystem(list(shifts), nuclei=list(nuclei))
