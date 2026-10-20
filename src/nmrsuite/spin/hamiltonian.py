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


class SpinKernel:
    """Precomputed structure of H for one spin-system *topology*.

    Everything that depends only on the number of spins, their nuclei and which pairs
    are coupled — block membership, the z-magnetization of every basis state, the
    flip-flop matrix positions and the F₋ index arrays between blocks — is computed
    once. :meth:`evaluate` then only fills in shifts and couplings and diagonalizes,
    which is what a lineshape fit does hundreds of times.
    """

    def __init__(
        self,
        n: int,
        nuclei: Sequence[str],
        pairs: Sequence[tuple[int, int]],
        observed: str = "1H",
    ) -> None:
        self.n = n
        self.pairs = [tuple(sorted(p)) for p in pairs]
        self.obs = [k for k, nuc in enumerate(nuclei) if nuc == observed]
        if not self.obs:
            raise ValueError(f"no {observed} spins in the system")
        obs_set = set(self.obs)
        het = [k for k in range(n) if k not in obs_set]
        all_states = np.arange(1 << n, dtype=np.int64)
        m2 = sum(2 * _bits(all_states, k) - 1 for k in self.obs)
        hbits = np.zeros_like(all_states)
        for pos, k in enumerate(het):
            hbits |= _bits(all_states, k) << pos
        index = np.full(1 << n, -1, dtype=np.int64)
        self.blocks: dict[tuple[int, int], dict] = {}
        for key2, hb in np.unique(np.stack([m2, hbits], axis=1), axis=0):
            states = all_states[(m2 == key2) & (hbits == hb)]
            index[states] = np.arange(states.size)
            mz = np.array([_bits(states, k) - 0.5 for k in range(n)])
            zz = (
                np.array([mz[a] * mz[b] for a, b in self.pairs])
                if self.pairs
                else np.zeros((0, states.size))
            )
            flips = []
            for p, (a, b) in enumerate(self.pairs):
                if a in obs_set and b in obs_set:
                    src = states[_bits(states, a) != _bits(states, b)]
                    dst = src ^ ((1 << a) | (1 << b))
                    flips.append((p, index[src].copy(), index[dst].copy()))
            self.blocks[(int(key2), int(hb))] = {
                "states": states, "mz_obs": mz[self.obs], "zz": zz, "flips": flips,
            }  # fmt: skip
        # F₋ = Σ_k I₋k between neighbouring blocks: (lower rows, upper cols) per spin k
        self.links = []
        for (m2_, hb), up in self.blocks.items():
            low = self.blocks.get((m2_ - 2, hb))
            if low is None:
                continue
            per_k = []
            for k in self.obs:
                src = up["states"][((up["states"] >> k) & 1) == 1]
                per_k.append((
                    k,
                    np.searchsorted(low["states"], src ^ (1 << k)),
                    np.searchsorted(up["states"], src),
                ))  # fmt: skip
            self.links.append(((m2_ - 2, hb), (m2_, hb), per_k))

    @classmethod
    def for_system(cls, system: SpinSystem, observed: str = "1H") -> SpinKernel:
        pairs = [(a, b) for a, b in combinations(range(system.n), 2) if system.j_matrix[a, b]]
        return cls(system.n, system.nuclei, pairs, observed)

    def evaluate(
        self, nu_hz: np.ndarray, j_pairs: np.ndarray
    ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
        """Eigenvalues and eigenvectors of every block for shifts ν (Hz, all spins) and
        couplings J (Hz, in the order of :attr:`pairs`)."""
        nu_obs = np.asarray(nu_hz, float)[self.obs]
        jv = np.asarray(j_pairs, float)
        out = {}
        for key, b in self.blocks.items():
            diag = nu_obs @ b["mz_obs"] + (jv @ b["zz"] if jv.size else 0.0)
            h = np.diag(diag)
            for p, rows, cols in b["flips"]:
                if jv[p]:
                    h[rows, cols] += 0.5 * jv[p]
            out[key] = np.linalg.eigh(h)
        return out


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


#: Bytes held per candidate transition while simulating (≈9 float64 working arrays).
BYTES_PER_TRANSITION = 72


def estimated_memory_mb(system: SpinSystem, observed: str = "1H") -> float:
    """Peak working memory of :func:`~nmrsuite.spin.simulate.transitions`, in MB.

    Every pair of neighbouring blocks (ΔM = −1) holds a dense n_lower × n_upper set of
    candidate transitions. Measured on this machine: 12 ¹H ≈ 185 MB (1 s), 13 ≈ 690 MB
    (7 s), 14 ≈ 2.6 GB (50 s, one thread); each extra spin costs ~3–4× more.
    """
    from math import comb

    n_obs = len(system.observed(observed))
    n_het = system.n - n_obs
    pairs = sum(comb(n_obs, k) * comb(n_obs, k + 1) for k in range(n_obs))
    return pairs * (1 << n_het) * BYTES_PER_TRANSITION / 1e6


def spins_from(items: Iterable[tuple[float, str]]) -> SpinSystem:
    """Convenience: ``spins_from([(7.26, "1H"), (−110.0, "19F")])`` without couplings."""
    shifts, nuclei = zip(*items, strict=True)
    return SpinSystem(list(shifts), nuclei=list(nuclei))
