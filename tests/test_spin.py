"""Spin engine: exact simulation validated against analytic solutions and brute force."""

from __future__ import annotations

import time
from functools import reduce

import numpy as np
import pytest

from nmrsuite.spin import SpinSystem, block_sizes, lorentzian, simulate, transitions
from nmrsuite.spin.simulate import LineList

SF = 500.0


def lines_of(system: SpinSystem, sf: float = SF, observed: str = "1H") -> LineList:
    return transitions(system, sf, observed)


def merge(lines: LineList, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Merge degenerate lines (equal frequency) into (freq, intensity)."""
    f, a = [], []
    for fi, ai in zip(lines.freq_hz, lines.intensity, strict=True):
        if f and abs(fi - f[-1]) < tol:
            a[-1] += ai
        else:
            f.append(fi)
            a.append(ai)
    return np.array(f), np.array(a)


# --- simplest cases --------------------------------------------------------------------


def test_single_spin() -> None:
    ll = lines_of(SpinSystem([7.26]))
    assert len(ll) == 1
    assert ll.freq_hz[0] == pytest.approx(7.26 * SF)
    assert ll.intensity[0] == pytest.approx(1.0)


def test_equivalent_pair_is_a_singlet() -> None:
    """A₂: coupling between equivalent spins is invisible."""
    f, a = merge(lines_of(SpinSystem([3.0, 3.0], {(0, 1): 7.0})))
    assert f == pytest.approx([1500.0]) and a == pytest.approx([2.0])


def test_first_order_doublet_of_doublets() -> None:
    """AMX far apart: each spin is a dd with the two J values (to ~J²/Δν)."""
    sys = SpinSystem([7.0, 4.0, 1.0], {(0, 1): 10.0, (0, 2): 4.0, (1, 2): 6.0})
    f, a = merge(lines_of(sys))
    a_lines = np.sort(f[f > 3000])
    expected = 3500 + np.array([-7.0, -3.0, 3.0, 7.0])
    np.testing.assert_allclose(a_lines, expected, atol=0.02)
    np.testing.assert_allclose(a[f > 3000], 0.25, atol=0.01)


# --- AB: analytic ----------------------------------------------------------------------


@pytest.mark.parametrize(("dnu", "j"), [(5.0, 15.0), (20.0, 7.0), (50.0, 16.0), (1.0, 12.0)])
def test_ab_quartet_exact(dnu, j) -> None:
    """Lines at ±(D ± J)/2 about the centre, D = √(Δν² + J²); inner : outer intensity
    (1 + J/D) : (1 − J/D) (roof effect)."""
    centre = 3.0 * SF
    sys = SpinSystem([3.0 + dnu / 2 / SF, 3.0 - dnu / 2 / SF], {(0, 1): j})
    f, a = merge(lines_of(sys))
    d = np.hypot(dnu, j)
    exp_f = centre + np.array([-(d + j) / 2, -(d - j) / 2, (d - j) / 2, (d + j) / 2])
    exp_a = np.array([1 - j / d, 1 + j / d, 1 + j / d, 1 - j / d]) / 2
    np.testing.assert_allclose(f, exp_f, atol=1e-8)
    np.testing.assert_allclose(a, exp_a, atol=1e-10)


# --- ABX: analytic ---------------------------------------------------------------------


def _ab_lines(nu_a: float, nu_b: float, j: float) -> tuple[np.ndarray, np.ndarray]:
    c, dnu = (nu_a + nu_b) / 2, nu_a - nu_b
    d = np.hypot(dnu, j)
    f = c + np.array([-(d + j) / 2, -(d - j) / 2, (d - j) / 2, (d + j) / 2])
    a = np.array([1 - j / d, 1 + j / d, 1 + j / d, 1 - j / d]) / 2
    return f, a


@pytest.mark.parametrize(
    ("dnu", "jab", "jax", "jbx"),
    [(12.0, 16.0, 8.0, 3.0), (30.0, 15.5, 10.0, 2.0), (6.0, 12.0, -3.0, 7.0)],
)
def test_abx_ab_part_with_heteronuclear_x(dnu, jab, jax, jbx) -> None:
    """With X = ¹⁹F the AB part is exactly two AB quartets with effective shifts
    ν_A ± J_AX/2, ν_B ± J_BX/2 (one per X spin state)."""
    nu_a, nu_b = 3.0 * SF + dnu / 2, 3.0 * SF - dnu / 2
    sys = SpinSystem(
        [nu_a / SF, nu_b / SF, -110.0],
        {(0, 1): jab, (0, 2): jax, (1, 2): jbx},
        nuclei=["1H", "1H", "19F"],
    )
    f, a = merge(lines_of(sys))
    exp = [_ab_lines(nu_a + s * jax / 2, nu_b + s * jbx / 2, jab) for s in (1, -1)]
    ef = np.concatenate([e[0] for e in exp])
    ea = np.concatenate([e[1] for e in exp]) / 2  # each X state holds half the molecules
    o = np.argsort(ef)
    np.testing.assert_allclose(f, ef[o], atol=1e-8)
    np.testing.assert_allclose(a, ea[o], atol=1e-10)


def test_abx_x_part_homonuclear_pople() -> None:
    """Homonuclear ABX with X far away: X lines at ν_X ± ½(J_AX + J_BX) and
    ν_X ± (D₊ − D₋), 2D± = √[(Δν_AB ± ½(J_AX − J_BX))² + J_AB²] (Pople, Schneider &
    Bernstein); exact up to terms of order J²/Δν_AX."""
    dnu, jab, jax, jbx = 12.0, 16.0, 8.0, 3.0
    nu_x = 7.0 * SF
    sys = SpinSystem(
        [3.0 + dnu / 2 / SF, 3.0 - dnu / 2 / SF, 7.0],
        {(0, 1): jab, (0, 2): jax, (1, 2): jbx},
    )
    ll = lines_of(sys)
    x = ll.freq_hz > 5 * SF
    xf, xa = merge(LineList(ll.freq_hz[x], ll.intensity[x], ll.width_hz[x], SF), tol=1e-3)
    strong = xf[xa > 0.05]
    dp = np.sqrt((dnu + (jax - jbx) / 2) ** 2 + jab**2) / 2
    dm = np.sqrt((dnu - (jax - jbx) / 2) ** 2 + jab**2) / 2
    expected = np.sort(nu_x + np.array([-1, 1, -2, 2]) / 2 * np.array(
        [jax + jbx, jax + jbx, dp - dm, dp - dm]))  # fmt: skip
    np.testing.assert_allclose(np.sort(strong), expected, atol=0.05)
    assert xa.sum() == pytest.approx(1.0, abs=1e-6)  # X is one proton


# --- independent brute-force check ----------------------------------------------------


def _brute_force(sys: SpinSystem, sf: float) -> tuple[np.ndarray, np.ndarray]:
    """Full 2^N Hamiltonian from Kronecker products (independent of the block code)."""
    sx = np.array([[0, 0.5], [0.5, 0]])
    sy = np.array([[0, -0.5j], [0.5j, 0]])
    sz = np.array([[0.5, 0], [0, -0.5]])
    sm = np.array([[0, 0], [1, 0]])  # I₋: α → β (basis order α, β)
    eye = np.eye(2)
    n = sys.n

    def op(o, k):
        return reduce(np.kron, [o if i == k else eye for i in range(n)])

    h = sum(sys.shifts_ppm[k] * sf * op(sz, k) for k in range(n)).astype(complex)
    for a in range(n):
        for b in range(a + 1, n):
            j = sys.j_matrix[a, b]
            if j:
                h = h + j * (op(sx, a) @ op(sx, b) + op(sy, a) @ op(sy, b) + op(sz, a) @ op(sz, b))
    e, v = np.linalg.eigh(h)
    fm = sum(op(sm, k) for k in range(n))
    t = v.conj().T @ fm @ v  # t[l, u] = ⟨l|F₋|u⟩
    inten = np.abs(t) ** 2 / 2 ** (n - 1)
    freq = e[None, :] - e[:, None]
    keep = inten > 1e-6 * inten.max()
    order = np.argsort(freq[keep])
    return freq[keep][order], inten[keep][order]


@pytest.mark.parametrize("seed", range(4))
def test_matches_brute_force_full_hamiltonian(seed) -> None:
    rng = np.random.default_rng(seed)
    n = 5
    shifts = 3.0 + rng.uniform(-0.05, 0.05, n)  # strongly coupled: Δν ~ J
    js = {(a, b): rng.uniform(-18, 18) for a in range(n) for b in range(a + 1, n)}
    sys = SpinSystem(list(shifts), js)
    f, a = merge(lines_of(sys), tol=1e-7)
    bf, ba = _brute_force(sys, SF)
    # merge the brute-force list the same way
    bl = LineList(bf, ba, np.ones_like(bf), SF)
    bf, ba = merge(bl, tol=1e-7)
    np.testing.assert_allclose(f, bf, atol=1e-7)
    np.testing.assert_allclose(a, ba, atol=1e-9)


# --- general properties ----------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_intensity_sum_rule(n) -> None:
    rng = np.random.default_rng(n)
    shifts = list(2.0 + rng.uniform(0, 0.1, n))
    js = {(a, b): rng.uniform(-15, 15) for a in range(n) for b in range(a + 1, n)}
    ll = transitions(SpinSystem(shifts, js), SF, min_relative=0.0)
    assert ll.total == pytest.approx(n, rel=1e-9)


def test_relabelling_spins_changes_nothing() -> None:
    shifts = [3.0, 3.02, 3.05]
    js = {(0, 1): 12.0, (0, 2): 5.0, (1, 2): 7.0}
    perm = [2, 0, 1]
    inv = {p: i for i, p in enumerate(perm)}
    s2 = [shifts[p] for p in perm]
    j2 = {(inv[a], inv[b]): v for (a, b), v in js.items()}
    f1, a1 = merge(lines_of(SpinSystem(shifts, js)), 1e-7)
    f2, a2 = merge(lines_of(SpinSystem(s2, j2)), 1e-7)
    np.testing.assert_allclose(f1, f2, atol=1e-8)
    np.testing.assert_allclose(a1, a2, atol=1e-10)


def test_second_order_depends_on_field_first_order_does_not() -> None:
    """J is field-independent; an AB pattern collapses towards first order at higher field."""
    sys = SpinSystem([3.0, 3.02], {(0, 1): 15.0})
    inner_outer = []
    for sf in (400.0, 600.0):
        f, a = merge(lines_of(sys, sf))
        assert f[1] - f[0] == pytest.approx(15.0, abs=1e-8)  # outer-inner spacing is J
        inner_outer.append(a[1] / a[0])
    assert inner_outer[0] > inner_outer[1] > 1  # roofing weaker at 600 MHz


def test_heteronuclear_couplings() -> None:
    """¹H–¹⁹F: the ¹H spectrum is a doublet; observing ¹⁹F gives the F doublet."""
    sys = SpinSystem([6.9, -115.0], {(0, 1): 48.0}, nuclei=["1H", "19F"])
    f, a = merge(lines_of(sys))
    np.testing.assert_allclose(f - 6.9 * SF, [-24.0, 24.0], atol=1e-9)
    np.testing.assert_allclose(a, [0.5, 0.5])
    ff, fa = merge(transitions(sys, 470.4, observed="19F"))
    np.testing.assert_allclose(ff - (-115.0 * 470.4), [-24.0, 24.0], atol=1e-9)
    assert fa.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        transitions(sys, SF, observed="31P")


def test_validation() -> None:
    with pytest.raises(ValueError):
        SpinSystem([])
    with pytest.raises(ValueError):
        SpinSystem([1.0] * 17)
    with pytest.raises(ValueError):
        SpinSystem([1.0, 2.0], {(0, 0): 3.0})
    with pytest.raises(ValueError):
        SpinSystem([1.0, 2.0], np.zeros((3, 3)))
    assert block_sizes(SpinSystem([1.0] * 4)) == [1, 4, 6, 4, 1]


# --- lineshape ---------------------------------------------------------------------


def test_lineshape_area_and_width() -> None:
    sys = SpinSystem([3.0], linewidths_hz=1.2)
    ppm, y = simulate(sys, SF, 3.5, 2.5, points=200001)
    hz = ppm * SF
    area = -np.trapezoid(y, hz)  # axis runs high → low
    assert area == pytest.approx(1.0, rel=2e-3)  # Lorentzian tails beyond ±250 Hz: 0.15 %
    half = y.max() / 2
    assert (y > half).sum() * abs(hz[1] - hz[0]) == pytest.approx(1.2, abs=0.02)


def test_direct_and_binned_lineshapes_agree() -> None:
    sys = SpinSystem([3.0, 3.03, 3.1], {(0, 1): 12.0, (1, 2): 6.0}, linewidths_hz=[0.8, 1.0, 1.4])
    ll = transitions(sys, SF)
    hz = np.linspace(1580, 1470, 8192)
    direct = lorentzian(ll, hz, method="direct")
    binned = lorentzian(ll, hz, method="binned")
    assert np.abs(direct - binned).max() < 0.01 * direct.max()
    with pytest.raises(ValueError):
        lorentzian(ll, hz, method="magic")


def test_phase_mixes_in_dispersion() -> None:
    ll = transitions(SpinSystem([3.0]), SF)
    hz = np.linspace(1510, 1490, 2001)
    absorptive = lorentzian(ll, hz)
    rotated = lorentzian(ll, hz, phase_deg=90.0)  # Re[(A + iD)·i] = −D
    np.testing.assert_allclose(rotated, -_dispersion(ll, hz), atol=1e-12)
    assert absorptive.argmax() == 1000


def _dispersion(ll: LineList, hz: np.ndarray) -> np.ndarray:
    hw = ll.width_hz[0] / 2
    x = hz - ll.freq_hz[0]
    return ll.intensity[0] / np.pi * x / (hw**2 + x**2)


def test_simulate_like_uses_experimental_axis() -> None:
    from conftest import synth_spectrum
    from nmrsuite.spin import simulate_like

    template = synth_spectrum([(7.0, 1.0)])
    sim = simulate_like(SpinSystem([7.0, 6.95], {(0, 1): 8.0}), template)
    assert sim.data.shape == template.data.shape
    np.testing.assert_allclose(sim.direct.ppm_axis(), template.direct.ppm_axis())
    assert sim.data.max() > 0


# --- performance (spec: exact for up to 12–14 spins) -----------------------------------


@pytest.mark.parametrize(("n", "budget_s"), [(8, 1.0), (10, 5.0)])
def test_simulation_speed(n, budget_s) -> None:
    rng = np.random.default_rng(0)
    shifts = list(1.0 + rng.uniform(0, 0.5, n))
    js = {(a, b): rng.uniform(-15, 15) for a in range(n) for b in range(a + 1, n) if b - a <= 3}
    t0 = time.perf_counter()
    ll = transitions(SpinSystem(shifts, js), SF)
    assert time.perf_counter() - t0 < budget_s
    assert ll.total == pytest.approx(n, rel=1e-3)


def test_memory_estimate_and_warning(monkeypatch) -> None:
    """Calibrated on the lab PC: 12 ¹H ≈ 185 MB, 14 ¹H ≈ 2.6 GB measured."""
    import importlib

    from nmrsuite.spin import estimated_memory_mb

    # the package also exports a *function* named simulate, which shadows the module
    sim = importlib.import_module("nmrsuite.spin.simulate")

    assert estimated_memory_mb(SpinSystem([1.0] * 12)) == pytest.approx(185, rel=0.1)
    assert estimated_memory_mb(SpinSystem([1.0] * 14)) == pytest.approx(2600, rel=0.1)
    het = SpinSystem([1.0, 2.0, -110.0], nuclei=["1H", "1H", "19F"])
    assert estimated_memory_mb(het) == pytest.approx(
        2 * estimated_memory_mb(SpinSystem([1.0, 2.0]))
    )
    monkeypatch.setattr(sim, "MEMORY_WARNING_MB", 0.0)
    with pytest.warns(ResourceWarning, match="exact simulation of 3 spins"):
        transitions(SpinSystem([1.0, 1.1, 1.2], {(0, 1): 7.0}), SF)


def test_line_frequency_gradients_match_finite_differences() -> None:
    """Hellmann–Feynman ∂ν/∂(ν_k, J_p) against central differences (non-degenerate
    strongly coupled system, so line order is stable under small steps)."""
    from nmrsuite.spin.simulate import kernel_for, transitions_from

    sys_ = SpinSystem([3.00, 3.031, 3.072], {(0, 1): 15.3, (0, 2): 4.1, (1, 2): 7.7},
                      linewidths_hz=[0.8, 1.1, 1.4])  # fmt: skip
    kern = kernel_for(sys_)
    nu = np.array(sys_.shifts_ppm) * SF
    jv = np.array([sys_.j_matrix[a, b] for a, b in kern.pairs])
    w = np.array(sys_.linewidths_hz)
    base = transitions_from(kern, nu, jv, w, SF, min_relative=0.0, gradients=True)
    h = 1e-4
    theta = np.concatenate([nu, jv])
    for t in range(theta.size):
        up, dn = theta.copy(), theta.copy()
        up[t] += h
        dn[t] -= h
        f_up = transitions_from(kern, up[:3], up[3:], w, SF, min_relative=0.0).freq_hz
        f_dn = transitions_from(kern, dn[:3], dn[3:], w, SF, min_relative=0.0).freq_hz
        np.testing.assert_allclose(base.dfreq[:, t], (f_up - f_dn) / (2 * h), atol=1e-6)
    # width shares sum to 1 for every line
    np.testing.assert_allclose(base.wweights.sum(axis=1), 1.0, atol=1e-12)


def test_lorentzian_jacobian_matches_finite_differences() -> None:
    from nmrsuite.spin.simulate import lorentzian_jacobian

    f = np.array([1500.0, 1507.3, 1512.1])
    ll = LineList(f, np.array([1.0, 0.6, 0.3]), np.array([0.9, 1.3, 0.7]), SF)
    hz = np.linspace(1520, 1490, 3001)
    phase = 7.0
    # θ = (shift of all lines together, width of all lines together)
    cf = np.column_stack([ll.intensity, np.zeros(3)])
    cw = np.column_stack([np.zeros(3), ll.intensity])
    spec, jac, dphi = lorentzian_jacobian(ll, hz, phase, cf, cw, chunk_elems=2000)
    np.testing.assert_allclose(spec, lorentzian(ll, hz, phase), atol=1e-12)
    h = 1e-5

    def shifted(df=0.0, dw=0.0, dp=0.0):
        return lorentzian(LineList(f + df, ll.intensity, ll.width_hz + dw, SF), hz, phase + dp)

    np.testing.assert_allclose(jac[:, 0], (shifted(df=h) - shifted(df=-h)) / (2 * h), atol=1e-6)
    np.testing.assert_allclose(jac[:, 1], (shifted(dw=h) - shifted(dw=-h)) / (2 * h), atol=1e-6)
    dp = np.rad2deg(h)
    np.testing.assert_allclose(dphi, (shifted(dp=dp) - shifted(dp=-dp)) / (2 * h), atol=1e-6)
