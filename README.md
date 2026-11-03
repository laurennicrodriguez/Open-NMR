# Open-NMR (nmrsuite)

Open-source NMR processing and analysis for small-molecule chemists: Mnova-level
everyday 1D/2D processing first, then quantum-mechanical multiplet fitting that
handles second-order, overlapped and long-range-coupled systems correctly.

The plan and requirements are in the
[spec](<NMR Analysis Suite — Product & Technical Spec.md>). Work follows its
implementation brief: core library and multiplet engine first, test-first, GUI last.

## Status

| Build step (spec) | State |
| --- | --- |
| 1. Bruker reader (1D, 2D) with group-delay removal; JCAMP-DX import/export | **done** |
| 2. 1D processing chain + replayable JSON history; `nmrsuite process` | **done** |
| 3. Peak picking, integration, deconvolution; first-order multiplets, ACS report | **done** |
| 4. Spin engine: Hamiltonian and exact simulation (AB/ABX validated) | **done** |
| 5. Lineshape fitting with global search and uncertainties | **done** |
| 6. Automatic second-order detection (Δν/J < 10) routing regions to the fit | next |
| 7–9. Benchmark harness vs Mnova, joint 500 + 600 MHz fits, GUI | not started |

### Lineshape fitting (step 5)

`nmrsuite.spin.fit_spin_region` fits a spin system to a spectral region. Shifts,
couplings and widths are numbers (fixed) or names (fitted); a shared name ties values,
which is how symmetry is written:

```python
from nmrsuite.spin import SpinModel, fit_spin_region

para = SpinModel(  # para-disubstituted benzene, AA'BB'
    shifts=["dA", "dB", "dB", "dA"],
    couplings={
        (0, 1): "Jo",
        (3, 2): "Jo",
        (0, 2): "Jp",
        (3, 1): "Jp",
        (0, 3): "Jm_a",
        (1, 2): "Jm_b",
    },
    linewidths="w",
)
start = {"dA": 7.745, "dB": 7.705, "Jo": 8.8, "Jp": 0.3, "Jm_a": 2.5, "Jm_b": 2.5, "w": 0.8}
fit = fit_spin_region(spectrum, 7.85, 7.60, para, start)
print(fit.table(500.3))  # values ± standard errors
```

- Bounded least squares on δ, J, widths and a residual phase; intensity scale and a
  linear baseline are solved exactly at each step (variable projection).
- Analytic Jacobian: line positions by Hellmann–Feynman, lineshape derivatives in closed
  form (intensity derivatives omitted while iterating; an exact numerical Jacobian at the
  minimum gives the standard errors).
- Multi-start global search (J ±30 %, small couplings sign-flipped), stopping early when
  two starts agree; stuck starts are abandoned.
- A ×2 zero-filled spectrum is fitted on every 2nd point, so residuals are independent.

Synthetic tests (AB, ABX, AA′BB′, AA′BB′C pyridinium, an 8-spin alkyl chain) recover
J within ±0.1 Hz and δ within ±0.001 ppm from deliberately wrong starts, and J within
±0.05 Hz on noise-free data (the spec's acceptance values).

On real data (LR-7-82-2, 500 MHz, CD₃CN) the pyridinium ring fits as AA′BB′C to 4.6 %
of the tallest line, giving δ 8.893 / 8.198 / 8.695 and J₂₃ 6.20, J₃₄ 7.89, J₂₄ 1.29,
J₃₅ 1.35, J₂₆ 1.42, J₂₅ 0.75 Hz; the para-aryl ring fits as AA′BB′ (J_o 8.65, J_p 0.45 Hz).
With regions drawn by hand, Mnova resolves only the para-H (tt, J = 7.9, 1.4 Hz) and reports
the other four signals as "m". A hand first-order reading of the line spacings gives 5.6, 7.1
and 7.9 Hz for the ortho couplings: H3's 7.1 Hz t is the average of J₂₃ and J₃₄, not a coupling.

**Known limitations**

- AA′BB′ spectra are identical under J_AA′ ↔ J_BB′ and J_AB ↔ J_AB′; the fit cannot (and
  no program can) tell these apart. Chemistry assigns them.
- Speed: a realistic 8-spin region (10 parameters) takes ~6.5 s on the lab PC, one
  thread (spec: < 5 s on a typical laptop). A fully asymmetric 8-spin system with 25
  free parameters is slow (~1 min per start) and needs good starting values; splitting
  into weakly coupled fragments is planned.
- Standard errors describe the fit to this spectrum; with very high S/N they are tiny
  and systematic lineshape errors dominate.

### Spin engine (step 4)

`nmrsuite.spin` simulates coupled spin-½ systems exactly:

```python
from nmrsuite.spin import SpinSystem, simulate, simulate_like, transitions

abx = SpinSystem([3.02, 2.98, 7.0], {(0, 1): 16.0, (0, 2): 8.0, (1, 2): 3.0}, linewidths_hz=0.8)
lines = transitions(abx, 500.0)  # frequencies, intensities, widths
ppm, y = simulate(abx, 500.0, 7.2, 2.8)  # spectrum
sim = simulate_like(abx, experimental)  # on an experimental spectrum's axis
```

- Hamiltonian built block by block (total F_z of the observed nucleus × heteronuclear
  spin states), so up to 14 coupled ¹H are exact. Heteronuclei (¹⁹F, ³¹P, …) couple
  through J·I_zI_z.
- Intensities normalized so each observed spin contributes 1; per-spin line widths.
- Validated against the analytic AB quartet and ABX (both parts), an independent
  brute-force 2ᴺ Hamiltonian, intensity sum rules, ¹H–¹⁹F couplings and field dependence.

Cost on the lab PC (one thread): 10 ¹H 0.1 s, 12 ¹H 1 s / 0.2 GB, 13 ¹H 7 s / 0.7 GB,
14 ¹H 50 s / 2.6 GB. `estimated_memory_mb()` predicts this, and simulations over
1.5 GB warn first. Fitting (step 5) works on multiplet regions, mostly ≤ 8 spins
(< 0.05 s per simulation).

### Analysis and reports (step 3)

`nmrsuite.analysis` and `nmrsuite.report`:

- **Peak picking** with noise from signal-free regions, sub-point positions and widths;
  **classification** as compound, solvent (each solvent's ¹H/¹³C pattern), impurity
  (common solvents and grease by NMR solvent, after Fulmer et al.; water always),
  ¹³C satellite (symmetric pairs at ±¹J(CH)/2) or artefact (spikes, broad humps).
- **Integration**: automatic regions, slope/bias correction, normalization, summed or
  deconvolution-based areas; CSV export.
- **Deconvolution**: pseudo-Voigt region fits with local baseline (seeded by the picked
  peaks or by hand) and whole-spectrum deconvolution; analytic areas.
- **First-order multiplets**: lines are clustered, then the coupling tree (J values with
  equivalent partners) that reproduces line positions *and* intensities is found and
  refined by least squares; unexplained clusters are `m` with a range. J matching across
  multiplets flags couplings without a partner. Manual `set_pattern`, `merge_multiplets`.
- **Reports**: ACS (default), RSC, Wiley or a custom `Style`, e.g.
  `¹H NMR (500 MHz, CDCl₃) δ 8.53 (d, J = 5.0 Hz, 1H), …, 1.24 (t, J = 7.8 Hz, 3H).`

```bash
nmrsuite multiplets "C:\data\HC-1-48\10"                      # report + tables
nmrsuite multiplets "C:\data\HC-1-48\10" --normalize 3.78=3 --style RSC --plain
```

Simulated multiplets (d … ddt, 13 patterns) are recovered with J within ±0.1 Hz.

**Known limitations**

- Proton counts without a formula are a suggestion: an incomplete multiplet can halve
  the scale. Use `--total-h` or `--normalize PPM=N` (Mnova also normalizes to a chosen
  multiplet).
- Second-order multiplets (AA′BB′, ABX, roofed AB) are not yet fitted; they come out as
  `m` or with apparent splittings. This is what the spin engine (steps 4–6) is for.
- ¹³C reports list shifts only; C–F/C–P doublets and DEPT editing labels come later.

### 1D processing (step 2)

Every step is a function `step(nmr, **params) -> new NMRData` that never modifies its
input and records itself, with all parameters, in `nmr.history`
(`nmrsuite.history`). A history is a recipe: save it as JSON, replay it on new data.

| Step | What it does |
| --- | --- |
| `remove_group_delay` | Bruker digital-filter delay, exact (step 1) |
| `dc_offset` | subtract the FID tail mean |
| `linear_prediction` | backward (repair first points) or forward (extend truncated FIDs) |
| `solvent_suppression` | convolution difference at any shift (default: carrier) |
| `apodize` | `em`, `gm`, `sine`, `qsine`, `traf`, `trap`; TopSpin definitions; `window()` for overlays |
| `zerofill` | to 2ⁿ, default ≥ 2 × TD |
| `fft` | complex FT into display order, ppm axis from O1/BF1 |
| `phase` / `autophase` | manual p0/p1 with pivot; automatic by peak symmetry + ACME entropy |
| `baseline` | `poly`, `airpls`, `whittaker` (ALS), manual `points`; exclusion regions |
| `reference` | residual solvent (Fulmer table), TMS, picked peak, offset, TopSpin's stored SR, Ξ ratios |
| `crop`, `magnitude`, `power`, `reverse` | spectral utilities |

`process_1d(raw)` runs the automatic chain: group delay → EM (LB 0.3 Hz ¹H, 1 Hz ¹³C) →
zero fill → FT → autophase → baseline (airPLS for ¹H, polynomial for ¹³C and for experiments
with negative peaks) → solvent referencing. Data-dependent
decisions (phase angles, reference shift) are recorded as results, so replaying a saved
recipe reproduces the spectrum exactly; `--rerun` re-decides them on new data.

On the group's 500 MHz data (15 1D sets: ¹H in CDCl₃/DMSO/MeOD, ¹³C, DEPT-135) the
automatic chain matches TopSpin's final spectra with correlation ≥ 0.98 for 14 of 15
and runs in 0.15–0.4 s per 64k-point spectrum (budget: 1 s).

¹H uses airPLS because the polynomial removes real signal on crowded spectra: LR-6-60's 18
aliphatic protons integrated to 10.8 H with it and 18.6 H with airPLS. Across 17 ¹H spectra
airPLS matched TopSpin's final spectrum at least as well (correlation ≥ 0.990).

**Known limitations**

- DEPT-135/APT phasing: with few peaks the first-order phase is underdetermined, and
  ACME without its negativity penalty is poorly conditioned (DEPT test set: 0.93).
- Referencing picks the residual solvent signal nearest its literature shift, preferring
  peaks with the solvent's own pattern (singlet CHCl₃, quintet DMSO-d₅ ...). When a
  singlet near 0 ppm (TMS — or silicone grease at ~0.07 ppm) disagrees by > 0.02 ppm,
  the solvent is used and a warning is recorded.
- 2D processing is not implemented yet (spec step 9); real (qf/qseq) FIDs are not
  transformed yet.

### Reading data (step 1)

- **Bruker TopSpin** (`nmrsuite.io.read_bruker`): raw `fid` and `ser` (int32 / float64,
  either byte order, 1 KiB row padding, partially acquired 2D), all acquisition
  parameters, direct-dimension mode (DQD/qsim/qseq/qf) and indirect mode (QF, TPPI,
  States, States-TPPI, echo-antiecho). Group delay from `GRPDLY`, or from the
  `DSPFVS`/`DECIM` table for older data.
- **TopSpin-processed data** (`read_bruker_processed`): `1r`/`1i` and tiled `2rr`, with
  ppm axes from `OFFSET`/`SW_p`/`SF`.
- **Group-delay removal** (`nmrsuite.processing.remove_group_delay`): exact
  frequency-domain shift; non-destructive (returns a new dataset).
- **Experiment detection** from pulse program and dimensions (1H, 13C{1H}, DEPT-135/90,
  APT, COSY, DQF-COSY, HSQC, HSQC-edited, HMBC, NOESY, ROESY, TOCSY, pseudo-2D, ...).
- **Folder scan** (`find_experiments`) for a parent folder of many experiments.
- **JCAMP-DX** (`nmrsuite.io.jcamp`): read XYDATA (AFFN and ASDF: SQZ/DIF/DUP with
  Y-checks), XYPOINTS and NTUPLES (real + imaginary pages); write real spectra as XYDATA
  and complex FIDs/spectra as NTUPLES, DIFDUP-compressed or AFFN.

### Conventions

- Shifts in ppm, couplings in Hz, spectrometer frequencies in MHz from the data
  (`SFO1`, `SF`), never hard-coded.
- Frequency-domain arrays are in display order: index 0 is the left edge (highest ppm).
- 2D arrays are `[F1, F2]`; the last axis is the directly detected one.
- Bruker FFT convention, validated against TopSpin on our 500 MHz data:
  `spectrum = fftshift(fft(conj(fid)))`, phase(i) = PHC0 + PHC1·i/SI degrees.

## Install (development)

Python 3.12 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

## Use

```python
from nmrsuite.history import replay, save_recipe
from nmrsuite.io import read_bruker
from nmrsuite.processing import apodize, autophase, fft, phase, process_1d, zerofill

raw = read_bruker(r"C:\data\HC-1-48\10")
print(raw.experiment, raw.summary())

spec = process_1d(raw)  # automatic chain
spec = phase(spec, p0=2.0, pivot_ppm=7.26)  # touch-up, also recorded
save_recipe(spec.history, "recipe.json")  # replay(other_raw, load_recipe(...))
ppm, intensity = spec.direct.ppm_axis(), spec.data.real
```

```bash
nmrsuite info  "C:\data\HC-1-48"                       # parameters of every experiment
nmrsuite convert "C:\data\HC-1-48\10" out.jdx           # raw FID to JCAMP-DX
nmrsuite process "C:\data\HC-1-48\10" -o spec.jdx --save-recipe recipe.json
nmrsuite process "C:\data\batch" --recipe recipe.json --out-dir out --format csv
```

## Tests

```bash
pytest
ruff check . && ruff format --check .
```

- **Unit tests** use synthetic FIDs with known answers (`tests/conftest.py`,
  `tests/test_processing.py`).
- **Committed real data** (`tests/data/chem2081`): four ¹H spectra of simple compounds
  from the Chem 2081 teaching set, cleared for public use and trimmed to `acqus`, `fid`
  and TopSpin's `procs`/`1r`. `tests/test_realdata.py` checks the automatic chain
  against TopSpin and against frozen outputs in `expected.json`; change those only on
  purpose, with `python tests/data/make_expected.py`, and review the diff.
- **Local-only data**: `tests/test_localdata.py` runs the same checks on any folders
  in `NMRSUITE_LOCAL_DATA` (separated by `;` on Windows, `:` elsewhere). Other group
  data stay out of the repository.

```powershell
$env:NMRSUITE_LOCAL_DATA = "D:\nmr\set1;D:\nmr\set2"; pytest tests/test_localdata.py -rs
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
