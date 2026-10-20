# NMR Analysis Suite — Product & Technical Spec

Oct 19, 2026 · @Lauren

## Overview

The goal is an open-source desktop NMR processing and analysis program that matches Mnova's everyday 1D/2D workflow first (Phase 1), then beats it on complex multiplet analysis (Phase 2) by fitting multiplets with a quantum-mechanical spin-system model instead of first-order pattern matching.

**Target users:** synthetic and physical organic chemists characterizing small molecules (roughly 100–1000 Da), writing SI, and running mechanistic or kinetic NMR studies.

**Goals**

- Process raw FIDs from the major vendors to publication-quality spectra with no Mnova step in between.
- Produce journal-format multiplet reports (ACS/RSC style) that are correct for second-order, overlapped and long-range-coupled systems.
- Make every processing step reproducible: a saved, scriptable processing history attached to each spectrum.

**Non-goals for v1**

- Spectrometer acquisition control, solid-state NMR, MRI/imaging.
- Full structure elucidation (CASE) and LC-MS integration.
- Enterprise database and LIMS features.

**Phases**

1. **Phase 1 — Parity:** import, processing and analysis features listed in the next three sections.
2. **Phase 2 — Multiplet engine:** spin-simulation fitting, J-coupling network solving, uncertainty reporting.
3. **Phase 3 — Extensions:** qNMR, reaction monitoring, prediction, automation.

## Data import and export

Import must read raw time-domain data and acquisition parameters directly, so processing never depends on vendor software.

| Format | Source | Priority | Notes |
| --- | --- | --- | --- |
| Bruker TopSpin (fid, ser, acqus, procs, pdata) | Bruker (our 500 and 600 MHz instruments) | P0 | Handle digital filter / group delay (GRPDLY, DSPFVS, DECIM); read processed 1r/2rr too |
| JCAMP-DX (FID and spectrum) | Any | P0 | Import and export |
| Varian/Agilent VnmrJ (fid, procpar) | Varian, Agilent | P1 | For collaborators' and literature data |
| nmrML, NMReDATA | Open standards | P1 | NMReDATA export pairs assignments with structure (SDF) |
| JEOL Delta (.jdf) | JEOL | P2 |  |
| Magritek Spinsolve / benchtop | Magritek, Nanalysis | P2 | For reaction monitoring |

**Requirements**

- Drag-and-drop an experiment folder or a parent folder of many experiments; auto-detect experiment type (1H, 13C, DEPT, COSY, HSQC, HMBC, NOESY, etc.) from pulse program and dimensions.
- Show all acquisition parameters in a searchable panel (SW, SFO1, O1P, TD, NS, D1, temperature, solvent, pulse program).
- Project file (single file) storing raw data, processing history, analysis results, structures and layout.
- Export: PNG/SVG/PDF (vector, publication sizes), CSV of peaks/integrals/multiplets, text report in journal format, JCAMP-DX, NMReDATA.

## Phase 1: core processing (Mnova parity)

Every step below must run automatically on import with sensible defaults, be adjustable interactively with live preview, and be recorded in the processing history.

### 1D time-domain

- **DC offset / drift correction** and digital-filter (group delay) removal.
- **Linear prediction:** backward LP to repair first points; forward LP to extend truncated FIDs (choose order and points).
- **Apodization:** exponential (LB), Gaussian (GB, LB), sine-bell and squared sine-bell (shift), Traf, trapezoid, first-point scaling. Show the window overlaid on the FID.
- **Zero filling** to 2^n with a recommended default (e.g. ×2 TD).
- **Solvent suppression (digital):** convolution-difference / Whittaker filter on the FID.

### 1D frequency-domain

- **Fourier transform** (complex, real, States/TPPI-aware).
- **Phase correction:** automatic (entropy-minimization and peak-symmetry methods, user-selectable) plus interactive zero-/first-order with pivot point dragged on the spectrum.
- **Baseline correction:** automatic (Whittaker smoother, Bernstein/polynomial, ablative/airPLS-type), and manual multipoint with spline or polynomial; option to exclude regions.
- **Referencing:** to TMS, residual solvent (built-in table of solvent shifts by nucleus), or a picked peak; indirect referencing of 13C/15N/19F/31P from 1H via Ξ ratios.
- **Spectral utilities:** reverse, shift, cut/crop, smoothing, magnitude and power modes, symmetrize.

### Multi-spectrum handling

- Stacked, superimposed and arrayed display; align spectra by a reference peak; difference spectra; normalize to an integral.
- Apply one processing template to a batch of spectra.

### 2D processing

- Per-dimension apodization, zero filling, LP (forward LP in F1 is essential for short t1).
- Acquisition modes: States, TPPI, States-TPPI, echo-antiecho, QF/magnitude.
- Phase correction per dimension; automatic 2D baseline correction; t1-noise reduction; symmetrization for COSY/TOCSY.
- Contour display with adjustable levels, positive/negative colours, 1D projections and external 1D traces on axes.
- Non-uniform sampling (NUS) reconstruction (IST or compressed sensing) as P1.

### Display and interaction

- Fast zoom/pan on million-point spectra; crosshair readout in ppm and Hz; distance (Hz) measurement between two cursors.
- Interactive integral and peak overlays; multiple spectra in tabs and in a single page layout for figure building.

## Phase 1: analysis (Mnova parity)

Phase 1 analysis must reproduce Mnova's peak, integral, deconvolution and first-order multiplet outputs closely enough that a user can switch for routine SI work.

### Peak picking

- Automatic picking with noise-based threshold (S/N estimated from a signal-free region), plus manual add/remove.
- Peak classification: compound, solvent, impurity, 13C satellite, artifact; known-impurity lookup (common solvents and grease by solvent, after Fulmer et al.).
- Peak table: ppm, Hz, intensity, width at half height, classification.

### Integration

- Automatic region detection and manual drag integration; integral curves with slope/bias correction.
- Normalize any integral to an integer proton count; sum integrals; integral table export.
- Deconvolution-based integration for overlapped regions (uses the fit below).

### Line fitting / deconvolution (GSD equivalent)

- Global spectral deconvolution of the full spectrum into Lorentzian/Gaussian/pseudo-Voigt lines.
- Region fitting with user-seeded lines; report position, width, area and fit residual.

### Multiplet analysis (first-order, parity level)

- Automatic multiplet detection from picked peaks and integrals.
- First-order classification: s, d, t, q, p, dd, dt, td, ddd, etc., with J values extracted from line spacings.
- Manual editing: split/merge multiplets, change pattern, set J values, mark as "m" with range.
- Journal-format report generated live, e.g. *¹H NMR (500 MHz, CDCl₃) δ 7.26 (dd, J = 8.1, 1.4 Hz, 1H), …*, with configurable style templates (ACS, RSC, Wiley, custom) and decimal places.
- J-matching across multiplets: flag coupling constants that should be shared between two protons.

### 13C and heteronuclear

- DEPT-135/90 editing aids (CH/CH₂/CH₃ labels); 13C report with C–F and C–P couplings reported as d/q with J.
- 19F and 31P with appropriate default referencing.

### 2D analysis

- 2D peak picking and volume integration; cross-peak tables.
- Link 2D cross-peaks to 1D multiplets and to atoms in a structure (COSY, HSQC, HMBC, NOESY correlations).

### Structure tools

- Draw or import a structure (MOL/SDF/SMILES), number atoms, and assign multiplets and 13C peaks to atoms; assignments flow into the report and NMReDATA export.

## Phase 2: complex multiplet analysis engine

The engine replaces line-spacing pattern matching with quantum-mechanical total-lineshape fitting: it builds a spin system, simulates it exactly, and fits shifts, couplings and linewidths to the experimental region, then reports values with uncertainties.

### Where first-order analysis fails

Automatic multiplet analysis (Mnova's included) reads J values from line spacings, which is only valid when Δν/J is large (roughly > 10). It breaks down in these cases, which are the engine's target set:

- **Second-order systems:** AB, ABX, ABC, AA′BB′, AA′XX′ — roofing, extra lines, spacings that are not J values.
- **Diastereotopic CH₂ groups**, e.g. ring methylenes in saturated N-heterocycles with large ²J and similar shifts.
- **Overlapped multiplets** from two or more protons within a few Hz of each other.
- **Unresolved small couplings** (⁴J, ⁵J, W-coupling) that broaden lines rather than split them.
- **Virtual coupling**, where a proton appears coupled to one it is not directly coupled to.
- **Heteronuclear couplings** (¹H–¹⁹F, ¹H–³¹P) mixed into ¹H patterns.

### Pipeline

1. **Define the spin system.** From an assigned structure (atoms, topology, chemical/magnetic equivalence from symmetry) or from user-selected multiplets when no structure is available. Include heteronuclei with nonzero coupling.
2. **Seed initial values.** Shifts from multiplet centres; J from (a) first-order extraction, (b) structure-based estimates (Karplus relations for ³J from a 3D conformer, typical ²J/⁴J tables), and (c) 2D constraints — a COSY or TOCSY cross-peak marks a J as nonzero, its absence bounds it as small.
3. **Simulate exactly.** Build the spin Hamiltonian, diagonalize, compute transition frequencies and intensities at the actual field (SFO1). Exact for up to about 12–14 spins; larger systems are split into weakly coupled fragments with the X-approximation at the boundaries. Optional ¹³C satellites.
4. **Fit the lineshape.** Minimize the residual between simulated and experimental spectrum over the selected region, fitting shifts, J values, per-spin linewidths, a residual phase term and a local baseline. Use multi-start or basin-hopping global optimization, because J-sign permutations and near-degenerate shifts create local minima.
5. **Quantify uncertainty.** Report each parameter ± from the fit covariance, with an optional Bayesian (MCMC) mode that exposes correlated or unidentifiable parameters (e.g. relative sign of two J values).
6. **Review and report.** Overlay experimental, simulated and residual traces; show a splitting tree; let the user lock or edit any parameter and refit live.

### Additional capabilities

- **Joint fitting across spectra:** one parameter set fitted to the same compound at two fields (e.g. 400 and 600 MHz), or to ¹H plus selective-decoupling, 1D TOCSY, J-resolved or pure-shift data. Field-dependent changes separate δ from J.
- **Coupling-network solver:** across all multiplets in a spectrum, match measured splittings into a consistent graph of proton pairs, flagging couplings that appear in only one partner.
- **Interactive simulation sandbox:** sliders for δ and J with live overlay, for teaching and for hand-guided fits.
- **Second-order-aware reporting:** δ and J reported from the simulation, labelled (e.g. "ABX system, values from simulation"), with the overlay figure exportable for the SI.

### Acceptance criteria

- On the benchmark set (see Validation), fitted J within ±0.1 Hz and δ within ±0.001 ppm of reference values for spin systems up to 8 coupled protons.
- Fit of an 8-spin region completes in under 5 s on a typical laptop; 12-spin in under 60 s.
- The engine never silently reports a first-order J for a region it classifies as second-order (Δν/J < 10); it either fits or flags.

## Other improvements and new features (Phase 3)

After the multiplet engine, the highest-value additions are reproducibility, quantitative and kinetic NMR, and automation of SI preparation.

| Feature | What it does | Why it beats current tools |
| --- | --- | --- |
| Scriptable processing history | Every step stored as a readable recipe (JSON + Python API); re-run on new data | Reproducible processing; shareable with co-authors and reviewers |
| SI builder | Batch-process a folder of compounds, generate spectra figures with structure, peak labels and reports into one PDF/DOCX | Removes hours of manual figure work per paper |
| qNMR module | Internal/external standard and ERETIC-style quantitation, purity with uncertainty, T1-aware recommendations | Integrates with deconvolution and multiplet fits for overlapped signals |
| Reaction monitoring | Arrayed or time-series spectra, auto-tracked integrals or fitted areas, concentration vs time export, simple kinetic model fitting | Supports mechanistic studies without spreadsheet hand-offs |
| dr / ee / ratio analysis | Deconvolution-based ratios for diastereomers, rotamers and CSA/derivatized ee samples with error bars | Overlapped diastereomer peaks are a common source of error |
| VT and exchange analysis | Lineshape fitting for two-site and multi-site exchange; coalescence and Eyring analysis | Rotamer and conformer barriers from VT series |
| Shift and J prediction | Plug-in predictor (empirical or ML) feeding the multiplet engine's initial guesses and flagging misassignments | Better starting points; consistency checks |
| Structure verification | Score a proposed structure against 1H/13C/HSQC data | Catches assignment errors before submission |
| Python plugin API | Users add processing steps, analyses and exporters | Extensible by the lab, not only the vendor |

## Architecture and tech stack

Recommended: a Python processing core with a Qt desktop GUI, Windows-first and cross-platform, with the spin-simulation kernel compiled for speed.

**Layers**

1. **Core library (no GUI):** data model (spectrum, dimension, processing step, analysis object), readers/writers, processing and analysis functions. Fully usable from scripts and Jupyter.
2. **Spin engine:** Hamiltonian builder, diagonalization, lineshape and fitting; isolated module with its own tests and a C++/Rust or Numba kernel.
3. **Application layer:** project files, processing history, undo/redo, background job queue for long fits.
4. **GUI:** spectrum canvas, panels (parameters, peaks, integrals, multiplets, structure), report and layout editor.
5. **Plugin API:** registered processing steps, analyses and exporters.

| Component | Choice | Notes |
| --- | --- | --- |
| Language | Python 3.12+ | Fast to build; matches scientific ecosystem |
| Numerics | NumPy, SciPy | FFT, linear algebra, optimization |
| Vendor I/O | nmrglue as a starting point | Reads Bruker, Varian/Agilent, NMRPipe, Sparky; extend for JEOL and others |
| Spin kernel | Numba first; C++ (pybind11) or Rust (PyO3) if needed | Eigen-decomposition of up to 2^14 states needs care with block-diagonalization by total Fz |
| Fitting | SciPy least\_squares + custom global search; emcee or similar for Bayesian mode |  |
| GUI | PySide6 (Qt 6) | LGPL; native-feeling on Windows |
| Plotting | pyqtgraph or VisPy | Smooth zoom on 64k–1M-point spectra and 2D contours |
| Chemistry | RDKit | Structure import, symmetry/equivalence, 3D conformers for Karplus estimates |
| Packaging | PyInstaller or Briefcase | Single installer for Windows; macOS later |

**Design rules**

- Processing is non-destructive: raw FID is never modified; the displayed spectrum is raw data + history.
- Every GUI action maps to a core-library call that is recorded in the history.
- Long computations run off the UI thread with progress and cancel.

## Validation, testing and benchmarks

The only direct comparison with Mnova is automatic multiplet analysis; processing is validated by its own unit and regression tests, not head-to-head. All test data come from the group's own Bruker 500 and 600 MHz spectra.

**Processing parity tests**

- A reference set of about 30 of the group's raw Bruker datasets (1H, 13C, DEPT, COSY, HSQC, HMBC, NOESY at 500 and 600 MHz), stored with a frozen expected-output file per dataset.
- Pass: output matches the stored expected output (peak positions, integrals) within numerical tolerance; any change to expected output requires an explicit, reviewed update.

**Multiplet engine tests**

- **Synthetic:** simulated spectra with known δ/J, added noise and linewidth, across AB → ABX → AA′BB′ → 8-spin systems. Measures accuracy, convergence rate and false local minima.
- **Experimental benchmark set:** about 20 of the group's compounds with hard multiplets — ortho-disubstituted aromatics (AA′BB′), vinyl groups (ABX), diastereotopic CH₂ in rings, fluorinated aromatics, overlapped aliphatic regions — each measured at both 500 and 600 MHz. Prioritize compounds whose Mnova multiplet reports have been wrong or needed manual correction.
- **Reference values:** for each compound, an expert manual analysis cross-checked by consistency between the 500 and 600 MHz data (J must not change with field; second-order patterns must).
- **Head-to-head (the one direct Mnova comparison):** on each benchmark compound, run Mnova's automatic multiplet analysis with default settings and the new program's automatic analysis with default settings; record multiplicity, δ and every J for both. Store the Mnova report text alongside the raw data.
- **Scoring:** per multiplet — multiplicity correct (y/n), δ error (ppm), J error (Hz, mean and max), and whether the program flagged it as uncertain. Report a summary table per compound and per pattern type (AB, ABX, AA′BB′, overlap, long-range).

**Engineering tests**

- Unit tests for each processing step against analytic or known outputs; regression tests on the reference datasets in CI.
- Performance budget: open and fully process a 64k-point 1H in under 1 s; 2D HSQC (1k × 256) in under 5 s.

## Milestones and open questions

Five milestones, each closed by a measurable gate; the spin engine (M4) can be prototyped as a standalone library in parallel with M2–M3, since it only needs 1D data.

&#91;embedded content: roadmap · 5 milestones, 4 gates\]

No calendar dates are set yet; each gate is the exit test for its milestone.

**Decisions and open questions**

- [x] Open source. License: BSD-3-Clause (compatible with NumPy, SciPy, nmrglue and RDKit; PySide6 is LGPL and used as a dynamic dependency).
- [x] Primary data: the department's Bruker 500 and 600 MHz spectrometers, so Bruker is the only P0 vendor reader.
- [x] Test and benchmark data: the group's own spectra.
- [x] Direct Mnova comparison: automatic multiplet analysis only.
- [x] Built with Claude Code from this spec (see Implementation brief).
- [ ] Desktop app (Python + Qt) or browser-based?
- [ ] Can the benchmark raw data be published in the repository, or must it stay private (shared test data vs a local-only folder)?
- [ ] Which Mnova version is the comparison baseline? Record it with every stored Mnova report.

## Implementation brief for Claude Code

Build the core library and the multiplet engine before any GUI, test-first, in the order below; each step ends with passing tests and a short README update.

**Repository layout**

```
nmr-suite/
  pyproject.toml          # BSD-3-Clause, Python >=3.12
  src/nmrsuite/
    io/                   # bruker.py, jcamp.py (readers/writers)
    processing/           # apodize.py, zerofill.py, fft.py, phase.py, baseline.py, reference.py
    analysis/             # peaks.py, integrate.py, deconvolve.py, multiplet_firstorder.py
    spin/                 # hamiltonian.py, simulate.py, fit.py, uncertainty.py
    report/               # journal-format multiplet reports (ACS default)
    history.py            # processing recipe (JSON), replayable
    cli.py                # nmrsuite process | multiplets | benchmark
  gui/                    # PySide6 app, built last
  tests/
    synthetic/            # generated spin systems with known δ/J
    data/                 # small committed Bruker test sets
  benchmark/
    compounds/<id>/       # raw 500 + 600 MHz data, reference.json, mnova_report.txt
    run_benchmark.py      # scores both programs, writes results table
```

**Build order**

1. Bruker reader (1D and 2D) with group-delay removal; JCAMP-DX import/export.
2. 1D processing chain with defaults and a replayable JSON history; CLI `nmrsuite process`.
3. Peak picking, integration, deconvolution; first-order multiplet analysis and ACS-style report.
4. Spin engine: Hamiltonian builder and exact simulation, validated against analytic AB and ABX solutions.
5. Lineshape fitting with global search and uncertainties; synthetic test suite.
6. Automatic second-order detection (Δν/J < 10) that routes regions to the spin engine.
7. Benchmark harness: CLI `nmrsuite benchmark` producing the head-to-head scoring table against stored Mnova reports.
8. Joint 500 + 600 MHz fitting.
9. GUI (PySide6 + pyqtgraph) over the finished core; then 2D processing.

**Conventions**

- Type hints throughout; pytest; ruff for lint and format; GitHub Actions CI on Windows and Linux.
- Core library has no GUI imports; every GUI action calls a core function recorded in the history.
- Units: shifts in ppm, couplings in Hz, field from the Bruker SFO1 parameter, never hard-coded.
- No Mnova code, file formats beyond its exported text reports, or UI assets are used; comparison is on outputs only.

**Definition of done for the multiplet engine**

- All synthetic tests pass (J within ±0.05 Hz on noise-free data).
- On the benchmark set, the summary table shows fewer multiplicity errors and lower mean J error than Mnova's automatic analysis, with J within ±0.1 Hz of reference for systems up to 8 coupled protons.
- Every second-order region is either fitted or explicitly flagged; none is reported with first-order J values silently.
