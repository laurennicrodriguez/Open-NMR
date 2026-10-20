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
| 2. 1D processing chain + replayable JSON history; `nmrsuite process` | next |
| 3–9. Analysis, spin engine, fitting, benchmark, GUI | not started |

### What works now

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
from nmrsuite.io import read_bruker
from nmrsuite.processing import remove_group_delay

fid = read_bruker(r"C:\data\HC-1-48\10")
print(fid.experiment, fid.summary())
fid = remove_group_delay(fid)
```

```bash
nmrsuite info  "C:\data\HC-1-48"              # parameters of every experiment found
nmrsuite convert "C:\data\HC-1-48\10" out.jdx  # raw FID to JCAMP-DX
```

## Tests

```bash
pytest
ruff check . && ruff format --check .
```

Unit tests use small synthetic datasets generated in `tests/conftest.py` and need no
data files. `tests/test_localdata.py` also runs regression tests on real spectra
(load every experiment, reproduce TopSpin's `1r` from the raw FID, JCAMP round trip)
when `NMRSUITE_LOCAL_DATA` points at one or more folders of Bruker experiments
(separated by `;` on Windows, `:` elsewhere). Real data are not committed until the
spec's open question on publishing benchmark data is settled.

```powershell
$env:NMRSUITE_LOCAL_DATA = "D:\nmr\set1;D:\nmr\set2"; pytest tests/test_localdata.py -rs
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
