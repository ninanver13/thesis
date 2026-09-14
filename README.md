# TeV Halo Detectability Study: HESS J1837-069 / PSR J1838-0655

Master's thesis project on the detectability and morphology of the candidate
TeV halo around PSR J1838-0655 / HESS J1837-069, using real archival data
(Fermi-LAT, HAWC, H.E.S.S.) together with forward-simulated observations for
next-generation instruments (CTA-South, SWGO).

The halo is modeled two ways throughout this repo:

- **Analytic (Abeysekara+2017)** — a fast, closed-form profile used for
  quick parameter scans.
- **Physical (cooling + spin-down + proper motion)** — a first-principles
  model that evolves the injection spectrum, IC cooling, and diffusion
  epoch-by-epoch; slower, but the one the thesis actually reports results
  from. All final numbers are calibrated against the real HESS-2006 measured
  size (0.12° × 0.05°, position angle 86.3°) rather than a generic benchmark.

## Repository structure

> Adjust this section to match your actual layout — the paths below are a
> suggested organization; rename/move as needed and update the tree.

```
.
├── thesis/
│   ├── thesis8.tex          # main thesis source
│   ├── thesis.bib
│   └── figures/
├── code/
│   ├── CT2-hess10-physical-model-fixed_x.py   # CTA: benchmark-source sims + scans
│   └── PeV-hess10-physical-model-fixed_x.py   # SWGO: benchmark-source sims + scans
├── notebooks/
│   └── TeV_Halo_Forward_Model_HESSJ1837_irfs_sigmap_fixed_2_PLUS_age_offset_scan.ipynb
│       # synthetic population generator + per-instrument (CTA/SWGO) detection
│       # pipeline, generalized from CT2/PeV to run over many systems
└── README.md
```

## What's in each piece

### `thesis/`
The written thesis: theoretical background (cosmic rays, PWNe, TeV halos),
instruments and methods, results (CTA/SWGO simulations, Fermi-LAT/HAWC/HESS
data analysis, joint SED fits), discussion, and conclusion. See in-repo
session notes for the change history if you keep them alongside the source.

### `code/CT2-*.py` and `code/PeV-*.py`
Single-system (the HESS J1837-069 benchmark) simulation and analysis
scripts for CTA and SWGO respectively. Each script:

1. Builds the instrument dataset (IRFs, background, geometry) for the
   benchmark source position.
2. Calibrates the physical model's diffusion coefficient (`D100`) against
   the real HESS-2006 measured size, and builds the primary halo template.
3. Runs the core hypothesis tests (halo-only vs. background;
   background+PWN+halo vs. background — the total-model H0/H1 test used as
   the primary detection result, with the PWN-marginal comparison reported
   separately as a diagnostic).
4. Runs a full suite of sensitivity scans (E_dat, age, distance, D0,
   halo size, offset, offset×background, 2D size×offset, age×D0, ...), all
   built on the same calibrated physical model rather than the fast
   analytic approximation.

Outputs (figures, `.npz` caches) are written to
`cta_output_true_halo_null/` / the script's configured `OUTPUT_DIR`.

### `notebooks/TeV_Halo_Forward_Model_*.ipynb`
1. Samples a synthetic population of TeV-halo systems from physically
   motivated priors and forward-simulates their gamma-ray counts.
2. Fits the real HESS J1837-069 broadband SED (Naima MCMC: IC-only and
   joint synchrotron+IC+hadronic) and compares the synthetic population
   against it.
3. Swaps in real per-instrument response functions (CTA-South, SWGO,
   Fermi-LAT) in place of the notebook's original generic instrument model.
4. Ports the CT2/PeV detection-significance pipeline (Section 14) so the
   same scan suite (age, size×offset, E_dat, D0, offset, offset×background,
   halo size, age×offset, ...) can be run per synthetic system or applied to
   the HESS J1837-069 benchmark specifically — all built on the physical
   model via a shared `build_scan_halo_template()` helper.

## Requirements

- Python 3.10+
- [gammapy](https://gammapy.org/) (datasets, IRFs, modeling/fitting)
- astropy, numpy, scipy, matplotlib, pandas
- [naima](https://naima.readthedocs.io/) (radiative-model MCMC fits, notebook only)
- Jupyter (for the notebook)

```bash
pip install gammapy astropy numpy scipy matplotlib pandas naima jupyter
```

IRF files, the diffuse emission template, and the Fermi-LAT SED (`.npy`) are
not included here — point the relevant `IRF_FILE` / `DIFFUSE_FILE` / SED path
variables near the top of each script/notebook at your own copies.

## Running

**Benchmark-source CTA/SWGO scans:**
```bash
python code/CT2-hess10-physical-model-fixed_x.py
python code/PeV-hess10-physical-model-fixed_x.py
```
Both cache scan results to an `.npz` file and skip re-running a scan whose
cache already exists, unless `FORCE_RERUN`/`FORCE_HALO_RERUN` is set.

**Population synthesis + per-instrument pipeline:**
Open the notebook in Jupyter and run top to bottom, or call
`run_population_scan_suite(...)` directly (Section 14.7/14.8) once the
per-system priors and instrument datasets are configured. The full scan
suite is expensive — start with `systems=labels.iloc[:N]` for a small N
before scaling to the whole population.

## Physical model notes

- The diffusion coefficient `D100` (at 100 TeV) is always calibrated against
  a *target size* via root-finding (`calibrate_physical_D100_cm2_s`), rather
  than assumed — this replaces the earlier generic Abeysekara+2017
  benchmark value everywhere it mattered for the reported results.
- Ellipticity (axis ratio 0.05/0.12) and position angle (86.3°, East of
  North) come from the HESS 2006 (Aharonian et al.) morphological fit and
  are held fixed across all scans and the primary template.
- Scans that vary the halo's *position* only (offset, offset×background)
  reuse the already-calibrated `D100` rather than recalibrating at every
  point, since size doesn't change; scans that vary *size* (directly, or via
  age/distance/E_dat/D0) recalibrate at each grid point.

## License / citation

_Add your preferred license and a citation entry here._
