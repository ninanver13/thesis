# # Fermi-LAT ROI analysis of HESS J1837-069 (4FGL J1836.5-0651e)


# %%
# ============================================================
# Imports and analysis constants
# ============================================================
from pathlib import Path
import os
import subprocess
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fermipy.gtanalysis import GTAnalysis

# Query/ROI center — HESS J1837-069 / 4FGL J1836.5-0651e region.
RA_TARGET  = 279.41
DEC_TARGET = -6.95

# LAT selections — match the values used for the data download.
EMIN       = 10000.0
EMAX       = 500000.0
ROI_RADIUS = 15.0
ZMAX       = 105.0
EVCLASS    = 128
EVTYPE     = 3
IRFS       = "P8R3_SOURCE_V3"

# 4FGL catalog name for the extended PWN.

TARGET_CATALOG_NAME = "FGES J1836.5-0651"

# ============================================================
# Project directories
# ============================================================
BASE      = Path("/project/ls-gruen/users/asu.uenver/fermi")
RAW       = BASE / "data_hess2"
PH_DIR    = RAW  / "PH"
SC_DIR    = RAW  / "SC"
DATA_DIR  = RAW
OUT_DIR   = BASE / "output_hess1837_0651"
ROI_OUT   = OUT_DIR / "roi_analysis"
PLOT_DIR  = OUT_DIR / "plots"

for directory in [PH_DIR, SC_DIR, DATA_DIR, OUT_DIR, ROI_OUT, PLOT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# Logging and figure-saving utilities
# ============================================================
LOG_FILE = OUT_DIR / "analysis-hess1837_0651.log"

def log(msg: str) -> None:
    print(msg)
    with LOG_FILE.open("a") as fh:
        fh.write(msg + "\n")

def savefig(name: str, dpi: int = 150) -> Path:
    path = PLOT_DIR / f"{name}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log(f"[plot] Saved: {path}")
    return path

def save_values(name: str, **kwargs) -> Path:
    path = OUT_DIR / f"{name}.txt"
    with path.open("w") as fh:
        for k, v in kwargs.items():
            line = f"{k} = {v}"
            fh.write(line + "\n")
            print(line)
    log(f"[values] Saved: {path}")
    return path

log(f"HESS J1837-069 query center: RA={RA_TARGET:.3f} deg, Dec={DEC_TARGET:.3f} deg")

# ============================================================
# Input file paths
# ============================================================

FT2          = SC_DIR  / "L260715145335F357373F10_SC00.fits"
PHOTON_FILES = sorted(PH_DIR.glob("*PH*.fits"))

PH_LIST      = RAW      / "downloadinfo.txt"
FT1_SELECTED = DATA_DIR / "h1837_binned_filtered.fits"
FT1_GTI      = DATA_DIR / "h1837_binned_filtered_gti.fits"

log(f"Photon files found: {len(PHOTON_FILES)}")
log(f"FT2 exists: {FT2.exists()}")
if not FT2.exists():
    log("WARNING: update FT2 above to point at your downloaded SC file.")

# ============================================================
# Helper: run a shell command
# ============================================================
RUN_COMMANDS = True

def run_command(cmd):
    cmd = [str(item) for item in cmd]
    log("$ " + " ".join(cmd))
    if RUN_COMMANDS:
        subprocess.run(cmd, check=True)

if PHOTON_FILES:
    PH_LIST.write_text("\n".join(str(p.resolve()) for p in PHOTON_FILES) + "\n")
    log(f"Wrote photon-file list: {PH_LIST}")
else:
    log("No photon files found — place LAT PH files in: " + str(PH_DIR))


# ## 1. Standard LAT event selection

if FT1_SELECTED.exists():
    log(f"[skip] {FT1_SELECTED} already exists — reusing, not re-running gtselect.")
else:
    run_command([
        "gtselect",
        f"infile=@{PH_LIST}",
        f"outfile={FT1_SELECTED}",
        f"ra={RA_TARGET}",
        f"dec={DEC_TARGET}",
        f"rad={ROI_RADIUS}",
        "tmin=INDEF",
        "tmax=INDEF",
        f"emin={EMIN}",
        f"emax={EMAX}",
        f"zmax={ZMAX}",
        f"evclass={EVCLASS}",
        f"evtype={EVTYPE}",
    ])

if FT1_GTI.exists():
    log(f"[skip] {FT1_GTI} already exists — reusing, not re-running gtmktime.")
else:
    run_command([
        "gtmktime",
        f"scfile={FT2}",
        'filter=(DATA_QUAL>0)&&(LAT_CONFIG==1)',
        "roicut=no",
        f"evfile={FT1_SELECTED}",
        f"outfile={FT1_GTI}",
    ])


# ## 2. Confirm the Fermipy config points at the selected event file


CONFIG_FILE = BASE / "hess1837069.yaml"

if not CONFIG_FILE.exists():
    raise FileNotFoundError(
        f"Config not found: {CONFIG_FILE}\n"
        "Create hess1837069.yaml there (adapted from hess1825137.yaml, "
        "with the ROI centre updated to RA=279.41, Dec=-6.95) before running."
    )

with CONFIG_FILE.open() as fh:
    cfg = yaml.safe_load(fh)
cfg_evfile = cfg.get("data", {}).get("evfile")
if cfg_evfile != str(FT1_GTI):
    log(f"WARNING: config evfile ({cfg_evfile}) != FT1_GTI ({FT1_GTI}). "
        "Update hess1837069.yaml data.evfile to match.")


# ## 3. Fermipy helper functions

def initialize(gta):
    print("Setting up ...", flush=True)
    gta.setup()
    print("Optimizing (initial pass) ...", flush=True)
    gta.optimize()
    # Standard free_sources pattern (matches supervisor's StdFermiAnalysis.pdf):
    # Free norms within 3° of ROI centre, both diffuse components,
    # all high-TS sources (TS>400); fix low-TS sources.
    gta.free_sources(distance=3.0, pars='norm')
    gta.free_source('galdiff')
    gta.free_source('isodiff')
    gta.free_sources(minmax_ts=[400, None], pars='norm')
    gta.free_sources(minmax_ts=[None, 400], free=False, pars='norm')
    gta.setup()   # rebuild likelihood with updated free/fixed states
    print("Optimizing (with free_sources) ...", flush=True)
    gta.optimize()
    gta.print_roi()

def produce_tsmap(gta, label):
    model = {'SpatialModel': 'PointSource', 'Index': 2.0, 'SpectrumType': 'PowerLaw'}
    print(f"TS map: {label} ...")
    m = gta.tsmap(label, model=model)
    print(f"TS map done: {label}")
    return m

def produce_residmap(gta, label):
    model = {'SpatialModel': 'PointSource', 'Index': 2.0}
    gta.residmap(label, model=model)

def produce_sed(gta, source_name, outfile):
    print(f"SED: {source_name} ...")
    sed = gta.sed(
        source_name,
        outfile=outfile,
        make_plots=True,
        write_fits=True,
        write_npy=True,
    )
    print(f"SED fluxes: {sed['eflux']}")
    print("SED done.")
    return sed

def delete_source(gta, catalog_name):
    """Remove the target source from the model and re-optimise."""
    print(f"Deleting {catalog_name} ...")
    gta.delete_source(catalog_name)
    gta.optimize()
    gta.print_roi()
    print("Source deleted.")

def find_sources(gta):
    print("Finding new sources ...")
    model = {'Index': 2.0, 'SpatialModel': 'PointSource'}
    srcs = gta.find_sources(model=model, sqrt_ts_threshold=4.0, min_separation=0.5)
    gta.print_roi()
    print("find_sources done.")
    return srcs


# ## 4. ROI likelihood analysis

log("\n" + "=" * 60)
log("ROI ANALYSIS — HESS J1837-069 (4FGL J1836.5-0651e)")
log("=" * 60)

gta = GTAnalysis(str(CONFIG_FILE), logging={"verbosity": 3})
initialize(gta)
gta.write_roi("hess1837_initial_fit", make_plots=True)

# Step 1: all-source baseline maps
produce_tsmap(gta, label="hess1837_allsource_ts")
produce_residmap(gta, label="hess1837_allsource_resid")

# Step 2: SED of the extended nebula (still in model)
produce_sed(gta, TARGET_CATALOG_NAME,
            outfile=str(ROI_OUT / "hess1837_sed.fits"))

# Step 3: delete the nebula, remap to reveal residual/additional emission
delete_source(gta, TARGET_CATALOG_NAME)
initialize(gta)
produce_tsmap(gta, label="hess1837_postdelete_ts")
produce_residmap(gta, label="hess1837_postdelete_resid")

# Step 4: blind source search in residuals
find_sources(gta)

# Step 5: save final state
gta.write_roi("hess1837_final_fit", make_plots=True)

log("\n=== Analysis complete. ===")
log(f"Plots  : {PLOT_DIR}")
log(f"Results: {ROI_OUT}")
log(f"Log    : {LOG_FILE}")