from __future__ import annotations
import logging, warnings, pathlib
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

from astropy.coordinates import SkyCoord
from scipy.stats import norm
from scipy.interpolate import interp1d

import gammapy
from gammapy.irf import load_irf_dict_from_file
from gammapy.maps import WcsGeom, Map, MapAxis
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    PowerLawSpectralModel,
    PowerLawNormSpectralModel,
    PointSpatialModel,
    GaussianSpatialModel,
    TemplateSpatialModel,
    SkyModel,
    FoVBackgroundModel,
    Models,
)
from gammapy.modeling import Fit
from gammapy.data import Observation
from gammapy.datasets import MapDataset
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.estimators import ExcessMapEstimator, FluxPointsEstimator

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)
print(f"Gammapy version: {gammapy.__version__}")
print("✓ Imports complete")


IRF_FILE     = "/project/ls-gruen/users/asu.uenver/templates/Prod5-South-20deg-SouthAz-14MSTs37SSTs.18000s-v0.1.fits"
DIFFUSE_FILE = "/project/ls-gruen/users/asu.uenver/templates/IEM_base_v2.fits"
LIVETIME     = 100 * u.hour
POINTING_OFF = 0.5 * u.deg  

SOURCE_CENTER = SkyCoord(ra=279.5130417 * u.deg, dec=-6.9259444 * u.deg, frame="icrs")
FOV           = 8.0 * u.deg
BINSZ         = 0.04           # deg / pixel
E_MIN         = 1.0 * u.TeV
E_REF       = 1.0 * u.TeV   
E_MAX         = 1000.0 * u.TeV
N_ENERGY      = 20


TEST_FLUX      = 1.779e-11 * u.Unit("cm-2 s-1 TeV-1")  # HESS 2018 (HGPS Tablo, dedike spektral fit)
SPECTRAL_INDEX = 2.54

#  TeV halo parameters 
HALO_ALPHA    = 2.54           # HESS J1837-069 mean spectral index (HESS 2018)
HALO_E0       = 0.95 * u.TeV  # HGPS SPECTRAL pivot energy (HESS 2018) -- normalises
HESS2006_SIGMA_MAJOR_DEG = 0.12
HALO_THETA_D0 = HESS2006_SIGMA_MAJOR_DEG * u.deg  # HESS 2006 (Aharonian) measured size
HALO_THETA_E0 = HALO_E0
HALO_DELTA    = 0.33            # kept only for the legacy analytic scan sections (see note near D0_cm2_s below)
HALO_E_MIN    = 1.0 * u.TeV
HALO_E_MAX    = 1000.0 * u.TeV
HALO_N0       = 2e-14 * u.Unit("cm-2 s-1 TeV-1 sr-1")
HALO_AMPLITUDE = 2.00e-11 * u.Unit("cm-2 s-1 TeV-1")  # HGPS: diff. flux at E0 (HESS 2018)

HALO_AXIS_RATIO     = 0.05 / 0.12     # HESS 2006 minor/major axis
HALO_POSITION_ANGLE = 86.3            # deg, East of North (ICRS) -- ellipse tilt
HALO_OFFSET_DEG     = 0.5 * HALO_THETA_D0.to_value(u.deg)
HALO_OFFSET_PA_DEG  = 270.0           # deg, East of North (ICRS) -- offset direction (pure right, no up)

DELTA_DIFF  = 0.33
D0_cm2_s    = 1.0e27           # cm²/s  (legacy Abeysekara+2017 value; scan sections only, see note above)
T_AGE_YR    = 23e3             # yr  ← PSR J1838-0655 characteristic age
DIST_KPC    = 6.6              # kpc ← assumed, via star-cluster association (HESS 2018)
E_DAT_TEV   = 10.0             # TeV  (electron reference energy for diffusion scan, independent of HALO_THETA_E0)
KPC_TO_CM   = 3.0857e21
YR_TO_S     = 3.1557e7
dist_cm     = DIST_KPC * KPC_TO_CM
t_age_s     = T_AGE_YR * YR_TO_S

HALO_TAU0_KYR         = 5.0     # kyr, initial spin-down timescale (n=3 braking)
HALO_U_TOTAL_EV_CM3   = 1.0     # eV/cm^3, total seed-photon energy density for IC cooling (CMB+FIR+NIR)
HALO_CUTOFF_TEV       = 100.0   # TeV, injection-spectrum cutoff energy
HALO_INJECTION_INDEX  = HALO_ALPHA  # reuse the HGPS-fitted index as the electron injection-index proxy
HALO_V_TRANSVERSE_KMS = 0.0     # km/s, pulsar transverse velocity (unmeasured here -> no proper-motion offset)
HALO_MOTION_PA_DEG    = 0.0     # deg, East of North -- proper-motion direction (unused while v=0)
HALO_N_TIME_SLICES    = 48      # lookback-time resolution of the injection-epoch mixture

SEED         = 42
BKG_NORM_SCALE = 1.0

OUTPUT_DIR   = pathlib.Path("cta_output_true_halo_final")   # all figures / fits / npz saved here
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE   = OUTPUT_DIR / "cta_halo_pwn_scans.npz"
FORCE_RERUN  = True

print("✓ Configuration set")
print(f"   Source        : HESS J1837-069 / PSR J1838-0655")
print(f"   Source centre : {SOURCE_CENTER.icrs}")
print(f"   Livetime      : {LIVETIME}")
print(f"   E range       : {E_MIN} – {E_MAX}")
print(f"   θ_d0          : {HALO_THETA_D0}  (at E={HALO_THETA_E0}, decoupled from spectral pivot {HALO_E0})")
print(f"   Distance      : {DIST_KPC} kpc")
print(f"   Age           : {T_AGE_YR/1e3:.0f} kyr  (PSR J1838-0655 spin-down)")
print(f"   E_dat         : {E_DAT_TEV} TeV")
print(f"   D₀            : {D0_cm2_s:.0e} cm²/s")


irfs = load_irf_dict_from_file(IRF_FILE)
print("✓ IRFs loaded:", list(irfs.keys()))

diffuse_emission_map = Map.read(DIFFUSE_FILE)
print("✓ Diffuse emission map loaded")


def build_cta_dataset(center, name="cta_dataset"):
    """Build a single CTA MapDataset from IRFs for the given sky centre."""
    center_icrs = center.icrs

    energy_axis = MapAxis.from_edges(
        np.geomspace(E_MIN.to_value(u.TeV), E_MAX.to_value(u.TeV), N_ENERGY + 1),
        unit="TeV", name="energy", interp="log",
    )
    energy_axis_true = MapAxis.from_edges(
        np.geomspace(0.1, 500, 2 * N_ENERGY + 1),
        unit="TeV", name="energy_true", interp="log",
    )
    geom = WcsGeom.create(
        skydir=center_icrs, binsz=BINSZ,
        width=(FOV.to_value(u.deg), FOV.to_value(u.deg)),
        frame="icrs", axes=[energy_axis],
    )
    pointing = SkyCoord(center_icrs.ra, center_icrs.dec + POINTING_OFF, frame="icrs")
    obs      = Observation.create(pointing=pointing, livetime=LIVETIME, irfs=irfs)

    empty      = MapDataset.create(geom, energy_axis_true=energy_axis_true, name=name)
    maker      = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
    mask_maker = SafeMaskMaker(methods=["offset-max"], offset_max=4.0 * u.deg)

    dataset = maker.run(empty, obs)
    dataset = mask_maker.run(dataset, obs)
    return dataset

dataset  = build_cta_dataset(SOURCE_CENTER, name="cta_pv_halo")
geom     = dataset.counts.geom
center_icrs = SOURCE_CENTER.icrs

print("✓ CTA dataset ready")
print(f"   Background total counts : {dataset.background.data.sum():.0f}")
print(f"   Geometry               : {geom.data_shape}")


fig, ax = plt.subplots(1, 1, figsize=(7, 5))
bkg_2d = dataset.background.smooth(0.05 * u.deg).reduce_over_axes()
bkg_2d.plot(ax=ax, add_cbar=True, stretch='linear', cmap='magma')
try:
    ax.images[-1].colorbar.set_label('Expected background counts  [dimensionless]', fontsize=9)
except Exception:
    pass
ax.set_title(
    f'{dataset.name}\nBackground (energy-integrated, smoothed 0.05°)',
    fontsize=10
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_background_map.png", dpi=150, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# Panel 0: 2D background map (linear)
ax0 = fig.add_subplot(gs[0, 0], projection=dataset.counts.geom.to_image().wcs)
bkg_2d = dataset.background.smooth(0.05 * u.deg).reduce_over_axes()
bkg_2d.plot(ax=ax0, add_cbar=True, stretch='linear', cmap='magma')
try:
    ax0.images[-1].colorbar.set_label('Expected background counts  [dimensionless]', fontsize=9)
except Exception:
    pass
ax0.set_title(f'{dataset.name}\nBackground (energy-integrated, linear)', fontsize=10)

# Panel 1: sqrt stretch (reveals faint structure)
ax1 = fig.add_subplot(gs[0, 1], projection=dataset.counts.geom.to_image().wcs)
bkg_2d_sqrt = dataset.background.smooth(0.05 * u.deg).reduce_over_axes()
bkg_2d_sqrt.plot(ax=ax1, add_cbar=True, stretch='sqrt', cmap='magma')
try:
    ax1.images[-1].colorbar.set_label('Expected background counts  [dimensionless]', fontsize=9)
except Exception:
    pass
ax1.set_title(f'{dataset.name}\nBackground (energy-integrated, sqrt stretch)', fontsize=10)

# Panel 2: Differential energy spectrum
ax2 = fig.add_subplot(gs[1, 0])
bkg_counts = dataset.background.data.sum(axis=(1, 2))
e_centers  = dataset.background.geom.axes['energy'].center.to(u.TeV).value
e_widths   = dataset.background.geom.axes['energy'].bin_width.to(u.TeV).value
ax2.loglog(e_centers, bkg_counts / e_widths, marker='o', ms=4, label=dataset.name)
ax2.set_xlabel('Reconstructed Energy  [TeV]')
ax2.set_ylabel('Background rate  [counts TeV⁻¹]')
ax2.set_title('Background energy spectrum\n(spatially integrated, differential)')
ax2.axvline(1.0, ls=':', color='gray', alpha=0.6, label='1 TeV')
ax2.legend(fontsize=9)
ax2.grid(True, which='both', alpha=0.3)


ax3 = fig.add_subplot(gs[1, 1])
bkg_nominal = bkg_counts / e_widths
ax3.fill_between(e_centers, bkg_nominal * 0.80, bkg_nominal * 1.20,
                 alpha=0.15, color="steelblue", label="±20% uncertainty band")
ax3.fill_between(e_centers, bkg_nominal * 0.90, bkg_nominal * 1.10,
                 alpha=0.35, color="steelblue", label="±10% uncertainty band")
ax3.loglog(e_centers, bkg_nominal, color="navy", lw=2.5, label="Nominal (×1.0)")
ax3.set_xlabel('Reconstructed Energy  [TeV]')
ax3.set_ylabel('Background rate  [counts TeV⁻¹]')
ax3.set_title(f'Background Uncertainty Bands\n(Dataset: {dataset.name})')
ax3.legend(fontsize=9, title='Bkg uncertainty')
ax3.grid(True, which='both', alpha=0.3)
ax3.text(0.02, 0.04, f'Current BKG_NORM_SCALE = {BKG_NORM_SCALE}',
         transform=ax3.transAxes, fontsize=9,
         bbox=dict(boxstyle='round', fc='white', alpha=0.7))

plt.suptitle('Background Model — Diagnostic Overview (CTA)', fontsize=13, fontweight='bold')
plt.savefig(OUTPUT_DIR / "cta_background_diagnostic.png", dpi=150, bbox_inches="tight")
plt.show()

if BKG_NORM_SCALE != 1.0:
    print(f'Scaling background by factor {BKG_NORM_SCALE}')
    dataset.background.data *= BKG_NORM_SCALE
    print('Done — background rescaled for all subsequent cells.')
else:
    print('BKG_NORM_SCALE = 1.0 → background unchanged.')

total_bkg = dataset.background.data.sum()
total_cnt = dataset.counts.data.sum() if dataset.counts is not None else float('nan')
print(f'  {dataset.name}: total bkg = {total_bkg:.0f} counts   total observed counts = {total_cnt:.0f}')

def ecpl_skymodel(flux, geom, index=2.54, e_ref=1.0*u.TeV, e_cutoff_tev=200.0, name="ecpl"):
    """ECPL point source at the FoV centre."""
    spectral = ExpCutoffPowerLawSpectralModel(
        index=index, amplitude=flux, reference=e_ref,
        lambda_=1/(e_cutoff_tev * u.TeV),
    )
    centre  = geom.center_skydir
    spatial = PointSpatialModel(lon_0=centre.ra, lat_0=centre.dec, frame="icrs")
    spatial.lon_0.frozen = True; spatial.lat_0.frozen = True
    return SkyModel(spatial_model=spatial, spectral_model=spectral, name=name)


def build_gaussian_template(geom_3d, center, sigma_deg):
    """HGPS tek-Gaussian modeli: sabit sigma, enerjiden bağımsız."""
    geom_img = geom_3d.to_image()
    coords = geom_img.get_coord()
    theta = coords.skycoord.separation(center).to_value(u.deg)
    gauss_2d = np.exp(-(theta**2) / (2 * sigma_deg**2))
    gauss_2d = gauss_2d / np.nanmax(gauss_2d)
    phi_map = Map.from_geom(geom_img)
    phi_map.quantity = gauss_2d * u.Unit("")
    return phi_map

print("✓ Model factories defined")


def pl_skymodel(flux, geom, index=2.54, e_ref=1.0*u.TeV, name='pl'):
    """Simple PowerLaw point source — used as null hypothesis in PL vs ECPL comparison."""
    spectral = PowerLawSpectralModel(index=index, amplitude=flux, reference=e_ref)
    centre   = geom.center_skydir
    spatial  = PointSpatialModel(lon_0=centre.ra, lat_0=centre.dec, frame='icrs')
    spatial.lon_0.frozen = True; spatial.lat_0.frozen = True
    return SkyModel(spatial_model=spatial, spectral_model=spectral, name=name)

print('✓ PL sky model factory defined')

def _make_pwn_model(name, geom_ds):
    """Helper: PL point source at SOURCE_CENTER (the PWN)."""
    _c = SOURCE_CENTER.icrs
    sp = PointSpatialModel(lon_0=_c.ra, lat_0=_c.dec, frame="icrs")
    sp.lon_0.frozen = True; sp.lat_0.frozen = True
    return SkyModel(
        spatial_model=sp,
        spectral_model=PowerLawSpectralModel(
            amplitude=TEST_FLUX, index=SPECTRAL_INDEX, reference=E_REF),
        name=name)

def diffusion_angle_deg(E_tev, D0, delta, t_s, d_cm, E0_tev):
    """Angular extension of TeV halo from diffusion physics."""
    D_E  = D0 * (E_tev / E0_tev) ** delta
    r_cm = np.sqrt(4 * D_E * t_s)
    return np.degrees(r_cm / d_cm)

def build_halo_template(geom_3d, center, n0, alpha, e0, theta_d0, delta, e_min, e_max,
                         axis_ratio=1.0, position_angle_deg=0.0,
                         offset_deg=0.0, offset_pa_deg=None, theta_e0=None):
    """Build 2D normalised TeV-halo template (Abeysekara+2017 diffusion profile).

    axis_ratio, position_angle_deg: optional ellipticity (minor/major axis
        ratio, and position angle in deg East-of-North in `center`'s frame).
        Defaults (1.0, 0.0) reproduce the original circular template exactly.
    offset_deg, offset_pa_deg: optional displacement of the halo's own centre
        away from `center` (e.g. the PWN position) -- the halo traces escaped
        particles, not the pulsar's current location, so real halos need not
        be centred on their PWN. Defaults to no offset.
    theta_e0: reference energy at which theta_d0 is defined (e.g. the 1 TeV
        Abeysekara+2017 pivot). Deliberately SEPARATE from `e0`, which sets
        the spectral power-law shape (E/e0)**-alpha. Defaults to `e0` for
        backward compatibility, but pass this explicitly whenever theta_d0's
        true reference energy differs from the spectral pivot -- otherwise
        the size-vs-energy scaling silently uses the wrong reference.
    """
    _theta_e0 = theta_e0 if theta_e0 is not None else e0
    geom_img = geom_3d.to_image()
    coords   = geom_img.get_coord()
    halo_center = center
    if offset_deg:
        _pa = offset_pa_deg if offset_pa_deg is not None else position_angle_deg
        halo_center = center.directional_offset_by(_pa * u.deg, offset_deg * u.deg)
    if axis_ratio != 1.0:
        _dlon, _dlat = halo_center.spherical_offsets_to(coords.skycoord)
        _dlon = _dlon.to_value(u.deg); _dlat = _dlat.to_value(u.deg)
        _pa_rad = np.radians(position_angle_deg)
        _x = _dlon * np.sin(_pa_rad) + _dlat * np.cos(_pa_rad)
        _y = -_dlon * np.cos(_pa_rad) + _dlat * np.sin(_pa_rad)
        theta = np.sqrt(_x**2 + (_y / axis_ratio)**2)
    else:
        theta = coords.skycoord.separation(halo_center).to_value(u.deg)
    e_centers = geom_3d.axes["energy"].center
    mask_e    = (e_centers >= e_min) & (e_centers <= e_max)
    n_e_tot   = geom_3d.axes["energy"].nbin
    ny, nx    = theta.shape
    phi_data  = np.zeros((n_e_tot, ny, nx))
    eps = 1e-7
    for ie, (use, E) in enumerate(zip(mask_e, e_centers)):
        if not use:
            continue
        spec  = float((E / e0).to_value("") ** (-alpha))
        td    = float((theta_d0 * (E / _theta_e0) ** (-delta)).to_value(u.deg))
        denom = np.pi**1.5 * td * (theta + 0.06 * td + eps)
        ang   = (1.22 / denom) * np.exp(-(theta**2) / td**2)
        phi_data[ie] = n0.value * spec * ang
    e_tev  = geom_3d.axes["energy"].center.to_value(u.TeV)
    phi_2d = np.trapezoid(phi_data, x=e_tev, axis=0)
    phi_2d = phi_2d / np.nanmax(phi_2d)
    phi_map          = Map.from_geom(geom_img)
    phi_map.quantity = phi_2d * u.Unit("")
    return phi_map


KYR_TO_S_PHYS       = 1.0e3 * 365.25 * 24.0 * 3600.0
PC_TO_CM_PHYS       = 3.085677581e18
RAD_TO_DEG_PHYS     = 180.0 / np.pi
KM_S_KYR_TO_PC_PHYS = 1.022712e-3


def _electron_energy_from_gamma_tev(Egamma_TeV):
    """IC-on-CMB Thomson-limit electron energy for a given gamma-ray energy."""
    return 17.6 * np.sqrt(np.asarray(Egamma_TeV, dtype=float))


def _cooling_time_kyr(Eelectron_TeV, u_total_eV_cm3):
    return 300.0 / (np.asarray(Eelectron_TeV, dtype=float) * u_total_eV_cm3)


def _diffusion_coefficient_cm2_s(Eelectron_TeV, D100_cm2_s, diffusion_index):
    return D100_cm2_s * (np.asarray(Eelectron_TeV, dtype=float) / 100.0) ** diffusion_index


def halo_spatial_template_physical(
    Egamma_TeV, xx_deg, yy_deg,
    age_kyr, distance_kpc, tau0_kyr,
    injection_index, cutoff_TeV,
    D100_cm2_s, diffusion_index, u_total_eV_cm3,
    v_transverse_kms=0.0, motion_angle_rad=0.0,
    axis_ratio=1.0, position_angle_deg=0.0,
    n_time_slices=48, pixel_size_deg=None,
):
    """Return a normalised 2D template for one gamma-ray energy bin, built
    from pulsar spin-down + electron cooling + proper motion + diffusion.

    `xx_deg`, `yy_deg` are sky-plane offset grids (deg) from the halo
    centre, using the same East-of-North convention as Gammapy's
    `SkyCoord.spherical_offsets_to`.

    `axis_ratio`, `position_angle_deg`: optional ellipticity (minor/major
    axis ratio, and position angle in deg East-of-North), applied to every
    lookback epoch's diffusion kernel -- same convention as
    `build_halo_template`'s ellipse transform, e.g. HESS 2006's
    HALO_AXIS_RATIO / HALO_POSITION_ANGLE. Defaults (1.0, 0.0) reproduce a
    plain circular template.
    """
    Ee = float(_electron_energy_from_gamma_tev(Egamma_TeV))
    tcool = float(_cooling_time_kyr(Ee, u_total_eV_cm3))

    # Avoid the exact cooling singularity.
    tmax = min(float(age_kyr), 0.98 * tcool)
    tmax = max(tmax, 1.0e-3)

    # Quadratic spacing resolves recent injection more finely.
    u_grid = np.linspace(0.0, 1.0, n_time_slices)
    lookback_kyr = tmax * u_grid**2
    dt = np.gradient(lookback_kyr)

    cooling_fraction = np.clip(1.0 - lookback_kyr / tcool, 0.02, None)
    Einj = Ee / cooling_fraction

    # n=3 magnetic-dipole spin-down: L(t) proportional to (1+t/tau0)^-2.
    source_age_at_injection = np.maximum(age_kyr - lookback_kyr, 0.0)
    spin_down_ratio = (
        (1.0 + age_kyr / tau0_kyr)
        / (1.0 + source_age_at_injection / tau0_kyr)
    ) ** 2

    # Q(Einj) dEinj/dE, at fixed observed E.
    injection_weight = (
        spin_down_ratio
        * (Einj / Ee) ** (-injection_index)
        * np.exp(-Einj / cutoff_TeV)
        / cooling_fraction**2
        * dt
    )
    injection_weight = np.nan_to_num(injection_weight, nan=0.0, posinf=0.0, neginf=0.0)
    if injection_weight.sum() <= 0.0:
        injection_weight = np.ones_like(injection_weight)
    injection_weight /= injection_weight.sum()

    D = _diffusion_coefficient_cm2_s(Einj, D100_cm2_s, diffusion_index)
    sigma_diff_pc = np.sqrt(2.0 * D * lookback_kyr * KYR_TO_S_PHYS) / PC_TO_CM_PHYS
    sigma_diff_deg = sigma_diff_pc / (1000.0 * distance_kpc) * RAD_TO_DEG_PHYS

    sigma_total_deg = sigma_diff_deg
    if pixel_size_deg is not None:
        sigma_total_deg = np.maximum(sigma_total_deg, 0.25 * pixel_size_deg)

    displacement_pc = v_transverse_kms * lookback_kyr * KM_S_KYR_TO_PC_PHYS
    center_x_deg = (
        -displacement_pc * np.cos(motion_angle_rad)
        / (1000.0 * distance_kpc) * RAD_TO_DEG_PHYS
    )
    center_y_deg = (
        -displacement_pc * np.sin(motion_angle_rad)
        / (1000.0 * distance_kpc) * RAD_TO_DEG_PHYS
    )

    template = np.zeros_like(xx_deg, dtype=float)
    _pa_rad = np.radians(position_angle_deg)
    for weight, cx, cy, sigma in zip(
        injection_weight, center_x_deg, center_y_deg, sigma_total_deg
    ):
        dx = xx_deg - cx
        dy = yy_deg - cy
        if axis_ratio != 1.0:
            x_rot = dx * np.sin(_pa_rad) + dy * np.cos(_pa_rad)
            y_rot = -dx * np.cos(_pa_rad) + dy * np.sin(_pa_rad)
            r2 = x_rot**2 + (y_rot / axis_ratio) ** 2
        else:
            r2 = dx**2 + dy**2
        template += weight * np.exp(-0.5 * r2 / sigma**2) / (2.0 * np.pi * sigma**2)

    template = np.clip(template, 0.0, None)
    norm = template.sum()
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Physical halo template normalization failed")
    return template / norm


def build_physical_halo_template(
    geom_3d, center,
    age_kyr, distance_kpc, tau0_kyr,
    injection_index, cutoff_TeV,
    D100_cm2_s, diffusion_index, u_total_eV_cm3,
    spectral_index, spectral_e0_TeV,
    v_transverse_kms=0.0, motion_angle_deg=0.0,
    axis_ratio=1.0, position_angle_deg=0.0,
    offset_deg=0.0, offset_pa_deg=None,
    n_time_slices=48,
):
    """Build the energy-dependent physical halo template on `geom_3d`'s grid
    and energy-integrate it into a single normalised 2D Map, matching the
    (spatial-template x separate spectral-model) convention used elsewhere
    in this script (`TemplateSpatialModel(phi_map, normalize=True)`).

    `spectral_index`, `spectral_e0_TeV`: the gamma-ray power-law index/pivot
    used to WEIGHT each energy bin's (already unit-normalised) morphology
    before summing over energy -- WITHOUT this, every bin contributes with
    equal probability mass regardless of how much flux it actually carries,
    and because cooling makes the morphology shrink toward a near
    delta-function at high energy, the energy-integrated map collapses onto
    a single pixel (a real bug found when the halo template rendered as an
    empty image). Use the same index/pivot as the PowerLawSpectralModel
    this template is paired with (e.g. HALO_ALPHA / HALO_E0) so the
    template's assumed spectral shape is self-consistent with the model
    that multiplies it.

    `axis_ratio`, `position_angle_deg`: optional ellipticity, same
    convention as `build_halo_template` (e.g. HESS 2006's HALO_AXIS_RATIO /
    HALO_POSITION_ANGLE). `offset_deg`, `offset_pa_deg`: optional
    displacement of the halo's own centre away from `center` (e.g. the PWN
    position), same convention as `build_halo_template`'s offset handling
    (e.g. HALO_OFFSET_DEG / HALO_OFFSET_PA_DEG).

    Returns `(phi_map, cube)`: `phi_map` is the energy-integrated 2D Map
    ready to drop into TemplateSpatialModel, and `cube` is the un-integrated,
    per-bin-normalised (energy, y, x) array (NOT spectrally weighted --
    useful for inspecting how the morphology's SHAPE alone changes with
    energy, independent of how much flux each bin carries).
    """
    halo_center = center
    if offset_deg:
        _pa = offset_pa_deg if offset_pa_deg is not None else position_angle_deg
        halo_center = center.directional_offset_by(_pa * u.deg, offset_deg * u.deg)

    geom_img = geom_3d.to_image()
    coords = geom_img.get_coord()
    dlon, dlat = halo_center.spherical_offsets_to(coords.skycoord)
    xx_deg = dlon.to_value(u.deg)
    yy_deg = dlat.to_value(u.deg)
    pixel_size_deg = geom_img.pixel_scales.mean().to_value(u.deg)

    e_centers = geom_3d.axes["energy"].center.to_value(u.TeV)
    motion_angle_rad = np.radians(motion_angle_deg)

    ny, nx = xx_deg.shape
    cube = np.zeros((len(e_centers), ny, nx))
    for ie, Egamma_TeV in enumerate(e_centers):
        cube[ie] = halo_spatial_template_physical(
            Egamma_TeV, xx_deg, yy_deg,
            age_kyr=age_kyr, distance_kpc=distance_kpc, tau0_kyr=tau0_kyr,
            injection_index=injection_index, cutoff_TeV=cutoff_TeV,
            D100_cm2_s=D100_cm2_s, diffusion_index=diffusion_index,
            u_total_eV_cm3=u_total_eV_cm3,
            v_transverse_kms=v_transverse_kms, motion_angle_rad=motion_angle_rad,
            axis_ratio=axis_ratio, position_angle_deg=position_angle_deg,
            n_time_slices=n_time_slices, pixel_size_deg=pixel_size_deg,
        )

    spectral_weight = (e_centers / spectral_e0_TeV) ** (-spectral_index)
    phi_2d = np.trapezoid(cube * spectral_weight[:, None, None], x=e_centers, axis=0)
    phi_2d = phi_2d / np.nanmax(phi_2d)
    phi_map = Map.from_geom(geom_img)
    phi_map.quantity = phi_2d * u.Unit("")
    return phi_map, cube


def _physical_template_effective_sigma_deg(
    Egamma_TeV, D100_cm2_s,
    age_kyr, distance_kpc, tau0_kyr,
    injection_index, cutoff_TeV, diffusion_index, u_total_eV_cm3,
    n_time_slices=48, half_width_deg=3.0, n_grid=241,
):
    """RMS radius (deg) of the CIRCULAR, un-offset physical template at one
    gamma-ray energy, for a trial diffusion coefficient -- used only to
    calibrate D100 against a measured size (see
    `calibrate_physical_D100_cm2_s`). Independent of the CTA geometry so it
    stays fast; ellipticity/offset are applied afterwards by
    `build_physical_halo_template` and don't change the calibration.
    """
    axis = np.linspace(-half_width_deg, half_width_deg, n_grid)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    _calib_pixel_deg = 2.0 * half_width_deg / (n_grid - 1)  # avoids a 0/0 at r=0 for tiny trial sigma
    template = halo_spatial_template_physical(
        Egamma_TeV, xx, yy,
        age_kyr=age_kyr, distance_kpc=distance_kpc, tau0_kyr=tau0_kyr,
        injection_index=injection_index, cutoff_TeV=cutoff_TeV,
        D100_cm2_s=D100_cm2_s, diffusion_index=diffusion_index,
        u_total_eV_cm3=u_total_eV_cm3, n_time_slices=n_time_slices,
        pixel_size_deg=_calib_pixel_deg,
    )
    r2 = xx**2 + yy**2
    return float(np.sqrt(np.sum(template * r2)))


def calibrate_physical_D100_cm2_s(
    target_sigma_deg, target_energy_TeV,
    age_kyr, distance_kpc, tau0_kyr,
    injection_index, cutoff_TeV, diffusion_index, u_total_eV_cm3,
    n_time_slices=48, log10_D100_bounds=(22.0, 30.0), half_width_deg=None,
):
    """Solve for the D100 (diffusion coefficient at 100 TeV) that makes the
    physical halo template's RMS radius at `target_energy_TeV` equal
    `target_sigma_deg`, given this source's own age/distance/spectral
    parameters -- replacing the old Abeysekara+2017 D0=1e27 cm^2/s
    benchmark with a value calibrated directly against a real measured size
    (e.g. the HESS-2006 semi-major axis, HESS2006_SIGMA_MAJOR_DEG).
    """
    from scipy.optimize import brentq

    _half_width_deg = half_width_deg if half_width_deg is not None else max(3.0, 4.0 * target_sigma_deg)

    def resid(log10_D100):
        D100 = 10.0 ** log10_D100
        sigma = _physical_template_effective_sigma_deg(
            target_energy_TeV, D100,
            age_kyr=age_kyr, distance_kpc=distance_kpc, tau0_kyr=tau0_kyr,
            injection_index=injection_index, cutoff_TeV=cutoff_TeV,
            diffusion_index=diffusion_index, u_total_eV_cm3=u_total_eV_cm3,
            n_time_slices=n_time_slices, half_width_deg=_half_width_deg,
        )
        return sigma - target_sigma_deg

    lo, hi = log10_D100_bounds
    resid_lo, resid_hi = resid(lo), resid(hi)
    if resid_lo > 0 or resid_hi < 0:
        raise RuntimeError(
            "calibrate_physical_D100_cm2_s: target size not bracketed in "
            f"log10(D100) in [{lo}, {hi}] (half_width_deg={_half_width_deg:.2f} deg) -- "
            f"sigma(lo)={resid_lo + target_sigma_deg:.3f} deg, sigma(hi)={resid_hi + target_sigma_deg:.3f} deg, "
            f"target={target_sigma_deg:.3f} deg. If sigma(hi) is still well below target, "
            "widen log10_D100_bounds; if sigma(hi) is close to target, increase half_width_deg."
        )
    log10_D100_sol = brentq(resid, lo, hi, xtol=1e-4)
    return 10.0 ** log10_D100_sol


def build_scan_halo_template(
    geom_3d, center, *,
    theta_d0_deg=None, D100_cm2_s=None,
    target_energy_TeV=None,
    offset_deg=0.0, offset_pa_deg=None,
    axis_ratio=None, position_angle_deg=None,
    age_kyr=None, distance_kpc=None,
):
    """Physical-model drop-in for every scan that used to call the analytic
    `build_halo_template()`. Pass EXACTLY ONE of:

    - `theta_d0_deg`: a target size in degrees -- this is calibrated into a
      D100 via `calibrate_physical_D100_cm2_s` (same procedure used for the
      primary phi_map), at `target_energy_TeV` (default: HALO_THETA_E0).
    - `D100_cm2_s`: a diffusion coefficient directly -- no calibration step
      at all (used by scans that sweep D0/D100 itself, e.g. the D0 scan).

    Everything else (ellipticity, age, distance, spectral shape, injection
    physics) defaults to the SAME globals used for the primary halo
    template (HALO_AXIS_RATIO=0.05/0.12, HALO_POSITION_ANGLE=86.3,
    T_AGE_YR, DIST_KPC, ...), so a scan point differs only in the one
    parameter actually being scanned. `offset_deg`/`offset_pa_deg` shift
    the halo centre away from `center`, same convention as
    HALO_OFFSET_DEG/HALO_OFFSET_PA_DEG.

    Returns `(phi_map, D100_cm2_s_used)`.
    """
    if (theta_d0_deg is None) == (D100_cm2_s is None):
        raise ValueError("build_scan_halo_template: pass exactly one of "
                          "theta_d0_deg or D100_cm2_s")

    _age_kyr   = age_kyr if age_kyr is not None else T_AGE_YR / 1e3
    _dist_kpc  = distance_kpc if distance_kpc is not None else DIST_KPC
    _e0_TeV    = target_energy_TeV if target_energy_TeV is not None else HALO_THETA_E0.to_value(u.TeV)

    if theta_d0_deg is not None:
        D100_cm2_s = calibrate_physical_D100_cm2_s(
            target_sigma_deg=float(theta_d0_deg),
            target_energy_TeV=_e0_TeV,
            age_kyr=_age_kyr, distance_kpc=_dist_kpc, tau0_kyr=HALO_TAU0_KYR,
            injection_index=HALO_INJECTION_INDEX, cutoff_TeV=HALO_CUTOFF_TEV,
            diffusion_index=DELTA_DIFF, u_total_eV_cm3=HALO_U_TOTAL_EV_CM3,
            n_time_slices=HALO_N_TIME_SLICES,
        )

    phi_map, _ = build_physical_halo_template(
        geom_3d, center,
        age_kyr=_age_kyr, distance_kpc=_dist_kpc, tau0_kyr=HALO_TAU0_KYR,
        injection_index=HALO_INJECTION_INDEX, cutoff_TeV=HALO_CUTOFF_TEV,
        D100_cm2_s=D100_cm2_s, diffusion_index=DELTA_DIFF,
        u_total_eV_cm3=HALO_U_TOTAL_EV_CM3,
        spectral_index=HALO_ALPHA, spectral_e0_TeV=HALO_E0.to_value(u.TeV),
        v_transverse_kms=HALO_V_TRANSVERSE_KMS, motion_angle_deg=HALO_MOTION_PA_DEG,
        axis_ratio=axis_ratio if axis_ratio is not None else HALO_AXIS_RATIO,
        position_angle_deg=position_angle_deg if position_angle_deg is not None else HALO_POSITION_ANGLE,
        offset_deg=offset_deg, offset_pa_deg=offset_pa_deg,
        n_time_slices=HALO_N_TIME_SLICES,
    )
    return phi_map, D100_cm2_s

print("✓ Physical (cooling + proper-motion) halo template function defined")
print("✓ build_scan_halo_template() wrapper defined -- all scans now use the physical model")


def run_halo_only(dataset_sim, phi_template, seed_offset=0):
    """Halo-only test: null=bkg, H0=halo+bkg. Returns ts_halo."""
    # Null: bkg only
    ds_n = dataset_sim.copy(name=f"n_{seed_offset}")
    ds_n.models = Models([FoVBackgroundModel(dataset_name=ds_n.name)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_n = Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_n])
    # H0: halo + bkg
    ds_0 = dataset_sim.copy(name=f"h0_{seed_offset}")
    h0   = SkyModel(spatial_model=TemplateSpatialModel(phi_template, normalize=True),
                    spectral_model=PowerLawSpectralModel(
                        amplitude=HALO_AMPLITUDE,
                        index=HALO_ALPHA, reference=HALO_E0),
                    name=f"halo-h0-{seed_offset}")
    ds_0.models = Models([h0, FoVBackgroundModel(dataset_name=ds_0.name)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_0 = Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_0])
        r_0 = Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_0])
    return max(0, r_n.total_stat - r_0.total_stat)

def run_pwn_halo(dataset_sim, phi_template, seed_offset=0):
    """
    Total-system detection test: null = bkg only (nothing there);
    alt = PWN + halo + bkg. Returns ts_det = stat(null) - stat(alt),
    the overall detection significance of the source complex against a
    pure-background null (H0 = bkg only, H1 = PWN + halo + bkg).
    """
    geom_ds = dataset_sim.counts.geom

    # Null: bkg only (nothing there)
    ds_n = dataset_sim.copy(name=f"n_{seed_offset}")
    ds_n.models = Models([FoVBackgroundModel(dataset_name=ds_n.name)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_n = Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_n])

    # Alt: PWN + halo + bkg
    ds_0 = dataset_sim.copy(name=f"h0_{seed_offset}")
    e0   = _make_pwn_model(f"pwn-h0-{seed_offset}", geom_ds)
    h0   = SkyModel(spatial_model=TemplateSpatialModel(phi_template, normalize=True),
                    spectral_model=PowerLawSpectralModel(
                        amplitude=HALO_AMPLITUDE,
                        index=HALO_ALPHA, reference=HALO_E0),
                    name=f"halo-h0-{seed_offset}")
    ds_0.models = Models([e0, h0, FoVBackgroundModel(dataset_name=ds_0.name)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_0 = Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_0])
        r_0 = Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_0])

    ts_det = max(0, r_n.total_stat - r_0.total_stat)
    return ts_det


E0_TeV  = HALO_E0.to_value(u.TeV)

theta_d0_phys_legacy_analytic = diffusion_angle_deg(E0_TeV, D0_cm2_s, DELTA_DIFF, t_age_s, dist_cm, E0_TeV)
print(f"[legacy analytic, unused by phi_map] θ_d0 = {theta_d0_phys_legacy_analytic:.3f}° "
      f"at d={DIST_KPC} kpc, t={T_AGE_YR/1e3:.0f} kyr (Abeysekara+2017 D0/delta benchmark)")

HALO_D100_CALIBRATED_CM2_S = calibrate_physical_D100_cm2_s(
    target_sigma_deg=HALO_THETA_D0.to_value(u.deg),
    target_energy_TeV=HALO_THETA_E0.to_value(u.TeV),
    age_kyr=T_AGE_YR / 1e3, distance_kpc=DIST_KPC, tau0_kyr=HALO_TAU0_KYR,
    injection_index=HALO_INJECTION_INDEX, cutoff_TeV=HALO_CUTOFF_TEV,
    diffusion_index=DELTA_DIFF, u_total_eV_cm3=HALO_U_TOTAL_EV_CM3,
    n_time_slices=HALO_N_TIME_SLICES,
)
print(f"✓ Calibrated D100 = {HALO_D100_CALIBRATED_CM2_S:.3e} cm²/s "
      f"(physical model reproduces θ={HALO_THETA_D0:.3f} at {HALO_THETA_E0:.2f} -- HESS 2006 size)")

phi_map, phi_cube_physical = build_physical_halo_template(
    geom, SOURCE_CENTER.icrs,
    age_kyr=T_AGE_YR / 1e3, distance_kpc=DIST_KPC, tau0_kyr=HALO_TAU0_KYR,
    injection_index=HALO_INJECTION_INDEX, cutoff_TeV=HALO_CUTOFF_TEV,
    D100_cm2_s=HALO_D100_CALIBRATED_CM2_S, diffusion_index=DELTA_DIFF,
    u_total_eV_cm3=HALO_U_TOTAL_EV_CM3,
    spectral_index=HALO_ALPHA, spectral_e0_TeV=HALO_E0.to_value(u.TeV),
    v_transverse_kms=HALO_V_TRANSVERSE_KMS, motion_angle_deg=HALO_MOTION_PA_DEG,
    axis_ratio=HALO_AXIS_RATIO, position_angle_deg=HALO_POSITION_ANGLE,
    offset_deg=HALO_OFFSET_DEG, offset_pa_deg=HALO_OFFSET_PA_DEG,
    n_time_slices=HALO_N_TIME_SLICES,
)
print(f"✓ Halo template built (physical cooling + proper-motion model) — shape: {phi_map.data.shape}")

fig, ax = plt.subplots(figsize=(6, 5))
phi_map.plot(ax=ax, cmap="inferno", add_cbar=True)
ax.set_title("TeV Halo Template (physical cooling + proper-motion model, normalised)\n"
             f"size calibrated to HESS 2006 θ = {HALO_THETA_D0:.3f} at {HALO_THETA_E0:.2f}")
plt.savefig(OUTPUT_DIR / "cta_halo_template.png", dpi=150, bbox_inches="tight")
plt.tight_layout(); plt.show()

# Energy-dependent morphology preview -- the physical model's shape (unlike
# the retired analytic profile) genuinely varies bin-to-bin.
_n_e_preview = min(4, len(geom.axes["energy"].center))
_preview_idx = np.linspace(0, len(geom.axes["energy"].center) - 1, _n_e_preview).astype(int)
fig, axes = plt.subplots(1, _n_e_preview, figsize=(3.2 * _n_e_preview, 3.4), constrained_layout=True)
for ax, ie in zip(np.atleast_1d(axes), _preview_idx):
    ax.imshow(phi_cube_physical[ie], origin="lower", cmap="inferno")
    ax.set_title(f"{geom.axes['energy'].center[ie]:.1f}")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Physical halo template -- morphology vs. energy bin")
plt.savefig(OUTPUT_DIR / "cta_halo_template_physical_vs_energy.png", dpi=150, bbox_inches="tight")
plt.show()


from gammapy.modeling.models import GaussianSpatialModel, DiskSpatialModel

# Simulate halo-only dataset, then fit both Gaussian and Disk, compare AIC.

ds_aic = dataset.copy(name="aic_test")
halo_sim_aic = SkyModel(
    spatial_model=TemplateSpatialModel(phi_map, normalize=True),
    spectral_model=PowerLawSpectralModel(
        amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
    name="halo-aic-sim")
bkg_aic = FoVBackgroundModel(dataset_name=ds_aic.name)
ds_aic.models = Models([halo_sim_aic, bkg_aic])
ds_aic.fake(random_state=SEED + 99)

# Fit 1: Radial Gaussian
ds_gauss = ds_aic.copy(name="aic_gauss")
gauss_sp  = GaussianSpatialModel(
    lon_0=SOURCE_CENTER.icrs.ra, lat_0=SOURCE_CENTER.icrs.dec,
    sigma=HALO_THETA_D0, frame="icrs")
gauss_sp.lon_0.frozen = True; gauss_sp.lat_0.frozen = True
halo_gauss = SkyModel(
    spatial_model=gauss_sp,
    spectral_model=PowerLawSpectralModel(
        amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
    name="halo-gauss")
ds_gauss.models = Models([halo_gauss, FoVBackgroundModel(dataset_name=ds_gauss.name)])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res_gauss = Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_gauss])
    res_gauss = Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_gauss])

# Fit 2: Disk
ds_disk = ds_aic.copy(name="aic_disk")
disk_sp  = DiskSpatialModel(
    lon_0=SOURCE_CENTER.icrs.ra, lat_0=SOURCE_CENTER.icrs.dec,
    r_0=HALO_THETA_D0, frame="icrs")
disk_sp.lon_0.frozen = True; disk_sp.lat_0.frozen = True
halo_disk = SkyModel(
    spatial_model=disk_sp,
    spectral_model=PowerLawSpectralModel(
        amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
    name="halo-disk")
ds_disk.models = Models([halo_disk, FoVBackgroundModel(dataset_name=ds_disk.name)])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res_disk = Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_disk])
    res_disk = Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_disk])

# AIC comparison
k_gauss = sum(not p.frozen for p in halo_gauss.parameters) + 1
k_disk  = sum(not p.frozen for p in halo_disk.parameters)  + 1
aic_gauss = 2*k_gauss + res_gauss.total_stat
aic_disk  = 2*k_disk  + res_disk.total_stat

print("=" * 50)
print(f"{'Model':<20}  {'Stat':>10}  {'k':>4}  {'AIC':>10}")
print("-" * 50)
print(f"{'Radial Gaussian':<20}  {res_gauss.total_stat:>10.2f}  {k_gauss:>4}  {aic_gauss:>10.2f}")
print(f"{'Disk':<20}  {res_disk.total_stat:>10.2f}  {k_disk:>4}  {aic_disk:>10.2f}")
print("=" * 50)
delta_aic_spatial = aic_disk - aic_gauss
USE_GAUSSIAN = delta_aic_spatial > 0  # Gaussian wins if AIC_disk > AIC_gauss
print(f"ΔAIC (Disk - Gauss) = {delta_aic_spatial:.1f}")
print(f"→ Preferred model: {'Radial Gaussian' if USE_GAUSSIAN else 'Disk'}")

sigma_gauss_fit = halo_gauss.spatial_model.sigma.value
r0_disk_fit     = halo_disk.spatial_model.r_0.value
print(f"   Gaussian σ fit  = {sigma_gauss_fit:.3f}°")
print(f"   Disk r₀ fit     = {r0_disk_fit:.3f}°")

FORCE_GAUSSIAN_THIS_RUN = False
if FORCE_GAUSSIAN_THIS_RUN:
    USE_GAUSSIAN = True
    print("→ OVERRIDE: forcing USE_GAUSSIAN = True for this run "
          "(AIC preference above is informational only)")


SPATIAL_MODEL = "gaussian" if USE_GAUSSIAN else "disk"
HALO_THETA_D0 = THETA_D0_OPT if 'THETA_D0_OPT' in dir() else HALO_THETA_D0  # updated after E_dat scan

print(f"✓ LOCKED: Spatial model = {SPATIAL_MODEL.upper()}")
print(f"   Gaussian σ fit = {sigma_gauss_fit:.3f}°  |  Disk r₀ fit = {r0_disk_fit:.3f}°")
print(f"   All subsequent fits will use: {SPATIAL_MODEL}")


# Use the winning spatial model for all subsequent halo-only fits

# True halo for simulation
sky_halo_true = SkyModel(
    spatial_model=TemplateSpatialModel(phi_map, normalize=True),
    spectral_model=PowerLawSpectralModel(
        amplitude=HALO_AMPLITUDE,
        index=HALO_ALPHA, reference=HALO_E0),
    name="halo-true")

diffuse_roi      = diffuse_emission_map.interp_to_geom(dataset.exposure.geom)
diffuse_template = TemplateSpatialModel(diffuse_roi, normalize=False)
diffuse_sim      = SkyModel(PowerLawNormSpectralModel(), diffuse_template, name="diffuse-sim")
bkg_sim          = FoVBackgroundModel(dataset_name=dataset.name)

# Halo-only simulation (no PWN)
dataset_sim_halo = dataset.copy(name="sim-halo")
dataset_sim_halo.models = Models([sky_halo_true, bkg_sim, diffuse_sim])
dataset_sim_halo.fake(random_state=SEED)

print("✓ Halo-only simulation complete")
print(f"   Total counts  : {dataset_sim_halo.counts.data.sum():.0f}")
print(f"   Background    : {dataset_sim_halo.background.data.sum():.0f}")
print(f"   Net excess    : {dataset_sim_halo.excess.data.sum():.0f}")


mu_halo_only = dataset_sim_halo.npred().data
N_halo_only  = dataset_sim_halo.counts.data
_mask_halo_only = mu_halo_only > 1.0
_z_halo_only = (N_halo_only[_mask_halo_only] - mu_halo_only[_mask_halo_only]) / np.sqrt(mu_halo_only[_mask_halo_only])

print(f"[halo_only] Poisson calibration check -- {_z_halo_only.size:,} bins with μ>1")
print(f"   Mean standardized residual z=(N-μ)/√μ : {_z_halo_only.mean():.3f}  (expect ≈0)")
print(f"   Std  standardized residual             : {_z_halo_only.std(ddof=1):.3f}  (expect ≈1)")

_mu_img_halo_only = mu_halo_only.sum(axis=0) if mu_halo_only.ndim > 2 else mu_halo_only
_n_img_halo_only  = N_halo_only.sum(axis=0) if N_halo_only.ndim > 2 else N_halo_only
_mask_img_halo_only = _mu_img_halo_only > 1.0
_z_img_halo_only = np.full(_mu_img_halo_only.shape, np.nan)
_z_img_halo_only[_mask_img_halo_only] = (_n_img_halo_only[_mask_img_halo_only] - _mu_img_halo_only[_mask_img_halo_only]) / np.sqrt(_mu_img_halo_only[_mask_img_halo_only])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(_z_halo_only, bins=60, range=(-5, 5), density=True, color="steelblue",
             alpha=0.75, label="Standardized residuals")
_zg = np.linspace(-5, 5, 200)
axes[0].plot(_zg, np.exp(-_zg**2/2)/np.sqrt(2*np.pi), "k--", lw=1.5, label="N(0,1)")
axes[0].set_xlabel(r"$z=(N-\mu)/\sqrt{\mu}$", fontsize=11)
axes[0].set_ylabel("Density", fontsize=11)
axes[0].set_title("Poisson calibration -- halo_only", fontsize=11, fontweight="bold")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

im = axes[1].imshow(_z_img_halo_only, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5)
fig.colorbar(im, ax=axes[1]).set_label(r"$(N-\mu)/\sqrt{\mu}$  (energy-summed)", fontsize=9)
axes[1].set_title("Poisson fluctuation map -- halo_only", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_poisson_calibration_halo_only.png", dpi=150, bbox_inches="tight")
plt.show()


ds_halo_H0 = dataset_sim_halo.copy(name="halo-H0")
diff_hH0   = SkyModel(PowerLawNormSpectralModel(),
                       TemplateSpatialModel(diffuse_emission_map.interp_to_geom(
                           ds_halo_H0.exposure.geom), normalize=False), name="diff-hH0")
bkg_hH0    = FoVBackgroundModel(dataset_name=ds_halo_H0.name)
ds_halo_H0.models = Models([bkg_hH0, diff_hH0])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res_halo_H0 = Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_halo_H0])
print(f"H₀ (bkg only) fit  success={res_halo_H0.success}  stat={res_halo_H0.total_stat:.2f}")

# Use AIC-selected spatial model
ds_halo_H1 = dataset_sim_halo.copy(name="halo-H1")

PRIMARY_MODEL_IS_HALO_TEMPLATE = True

if PRIMARY_MODEL_IS_HALO_TEMPLATE:
    _sp_h1 = TemplateSpatialModel(phi_map, normalize=True)
elif USE_GAUSSIAN:
    _sp_h1 = GaussianSpatialModel(
        lon_0=SOURCE_CENTER.icrs.ra, lat_0=SOURCE_CENTER.icrs.dec,
        sigma=HALO_THETA_D0, frame="icrs")
    _sp_h1.lon_0.frozen = True; _sp_h1.lat_0.frozen = True
else:
    _sp_h1 = DiskSpatialModel(
        lon_0=SOURCE_CENTER.icrs.ra, lat_0=SOURCE_CENTER.icrs.dec,
        r_0=HALO_THETA_D0, frame="icrs")
    _sp_h1.lon_0.frozen = True; _sp_h1.lat_0.frozen = True

halo_H1 = SkyModel(
    spatial_model=_sp_h1,
    spectral_model=PowerLawSpectralModel(
        amplitude=1e-13*u.Unit("cm-2 s-1 TeV-1"),
        index=HALO_ALPHA, reference=HALO_E0),
    name="halo-H1")
diff_hH1 = SkyModel(PowerLawNormSpectralModel(),
                     TemplateSpatialModel(diffuse_emission_map.interp_to_geom(
                         ds_halo_H1.exposure.geom), normalize=False), name="diff-hH1")
bkg_hH1  = FoVBackgroundModel(dataset_name=ds_halo_H1.name)
ds_halo_H1.models = Models([halo_H1, bkg_hH1, diff_hH1])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res_halo_H1 = Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_halo_H1])
    res_halo_H1 = Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_halo_H1])
print(f"H₁ (halo) fit  success={res_halo_H1.success}  stat={res_halo_H1.total_stat:.2f}")


ts_halo_det   = max(0, res_halo_H0.total_stat - res_halo_H1.total_stat)
sigma_halo    = np.sqrt(ts_halo_det)
k_H0h = sum(not p.frozen for p in ds_halo_H0.models.parameters)
k_H1h = sum(not p.frozen for p in ds_halo_H1.models.parameters)
aic_H0h = 2*k_H0h + res_halo_H0.total_stat
aic_H1h = 2*k_H1h + res_halo_H1.total_stat
delta_aic_h = aic_H0h - aic_H1h

if PRIMARY_MODEL_IS_HALO_TEMPLATE:
    sp_label    = "Halo Template"
    fitted_size = HALO_THETA_D0.to_value(u.deg)
    fitted_size_str = f"{HALO_THETA_D0.to_value(u.deg):.3f}°x{(HALO_THETA_D0*HALO_AXIS_RATIO).to_value(u.deg):.3f}°"
elif USE_GAUSSIAN:
    sp_label    = "Gaussian"
    fitted_size = halo_H1.spatial_model.sigma.value
    fitted_size_str = f"{fitted_size:.3f}°"
else:
    sp_label    = "Disk"
    fitted_size = halo_H1.spatial_model.r_0.value
    fitted_size_str = f"{fitted_size:.3f}°"

print("=" * 60)
print("HALO-ONLY — MODEL COMPARISON")
print(f"{'Model':<28}  {'Stat':>10}  {'k':>4}  {'AIC':>10}")
print("-" * 60)
print(f"{'H₀ (bkg only)':<28}  {res_halo_H0.total_stat:>10.2f}  {k_H0h:>4}  {aic_H0h:>10.2f}")
print(f"{'H₁ (halo, '+sp_label+')':<28}  {res_halo_H1.total_stat:>10.2f}  {k_H1h:>4}  {aic_H1h:>10.2f}")
print("=" * 60)
print(f"  TS_halo      = {ts_halo_det:.1f}  ({sigma_halo:.1f}σ)")
print(f"  ΔAIC         = {delta_aic_h:.1f}")
print(f"  Fitted size  = {fitted_size_str}  ({sp_label})")


from gammapy.estimators import ExcessMapEstimator
import matplotlib.colors as mcolors

est = ExcessMapEstimator(
    correlation_radius='0.05 deg',
    selection_optional=[],
    energy_edges=[E_MIN.to_value(u.TeV), E_MAX.to_value(u.TeV)] * u.TeV,
)

# Detection map — bkg-only null on H₀ data
ds_det_H0 = ds_halo_H0.copy(name="det-halo-H0")
ds_det_H0.models = Models([FoVBackgroundModel(dataset_name=ds_det_H0.name)])
res_det_H0 = est.run(ds_det_H0)

# Residual map — H₁ fitted
ds_det_H1 = ds_halo_H1.copy(name="det-halo-H1")
res_det_H1_haloonly = est.run(ds_det_H1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
res_det_H0['sqrt_ts'].plot(ax=axes[0], add_cbar=True, vmin=-5, vmax=5, cmap='PuOr')
axes[0].set_title('Significance — H₀ (bkg only)  [σ]', fontsize=10, fontweight='bold')
res_det_H1_haloonly['sqrt_ts'].plot(ax=axes[1], add_cbar=True, vmin=-5, vmax=5, cmap='PuOr')
axes[1].set_title('Significance — H₁ (halo fitted)  [σ]', fontsize=10, fontweight='bold')
plt.suptitle(
    f'Significance Maps — CTA Halo Only  [σ]\n'
    f'TS_halo = {ts_halo_det:.1f}  ({sigma_halo:.1f}σ)  |  '
    f'ΔAIC = {delta_aic_h:.1f}  |  θ_d0 = {HALO_THETA_D0}',
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_halo_only_significance_maps.png", dpi=150, bbox_inches="tight")
plt.show()


sig_data = res_det_H1_haloonly['sqrt_ts'].data.flatten()
sig_data = sig_data[np.isfinite(sig_data)]
from scipy.stats import norm as scipy_norm
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(sig_data, bins=50, density=True, alpha=0.7, color='steelblue', label='Data')
x = np.linspace(-6, 6, 200)
ax.plot(x, scipy_norm.pdf(x), 'r-', lw=2, label='N(0,1)')
ax.set_xlabel('Significance [σ]'); ax.set_ylabel('Density')
ax.set_title('Significance distribution — H₁ halo fit', fontweight='bold')
plt.savefig(OUTPUT_DIR / "cta_significance_distribution.png", dpi=150, bbox_inches="tight")
ax.legend(); plt.tight_layout(); plt.show()


fig, ax = plt.subplots(figsize=(7, 5))
dataset_sim_halo.counts.reduce_over_axes().smooth(0.05 * u.deg).plot(
    ax=ax, add_cbar=True, stretch='sqrt', cmap='inferno')
try:
    ax.images[-1].colorbar.set_label('Counts per bin  [dimensionless]', fontsize=9)
except Exception:
    pass
ax.set_title(f'{dataset_sim_halo.name}\nCounts map (energy-integrated, smoothed 0.05°)', fontsize=10)
plt.savefig(OUTPUT_DIR / "cta_counts_map.png", dpi=150, bbox_inches="tight")
plt.tight_layout(); plt.show()


estimator_ebands = ExcessMapEstimator(
    correlation_radius='0.05 deg',
    selection_optional=[],
    energy_edges=[1, 5, 50, 300] * u.TeV,
)
result_ebands = estimator_ebands.run(ds_halo_H1)
result_ebands['sqrt_ts'].plot_grid(
    figsize=(14, 4), cmap='PuOr', add_cbar=True, vmin=-5, vmax=5, ncols=3)
plt.suptitle('Residual significance maps per energy band  [σ] — Halo Only (H₁ fitted)',
             fontsize=11, fontweight='bold')
plt.savefig(OUTPUT_DIR / "cta_residual_maps_multi_energy_halo_only.png", dpi=150, bbox_inches="tight")
plt.tight_layout(); plt.show()


import inspect
from gammapy.estimators import TSMapEstimator
print(inspect.signature(TSMapEstimator.__init__))
from gammapy.estimators import TSMapEstimator
estimator_ts = TSMapEstimator(
    model=SkyModel(
        spatial_model=PointSpatialModel(),
        spectral_model=PowerLawSpectralModel(index=2.54, amplitude=1e-12*u.Unit("cm-2 s-1 TeV-1"),
                                              reference=1*u.TeV)),
    kernel_width="0.1 deg", selection_optional=[], sum_over_energy_groups=False,
    energy_edges=[1, 300] * u.TeV)

ts_map_H0 = estimator_ts.run(ds_halo_H0)
ts_map_H1 = estimator_ts.run(ds_halo_H1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, ts_result, label, note, det_ts in zip(
    axes,
    [ts_map_H0, ts_map_H1],
    ['Bkg only (H₀)', 'Halo fitted (H₁)'],
    ['Halo unabsorbed → residual ring visible', 'Halo model fitted → should be flat'],
    [ts_halo_det, ts_halo_det],
):
    sqrt_ts = ts_result['sqrt_ts'].data[0]
    vmax    = max(np.nanpercentile(np.abs(sqrt_ts[np.isfinite(sqrt_ts)]), 99.5), 5)
    im = ax.imshow(sqrt_ts, origin='lower', cmap='PuOr', vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax).set_label('sqrt(TS)  [σ]', fontsize=9)
    ax.set_title(f'{label}\n{note}\nTS_halo = {det_ts:.1f}  ({np.sqrt(det_ts):.1f}σ)',
                 fontsize=10, fontweight='bold')

plt.suptitle(
    f'TS Maps — CTA Halo Only\n'
    f'TS_halo = {ts_halo_det:.1f}  ({sigma_halo:.1f}σ)  |  θ_d0 = {HALO_THETA_D0}',
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_halo_only_ts_maps.png", dpi=150, bbox_inches="tight")
plt.show()


from gammapy.estimators import TSMapEstimator

def fit_shape(dataset_sim, spatial_model, name):
    ds = dataset_sim.copy(name=f"shapecmp-{name}")
    diff_tmpl = TemplateSpatialModel(
        diffuse_emission_map.interp_to_geom(ds.exposure.geom), normalize=False)
    diff_sky  = SkyModel(PowerLawNormSpectralModel(), diff_tmpl, name=f"diff-{name}")
    sky = SkyModel(
        spatial_model=spatial_model,
        spectral_model=PowerLawSpectralModel(
            amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
        name=f"halo-{name}")
    bkg = FoVBackgroundModel(dataset_name=ds.name)
    ds.models = Models([sky, bkg, diff_sky])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds])
        res = Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds])
    return ds, sky, res

# (a) physical template shape (fixed shape -> only norm/index free)
tmpl_shape = TemplateSpatialModel(phi_map, normalize=True)
ds_tmpl_fit, sky_tmpl_fit, res_tmpl_fit = fit_shape(dataset_sim_halo, tmpl_shape, "template")

# (b) Gaussian shape (1 extra free morphology param: sigma)
gauss_shape = GaussianSpatialModel(
    lon_0=SOURCE_CENTER.icrs.ra, lat_0=SOURCE_CENTER.icrs.dec,
    sigma=HALO_THETA_D0, frame="icrs")
gauss_shape.lon_0.frozen = True; gauss_shape.lat_0.frozen = True
ds_gauss_fit, sky_gauss_fit, res_gauss_fit = fit_shape(dataset_sim_halo, gauss_shape, "gaussian")

k_tmpl  = sum(not p.frozen for p in ds_tmpl_fit.models.parameters)
k_gauss = sum(not p.frozen for p in ds_gauss_fit.models.parameters)
aic_tmpl   = 2 * k_tmpl  + res_tmpl_fit.total_stat
aic_gauss  = 2 * k_gauss + res_gauss_fit.total_stat
delta_stat = res_gauss_fit.total_stat - res_tmpl_fit.total_stat  # >0 -> template fits better
delta_aic  = aic_gauss - aic_tmpl

print("=" * 66)
print("HALO SHAPE COMPARISON -- physical template vs 2D Gaussian (same data)")
print(f"{'Model':<24}  {'Stat (-2lnL)':>13}  {'k':>4}  {'AIC':>10}")
print("-" * 66)
print(f"{'Physical template':<24}  {res_tmpl_fit.total_stat:>13.2f}  {k_tmpl:>4}  {aic_tmpl:>10.2f}")
print(f"{'2D Gaussian':<24}  {res_gauss_fit.total_stat:>13.2f}  {k_gauss:>4}  {aic_gauss:>10.2f}")
print("=" * 66)
print(f"  Δstat (Gauss - template) = {delta_stat:.2f}   "
      f"({'template' if delta_stat > 0 else 'Gaussian'} has higher likelihood)")
print(f"  ΔAIC  (Gauss - template) = {delta_aic:.2f}   "
      f"(AIC alone is not decisive here -- see TS maps & radial profile below)")

_ts_est_cmp = TSMapEstimator(
    model=SkyModel(
        spatial_model=PointSpatialModel(),
        spectral_model=PowerLawSpectralModel(
            index=2.54, amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV)),
    kernel_width="0.1 deg", selection_optional=[], sum_over_energy_groups=False,
    energy_edges=[1, 300] * u.TeV)

ts_map_tmpl_fit  = _ts_est_cmp.run(ds_tmpl_fit)
ts_map_gauss_fit = _ts_est_cmp.run(ds_gauss_fit)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, ts_result, label, res in zip(
    axes, [ts_map_tmpl_fit, ts_map_gauss_fit],
    ["Physical template fitted", "2D Gaussian fitted"],
    [res_tmpl_fit, res_gauss_fit],
):
    sqrt_ts = ts_result["sqrt_ts"].data[0]
    vmax = max(np.nanpercentile(np.abs(sqrt_ts[np.isfinite(sqrt_ts)]), 99.5), 5)
    im = ax.imshow(sqrt_ts, origin="lower", cmap="PuOr", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax).set_label("sqrt(TS) residual  [σ]", fontsize=9)
    ax.set_title(f"{label}\nstat={res.total_stat:.1f}  (should look flat if shape is correct)",
                 fontsize=10, fontweight="bold")

plt.suptitle("Residual TS maps after fitting each shape — CTA halo-only sim\n"
             "Ring-like residuals = shape mismatch; flat = good fit",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_gauss_vs_template_ts_maps.png", dpi=150, bbox_inches="tight")
plt.show()


def deg_to_pc(theta_deg, distance_kpc=DIST_KPC):
    """Convert an angular radius [deg] to a projected physical radius [pc]."""
    return distance_kpc * 1000.0 * np.tan(np.radians(theta_deg))

def radial_profile_density(data_map, center, n_bins=25, r_max_deg=None):
    """
    Azimuthally bin a counts-like Map into a surface density [counts / deg^2]
    per radial annulus, with Poisson errors -- this is what "surface brightness"
    profile plots (e.g. HAWC technotes) actually show, as opposed to a mean
    significance per pixel.
    """
    geom_img = data_map.geom.to_image()
    coords   = data_map.geom.get_coord()
    sep_deg  = coords.skycoord.separation(center).to_value(u.deg)
    if sep_deg.ndim > 2:
        sep_deg = sep_deg[0]
    data = data_map.data
    if data.ndim > 2:
        data = data.sum(axis=0)  # sum over energy axis
    solid_angle = geom_img.solid_angle().to_value(u.deg ** 2)
    if r_max_deg is None:
        r_max_deg = np.nanpercentile(sep_deg, 95)
    edges = np.linspace(0, r_max_deg, n_bins + 1)
    centers_deg = 0.5 * (edges[:-1] + edges[1:])
    sb, sb_err, n_pix = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (sep_deg >= lo) & (sep_deg < hi) & np.isfinite(data)
        counts_sum = np.nansum(data[sel])
        area_sum   = np.nansum(solid_angle[sel])
        sb.append(counts_sum / area_sum if area_sum > 0 else np.nan)
        sb_err.append(np.sqrt(max(counts_sum, 0)) / area_sum if area_sum > 0 else np.nan)
        n_pix.append(int(sel.sum()))
    return centers_deg, np.array(sb), np.array(sb_err), np.array(n_pix)

def predicted_excess_map(ds_fit, sky_model_only):
    """
    Predicted (background-subtracted) counts map for a SINGLE spatial+spectral
    model, using the same exposure/PSF/edisp as ds_fit and the dataset's fitted
    background normalisation. Because it is evaluated on the full pixel grid
    with no Poisson noise, this radially bins into a smooth curve -- the
    "expected excess" analog of the green/red model lines in the reference plot.
    """
    bkg_model = [m for m in ds_fit.models if isinstance(m, FoVBackgroundModel)][0]
    ds_pred = ds_fit.copy(name=f"pred-{sky_model_only.name}")
    ds_pred.models = Models([sky_model_only, bkg_model])
    npred_total = ds_pred.npred()
    npred_bkg   = ds_fit.background.data * bkg_model.spectral_model.norm.value
    m = npred_total.copy()
    m.data = np.clip(npred_total.data - npred_bkg, a_min=0, a_max=None)
    return m

r_deg_data, sb_data, sb_err_data, _ = radial_profile_density(
    dataset_sim_halo.excess, SOURCE_CENTER.icrs, n_bins=25)
r_pc_data = deg_to_pc(r_deg_data)

model_tmpl_map  = predicted_excess_map(ds_tmpl_fit,  sky_tmpl_fit)
model_gauss_map = predicted_excess_map(ds_gauss_fit, sky_gauss_fit)

r_deg_tmpl,  sb_tmpl,  _, _ = radial_profile_density(model_tmpl_map,  SOURCE_CENTER.icrs, n_bins=25)
r_deg_gauss, sb_gauss, _, _ = radial_profile_density(model_gauss_map, SOURCE_CENTER.icrs, n_bins=25)

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(r_deg_gauss, sb_gauss, "-", color="#2ca02c", lw=2.2,
        label="Gaussian-model expected excess")
ax.plot(r_deg_tmpl,  sb_tmpl,  "-", color="#b2182b", lw=2.2,
        label="Halo-model expected excess")
ax.errorbar(r_deg_data, sb_data, yerr=sb_err_data, fmt="o", color="steelblue",
            ms=5, elinewidth=1, capsize=2.5, label="Measured source excess profile")
ax.axhline(0, color="grey", lw=1, ls=":")
ax.set_xlabel("Distance from source [deg]", fontsize=11)
ax.set_ylabel("Excess surface density  [counts deg$^{-2}$]", fontsize=11)
ax.set_title("Radial excess profile — CTA halo-only sim\n"
             "Gaussian and halo expectations compared with the measured excess profile",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)

def _deg2pc_fwd(x): return deg_to_pc(x)
def _pc2deg_inv(x): return np.degrees(np.arctan(x / (DIST_KPC * 1000.0)))
secax = ax.secondary_xaxis("top", functions=(_deg2pc_fwd, _pc2deg_inv))
secax.set_xlabel(f"Projected distance [pc]  (assuming d = {DIST_KPC} kpc)", fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_radial_profile_excess.png", dpi=150, bbox_inches="tight")
plt.show()

print("Radial profile bin centers [deg] -> [pc]:")
for rd, rp in zip(r_deg_data, r_pc_data):
    print(f"   {rd:6.3f}°  ->  {rp:8.2f} pc")


energy_edges_fp = np.geomspace(E_MIN.to_value(u.TeV), E_MAX.to_value(u.TeV), 14) * u.TeV

for model in ds_halo_H1.models:
    if isinstance(model, FoVBackgroundModel):
        model.spectral_model.norm.frozen = True

fpe_halo = FluxPointsEstimator(energy_edges=energy_edges_fp,
                                source="halo-H1", selection_optional=["ul"])
fp_halo  = fpe_halo.run([ds_halo_H1])

fig, ax = plt.subplots(figsize=(9, 6))
fp_halo.plot(ax=ax, sed_type="e2dnde", color="#b2182b",
             marker="s", markersize=5, elinewidth=0.9, capsize=2.5,
             label="Flux points — TeV Halo (H₁)")
halo_H1.spectral_model.plot(
    ax=ax, energy_bounds=[HALO_E_MIN, HALO_E_MAX], energy_power=2,
    color="#b2182b", linewidth=1.8, label=f"Halo {sp_label} best-fit")
halo_H1.spectral_model.plot_error(
    ax=ax, energy_bounds=[HALO_E_MIN, HALO_E_MAX], energy_power=2,
    facecolor="#b2182b", alpha=0.18)
sky_halo_true.spectral_model.plot(
    ax=ax, energy_bounds=[HALO_E_MIN, HALO_E_MAX], energy_power=2,
    color="black", linewidth=1.2, linestyle="--", label="True halo (injected)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Energy [TeV]", fontsize=12)
ax.set_ylabel(r"$E^2\,dN/dE$  [TeV cm$^{-2}$ s$^{-1}$]", fontsize=12)
ax.set_xlim(0.5, 1200); ax.set_ylim(1e-14, 3e-11)
ax.set_title(
    f"Halo-Only SED (CTA)\n"
    f"TS_halo = {ts_halo_det:.1f}  ({np.sqrt(ts_halo_det):.1f}σ)  |  "
    f"Fitted size = {fitted_size_str} ({sp_label})",
    fontsize=11, fontweight="bold")
ax.grid(which="both", alpha=0.3); ax.legend(fontsize=10)

_fit_idx    = halo_H1.spectral_model.index.value
_fit_amp    = halo_H1.spectral_model.amplitude.value
_fit_ref    = halo_H1.spectral_model.reference.quantity.to_value(u.TeV)
_param_txt  = (
    f"Fitted:  index = {_fit_idx:.2f}   "
    f"amp @ {_fit_ref:.2g} TeV = {_fit_amp:.2e} cm$^{{-2}}$s$^{{-1}}$TeV$^{{-1}}$   "
    f"size ({sp_label}) = {fitted_size_str}\n"
    f"HESS J1837-069 (injected):  index = {HALO_ALPHA:.2f}   "
    f"amp @ {HALO_E0.to_value(u.TeV):.2g} TeV = {HALO_AMPLITUDE.value:.2e} cm$^{{-2}}$s$^{{-1}}$TeV$^{{-1}}$   "
    f"θ_d0 = {HALO_THETA_D0.value:.3f}°"
)
ax.text(0.02, 0.02, _param_txt, transform=ax.transAxes, fontsize=8,
        ha="left", va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.88))

_hess_pwn_data = np.array([
    [0.222308, 2.82024e-10, 1.44221e-10, 1.54296e-10],
    [0.288461, 3.93396e-11, 2.05788e-11, 2.13974e-11],
    [0.422034, 3.21142e-11, 6.44578e-12, 6.67094e-12],
    [0.610243, 2.07182e-11, 2.93353e-12, 3.04057e-12],
    [0.884837, 6.57914e-12, 1.12113e-12, 1.17069e-12],
    [1.29498,  3.28387e-12, 5.30865e-13, 5.58275e-13],
    [1.88953,  1.089e-12,   2.4159e-13,  2.56409e-13],
    [2.74707,  4.72377e-13, 1.34714e-13, 1.44667e-13],
    [4.19217,  2.81362e-13, 6.63316e-14, 7.24583e-14],
    [5.99275,  8.35974e-14, 2.83258e-14, 3.21689e-14],
    [8.54779,  3.40375e-14, 1.2569e-14,  1.49668e-14],
    [16.4452,  5.42638e-15, 2.47605e-15, 3.52043e-15],
])
_e_hess        = _hess_pwn_data[:, 0]
_e2dnde_hess   = _e_hess**2 * _hess_pwn_data[:, 1]
_e2dnde_err_lo = _e_hess**2 * _hess_pwn_data[:, 2]
_e2dnde_err_hi = _e_hess**2 * _hess_pwn_data[:, 3]
ax.errorbar(_e_hess, _e2dnde_hess, yerr=[_e2dnde_err_lo, _e2dnde_err_hi],
            fmt="D", color="black", ms=6, elinewidth=1.3, capsize=3, alpha=0.85,
            label="H.E.S.S. (real, total system)", zorder=5)
ax.legend(fontsize=10)

plt.savefig(OUTPUT_DIR / "cta_halo_only_sed.png", dpi=150, bbox_inches="tight")
plt.tight_layout(); plt.show()


# Band: fitted SED envelope. Flux points: one set per background scale.
import warnings
warnings.filterwarnings("ignore", message="The filename is not defined")

energy_edges_fp_bkg = np.geomspace(E_MIN.to_value(u.TeV), E_MAX.to_value(u.TeV), 10) * u.TeV
_E_DENSE = np.geomspace(E_MIN.to_value(u.TeV) * 0.8, E_MAX.to_value(u.TeV) * 1.2, 150) * u.TeV

bkg_scales_sed = [0.80, 1.00, 1.20]
_e2dnde_sed    = {}
_fp_per_scale  = {}   # bkg_scale → FluxPoints
_fp_cols       = {0.80: "#2166ac", 1.00: "black", 1.20: "#d6604d"}
_fp_labels     = {0.80: "Flux points (bkg=0.80×)", 1.00: "Flux points (nominal)", 1.20: "Flux points (bkg=1.20×)"}

for bkg_s in bkg_scales_sed:
    ds_s = dataset.copy(name=f"sed_bkg{int(bkg_s*100)}")
    if ds_s.mask_fit is None:
        ds_s.mask_fit = ds_s.mask_safe.copy()
    h_s = SkyModel(
        spatial_model=TemplateSpatialModel(phi_map, normalize=True),
        spectral_model=PowerLawSpectralModel(
            amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
        name=f"halo-sed-{int(bkg_s*100)}")
    bkg_true = FoVBackgroundModel(dataset_name=ds_s.name)
    bkg_true.spectral_model.norm.value  = bkg_s
    bkg_true.spectral_model.norm.frozen = True
    ds_s.models = Models([h_s, bkg_true])
    ds_s.fake(random_state=SEED + int(bkg_s * 100))
    bkg_true.spectral_model.norm.value  = 1.0
    bkg_true.spectral_model.norm.frozen = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds_s])
    _dnde   = h_s.spectral_model(_E_DENSE).to("cm-2 s-1 TeV-1")
    _e2dnde = (_dnde * _E_DENSE.to("TeV")**2).to("TeV cm-2 s-1")
    _e2dnde_sed[bkg_s] = _e2dnde.value
    # Flux points for each scale
    fpe = FluxPointsEstimator(
        energy_edges=energy_edges_fp_bkg, source=h_s.name, selection_optional=["ul"])
    _fp_per_scale[bkg_s] = fpe.run([ds_s])
    print(f"  bkg_scale={bkg_s:.2f} → fitted, flux points computed")

fig, ax = plt.subplots(figsize=(10, 6))

# True injected spectrum
sky_halo_true.spectral_model.plot(
    ax=ax, energy_bounds=[HALO_E_MIN, HALO_E_MAX], energy_power=2,
    color="black", lw=1.5, ls=":", label="Injected spectrum (true, fixed -- does not vary with bkg)")

# ±20% bkg uncertainty band
ax.fill_between(_E_DENSE.to_value(u.TeV),
                _e2dnde_sed[0.80], _e2dnde_sed[1.20],
                alpha=0.20, color="steelblue", label="Fitted SED range (bkg scaled 0.8x-1.2x)")

# Nominal fitted SED line
ax.plot(_E_DENSE.to_value(u.TeV), _e2dnde_sed[1.00],
        color="darkorange", lw=2, ls="-", label="Nominal fitted SED")

# Flux points for each bkg scale
_markers = {0.80: "^", 1.00: "o", 1.20: "v"}
for bkg_s in bkg_scales_sed:
    _fp_per_scale[bkg_s].plot(
        ax=ax, sed_type="e2dnde", color=_fp_cols[bkg_s],
        marker=_markers[bkg_s], markersize=5, elinewidth=0.9, capsize=2.5, ls="none",
        label=_fp_labels[bkg_s])

ax.text(0.02, 0.03,
        "Shaded band = spread of FITTED SEDs across background\n"
        "scale scenarios (0.8x-1.2x). Dotted line = fixed injected\n"
        "truth shown for reference -- it does not vary with bkg.",
        transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
        bbox=dict(boxstyle="round", fc="white", alpha=0.85))

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Energy [TeV]", fontsize=13)
ax.set_ylabel(r"$E^2\,dN/dE$  [TeV cm$^{-2}$ s$^{-1}$]", fontsize=13)
ax.set_xlim(E_MIN.to_value(u.TeV) * 0.8, E_MAX.to_value(u.TeV) * 1.2)
ax.set_title(
    "Effect of ±20% Background Scaling on Halo SED — CTA | HESS J1837-069\n"
    f"Band = fitted SED envelope  |  θ_d0={HALO_THETA_D0.value:.2f}°"
    f"  |  d={DIST_KPC} kpc  |  t={T_AGE_YR/1e3:.0f} kyr  [FIXED]",
    fontsize=11, fontweight="bold")
ax.legend(fontsize=10); ax.grid(which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_bkg_sed_band.png", dpi=150, bbox_inches="tight")
plt.show()


import warnings
warnings.filterwarnings("ignore", message="The filename is not defined")

_e_edges_sig  = np.geomspace(1, 300, 7) * u.TeV
e_centers_sig = np.array([
    (_e_edges_sig[i].value * _e_edges_sig[i+1].value)**0.5
    for i in range(len(_e_edges_sig) - 1)])

bkg_scales_sig = [0.80, 1.00, 1.20]
_sig_per_bin   = {}

for bkg_s in bkg_scales_sig:
    ds_sig = dataset.copy(name=f"sig_en_{int(bkg_s*100)}")
    if ds_sig.mask_fit is None:
        ds_sig.mask_fit = ds_sig.mask_safe.copy()
    h_sig  = SkyModel(
        spatial_model=TemplateSpatialModel(phi_map, normalize=True),
        spectral_model=PowerLawSpectralModel(
            amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
        name=f"halo-sig-{int(bkg_s*100)}")
    bkg_sig = FoVBackgroundModel(dataset_name=ds_sig.name)
    bkg_sig.spectral_model.norm.value  = bkg_s
    bkg_sig.spectral_model.norm.frozen = True
    ds_sig.models = Models([h_sig, bkg_sig])
    ds_sig.fake(random_state=SEED + int(bkg_s * 100) + 50)
    bkg_sig.spectral_model.norm.value  = 1.0
    bkg_sig.spectral_model.norm.frozen = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds_sig])
    # FPE: per-bin TS = likelihood ratio (amplitude=0 vs best-fit amplitude)
    fpe_sig = FluxPointsEstimator(
        energy_edges=_e_edges_sig, source=h_sig.name, selection_optional=[])
    fp_sig  = fpe_sig.run([ds_sig])
    try:
        _ts = np.asarray(fp_sig.ts).flatten()
    except Exception:
        _ts = np.asarray(fp_sig.to_table()["ts"]).flatten()
    _sig_per_bin[bkg_s] = np.sqrt(np.maximum(_ts, 0))
    print(f"  bkg={bkg_s:.2f}: σ per bin = {np.round(_sig_per_bin[bkg_s], 1)}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(e_centers_sig,
                _sig_per_bin[0.80], _sig_per_bin[1.20],
                alpha=0.25, color="steelblue", label="±20% bkg uncertainty band")
ax.semilogx(e_centers_sig, _sig_per_bin[1.00], "o-",
            color="black", lw=2.5, ms=7, label="Nominal σ per energy bin")
ax.axhline(5, ls="--", c="firebrick", lw=1.5, label="5σ threshold")
ax.fill_between([e_centers_sig[0]*0.5, e_centers_sig[-1]*2], 0, 5,
                alpha=0.08, color="firebrick")
ax.set_xlabel("Energy [TeV]", fontsize=13)
ax.set_ylabel("Halo significance σ per energy bin", fontsize=13)
ax.set_title(
    "Halo TS significance per energy bin under ±20% bkg scaling — CTA\n"
    f"(FluxPointsEstimator LR-TS)  |  θ_d0={HALO_THETA_D0.value:.2f}°"
    f"  |  d={DIST_KPC} kpc  |  t={T_AGE_YR/1e3:.0f} kyr  [FIXED]",
    fontsize=11, fontweight="bold")
ax.legend(fontsize=11); ax.grid(alpha=0.3)
ax.set_xlim(e_centers_sig[0]*0.5, e_centers_sig[-1]*2)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_sig_vs_ebin_fpe.png", dpi=150, bbox_inches="tight")
plt.show()


AMP_SCALE_LOW = 0.02
amp_low = HALO_AMPLITUDE * AMP_SCALE_LOW

ds_lowamp = dataset.copy(name="halo_lowamp_test_cta")
if ds_lowamp.mask_fit is None:
    ds_lowamp.mask_fit = ds_lowamp.mask_safe.copy()
h_lowamp_true = SkyModel(
    spatial_model=TemplateSpatialModel(phi_map, normalize=True),
    spectral_model=PowerLawSpectralModel(amplitude=amp_low, index=HALO_ALPHA, reference=HALO_E0),
    name="halo-lowamp-true-cta")
ds_lowamp.models = Models([h_lowamp_true, FoVBackgroundModel(dataset_name=ds_lowamp.name)])
ds_lowamp.fake(random_state=SEED + 777)

h_lowamp_fit = SkyModel(
    spatial_model=TemplateSpatialModel(phi_map, normalize=True),
    spectral_model=PowerLawSpectralModel(amplitude=amp_low, index=HALO_ALPHA, reference=HALO_E0),
    name="halo-lowamp-fit-cta")
ds_lowamp.models = Models([h_lowamp_fit, FoVBackgroundModel(dataset_name=ds_lowamp.name)])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds_lowamp])

ds_null_lowamp = ds_lowamp.copy(name="null_lowamp_cta")
ds_null_lowamp.models = Models([FoVBackgroundModel(dataset_name=ds_null_lowamp.name)])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds_null_lowamp])
ts_det_lowamp = max(0.0, ds_null_lowamp.stat_sum() - ds_lowamp.stat_sum())
sigma_lowamp = np.sqrt(ts_det_lowamp)

print(f"[CTA] Amplitude = {AMP_SCALE_LOW*100:.0f}% of nominal ({amp_low:.2e})")
print(f"  TS_det = {ts_det_lowamp:.2f}  ({sigma_lowamp:.2f}sigma)")
print(f"  Fitted amplitude = {h_lowamp_fit.spectral_model.amplitude.value:.3e} "
      f"(true = {amp_low.value:.3e})")
print(f"  Fitted index     = {h_lowamp_fit.spectral_model.index.value:.2f} "
      f"(true = {HALO_ALPHA})")

fig, ax = plt.subplots(figsize=(9, 6))
energy_edges_lowamp = np.geomspace(E_MIN.to_value(u.TeV), E_MAX.to_value(u.TeV), 10) * u.TeV
fpe_lowamp = FluxPointsEstimator(energy_edges=energy_edges_lowamp,
                                  source="halo-lowamp-fit-cta", selection_optional=["ul"])
fp_lowamp = fpe_lowamp.run([ds_lowamp])
fp_lowamp.plot(ax=ax, sed_type="e2dnde", color="crimson",
               marker="o", markersize=5, elinewidth=0.9, capsize=2.5,
               label=f"Flux points (amp={AMP_SCALE_LOW*100:.0f}% nominal)")
h_lowamp_true.spectral_model.plot(ax=ax, energy_bounds=[HALO_E_MIN, HALO_E_MAX], energy_power=2,
                                   color="black", lw=1.5, ls=":", label="Injected (true, 2% amp)")
h_lowamp_fit.spectral_model.plot(ax=ax, energy_bounds=[HALO_E_MIN, HALO_E_MAX], energy_power=2,
                                  color="crimson", lw=2, label="Recovered fit (2% amp)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Energy [TeV]", fontsize=13)
ax.set_ylabel(r"$E^2\,dN/dE$  [TeV cm$^{-2}$ s$^{-1}$]", fontsize=13)
ax.set_title(f"Low-Amplitude Recovery Test -- CTA | HESS J1837-069\n"
             f"Amplitude = {AMP_SCALE_LOW*100:.0f}% of nominal  |  "
             f"TS_det={ts_det_lowamp:.1f} ({sigma_lowamp:.1f}sigma)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10); ax.grid(which="both", alpha=0.3)

_param_txt_lowamp = (
    f"Fitted:  amp={h_lowamp_fit.spectral_model.amplitude.value:.2e}   "
    f"index={h_lowamp_fit.spectral_model.index.value:.2f}\n"
    f"HESS J1837-069 (injected, {AMP_SCALE_LOW*100:.0f}% amp):  "
    f"amp={amp_low.value:.2e}   index={HALO_ALPHA:.2f}"
)
ax.text(0.02, 0.02, _param_txt_lowamp, transform=ax.transAxes, fontsize=8,
        ha="left", va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.88))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_lowamp_recovery_test.png", dpi=150, bbox_inches="tight")
plt.show()


INDEX_SCAN_VALUES = [1.8, 2.2, 2.54, 3.0, 3.5]
index_scan_results_cta = {}


for true_index in INDEX_SCAN_VALUES:
    _tagstr = f"{true_index:.2f}".replace(".", "p")
    ds_idx = dataset.copy(name=f"idxscan_cta_{_tagstr}")
    if ds_idx.mask_fit is None:
        ds_idx.mask_fit = ds_idx.mask_safe.copy()
    h_idx_true = SkyModel(
        spatial_model=TemplateSpatialModel(phi_map, normalize=True),
        spectral_model=PowerLawSpectralModel(amplitude=HALO_AMPLITUDE, index=true_index, reference=HALO_E0),
        name=f"halo-idx-true-cta-{_tagstr}")
    ds_idx.models = Models([h_idx_true, FoVBackgroundModel(dataset_name=ds_idx.name)])
    ds_idx.fake(random_state=SEED + int(true_index * 100))

    h_idx_fit = SkyModel(
        spatial_model=TemplateSpatialModel(phi_map, normalize=True),
        spectral_model=PowerLawSpectralModel(amplitude=HALO_AMPLITUDE, index=2.54, reference=HALO_E0),
        name=f"halo-idx-fit-cta-{_tagstr}")
    ds_idx.models = Models([h_idx_fit, FoVBackgroundModel(dataset_name=ds_idx.name)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds_idx])

    index_scan_results_cta[true_index] = dict(
        fitted_index=h_idx_fit.spectral_model.index.value,
        fitted_index_err=h_idx_fit.spectral_model.index.error,
        fitted_amplitude=h_idx_fit.spectral_model.amplitude.value,
    )
    print(f"[CTA] true_index={true_index:.2f} -> fitted_index="
          f"{index_scan_results_cta[true_index]['fitted_index']:.2f} "
          f"+/- {index_scan_results_cta[true_index]['fitted_index_err']:.2f}")

np.save(str(OUTPUT_DIR / "cta_index_scan_results.npy"), index_scan_results_cta, allow_pickle=True)

fig, ax = plt.subplots(figsize=(7, 6))
true_vals = list(index_scan_results_cta.keys())
fit_vals  = [index_scan_results_cta[t]["fitted_index"] for t in true_vals]
fit_errs  = [index_scan_results_cta[t]["fitted_index_err"] for t in true_vals]
ax.errorbar(true_vals, fit_vals, yerr=fit_errs, fmt="o-", color="#00883A",
            capsize=4, ms=8, lw=2, label="CTA recovered index")
ax.plot([1.5, 3.7], [1.5, 3.7], "k--", lw=1, label="1:1 (perfect recovery)")

if HALO_ALPHA in index_scan_results_cta:
    ax.scatter([HALO_ALPHA], [index_scan_results_cta[HALO_ALPHA]["fitted_index"]],
               marker="*", s=300, color="gold", edgecolors="black", linewidths=1.2,
               zorder=10, label=f"HESS J1837-069 (index={HALO_ALPHA})")
else:
    ax.axvline(HALO_ALPHA, ls=":", c="green", lw=1.5,
               label=f"HESS J1837-069 index={HALO_ALPHA}")

ax.set_xlabel("Injected (true) spectral index", fontsize=13)
ax.set_ylabel("Fitted (recovered) spectral index", fontsize=13)
ax.set_title("CTA -- Recovered vs. Injected Spectral Index\n(amplitude fixed at nominal)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_index_recovery_scan.png", dpi=150, bbox_inches="tight")
plt.show()


from IPython.utils.capture import capture_output as _capture

_edat_key     = "edat_scan"
_cache_loaded = False

E_REF         = 1.0 * u.TeV

E_DAT_VALUES = np.geomspace(10, 1000, 12)  # TeV

if CACHE_FILE.exists() and not FORCE_RERUN:
    try:
        _c = np.load(CACHE_FILE, allow_pickle=True)
        if _edat_key + "_edat" in _c:
            E_DAT_VALUES      = _c[_edat_key + "_edat"]
            theta_true_edat   = list(_c[_edat_key + "_theta_true"])
            theta_fit_edat    = list(_c[_edat_key + "_theta_fit"])
            ts_edat           = list(_c[_edat_key + "_ts"])
            ts_edat_pwn       = list(_c[_edat_key + "_ts_pwn"]) if _edat_key+"_ts_pwn" in _c else [0]*len(ts_edat)
            _cache_loaded = True
            print(f"✓ E_dat scan loaded from cache")
    except Exception as e:
        print(f"  Cache read failed ({e}), re-running ...")

if not _cache_loaded:
    theta_true_edat = []
    theta_fit_edat  = []
    ts_edat         = []
    ts_edat_pwn     = []

    print(f"{'E_dat [TeV]':>12}  {'θ_true [°]':>11}  {'TS_halo':>9}  {'TS_pwn':>9}")
    print("-" * 55)

    for i_e, edat in enumerate(E_DAT_VALUES):
        theta_phys = diffusion_angle_deg(edat, D0_cm2_s, DELTA_DIFF, t_age_s, dist_cm, E0_TeV)
        theta_true_edat.append(theta_phys)

        phi_e, _ = build_scan_halo_template(geom, SOURCE_CENTER.icrs, theta_d0_deg=theta_phys)

        ds_e = dataset.copy(name=f"edat_{i_e}")
        halo_e_sim = SkyModel(
            spatial_model=TemplateSpatialModel(phi_e, normalize=True),
            spectral_model=PowerLawSpectralModel(
                amplitude=HALO_AMPLITUDE,
                index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-edat-sim-{i_e}")
        with _capture():
            ds_e.models = Models([halo_e_sim, FoVBackgroundModel(dataset_name=ds_e.name)])
        ds_e.fake(random_state=SEED + 200 + i_e)

        # Fit extension
        ds_efit = ds_e.copy(name=f"edat_fit_{i_e}")
        if USE_GAUSSIAN:
            _sp_e = GaussianSpatialModel(
                lon_0=SOURCE_CENTER.icrs.ra, lat_0=SOURCE_CENTER.icrs.dec,
                sigma=max(0.05, theta_phys)*u.deg, frame="icrs")
        else:
            _sp_e = DiskSpatialModel(
                lon_0=SOURCE_CENTER.icrs.ra, lat_0=SOURCE_CENTER.icrs.dec,
                r_0=max(0.05, theta_phys)*u.deg, frame="icrs")
        _sp_e.lon_0.frozen = True; _sp_e.lat_0.frozen = True
        halo_efit = SkyModel(
            spatial_model=_sp_e,
            spectral_model=PowerLawSpectralModel(
                amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-edat-fit-{i_e}")
        ds_efit.models = Models([halo_efit, FoVBackgroundModel(dataset_name=ds_efit.name)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_efit])
            Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_efit])
        fitted_sz = (_sp_e.sigma.value if USE_GAUSSIAN else _sp_e.r_0.value)
        theta_fit_edat.append(fitted_sz)

        ts_e = run_halo_only(ds_e, phi_e, seed_offset=i_e*50)
        ts_edat.append(ts_e)

        ds_e_pwn = dataset.copy(name=f"edat_pwn_{i_e}")
        ec_e_pwn = _make_pwn_model(f"pwn-edat-{i_e}", ds_e_pwn.counts.geom)
        halo_e_pwn = SkyModel(
            spatial_model=TemplateSpatialModel(phi_e, normalize=True),
            spectral_model=PowerLawSpectralModel(
                amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-edatpwn-{i_e}")
        with _capture():
            ds_e_pwn.models = Models([ec_e_pwn, halo_e_pwn,
                                      FoVBackgroundModel(dataset_name=ds_e_pwn.name)])
        ds_e_pwn.fake(random_state=SEED + 700 + i_e)
        ts_e_pwn = run_pwn_halo(ds_e_pwn, phi_e, seed_offset=i_e*50 + 500)
        ts_edat_pwn.append(ts_e_pwn)

        print(f"{edat:12.1f}  {theta_phys:11.3f}  {ts_e:9.2f}  {ts_e_pwn:9.2f}")

    _ex = dict(np.load(CACHE_FILE, allow_pickle=True)) if CACHE_FILE.exists() else {}
    _ex[_edat_key+"_edat"]       = np.array(E_DAT_VALUES)
    _ex[_edat_key+"_theta_true"] = np.array(theta_true_edat)
    _ex[_edat_key+"_theta_fit"]  = np.array(theta_fit_edat)
    _ex[_edat_key+"_ts"]         = np.array(ts_edat)
    _ex[_edat_key+"_ts_pwn"]     = np.array(ts_edat_pwn)
    np.savez(CACHE_FILE, **_ex)
    print(f"\n✓ E_dat scan saved")

E_DAT_VALUES    = np.array(E_DAT_VALUES)
theta_true_edat = np.array(theta_true_edat)
theta_fit_edat  = np.array(theta_fit_edat)
ts_edat         = np.array(ts_edat)
ts_edat_pwn     = np.array(ts_edat_pwn)
sigma_edat      = np.sqrt(ts_edat)
sigma_edat_pwn  = np.sqrt(ts_edat_pwn)


_ptxt = (f"D₀={D0_cm2_s:.0e} cm²/s  |  d={DIST_KPC} kpc  |  "
         f"age={T_AGE_YR/1e3:.0f} kyr  [FIXED]")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Panel 0: true vs fitted extension
ax = axes[0]
ax.semilogx(E_DAT_VALUES, theta_true_edat, "o-", color="steelblue", lw=2, ms=7, label="True θ_d0")
ax.semilogx(E_DAT_VALUES, theta_fit_edat,  "s--", color="darkorange", lw=2, ms=7, label="Fitted θ")
ax.axhline(HALO_THETA_D0.to_value(u.deg), ls="--", c="green", lw=1.5,
           label=f"HESS J1837-069 θ_d0={HALO_THETA_D0.value:.2f}°")
ax.set_xlabel("Electron energy  E_dat  [TeV]\n(1 TeV = 10¹² eV)", fontsize=11); ax.set_ylabel("Extension [°]", fontsize=12)
ax.set_title("True & Fitted Extension vs E_dat", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 1: detected extension colored by significance
ax = axes[1]
sc = ax.scatter(E_DAT_VALUES, theta_fit_edat, c=sigma_edat, cmap='YlOrRd',
                s=100, zorder=5, edgecolors='k', linewidths=0.5)
ax.semilogx(E_DAT_VALUES, theta_fit_edat, "-", color="grey", lw=1, alpha=0.5)
ax.axhline(HALO_THETA_D0.to_value(u.deg), ls="--", c="green", lw=1.5,
           label=f"HESS J1837-069 θ_d0={HALO_THETA_D0.value:.2f}°")
fig.colorbar(sc, ax=ax).set_label("Significance (halo-only) [σ]", fontsize=9)
ax.set_xlabel("Electron energy  E_dat  [TeV]\n(1 TeV = 10¹² eV)", fontsize=11); ax.set_ylabel("Detected extension [°]", fontsize=12)
ax.set_title("Detected Extension vs E_dat", fontsize=11, fontweight="bold"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 2: significance vs E_dat — HALO ONLY + PWN+HALO on same axes
ax = axes[2]
ax.semilogx(E_DAT_VALUES, sigma_edat,     "o-",  color="steelblue",  lw=2, ms=8, label="Halo-only σ")
ax.semilogx(E_DAT_VALUES, sigma_edat_pwn, "s--", color="darkorange", lw=2, ms=8, label="PWN+Halo σ")
ax.axhline(5, ls="--", c="black", lw=1.5, label="5σ threshold")
ax.axvline(E_DAT_VALUES[np.argmax(ts_edat)], ls=":", c="steelblue", lw=1.5,
           label=f"Opt E_dat (halo)={E_DAT_VALUES[np.argmax(ts_edat)]:.0f} TeV")
ax.set_xlabel("Electron energy  E_dat  [TeV]\n(1 TeV = 10¹² eV)", fontsize=11); ax.set_ylabel("Detection significance [σ]", fontsize=12)
ax.set_title("Significance vs E_dat\nHalo-only vs PWN+Halo", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

axes[0].text(0.02, 0.97, _ptxt, transform=axes[0].transAxes, fontsize=9,
             ha='left', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

plt.suptitle(
    f"E_dat Scan — CTA  |  HESS J1837-069  (D₀={D0_cm2_s:.0e} cm²/s, d={DIST_KPC} kpc)",
    fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_edat_scan.png", dpi=150, bbox_inches="tight")
plt.show()


i_opt        = np.argmax(ts_edat)
E_DAT_OPT    = E_DAT_VALUES[i_opt]
THETA_D0_OPT = theta_true_edat[i_opt] * u.deg
print(f"\n→ Optimal E_dat = {E_DAT_OPT:.1f} TeV")
print(f"   Fitted θ_d0   = {theta_fit_edat[i_opt]:.3f}°")
print(f"   TS at optimum = {ts_edat[i_opt]:.1f}  ({sigma_edat[i_opt]:.1f}σ)  [halo-only]")
print(f"   TS at optimum = {ts_edat_pwn[i_opt]:.1f}  ({sigma_edat_pwn[i_opt]:.1f}σ)  [PWN+halo]")


print(f"   E_dat-scan argmax (diagnostic only): E_dat={E_DAT_OPT:.1f} TeV, "
      f"θ_d0={THETA_D0_OPT:.3f} -- NOT applied; fixed size {HALO_THETA_D0:.3f} "
      f"(D100={HALO_D100_CALIBRATED_CM2_S:.3e} cm²/s) is used for all scans below.")


# E_dat and D₀ are FIXED — only age varies
# E_DAT_TEV (locked), D0_cm2_s (locked), DIST_KPC (locked)
_age_key      = "age_scan_halo"
_cache_loaded = False
AGE_VALUES    = np.linspace(10e3, 500e3, 12)  # yr

if CACHE_FILE.exists() and not FORCE_RERUN:
    try:
        _c = np.load(CACHE_FILE, allow_pickle=True)
        if _age_key + "_ages" in _c:
            AGE_VALUES     = _c[_age_key + "_ages"]
            ts_age_halo    = list(_c[_age_key + "_ts"])
            theta_age_vals = list(_c[_age_key + "_theta"])
            _cache_loaded  = True
            print(f"✓ Age scan loaded from cache")
    except Exception as e:
        print(f"  Cache read failed ({e}), re-running ...")

if not _cache_loaded:
    ts_age_halo    = []
    theta_age_vals = []

    print(f"{'Age [kyr]':>10}  {'θ_d0 [°]':>9}  {'TS':>9}  {'σ':>7}")
    print("-" * 44)

    for i_age, t_yr in enumerate(AGE_VALUES):
        t_s_age    = t_yr * YR_TO_S
        theta_this = diffusion_angle_deg(E_DAT_OPT, D0_cm2_s, DELTA_DIFF, t_s_age, dist_cm, E0_TeV)
        theta_age_vals.append(theta_this)

        phi_age, _ = build_scan_halo_template(geom, SOURCE_CENTER.icrs, theta_d0_deg=theta_this)

        ds_age = dataset.copy(name=f"age_{i_age}")
        halo_age_sim = SkyModel(
            spatial_model=TemplateSpatialModel(phi_age, normalize=True),
            spectral_model=PowerLawSpectralModel(
                amplitude=HALO_AMPLITUDE,
                index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-age-{i_age}")
        with _capture():
            ds_age.models = Models([halo_age_sim, FoVBackgroundModel(dataset_name=ds_age.name)])
        ds_age.fake(random_state=SEED + 50 + i_age)

        ts_h = run_halo_only(ds_age, phi_age, seed_offset=i_age*500)
        ts_age_halo.append(ts_h)
        print(f"{t_yr/1e3:>10.1f}  {theta_this:>9.3f}  {ts_h:>9.2f}  {np.sqrt(ts_h):>7.1f}σ")

    _ex = dict(np.load(CACHE_FILE, allow_pickle=True)) if CACHE_FILE.exists() else {}
    _ex[_age_key+"_ages"]  = np.array(AGE_VALUES)
    _ex[_age_key+"_ts"]    = np.array(ts_age_halo)
    _ex[_age_key+"_theta"] = np.array(theta_age_vals)
    np.savez(CACHE_FILE, **_ex)
    print(f"\n✓ Age scan saved")

AGE_VALUES     = np.array(AGE_VALUES)
ts_age_halo    = np.array(ts_age_halo)
theta_age_vals = np.array(theta_age_vals)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(AGE_VALUES/1e3, theta_age_vals, "o-", color="steelblue", lw=2, ms=7)
axes[0].set_xlabel("Pulsar age [kyr]", fontsize=12)
axes[0].set_ylabel("θ_d0 [°]", fontsize=12)
axes[0].set_title("Extension vs age  (θ_d0 ∝ √(D₀·t)·E_dat^(−δ)/d)",
                   fontsize=11, fontweight="bold")
axes[0].axvline(T_AGE_YR/1e3, ls=":", c="grey", lw=1.5, label=f"Locked={T_AGE_YR/1e3:.0f} kyr")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

axes[1].plot(AGE_VALUES/1e3, np.sqrt(ts_age_halo), "o-", color="pink", lw=2, ms=8, label="Halo σ")
axes[1].axhline(5, ls="--", c="black", lw=1.5, label="5σ")
axes[1].axvline(T_AGE_YR/1e3, ls=":", c="grey", lw=1.5, label=f"Ref age={T_AGE_YR/1e3:.0f} kyr")
axes[1].set_xlabel("Pulsar age [kyr]", fontsize=12)
axes[1].set_ylabel("Detection significance [σ]", fontsize=12)
axes[1].set_title("Halo detectability vs age\n(larger halo → lower surface brightness → harder at large ages)",
                   fontsize=11, fontweight="bold")
axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)

_ptxt_age = f"E_dat={E_DAT_TEV:.0f} TeV  |  D₀={D0_cm2_s:.0e}  |  d={DIST_KPC} kpc  [FIXED]"
axes[0].text(0.98, 0.97, _ptxt_age, transform=axes[0].transAxes, fontsize=8,
             ha='right', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

plt.suptitle(
    f"Age Scan — Halo Only (E_dat={E_DAT_OPT:.0f} TeV, D₀={D0_cm2_s:.0e} cm²/s, d={DIST_KPC} kpc)",
    fontsize=12, fontweight="bold")
plt.savefig(OUTPUT_DIR / "cta_age_scan_halo_only.png", dpi=150, bbox_inches="tight")
plt.tight_layout(); plt.show()

ts_age_halo_only   = ts_age_halo.copy()     # preserved for combined halo vs joint plot
theta_age_only     = theta_age_vals.copy()
AGE_VALUES_HALOONLY = AGE_VALUES.copy()


# Physical cooling time limit: above t_cool, halo size saturates (age-independent)
# t_cool = 1 / (b0 * E_dat)  where b0 ≈ 1.02e-3 TeV^-1 Myr^-1 (combined IC+sync at B=3μG)
b0_Myr = 1.02e-3  # TeV^-1 Myr^-1
t_cool_kyr = (1.0 / (b0_Myr * E_DAT_TEV)) * 1e3  # convert Myr → kyr
print(f"Cooling time limit at E_dat={E_DAT_TEV:.0f} TeV: t_cool ≈ {t_cool_kyr:.0f} kyr")
print(f"Above this age, halo size is age-independent (electrons have all cooled)")

i_age_opt  = np.argmax(ts_age_halo)
T_AGE_YR   = AGE_VALUES[i_age_opt]      # overwrite config default
t_age_s    = T_AGE_YR * YR_TO_S         # update derived quantity

print(f"✓ LOCKED: T_AGE_YR = {T_AGE_YR/1e3:.0f} kyr  (TS = {ts_age_halo[i_age_opt]:.1f})")
print(f"   Used in: distance scan, PWN+halo simulation, D₀ scan")


# E_dat and D₀ are FIXED — only distance varies
from IPython.utils.capture import capture_output as _capture
_dist_key     = "dist_scan"
_cache_loaded = False
DIST_VALUES   = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0])  # kpc

if CACHE_FILE.exists() and not FORCE_RERUN:
    try:
        _c = np.load(CACHE_FILE, allow_pickle=True)
        if _dist_key + "_dists" in _c:
            DIST_VALUES    = _c[_dist_key + "_dists"]
            ts_dist        = list(_c[_dist_key + "_ts"])
            ts_dist_pwn    = list(_c[_dist_key + "_ts_pwn"]) if _dist_key+"_ts_pwn" in _c else [0]*len(ts_dist)
            theta_dist     = list(_c[_dist_key + "_theta"])
            _cache_loaded  = True
            print(f"✓ Distance scan loaded from cache")
    except Exception as e:
        print(f"  Cache read failed ({e}), re-running ...")

if not _cache_loaded:
    ts_dist     = []
    ts_dist_pwn = []
    theta_dist  = []

    print(f"{'Dist [kpc]':>10}  {'θ_d0 [°]':>9}  {'TS_halo':>9}  {'TS_pwn':>9}")
    print("-" * 46)

    for i_d, dist_kpc in enumerate(DIST_VALUES):
        d_cm_scan  = dist_kpc * KPC_TO_CM
        theta_this = diffusion_angle_deg(E_DAT_OPT, D0_cm2_s, DELTA_DIFF, t_age_s, d_cm_scan, E0_TeV)
        theta_dist.append(theta_this)

        phi_dist, _ = build_scan_halo_template(geom, SOURCE_CENTER.icrs, theta_d0_deg=theta_this,
                                                distance_kpc=dist_kpc)

        ds_dist = dataset.copy(name=f"dist_{i_d}")
        halo_dist_sim = SkyModel(
            spatial_model=TemplateSpatialModel(phi_dist, normalize=True),
            spectral_model=PowerLawSpectralModel(
                amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-dist-{i_d}")
        with _capture():
            ds_dist.models = Models([halo_dist_sim, FoVBackgroundModel(dataset_name=ds_dist.name)])
        ds_dist.fake(random_state=SEED + 300 + i_d)
        ts_d = run_halo_only(ds_dist, phi_dist, seed_offset=i_d*600)
        ts_dist.append(ts_d)

        ds_dist_pwn = dataset.copy(name=f"dist_pwn_{i_d}")
        ec_dist = _make_pwn_model(f"pwn-dist-{i_d}", ds_dist_pwn.counts.geom)
        halo_dist_pwn = SkyModel(
            spatial_model=TemplateSpatialModel(phi_dist, normalize=True),
            spectral_model=PowerLawSpectralModel(
                amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-distpwn-{i_d}")
        with _capture():
            ds_dist_pwn.models = Models([ec_dist, halo_dist_pwn,
                                         FoVBackgroundModel(dataset_name=ds_dist_pwn.name)])
        ds_dist_pwn.fake(random_state=SEED + 800 + i_d)
        ts_d_pwn = run_pwn_halo(ds_dist_pwn, phi_dist, seed_offset=i_d*600 + 500)
        ts_dist_pwn.append(ts_d_pwn)
        print(f"{dist_kpc:>10.2f}  {theta_this:>9.3f}  {ts_d:>9.2f}  {ts_d_pwn:>9.2f}")

    _ex = dict(np.load(CACHE_FILE, allow_pickle=True)) if CACHE_FILE.exists() else {}
    _ex[_dist_key+"_dists"]  = np.array(DIST_VALUES)
    _ex[_dist_key+"_ts"]     = np.array(ts_dist)
    _ex[_dist_key+"_ts_pwn"] = np.array(ts_dist_pwn)
    _ex[_dist_key+"_theta"]  = np.array(theta_dist)
    np.savez(CACHE_FILE, **_ex)
    print(f"\n✓ Distance scan saved")

DIST_VALUES = np.array(DIST_VALUES)
ts_dist     = np.array(ts_dist)
ts_dist_pwn = np.array(ts_dist_pwn)
theta_dist  = np.array(theta_dist)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(DIST_VALUES, theta_dist, "o-", color="steelblue", lw=2, ms=7)
axes[0].axvline(DIST_KPC, ls=":", c="grey", lw=1.5, label=f"d={DIST_KPC} kpc")
axes[0].set_xlabel("Distance [kpc]", fontsize=12); axes[0].set_ylabel("θ_d0 [°]", fontsize=12)
axes[0].set_title("Extension vs distance", fontsize=11, fontweight="bold")
axes[0].legend(fontsize=10); axes[0].grid(alpha=0.3)

axes[1].plot(DIST_VALUES, np.sqrt(ts_dist),     "o-",  color="steelblue",  lw=2, ms=8, label="Halo-only σ")
axes[1].plot(DIST_VALUES, np.sqrt(ts_dist_pwn), "s--", color="darkorange", lw=2, ms=8, label="PWN+Halo σ")
axes[1].axhline(5, ls="--", c="black", lw=1.5, label="5σ threshold")
axes[1].axvline(DIST_KPC, ls=":", c="grey", lw=1.5, label=f"d={DIST_KPC} kpc")
axes[1].set_xlabel("Distance [kpc]", fontsize=12); axes[1].set_ylabel("Detection significance [σ]", fontsize=12)
axes[1].set_title("Detectability vs distance\nHalo-only vs PWN+Halo", fontsize=11, fontweight="bold")
axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)

_ptxt = f"E_dat={E_DAT_TEV:.0f} TeV  |  D₀={D0_cm2_s:.0e}  |  age={T_AGE_YR/1e3:.0f} kyr"
for ax in axes:
    ax.text(0.98, 0.05, _ptxt, transform=ax.transAxes, fontsize=8,
            ha='right', va='bottom', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
plt.suptitle(
    f"Distance Scan — CTA  |  HESS J1837-069  (E_dat={E_DAT_OPT:.0f} TeV, D₀={D0_cm2_s:.0e} cm²/s)",
    fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_dist_scan.png", dpi=150, bbox_inches="tight")
plt.show()


from IPython.utils.capture import capture_output as _capture
_bkg_halo_key = "bkg_halo_only"
_cache_loaded = False

BKG_SCALE_FACTORS = np.linspace(1.0, 5.0, num=5).tolist()

if CACHE_FILE.exists() and not FORCE_RERUN:
    try:
        _c = np.load(CACHE_FILE, allow_pickle=True)
        if _bkg_halo_key + "_results" in _c:
            bkg_halo_results = dict(_c[_bkg_halo_key + "_results"].item())
            _cache_loaded = True
            print("✓ Bkg scan (halo-only) loaded from cache")
    except Exception as e:
        print(f"  Cache read failed ({e}), re-running ...")

if not _cache_loaded:
    BKG_SCALE_FACTORS = np.linspace(1.0, 5.0, num=5).tolist()
    _bkg_scenarios = {
        "bkg_p10": [s * 1.10 for s in BKG_SCALE_FACTORS],
        "bkg_m10": [s * 0.90 for s in BKG_SCALE_FACTORS],
        "bkg_p20": [s * 1.20 for s in BKG_SCALE_FACTORS],
        "bkg_m20": [s * 0.80 for s in BKG_SCALE_FACTORS],
    }
    bkg_halo_results = {}

    for _bkg_key, scale_factors in _bkg_scenarios.items():
        ts_list  = []
        sig_list = []
        print(f"\n── Scenario: {_bkg_key} ──")
        print(f"{'Bkg scale':>10}  {'TS':>10}  {'sigma':>8}  {'Detected':>10}")
        print("-" * 46)
        for i_bkg, bkg_scale in enumerate(scale_factors):
            ds_bh = dataset.copy(name=f"bkgh_{_bkg_key}_{i_bkg}")
            sky_bh = SkyModel(
                spatial_model=TemplateSpatialModel(phi_map, normalize=True),
                spectral_model=PowerLawSpectralModel(
                    amplitude=HALO_AMPLITUDE,
                    index=HALO_ALPHA, reference=HALO_E0),
                name=f"halo-bkg-{_bkg_key}-{i_bkg}")
            bkg_bh = FoVBackgroundModel(dataset_name=ds_bh.name)
            bkg_bh.spectral_model.norm.value = bkg_scale
            with _capture():
                ds_bh.models = Models([sky_bh, bkg_bh])
            ds_bh.fake(random_state=SEED + 800 + i_bkg)
            ts_b = run_halo_only(ds_bh, phi_map, seed_offset=i_bkg*900)
            sig_b = np.sqrt(ts_b)
            ts_list.append(ts_b); sig_list.append(sig_b)
            print(f"{bkg_scale:>10.2f}  {ts_b:>10.2f}  {sig_b:>8.2f}  {'✓ YES' if ts_b>=25 else '✗ NO':>10}")
        bkg_halo_results[_bkg_key] = {
            "scales": scale_factors, "ts": ts_list, "sigma": sig_list}

    _ex = dict(np.load(CACHE_FILE, allow_pickle=True)) if CACHE_FILE.exists() else {}
    _ex[_bkg_halo_key + "_results"] = np.array(bkg_halo_results, dtype=object)
    np.savez(CACHE_FILE, **_ex)
    print("\n✓ Bkg scan (halo-only) saved")

sig_halo_p10 = np.array(bkg_halo_results["bkg_p10"]["sigma"])
sig_halo_m10 = np.array(bkg_halo_results["bkg_m10"]["sigma"])
sig_halo_p20 = np.array(bkg_halo_results["bkg_p20"]["sigma"])
sig_halo_m20 = np.array(bkg_halo_results["bkg_m20"]["sigma"])
sig_halo_mid = (sig_halo_p10 + sig_halo_m10) / 2


fig, ax = plt.subplots(figsize=(8, 5))

ax.fill_between(BKG_SCALE_FACTORS, sig_halo_m20, sig_halo_p20,
                alpha=0.15, color="steelblue", label="±20% bkg uncertainty")
ax.fill_between(BKG_SCALE_FACTORS, sig_halo_m10, sig_halo_p10,
                alpha=0.40, color="steelblue", label="±10% bkg uncertainty")
ax.plot(BKG_SCALE_FACTORS, sig_halo_mid, "s-", color="steelblue",
        lw=2.5, ms=8, label="Mean significance")
ax.axhline(5, ls="--", c="firebrick", lw=2, label="5σ threshold")
ax.fill_between(BKG_SCALE_FACTORS, 0, 5,
                where=sig_halo_mid < 5, alpha=0.10, color="firebrick",
                label="Below detection threshold")
ax.set_xlabel("Background scale factor", fontsize=12)
ax.set_ylabel("Detection significance [σ]", fontsize=12)
ax.set_title(
    "Effect of Background Systematics on Halo Detection (halo-only)\n"
    "±10% and ±20% bkg scaling — band style",
    fontsize=11, fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
_ptxt = f"E_dat={E_DAT_TEV:.0f} TeV  |  θ_d0={HALO_THETA_D0.value:.3f}°  |  d={DIST_KPC} kpc"
ax.text(0.98, 0.05, _ptxt, transform=ax.transAxes, fontsize=8,
        ha='right', va='bottom', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_bkg_halo_only.png", dpi=150, bbox_inches="tight")
plt.show()

dist_cm = DIST_KPC * KPC_TO_CM   # recompute to be safe

print(f"✓ LOCKED: DIST_KPC = {DIST_KPC} kpc  (fixed throughout)")
print(f"   dist_cm = {dist_cm:.3e} cm")
print(f"   All PWN+halo scans will use this distance.")

print()
print("=" * 55)
print("LOCKED PARAMETERS — used for all remaining sections")
print("=" * 55)
print(f"  Spatial model : {SPATIAL_MODEL.upper()}")
print(f"  θ_d0          : {HALO_THETA_D0:.3f}")
print(f"  E_dat         : {E_DAT_TEV:.1f} TeV")
print(f"  Age           : {T_AGE_YR/1e3:.0f} kyr")
print(f"  Distance      : {DIST_KPC} kpc")
print(f"  D₀            : {D0_cm2_s:.0e} cm²/s")
print("=" * 55)


phi_map_pwn = phi_map

# True PWN
pwn_spatial_true = PointSpatialModel(
    lon_0=SOURCE_CENTER.icrs.ra, lat_0=SOURCE_CENTER.icrs.dec, frame="icrs")
pwn_spatial_true.lon_0.frozen = True; pwn_spatial_true.lat_0.frozen = True
pwn_spectral_true = PowerLawSpectralModel(amplitude=TEST_FLUX, index=SPECTRAL_INDEX, reference=E_REF)
pwn_true = SkyModel(spatial_model=pwn_spatial_true, spectral_model=pwn_spectral_true, name="pwn-true")

# True halo
sky_halo_pwn_true = SkyModel(
    spatial_model=TemplateSpatialModel(phi_map_pwn, normalize=True),
    spectral_model=PowerLawSpectralModel(
        amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
    name="halo-pwn-true")

diffuse_pwn    = SkyModel(PowerLawNormSpectralModel(),
                           TemplateSpatialModel(diffuse_emission_map.interp_to_geom(
                               dataset.exposure.geom), normalize=False), name="diff-pwn-sim")
bkg_pwn_sim    = FoVBackgroundModel(dataset_name=dataset.name)
dataset_sim    = dataset.copy(name="sim-pwn-halo")
dataset_sim.models = Models([pwn_true, sky_halo_pwn_true, bkg_pwn_sim, diffuse_pwn])
dataset_sim.fake(random_state=SEED)

print("✓ PWN+Halo simulation complete")
print(f"   Total counts : {dataset_sim.counts.data.sum():.0f}")
print(f"   Net excess   : {dataset_sim.excess.data.sum():.0f}")


mu_pwn_halo = dataset_sim.npred().data
N_pwn_halo  = dataset_sim.counts.data
_mask_pwn_halo = mu_pwn_halo > 1.0
_z_pwn_halo = (N_pwn_halo[_mask_pwn_halo] - mu_pwn_halo[_mask_pwn_halo]) / np.sqrt(mu_pwn_halo[_mask_pwn_halo])

print(f"[pwn_halo] Poisson calibration check -- {_z_pwn_halo.size:,} bins with μ>1")
print(f"   Mean standardized residual z=(N-μ)/√μ : {_z_pwn_halo.mean():.3f}  (expect ≈0)")
print(f"   Std  standardized residual             : {_z_pwn_halo.std(ddof=1):.3f}  (expect ≈1)")

_mu_img_pwn_halo = mu_pwn_halo.sum(axis=0) if mu_pwn_halo.ndim > 2 else mu_pwn_halo
_n_img_pwn_halo  = N_pwn_halo.sum(axis=0) if N_pwn_halo.ndim > 2 else N_pwn_halo
_mask_img_pwn_halo = _mu_img_pwn_halo > 1.0
_z_img_pwn_halo = np.full(_mu_img_pwn_halo.shape, np.nan)
_z_img_pwn_halo[_mask_img_pwn_halo] = (_n_img_pwn_halo[_mask_img_pwn_halo] - _mu_img_pwn_halo[_mask_img_pwn_halo]) / np.sqrt(_mu_img_pwn_halo[_mask_img_pwn_halo])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(_z_pwn_halo, bins=60, range=(-5, 5), density=True, color="steelblue",
             alpha=0.75, label="Standardized residuals")
_zg = np.linspace(-5, 5, 200)
axes[0].plot(_zg, np.exp(-_zg**2/2)/np.sqrt(2*np.pi), "k--", lw=1.5, label="N(0,1)")
axes[0].set_xlabel(r"$z=(N-\mu)/\sqrt{\mu}$", fontsize=11)
axes[0].set_ylabel("Density", fontsize=11)
axes[0].set_title("Poisson calibration -- pwn_halo", fontsize=11, fontweight="bold")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

im = axes[1].imshow(_z_img_pwn_halo, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5)
fig.colorbar(im, ax=axes[1]).set_label(r"$(N-\mu)/\sqrt{\mu}$  (energy-summed)", fontsize=9)
axes[1].set_title("Poisson fluctuation map -- pwn_halo", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_poisson_calibration_pwn_halo.png", dpi=150, bbox_inches="tight")
plt.show()


ds_H0   = dataset_sim.copy(name="H0")
_sp_H0  = PointSpatialModel(lon_0=SOURCE_CENTER.icrs.ra, lat_0=SOURCE_CENTER.icrs.dec, frame="icrs")
_sp_H0.lon_0.frozen = True; _sp_H0.lat_0.frozen = True
pwn_H0  = SkyModel(
    spatial_model=_sp_H0,
    spectral_model=PowerLawSpectralModel(amplitude=TEST_FLUX, index=SPECTRAL_INDEX, reference=E_REF),
    name="pwn-H0")
diff_H0_roi  = diffuse_emission_map.interp_to_geom(ds_H0.exposure.geom)
diff_H0_tmpl = TemplateSpatialModel(diff_H0_roi, normalize=False)
diff_H0      = SkyModel(PowerLawNormSpectralModel(), diff_H0_tmpl, name="diff-H0")
bkg_H0       = FoVBackgroundModel(dataset_name=ds_H0.name)
ds_H0.models = Models([pwn_H0, bkg_H0, diff_H0])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fit_H0  = Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2})
    res_H0  = fit_H0.run([ds_H0])
    res_H0  = fit_H0.run([ds_H0])
print(f"H₀ (PWN only) fit  success={res_H0.success}  stat={res_H0.total_stat:.2f}")

ds_H1   = dataset_sim.copy(name="H1")
_sp_H1  = PointSpatialModel(lon_0=SOURCE_CENTER.icrs.ra, lat_0=SOURCE_CENTER.icrs.dec, frame="icrs")
_sp_H1.lon_0.frozen = True; _sp_H1.lat_0.frozen = True
pwn_H1_src = SkyModel(
    spatial_model=_sp_H1,
    spectral_model=PowerLawSpectralModel(amplitude=TEST_FLUX, index=SPECTRAL_INDEX, reference=E_REF),
    name="pwn-H1")
halo_H1 = SkyModel(
    spatial_model=TemplateSpatialModel(phi_map, normalize=True),
    spectral_model=PowerLawSpectralModel(
        amplitude=1e-13 * u.Unit("cm-2 s-1 TeV-1"),
        index=HALO_ALPHA, reference=HALO_E0),
    name="halo-H1",
)
diff_H1_roi  = diffuse_emission_map.interp_to_geom(ds_H1.exposure.geom)
diff_H1_tmpl = TemplateSpatialModel(diff_H1_roi, normalize=False)
diff_H1      = SkyModel(PowerLawNormSpectralModel(), diff_H1_tmpl, name="diff-H1")
bkg_H1       = FoVBackgroundModel(dataset_name=ds_H1.name)
ds_H1.models = Models([pwn_H1_src, halo_H1, bkg_H1, diff_H1])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fit_H1  = Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2})
    res_H1  = fit_H1.run([ds_H1])
    res_H1  = fit_H1.run([ds_H1])
print(f"H₁ fit  success={res_H1.success}  stat={res_H1.total_stat:.2f}")

ds_null     = dataset_sim.copy(name="null")
diff_null_roi  = diffuse_emission_map.interp_to_geom(ds_null.exposure.geom)
diff_null_tmpl = TemplateSpatialModel(diff_null_roi, normalize=False)
diff_null      = SkyModel(PowerLawNormSpectralModel(), diff_null_tmpl, name="diff-null")
bkg_null       = FoVBackgroundModel(dataset_name=ds_null.name)
ds_null.models = Models([bkg_null, diff_null])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res_null = Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds_null])


ts_det_H0    = max(0, res_null.total_stat - res_H0.total_stat)
ts_det_H1    = max(0, res_null.total_stat - res_H1.total_stat)
ts_halo      = max(0, res_H0.total_stat   - res_H1.total_stat)
sigma_det_H0 = np.sqrt(ts_det_H0)
sigma_det_H1 = np.sqrt(ts_det_H1)
sigma_halo   = np.sqrt(ts_halo)

k_H0 = sum(1 for p in ds_H0.models.parameters if not p.frozen)
k_H1 = sum(1 for p in ds_H1.models.parameters if not p.frozen)
aic_H0    = 2 * k_H0 + res_H0.total_stat
aic_H1    = 2 * k_H1 + res_H1.total_stat
delta_aic = aic_H0 - aic_H1

pwn_index_H0 = pwn_H0.spectral_model.index.value
pwn_index_H1 = pwn_H1_src.spectral_model.index.value

print("=" * 68)
print("CTA PWN + TEV HALO — MODEL COMPARISON")
print("=" * 68)
print(f"{'Model':<28}  {'Stat':>10}  {'k':>4}  {'AIC':>10}")
print("-" * 68)
print(f"{'PWN only (H₀)':<28}  {res_H0.total_stat:>10.2f}  {k_H0:>4}  {aic_H0:>10.2f}")
print(f"{'PWN + halo (H₁)':<28}  {res_H1.total_stat:>10.2f}  {k_H1:>4}  {aic_H1:>10.2f}")
print("=" * 68)
print()
print(f"  TS_det (H₀)  = {ts_det_H0:.2f}  →  {sigma_det_H0:.1f} σ")
print(f"  TS_det (H₁)  = {ts_det_H1:.2f}  →  {sigma_det_H1:.1f} σ")
print(f"  TS_halo       = {ts_halo:.2f}  →  {np.sqrt(ts_halo):.1f} σ")
print(f"  ΔAIC (H₀−H₁)  = {delta_aic:.2f}  "
      f"({'STRONG' if delta_aic>10 else 'MODERATE' if delta_aic>2 else 'WEAK'} preference for H₁)")
print()
print(f"  PWN spectral index (H₀ fit): {pwn_index_H0:.3f}")
print(f"  PWN spectral index (H₁ fit): {pwn_index_H1:.3f}")


from gammapy.estimators import ExcessMapEstimator

est = ExcessMapEstimator(
    correlation_radius='0.05 deg',
    selection_optional=[],
    energy_edges=[E_MIN.to_value(u.TeV), E_MAX.to_value(u.TeV)] * u.TeV,
)

# Detection map — bkg-only null on H₀ data
ds_det_H0 = ds_H0.copy(name="det-H0")
ds_det_H0.models = Models([FoVBackgroundModel(dataset_name=ds_det_H0.name)])
res_det_H0 = est.run(ds_det_H0)

# Detection map — bkg-only null on H₁ data
ds_det_H1 = ds_H1.copy(name="det-H1")
ds_det_H1.models = Models([FoVBackgroundModel(dataset_name=ds_det_H1.name)])
res_det_H1 = est.run(ds_det_H1)

# Residual map — H₀ fitted model (ECPL only)
ds_res_H0 = ds_H0.copy(name="res-H0")
ds_res_H0.models = ds_H0.models.copy()
res_res_H0 = est.run(ds_res_H0)

# Residual map — H₁ fitted model (PWN + halo)
ds_res_H1 = ds_H1.copy(name="res-H1")
ds_res_H1.models = ds_H1.models.copy()
res_res_H1 = est.run(ds_res_H1)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Top row: H₀
ds_H0.plot_residuals_spatial(ax=axes[0,0], method='diff/sqrt(model)', vmin=-3, vmax=3, cmap='PuOr')
axes[0,0].set_title('Residuals — H₀ (ECPL only)  [σ]', fontsize=10, fontweight='bold')

res_det_H0['sqrt_ts'].plot(ax=axes[0,1], add_cbar=True, vmin=-5, vmax=20, cmap='PuOr')
axes[0,1].set_title('Detection significance — H₀ (null model)  [σ]', fontsize=10, fontweight='bold')

res_res_H0['sqrt_ts'].plot(ax=axes[0,2], add_cbar=True, vmin=-5, vmax=5, cmap='PuOr')
axes[0,2].set_title('Residual significance — H₀ (ECPL fitted)  [σ]', fontsize=10, fontweight='bold')

# Bottom row: H₁
ds_H1.plot_residuals_spatial(ax=axes[1,0], method='diff/sqrt(model)', vmin=-3, vmax=3, cmap='PuOr')
axes[1,0].set_title('Residuals — H₁ (PWN + halo)  [σ]', fontsize=10, fontweight='bold')

res_det_H1['sqrt_ts'].plot(ax=axes[1,1], add_cbar=True, vmin=-5, vmax=20, cmap='PuOr')
axes[1,1].set_title('Detection significance — H₁ (null model)  [σ]', fontsize=10, fontweight='bold')

res_res_H1['sqrt_ts'].plot(ax=axes[1,2], add_cbar=True, vmin=-5, vmax=5, cmap='PuOr')
axes[1,2].set_title('Residual significance — H₁ (halo fitted)  [σ]', fontsize=10, fontweight='bold')

plt.suptitle(
    f'Significance Maps — CTA PWN + Halo  [σ]\n'
    f'TS_halo = {ts_halo:.1f}  ({np.sqrt(ts_halo):.1f}σ)  |  '
    f'ΔAIC = {delta_aic:.1f}  |  θ_d0 = {HALO_THETA_D0.value}°',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_pwn_halo_significance_maps.png", dpi=150, bbox_inches="tight")
plt.show()

res_det_H0['sqrt_ts'].write(OUTPUT_DIR / 'cta_sig_map_pwn_halo.fits.gz', overwrite=True)
res_res_H0['sqrt_ts'].write(OUTPUT_DIR / 'cta_sig_map_pwn_halo_isolated.fits.gz', overwrite=True)
print('✓ CTA PWN+Halo significance maps saved (total-system + halo-isolated)')

from scipy.stats import norm as scipy_norm

significance_data = res_det_H1['sqrt_ts'].data
selection         = np.isfinite(significance_data) & ~(significance_data == 0)
sig_flat          = significance_data[selection]

fig, ax = plt.subplots(figsize=(7, 4))
ax.set_yscale('log')
ax.set_xscale('symlog', linthresh=1)

ax.hist(sig_flat, density=True, alpha=0.7, color='pink', bins=50,
        label='Significance')

mu, std = scipy_norm.fit(sig_flat)
x = np.linspace(sig_flat.min(), sig_flat.max(), 500)
ax.plot(x, scipy_norm.pdf(x, mu, std), lw=2, color='black',
        label=rf'$\mu={mu:.2f}$, $\sigma={std:.2f}$')
ax.plot(x, scipy_norm.pdf(x, 0, 1), lw=1.5, ls='--', color='gray',
        label='N(0,1)')

ax.legend(fontsize=13)
ax.set_xlabel('Significance [σ]')
ax.set_ylabel('Probability density')
ax.set_title('Significance distribution (CTA — detection map)')
ax.set_ylim(1e-5, 10)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_significance.png", dpi=150, bbox_inches="tight")
plt.show()

print(f'Fitted: μ={mu:.3f}, σ={std:.3f}')
print('A good background model gives μ≈0 and σ≈1.')
if abs(mu) > 0.3:
    print('WARNING: |μ| > 0.3 — possible background normalisation bias.')
if abs(std - 1) > 0.2:
    print('WARNING: σ deviates from 1 — background model may need adjustment.')

estimator_ebands = ExcessMapEstimator(
    correlation_radius='0.05 deg',
    selection_optional=[],
    energy_edges=[1, 5, 50, 300] * u.TeV,
)
result_ebands = estimator_ebands.run(ds_H1)
result_ebands['sqrt_ts'].plot_grid(
    figsize=(14, 4), cmap='PuOr', add_cbar=True, vmin=-5, vmax=5, ncols=3)
plt.suptitle('Residual significance maps per energy band  [σ] (CTA — H₁ fitted)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_residual_maps_multi_energy_pwn_halo.png", dpi=150, bbox_inches="tight")
plt.show()


from gammapy.estimators import TSMapEstimator

test_model_ts = SkyModel(
    spatial_model=PointSpatialModel(),
    spectral_model=PowerLawSpectralModel(
        index=2.54,
        amplitude=1e-11 * u.Unit('cm-2 s-1 TeV-1'),
        reference=1 * u.TeV,
    ),
    name='ts_test',
)
estimator_ts = TSMapEstimator(
    model=test_model_ts,
    kernel_width='1.0 deg',
    energy_edges=[1, 300] * u.TeV,
    sum_over_energy_groups=True,
)

ts_map_H0 = estimator_ts.run(ds_H0)
ts_map_H1 = estimator_ts.run(ds_H1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, ts_result, label, note, det_ts, det_sig in zip(
    axes,
    [ts_map_H0, ts_map_H1],
    ['ECPL only (H₀)', 'PWN + halo (H₁)'],
    ['Halo unabsorbed → residual ring visible', 'Full model fitted → should be flat'],
    [ts_det_H0, ts_det_H1],
    [sigma_det_H0, sigma_det_H1],
):
    sqrt_ts = ts_result['sqrt_ts'].data[0]
    vmax    = max(np.nanpercentile(np.abs(sqrt_ts[np.isfinite(sqrt_ts)]), 99.5), 5)
    im      = ax.imshow(sqrt_ts, origin='lower', cmap="PuOr", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.85, label='sqrt(TS)  [σ]')
    peak = np.nanmax(sqrt_ts)
    ax.set_title(
        f'Significance map — {label}\n{note}\n'
        f'TS_det = {det_ts:.1f}  ({det_sig:.1f}σ)  |  peak = {peak:.1f}σ',
        fontsize=10, fontweight='bold')
    ax.set_xlabel('Pixel X'); ax.set_ylabel('Pixel Y')

plt.suptitle(
    f'TS Maps — CTA PWN + Halo  [σ]\n'
    f'TS_halo = {ts_halo:.1f}  ({np.sqrt(ts_halo):.1f}σ)  |  θ_d0 = {HALO_THETA_D0}',
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_ts.png", dpi=150, bbox_inches="tight")
plt.show()

energy_edges_fp = np.geomspace(E_MIN.to_value(u.TeV), E_MAX.to_value(u.TeV), 14) * u.TeV

for model in ds_H1.models:
    if isinstance(model, FoVBackgroundModel):
        model.spectral_model.norm.frozen = True

fpe_pwn = FluxPointsEstimator(energy_edges=energy_edges_fp,
                               source="pwn-H1", selection_optional=["ul"])
fp_pwn  = fpe_pwn.run([ds_H1])

fpe_halo = FluxPointsEstimator(energy_edges=energy_edges_fp,
                                source="halo-H1", selection_optional=["ul"])
fp_halo  = fpe_halo.run([ds_H1])

_tbl_diag = fp_halo.to_table(sed_type="e2dnde")
print("Columns:", _tbl_diag.colnames)
for _row in _tbl_diag:
    _e = _row["e_ref"] if "e_ref" in _tbl_diag.colnames else float("nan")
    _ul = _row["is_ul"] if "is_ul" in _tbl_diag.colnames else "?"
    _ts = _row["ts"] if "ts" in _tbl_diag.colnames else float("nan")
    print(f"  E={_e:.3g} TeV | is_ul={_ul} | TS={_ts:.2f}")


fig, ax = plt.subplots(figsize=(10, 6))

fp_pwn.plot(ax=ax, sed_type="e2dnde", color="royalblue",
            marker="o", markersize=5, elinewidth=0.9, capsize=2.5,
            label="Flux points — PWN (PL)")
pwn_H1_src.spectral_model.plot(
    ax=ax, energy_bounds=[1, 1000]*u.TeV, energy_power=2,
    color="royalblue", linewidth=1.8, label="PWN PL best-fit")
pwn_H1_src.spectral_model.plot_error(
    ax=ax, energy_bounds=[1, 1000]*u.TeV, energy_power=2,
    facecolor="royalblue", alpha=0.18)

fp_halo.plot(ax=ax, sed_type="e2dnde", color="green",
             marker="s", markersize=5, elinewidth=0.9, capsize=2.5,
             label="Flux points — TeV Halo (H₁)")
halo_H1.spectral_model.plot(
    ax=ax, energy_bounds=[HALO_E_MIN, HALO_E_MAX], energy_power=2,
    color="#b2182b", linewidth=1.8, label="Halo PL best-fit")
halo_H1.spectral_model.plot_error(
    ax=ax, energy_bounds=[HALO_E_MIN, HALO_E_MAX], energy_power=2,
    facecolor="#b2182b", alpha=0.18)

pwn_true.spectral_model.plot(
    ax=ax, energy_bounds=[1, 1000]*u.TeV, energy_power=2,
    color="black", linewidth=1.2, linestyle="--", label="True PWN PL (injected)")
sky_halo_pwn_true.spectral_model.plot(
    ax=ax, energy_bounds=[HALO_E_MIN, HALO_E_MAX], energy_power=2,
    color="#666666", linewidth=1.2, linestyle="--", label="True Halo PL (injected)")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Energy [TeV]", fontsize=12)
ax.set_ylabel(r"$E^2\,dN/dE$  [TeV cm$^{-2}$ s$^{-1}$]", fontsize=12)
ax.set_xlim(0.5, 1200); ax.set_ylim(1e-14, 3e-11)
ax.set_title(
    f"Combined SED — PWN (PL) + TeV Halo (CTA)\n"
    f"TS_halo = {ts_halo:.1f}  ({np.sqrt(ts_halo):.1f}σ)  |  "
    f"TS_det(H₁) = {ts_det_H1:.1f}  ({sigma_det_H1:.1f}σ)",
    fontsize=11, fontweight="bold",
)
ax.grid(which="both", alpha=0.3)
ax.legend(fontsize=10)

_pwn_fit_idx  = pwn_H1_src.spectral_model.index.value
_pwn_fit_amp  = pwn_H1_src.spectral_model.amplitude.value
_pwn_fit_ref  = pwn_H1_src.spectral_model.reference.quantity.to_value(u.TeV)
_halo_fit_idx = halo_H1.spectral_model.index.value
_halo_fit_amp = halo_H1.spectral_model.amplitude.value
_halo_fit_ref = halo_H1.spectral_model.reference.quantity.to_value(u.TeV)
_param_txt = (
    f"Fitted:  PWN index={_pwn_fit_idx:.2f} amp@{_pwn_fit_ref:.2g}TeV={_pwn_fit_amp:.2e}   |   "
    f"Halo index={_halo_fit_idx:.2f} amp@{_halo_fit_ref:.2g}TeV={_halo_fit_amp:.2e}\n"
    f"HESS J1837-069 (injected):  PWN index={SPECTRAL_INDEX:.2f} amp@{E_REF.to_value(u.TeV):.2g}TeV={TEST_FLUX.value:.2e}   |   "
    f"Halo index={HALO_ALPHA:.2f} amp@{HALO_E0.to_value(u.TeV):.2g}TeV={HALO_AMPLITUDE.value:.2e}  θ_d0={HALO_THETA_D0.value:.3f}°"
)
ax.text(0.02, 0.02, _param_txt, transform=ax.transAxes, fontsize=7.3,
        ha="left", va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.88))

_hess_pwn_data = np.array([
    [0.222308, 2.82024e-10, 1.44221e-10, 1.54296e-10],
    [0.288461, 3.93396e-11, 2.05788e-11, 2.13974e-11],
    [0.422034, 3.21142e-11, 6.44578e-12, 6.67094e-12],
    [0.610243, 2.07182e-11, 2.93353e-12, 3.04057e-12],
    [0.884837, 6.57914e-12, 1.12113e-12, 1.17069e-12],
    [1.29498,  3.28387e-12, 5.30865e-13, 5.58275e-13],
    [1.88953,  1.089e-12,   2.4159e-13,  2.56409e-13],
    [2.74707,  4.72377e-13, 1.34714e-13, 1.44667e-13],
    [4.19217,  2.81362e-13, 6.63316e-14, 7.24583e-14],
    [5.99275,  8.35974e-14, 2.83258e-14, 3.21689e-14],
    [8.54779,  3.40375e-14, 1.2569e-14,  1.49668e-14],
    [16.4452,  5.42638e-15, 2.47605e-15, 3.52043e-15],
])
_e_hess        = _hess_pwn_data[:, 0]
_e2dnde_hess   = _e_hess**2 * _hess_pwn_data[:, 1]
_e2dnde_err_lo = _e_hess**2 * _hess_pwn_data[:, 2]
_e2dnde_err_hi = _e_hess**2 * _hess_pwn_data[:, 3]
ax.errorbar(_e_hess, _e2dnde_hess, yerr=[_e2dnde_err_lo, _e2dnde_err_hi],
            fmt="D", color="black", ms=6, elinewidth=1.3, capsize=3, alpha=0.85,
            label="H.E.S.S. (real, total system)", zorder=5)
ax.legend(fontsize=10)

plt.tight_layout()

fp_pwn.write(OUTPUT_DIR / "cta_fp_pwn_1.fits", overwrite=True)
fp_halo.write(OUTPUT_DIR / "cta_fp_halo_1.fits", overwrite=True)
print("✓ CTA flux points saved")

Models([pwn_H1_src]).write(OUTPUT_DIR / 'cta_pwn_fit.yaml', overwrite=True)

phi_map_pwn.write(OUTPUT_DIR / "cta_halo_template.fits", overwrite=True)
halo_H1_save = SkyModel(
    spatial_model=TemplateSpatialModel.read(OUTPUT_DIR / "cta_halo_template.fits", normalize=True),
    spectral_model=halo_H1.spectral_model.copy(),
    name=halo_H1.name,
)
Models([halo_H1_save]).write(OUTPUT_DIR / 'cta_halo_fit.yaml', overwrite=True)
print("✓ Model YAML files saved")

plt.savefig(OUTPUT_DIR / "cta_sed_pwn_halo.png", dpi=150, bbox_inches="tight")
plt.show()


HESSJ1837_INTFLUX_1TEV  = 1.15e-11 * u.Unit("cm-2 s-1")   # HGPS integral flux >1 TeV (HESS 2018)
CRAB_INTFLUX_1TEV       = 2.26e-11 * u.Unit("cm-2 s-1")   # HESS Crab integral flux >1 TeV (Aharonian+2006)
HESSJ1837_CRAB_FRACTION = (HESSJ1837_INTFLUX_1TEV / CRAB_INTFLUX_1TEV).to_value("")  # ~0.509 (TeVCat quotes 0.531)
CRAB_FRACTION_TARGET    = 0.02
STEEP_INDEX             = 2.54   # HESS J1837-069's own index -- unchanged throughout

amp_2crab = TEST_FLUX * (CRAB_FRACTION_TARGET / HESSJ1837_CRAB_FRACTION)

print(f"HESS J1837-069 catalogue brightness = {HESSJ1837_CRAB_FRACTION*100:.1f}% Crab "
      f"(HGPS {HESSJ1837_INTFLUX_1TEV:.2e} / HESS-Crab {CRAB_INTFLUX_1TEV:.2e}, both >1 TeV)")
print(f"Scaling down to {CRAB_FRACTION_TARGET*100:.0f}% Crab -> "
      f"amp_2crab = TEST_FLUX x ({CRAB_FRACTION_TARGET}/{HESSJ1837_CRAB_FRACTION:.3f})")

# Context: how faint is this vs. the REAL measured HESS J1837-069 flux
# (TEST_FLUX, HGPS spectral table: 1.78e-11 cm-2 s-1 TeV-1 at 1 TeV)?
_ratio_to_real = (amp_2crab / TEST_FLUX).to_value("")
print(f"2% Crab test amplitude = {amp_2crab:.3e}")
print(f"Real HESS J1837-069 flux (TEST_FLUX) = {TEST_FLUX:.3e}")
print(f"This test represents {_ratio_to_real*100:.1f}% of the REAL measured "
      f"HESS J1837-069 flux -- i.e. ~{1/_ratio_to_real:.0f}x fainter than the actual source.")

ds_2crab = dataset.copy(name="halo_2crab_steep_test_cta")
if ds_2crab.mask_fit is None:
    ds_2crab.mask_fit = ds_2crab.mask_safe.copy()
h_2crab_true = SkyModel(
    spatial_model=TemplateSpatialModel(phi_map, normalize=True),
    spectral_model=PowerLawSpectralModel(amplitude=amp_2crab, index=STEEP_INDEX, reference=E_REF),
    name="halo-2crab-true-cta")
ds_2crab.models = Models([h_2crab_true, FoVBackgroundModel(dataset_name=ds_2crab.name)])
ds_2crab.fake(random_state=SEED + 999)

h_2crab_fit = SkyModel(
    spatial_model=TemplateSpatialModel(phi_map, normalize=True),
    spectral_model=PowerLawSpectralModel(amplitude=amp_2crab, index=2.54, reference=E_REF),
    name="halo-2crab-fit-cta")
ds_2crab.models = Models([h_2crab_fit, FoVBackgroundModel(dataset_name=ds_2crab.name)])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds_2crab])

ds_null_2crab = ds_2crab.copy(name="null_2crab_cta")
ds_null_2crab.models = Models([FoVBackgroundModel(dataset_name=ds_null_2crab.name)])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds_null_2crab])
ts_det_2crab = max(0.0, ds_null_2crab.stat_sum() - ds_2crab.stat_sum())
sigma_2crab = np.sqrt(ts_det_2crab)

print(f"[CTA] 2% Crab @ 1 TeV, index={STEEP_INDEX}: amplitude={amp_2crab:.3e}")
print(f"  TS_det = {ts_det_2crab:.2f}  ({sigma_2crab:.2f}sigma)")
print(f"  Fitted index = {h_2crab_fit.spectral_model.index.value:.2f} (true={STEEP_INDEX})")

energy_edges_2crab = np.geomspace(1.0, 1000.0, 10) * u.TeV
fpe_2crab = FluxPointsEstimator(energy_edges=energy_edges_2crab,
                                 source="halo-2crab-fit-cta", selection_optional=["ul"])
fp_2crab = fpe_2crab.run([ds_2crab])
fp_2crab.write(OUTPUT_DIR / "cta_fp_2crab_steep.fits", overwrite=True)

fig, ax = plt.subplots(figsize=(9, 6))
fp_2crab.plot(ax=ax, sed_type="e2dnde", color="purple",
              marker="D", markersize=5, elinewidth=0.9, capsize=2.5,
              label=f"Flux points (2% Crab, Gamma={STEEP_INDEX})")
h_2crab_true.spectral_model.plot(ax=ax, energy_bounds=[1, 1000]*u.TeV, energy_power=2,
                                  color="black", lw=1.5, ls=":", label="Injected (true)")
h_2crab_fit.spectral_model.plot(ax=ax, energy_bounds=[1, 1000]*u.TeV, energy_power=2,
                                 color="purple", lw=2, label="Recovered fit")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Energy [TeV]", fontsize=13)
ax.set_ylabel(r"$E^2\,dN/dE$  [TeV cm$^{-2}$ s$^{-1}$]", fontsize=13)
ax.set_xlim(0.8, 1200)
ax.set_title(f"CTA -- 2% Crab Flux, Steep Index (Gamma={STEEP_INDEX}) Recovery Test\n"
             f"TS_det={ts_det_2crab:.1f} ({sigma_2crab:.1f}sigma)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10); ax.grid(which="both", alpha=0.3)

_param_txt_2crab = (
    f"Fitted:  amp={h_2crab_fit.spectral_model.amplitude.value:.2e}   "
    f"index={h_2crab_fit.spectral_model.index.value:.2f}\n"
    f"HESS J1837-069 (injected, {CRAB_FRACTION_TARGET*100:.0f}% Crab-scaled):  "
    f"amp={amp_2crab.value:.2e}   index={STEEP_INDEX:.2f}"
)
ax.text(0.02, 0.02, _param_txt_2crab, transform=ax.transAxes, fontsize=8,
        ha="left", va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.88))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_2crab_steep_test.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: cta_fp_2crab_steep.fits, cta_2crab_steep_test.png")


def _make_pwn_model(name, geom_ds):
    """Helper: PL point source at SOURCE_CENTER (the PWN)."""
    _c = SOURCE_CENTER.icrs
    sp = PointSpatialModel(lon_0=_c.ra, lat_0=_c.dec, frame="icrs")
    sp.lon_0.frozen = True; sp.lat_0.frozen = True
    return SkyModel(
        spatial_model=sp,
        spectral_model=PowerLawSpectralModel(
            amplitude=TEST_FLUX, index=SPECTRAL_INDEX, reference=E_REF),
        name=name)

def run_pwn_halo(dataset_sim, phi_template, seed_offset=0):
    """
    Total-system detection test: null = bkg only (nothing there);
    alt = PWN + halo + bkg. Returns ts_det = stat(null) - stat(alt),
    the overall detection significance of the source complex against a
    pure-background null (H0 = bkg only, H1 = PWN + halo + bkg).
    """
    geom_ds = dataset_sim.counts.geom

    # Null: bkg only (nothing there)
    ds_n = dataset_sim.copy(name=f"n_{seed_offset}")
    ds_n.models = Models([FoVBackgroundModel(dataset_name=ds_n.name)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_n = Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_n])

    # Alt: PWN + halo + bkg
    ds_0 = dataset_sim.copy(name=f"h0_{seed_offset}")
    e0   = _make_pwn_model(f"pwn-h0-{seed_offset}", geom_ds)
    h0   = SkyModel(spatial_model=TemplateSpatialModel(phi_template, normalize=True),
                    spectral_model=PowerLawSpectralModel(
                        amplitude=HALO_AMPLITUDE,
                        index=HALO_ALPHA, reference=HALO_E0),
                    name=f"halo-h0-{seed_offset}")
    ds_0.models = Models([e0, h0, FoVBackgroundModel(dataset_name=ds_0.name)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_0 = Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_0])
        r_0 = Fit(optimize_opts={"print_level":0,"tol":6,"strategy":2}).run([ds_0])

    ts_det = max(0, r_n.total_stat - r_0.total_stat)
    return ts_det

print("✓ Scan helper defined")

# Using locked values: D0_cm2_s, DIST_KPC, T_AGE_YR, E_DAT_TEV
# D₀ scan: runs BOTH halo-only and PWN+halo at each D₀ value
from IPython.utils.capture import capture_output as _capture
_d0_key      = "d0_scan"
_cache_loaded = False

E0_TeV = HALO_E0.to_value(u.TeV)

if CACHE_FILE.exists() and not FORCE_RERUN:
    try:
        _c = np.load(CACHE_FILE, allow_pickle=True)
        if _d0_key + "_D0_values" in _c:
            D0_VALUES         = list(_c[_d0_key + "_D0_values"])
            ts_d0_halo        = list(_c[_d0_key + "_ts_halo"])
            ts_d0_halo_only   = list(_c[_d0_key + "_ts_halo_only"]) if _d0_key+"_ts_halo_only" in _c else list(_c[_d0_key + "_ts_halo"])
            theta_d0_vals     = list(_c[_d0_key + "_theta_d0"])
            _cache_loaded = True
            print(f"✓ D₀ scan loaded from cache ({CACHE_FILE})")
    except Exception as e:
        print(f"  Cache read failed ({e}), re-running ...")

if not _cache_loaded:
    D0_VALUES       = np.logspace(25, 27, 9)
    ts_d0_halo      = []   # PWN+halo test
    ts_d0_halo_only = []   # halo-only test
    theta_d0_vals   = []

    center_icrs = SOURCE_CENTER.icrs
    print(f"{'D0 [cm2/s]':>14}  {'theta_d0':>9}  {'TS_pwn':>9}  {'TS_halo_only':>13}")
    print("-" * 54)

    for i_d0, D0 in enumerate(D0_VALUES):
        theta_this = diffusion_angle_deg(E0_TeV, D0, DELTA_DIFF, t_age_s, dist_cm, E0_TeV)
        theta_d0_vals.append(theta_this)

        phi_d0, _ = build_scan_halo_template(geom, center_icrs, theta_d0_deg=theta_this)

        ds_d0 = dataset.copy(name=f"d0_sim_{i_d0}")
        ec_d0 = _make_pwn_model(f"pwn-d0-{i_d0}", geom)
        sky_hd = SkyModel(
            spatial_model=TemplateSpatialModel(phi_d0, normalize=True),
            spectral_model=PowerLawSpectralModel(
                amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-d0-{i_d0}")
        bkg_d0 = FoVBackgroundModel(dataset_name=ds_d0.name)
        with _capture():
            ds_d0.models = Models([ec_d0, sky_hd, bkg_d0])
        ds_d0.fake(random_state=SEED + 60 + i_d0)
        ts_h = run_pwn_halo(ds_d0, phi_d0, seed_offset=i_d0*100)
        ts_d0_halo.append(ts_h)

        ds_d0_only = dataset.copy(name=f"d0_haloonly_{i_d0}")
        sky_only = SkyModel(
            spatial_model=TemplateSpatialModel(phi_d0, normalize=True),
            spectral_model=PowerLawSpectralModel(
                amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-only-d0-{i_d0}")
        with _capture():
            ds_d0_only.models = Models([sky_only, FoVBackgroundModel(dataset_name=ds_d0_only.name)])
        ds_d0_only.fake(random_state=SEED + 560 + i_d0)
        ts_h_only = run_halo_only(ds_d0_only, phi_d0, seed_offset=i_d0*100 + 500)
        ts_d0_halo_only.append(ts_h_only)

        print(f"{D0:14.2e}  {theta_this:9.3f}  {ts_h:9.2f}  {ts_h_only:13.2f}")

    _ex = dict(np.load(CACHE_FILE, allow_pickle=True)) if CACHE_FILE.exists() else {}
    _ex[_d0_key+"_D0_values"]      = np.array(D0_VALUES)
    _ex[_d0_key+"_ts_halo"]        = np.array(ts_d0_halo)
    _ex[_d0_key+"_ts_halo_only"]   = np.array(ts_d0_halo_only)
    _ex[_d0_key+"_theta_d0"]       = np.array(theta_d0_vals)
    np.savez(CACHE_FILE, **_ex)
    print(f"\n✓ D₀ scan saved to {CACHE_FILE}")

D0_VALUES       = np.array(D0_VALUES)
ts_d0_halo      = np.array(ts_d0_halo)
ts_d0_halo_only = np.array(ts_d0_halo_only)
theta_d0_vals   = np.array(theta_d0_vals)
sigma_d0_halo      = np.sqrt(ts_d0_halo)
sigma_d0_halo_only = np.sqrt(ts_d0_halo_only)
print("✓ D₀ scan complete")


# D₀ plots: significance (combined) + analytical extension lines 
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
norm_ext = plt.Normalize(vmin=np.min(theta_d0_vals), vmax=np.max(theta_d0_vals))
cmap_ext = plt.cm.plasma

sc1 = ax.scatter(D0_VALUES, sigma_d0_halo_only,
                 c=theta_d0_vals, cmap=cmap_ext, norm=norm_ext,
                 s=100, marker="o", edgecolors="steelblue", linewidths=1.5, zorder=5)
ax.plot(D0_VALUES, sigma_d0_halo_only, "-", color="steelblue", lw=1.5, alpha=0.7,
        label="Halo-only σ")
sc2 = ax.scatter(D0_VALUES, sigma_d0_halo,
                 c=theta_d0_vals, cmap=cmap_ext, norm=norm_ext,
                 s=100, marker="D", edgecolors="darkorange", linewidths=1.5, zorder=5)
ax.plot(D0_VALUES, sigma_d0_halo, "--", color="darkorange", lw=1.5, alpha=0.7,
        label="PWN+Halo σ")

cb = fig.colorbar(sc1, ax=ax)
cb.set_label("Halo extension θ_d0 [°]", fontsize=10)

for d0, sig1, sig2, theta in zip(D0_VALUES, sigma_d0_halo_only, sigma_d0_halo, theta_d0_vals):
    ax.annotate(f"θ={theta:.2f}°", (d0, max(sig1, sig2)),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=7.5, color="black")

ax.axhline(5, ls="--", c="firebrick", lw=1.8, label="5σ threshold")
ax.fill_between([D0_VALUES.min()*0.3, D0_VALUES.max()*3], 0, 5,
                alpha=0.08, color="firebrick")
ax.axvline(D0_cm2_s, ls=":", c="grey", lw=1.5, label=f"Locked D₀={D0_cm2_s:.0e}")
ax.set_xscale("log")
ax.set_xlim(D0_VALUES.min()*0.3, D0_VALUES.max()*3)
ax.set_xlabel("Diffusion coefficient D₀ [cm² s⁻¹]", fontsize=12)
ax.set_ylabel("Detection significance σ", fontsize=12)
ax.set_title("D₀ vs Significance\nHalo-only (○) vs PWN+Halo (◆)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")


ax = axes[1]
D0_LINE    = np.logspace(24.5, 27.5, 80)
age_curves = [10e3, 23e3, 50e3, 100e3, 200e3, 500e3]
age_colors = ["#d62728", "#e6550d", "#fd8d3c", "#fdae6b", "#3182bd", "#6baed6"]
_E0 = HALO_E0.to_value(u.TeV)

for t_yr, c_col in zip(age_curves, age_colors):
    theta_curve = [diffusion_angle_deg(_E0, D0, DELTA_DIFF, t_yr*YR_TO_S, dist_cm, _E0)
                   for D0 in D0_LINE]
    ax.semilogx(D0_LINE, theta_curve, lw=2, color=c_col,
                label=f"age = {t_yr/1e3:.0f} kyr")


ax.scatter(D0_VALUES, theta_d0_vals, zorder=6, s=70, c="k", marker="D",
           label="Simulated points")
ax.axvline(D0_cm2_s, ls=":", c="grey", lw=1.8, label=f"Locked D₀={D0_cm2_s:.0e}")
ax.set_xlabel("D₀ [cm²/s]", fontsize=12); ax.set_ylabel("Extension θ_d0 [°]", fontsize=12)
ax.set_title(f"Extension vs D₀  (different ages)\nd={DIST_KPC} kpc  |  δ={DELTA_DIFF}",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

plt.suptitle(f"D₀ Scan — CTA  |  HESS J1837-069", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_d0_scan.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"{'D0 [cm²/s]':>14}  {'θ_d0 [°]':>10}  {'σ halo-only':>12}  {'σ PWN+halo':>11}")
print("-" * 55)
for d0, theta, s1, s2 in zip(D0_VALUES, theta_d0_vals, sigma_d0_halo_only, sigma_d0_halo):
    flag = "  ← detectable" if s1 >= 5 or s2 >= 5 else ""
    print(f"{d0:>14.2e}  {theta:>10.3f}  {s1:>12.2f}  {s2:>11.2f}{flag}")


from IPython.utils.capture import capture_output as _capture

_extd0_key    = "ext_d0_scan"
_cache_loaded = False

D0_EXT_VALUES  = np.logspace(25, 27, 7)
AGE_KYR_EXT    = [10, 21, 50, 100, 200]
AGE_COLORS_EXT = ["#d62728", "#e6550d", "#fd8d3c", "#3182bd", "#6baed6"]

if CACHE_FILE.exists() and not FORCE_RERUN:
    try:
        _c = np.load(CACHE_FILE, allow_pickle=True)
        if _extd0_key + "_ts_ho" in _c:
            D0_EXT_VALUES = _c[_extd0_key + "_D0"]
            AGE_KYR_EXT   = list(_c[_extd0_key + "_ages"])
            ts_extd0_ho   = _c[_extd0_key + "_ts_ho"]   # shape (n_age, n_D0)
            ts_extd0_ph   = _c[_extd0_key + "_ts_ph"]
            theta_extd0   = _c[_extd0_key + "_theta"]
            _cache_loaded = True
            print("✓ Sigma vs D₀ (extension curves) loaded from cache")
    except Exception as e:
        print(f"  Cache read failed ({e}), re-running ...")

if not _cache_loaded:
    n_age = len(AGE_KYR_EXT)
    n_D0  = len(D0_EXT_VALUES)
    ts_extd0_ho = np.zeros((n_age, n_D0))
    ts_extd0_ph = np.zeros((n_age, n_D0))
    theta_extd0 = np.zeros((n_age, n_D0))

    _E0 = HALO_E0.to_value(u.TeV)

    for i_a, age_kyr in enumerate(AGE_KYR_EXT):
        t_s_a = age_kyr * 1e3 * YR_TO_S
        print(f"\n── Age = {age_kyr} kyr ──")
        print(f"  {'D0':>12}  {'θ_d0 [°]':>10}  {'TS_halo_only':>13}  {'TS_pwn_halo':>12}")
        for i_d, D0 in enumerate(D0_EXT_VALUES):
            theta_this = diffusion_angle_deg(_E0, D0, DELTA_DIFF, t_s_a, dist_cm, _E0)
            theta_extd0[i_a, i_d] = theta_this

            phi_ea, _ = build_scan_halo_template(geom, SOURCE_CENTER.icrs, theta_d0_deg=theta_this)

            # Halo-only
            ds_ho = dataset.copy(name=f"extd0_ho_{i_a}_{i_d}")
            sky_ho = SkyModel(
                spatial_model=TemplateSpatialModel(phi_ea, normalize=True),
                spectral_model=PowerLawSpectralModel(
                    amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
                name=f"halo-extd0-ho-{i_a}-{i_d}")
            with _capture():
                ds_ho.models = Models([sky_ho, FoVBackgroundModel(dataset_name=ds_ho.name)])
            ds_ho.fake(random_state=SEED + 1000 + i_a*100 + i_d)
            ts_ho = run_halo_only(ds_ho, phi_ea, seed_offset=i_a*100+i_d)
            ts_extd0_ho[i_a, i_d] = ts_ho

            # PWN+halo
            ds_ph = dataset.copy(name=f"extd0_ph_{i_a}_{i_d}")
            ec_ea = _make_pwn_model(f"pwn-extd0-{i_a}-{i_d}", geom)
            sky_ph = SkyModel(
                spatial_model=TemplateSpatialModel(phi_ea, normalize=True),
                spectral_model=PowerLawSpectralModel(
                    amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
                name=f"halo-extd0-ph-{i_a}-{i_d}")
            with _capture():
                ds_ph.models = Models([ec_ea, sky_ph,
                                       FoVBackgroundModel(dataset_name=ds_ph.name)])
            ds_ph.fake(random_state=SEED + 1500 + i_a*100 + i_d)
            ts_ph = run_pwn_halo(ds_ph, phi_ea, seed_offset=i_a*100+i_d+500)
            ts_extd0_ph[i_a, i_d] = ts_ph

            print(f"  {D0:>12.2e}  {theta_this:>10.3f}  {ts_ho:>13.2f}  {ts_ph:>12.2f}")

    _ex = dict(np.load(CACHE_FILE, allow_pickle=True)) if CACHE_FILE.exists() else {}
    _ex[_extd0_key+"_D0"]    = np.array(D0_EXT_VALUES)
    _ex[_extd0_key+"_ages"]  = np.array(AGE_KYR_EXT)
    _ex[_extd0_key+"_ts_ho"] = ts_extd0_ho
    _ex[_extd0_key+"_ts_ph"] = ts_extd0_ph
    _ex[_extd0_key+"_theta"] = theta_extd0
    np.savez(CACHE_FILE, **_ex)
    print("\n✓ Sigma vs D₀ (extension curves) saved")

AGE_KYR_EXT   = list(AGE_KYR_EXT)
sigma_extd0_ho = np.sqrt(ts_extd0_ho)
sigma_extd0_ph = np.sqrt(ts_extd0_ph)

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

for ax, sigma_2d, title in zip(
        axes,
        [sigma_extd0_ho, sigma_extd0_ph],
        ["Halo-only", "PWN + Halo"]):

    for i_a, (age_kyr, col) in enumerate(zip(AGE_KYR_EXT, AGE_COLORS_EXT)):
        theta_label = theta_extd0[i_a, -1]   # extension at largest D0 for legend
        ax.semilogx(D0_EXT_VALUES, sigma_2d[i_a],
                    "o-", color=col, lw=2, ms=7,
                    label=f"age={age_kyr} kyr  (θ={theta_label:.2f}° @ D₀=10²⁷)")

    ax.axhline(5, ls="--", c="firebrick", lw=1.5, label="5σ threshold")
    ax.axvline(D0_cm2_s, ls=":", c="grey", lw=1.5, label=f"Locked D₀={D0_cm2_s:.0e}")
    ax.set_xlabel("D₀ [cm²/s]", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

axes[0].set_ylabel("Detection significance σ", fontsize=12)
plt.suptitle(
    f"σ vs D₀ for different extension values — CTA  |  HESS J1837-069\n"
    f"(d={DIST_KPC} kpc, each curve = different age → different θ_d0 at each D₀)",
    fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_sigma_vs_d0_ext_curves.png", dpi=150, bbox_inches="tight")
plt.show()


from IPython.utils.capture import capture_output as _capture
_extfix_key   = "extfix_d0_scan"
_cache_loaded = False
D0_EXT_VALUES   = np.logspace(25, 27, 7)
THETA_FIX_DEG   = [0.05, 0.1, 0.3, 0.5, 0.9, 1.5, 2.0]
THETA_FIX_COLS  = ["#7f2704", "#d62728", "#fd8d3c", "#74c476", "#3182bd", "#6baed6",
                    "#2ca25f"]
if CACHE_FILE.exists() and not FORCE_RERUN:
    try:
        _c = np.load(CACHE_FILE, allow_pickle=True)
        if _extfix_key + "_ts_ho" in _c:
            D0_EXT_VALUES = _c[_extfix_key + "_D0"]
            THETA_FIX_DEG = list(_c[_extfix_key + "_theta_fix"])
            ts_extfix_ho  = _c[_extfix_key + "_ts_ho"]   # shape (n_theta, n_D0)
            ts_extfix_ph  = _c[_extfix_key + "_ts_ph"]
            _cache_loaded = True
            print("✓ Sigma vs D₀ (fixed extension curves) loaded from cache")
    except Exception as e:
        print(f"  Cache read failed ({e}), re-running ...")
if not _cache_loaded:
    n_th = len(THETA_FIX_DEG)
    n_D0 = len(D0_EXT_VALUES)
    ts_extfix_ho = np.zeros((n_th, n_D0))
    ts_extfix_ph = np.zeros((n_th, n_D0))
    for i_th, theta_deg in enumerate(THETA_FIX_DEG):
        phi_fix, _ = build_scan_halo_template(geom, SOURCE_CENTER.icrs, theta_d0_deg=theta_deg)
        print(f"\n── θ_d0 = {theta_deg}° ──")
        print(f"  {'D0':>12}  {'TS_halo_only':>13}  {'TS_pwn_halo':>12}")
        for i_d, D0 in enumerate(D0_EXT_VALUES):
            # Halo-only
            ds_ho = dataset.copy(name=f"extfix_ho_{i_th}_{i_d}")
            sky_ho = SkyModel(
                spatial_model=TemplateSpatialModel(phi_fix, normalize=True),
                spectral_model=PowerLawSpectralModel(
                    amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
                name=f"halo-extfix-ho-{i_th}-{i_d}")
            with _capture():
                ds_ho.models = Models([sky_ho, FoVBackgroundModel(dataset_name=ds_ho.name)])
            ds_ho.fake(random_state=SEED + 2000 + i_th*100 + i_d)
            ts_ho = run_halo_only(ds_ho, phi_fix, seed_offset=i_th*100+i_d)
            ts_extfix_ho[i_th, i_d] = ts_ho
            # PWN+halo
            ds_ph = dataset.copy(name=f"extfix_ph_{i_th}_{i_d}")
            ec_fix = _make_pwn_model(f"pwn-extfix-{i_th}-{i_d}", geom)
            sky_ph = SkyModel(
                spatial_model=TemplateSpatialModel(phi_fix, normalize=True),
                spectral_model=PowerLawSpectralModel(
                    amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
                name=f"halo-extfix-ph-{i_th}-{i_d}")
            with _capture():
                ds_ph.models = Models([ec_fix, sky_ph,
                                       FoVBackgroundModel(dataset_name=ds_ph.name)])
            ds_ph.fake(random_state=SEED + 2500 + i_th*100 + i_d)
            ts_ph = run_pwn_halo(ds_ph, phi_fix, seed_offset=i_th*100+i_d+500)
            ts_extfix_ph[i_th, i_d] = ts_ph
            print(f"  {D0:>12.2e}  {ts_ho:>13.2f}  {ts_ph:>12.2f}")
    _ex = dict(np.load(CACHE_FILE, allow_pickle=True)) if CACHE_FILE.exists() else {}
    _ex[_extfix_key+"_D0"]        = np.array(D0_EXT_VALUES)
    _ex[_extfix_key+"_theta_fix"] = np.array(THETA_FIX_DEG)
    _ex[_extfix_key+"_ts_ho"]     = ts_extfix_ho
    _ex[_extfix_key+"_ts_ph"]     = ts_extfix_ph
    np.savez(CACHE_FILE, **_ex)
    print("\n✓ Fixed-extension scan saved")
sigma_extfix_ho = np.sqrt(ts_extfix_ho)
sigma_extfix_ph = np.sqrt(ts_extfix_ph)

_params_txt = (f"d = {DIST_KPC} kpc  |  t = {T_AGE_YR/1e3:.0f} kyr  |  "
               f"E_dat = {E_DAT_TEV:.0f} TeV  |  D₀ scanned  [HESS J1837-069]")
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
for i_ax, (ax, sigma_2d, title) in enumerate(zip(
        axes,
        [sigma_extfix_ho, sigma_extfix_ph],
        ["Halo-only", "PWN + Halo"])):
    for i_th, (theta_deg, col) in enumerate(zip(THETA_FIX_DEG, THETA_FIX_COLS)):
        ax.semilogx(D0_EXT_VALUES, sigma_2d[i_th],
                    "o-", color=col, lw=2, ms=7,
                    label=f"θ_d0 = {theta_deg}°")
    ax.axhline(5, ls="--", c="firebrick", lw=1.5, label="5σ threshold")
    ax.axvline(D0_cm2_s, ls=":", c="grey", lw=1.5, label=f"Locked D₀={D0_cm2_s:.0e}")
    ax.set_xlabel("D₀  [cm² s⁻¹]", fontsize=13)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, ncol=2); ax.grid(alpha=0.3, which="both")

axes[0].set_ylabel("Detection significance σ", fontsize=13)
axes[0].text(0.02, 0.02, _params_txt, transform=axes[0].transAxes,
             fontsize=10, ha='left', va='bottom',
             bbox=dict(boxstyle='round', fc='white', alpha=0.85))
plt.suptitle(
    f"σ vs D₀ for fixed extension values — CTA  |  HESS J1837-069\n"
    f"(each curve = fixed θ_d0 template; flat = D₀ doesn't affect pre-built template)",
    fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_sigma_vs_d0_fixed_ext.png", dpi=150, bbox_inches="tight")
plt.show()


from IPython.utils.capture import capture_output as _capture
_amp_key     = "amp_scan"
_cache_loaded = False

if CACHE_FILE.exists() and not FORCE_RERUN:
    try:
        _c = np.load(CACHE_FILE, allow_pickle=True)
        if _amp_key + "_amplitudes" in _c and _amp_key + "_ts_ho" in _c:
            HALO_AMPLITUDES  = _c[_amp_key + "_amplitudes"]
            ts_amp           = np.array(list(_c[_amp_key + "_ts"]))
            sigma_det_amp    = np.array(list(_c[_amp_key + "_sigma"]))
            ts_amp_ho        = np.array(list(_c[_amp_key + "_ts_ho"]))
            sigma_det_amp_ho = np.array(list(_c[_amp_key + "_sigma_ho"]))
            _cache_loaded    = True
            print(f"✓ Amplitude scan loaded from cache ({CACHE_FILE})")
    except Exception as e:
        print(f"  Cache read failed ({e}), re-running ...")

if not _cache_loaded:
    center_icrs      = SOURCE_CENTER.icrs
    HALO_AMPLITUDES  = np.logspace(-13, -11, 10)
    ts_amp           = []
    sigma_det_amp    = []
    ts_amp_ho        = []
    sigma_det_amp_ho = []

    print(f"{'Halo Amplitude':>16}  {'TS_pwn_halo':>12}  {'σ_ph':>6}  {'TS_halo_only':>13}  {'σ_ho':>6}")
    print("-" * 65)

    for i_amp, amp in enumerate(HALO_AMPLITUDES):
        amp_q = amp * u.Unit("cm-2 s-1 TeV-1")
        ds_sc = dataset.copy(name=f"sc_{i_amp}")
        ec_sc = _make_pwn_model(f"pwn-sc-{i_amp}", geom)
        sky_hs = SkyModel(
            spatial_model=TemplateSpatialModel(phi_map, normalize=True),
            spectral_model=PowerLawSpectralModel(
                amplitude=amp_q, index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-sc-{i_amp}")
        diff_sc_roi  = diffuse_emission_map.interp_to_geom(ds_sc.exposure.geom)
        diff_sc_tmpl = TemplateSpatialModel(diff_sc_roi, normalize=False)
        diff_sc      = SkyModel(PowerLawNormSpectralModel(), diff_sc_tmpl, name=f"diff-sc-{i_amp}")
        bkg_sc       = FoVBackgroundModel(dataset_name=ds_sc.name)
        with _capture():
            ds_sc.models = Models([ec_sc, sky_hs, diff_sc, bkg_sc])
        ds_sc.fake(random_state=SEED + 10 + i_amp)
        ts_h = run_pwn_halo(ds_sc, phi_map, seed_offset=i_amp*200)
        sig_h = np.sqrt(ts_h)
        ts_amp.append(ts_h); sigma_det_amp.append(sig_h)

        ds_sc_ho = dataset.copy(name=f"sc_ho_{i_amp}")
        sky_hs_ho = SkyModel(
            spatial_model=TemplateSpatialModel(phi_map, normalize=True),
            spectral_model=PowerLawSpectralModel(
                amplitude=amp_q, index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-sc-ho-{i_amp}")
        bkg_sc_ho = FoVBackgroundModel(dataset_name=ds_sc_ho.name)
        with _capture():
            ds_sc_ho.models = Models([sky_hs_ho, bkg_sc_ho])
        ds_sc_ho.fake(random_state=SEED + 810 + i_amp)
        ts_ho = run_halo_only(ds_sc_ho, phi_map, seed_offset=i_amp*200 + 800)
        sig_ho = np.sqrt(ts_ho)
        ts_amp_ho.append(ts_ho); sigma_det_amp_ho.append(sig_ho)

        print(f"{amp:16.2e}  {ts_h:12.2f}  {sig_h:6.2f}σ  {ts_ho:13.2f}  {sig_ho:6.2f}σ")

    ts_amp           = np.array(ts_amp)
    sigma_det_amp    = np.array(sigma_det_amp)
    ts_amp_ho        = np.array(ts_amp_ho)
    sigma_det_amp_ho = np.array(sigma_det_amp_ho)
    _ex = dict(np.load(CACHE_FILE, allow_pickle=True)) if CACHE_FILE.exists() else {}
    _ex[_amp_key+"_amplitudes"] = HALO_AMPLITUDES
    _ex[_amp_key+"_ts"]         = ts_amp
    _ex[_amp_key+"_sigma"]      = sigma_det_amp
    _ex[_amp_key+"_ts_ho"]      = ts_amp_ho
    _ex[_amp_key+"_sigma_ho"]   = sigma_det_amp_ho
    np.savez(CACHE_FILE, **_ex)
    print(f"\n✓ Amplitude scan saved to {CACHE_FILE}")

HALO_AMPLITUDES  = np.array(HALO_AMPLITUDES)
ts_amp           = np.array(ts_amp)
sigma_det_amp    = np.array(sigma_det_amp)
ts_amp_ho        = np.array(ts_amp_ho)
sigma_det_amp_ho = np.array(sigma_det_amp_ho)

from scipy.interpolate import interp1d
try:
    interp_fn_ph  = interp1d(sigma_det_amp, HALO_AMPLITUDES, kind="linear", fill_value="extrapolate")
    amp_5sigma_ph = float(interp_fn_ph(5.0))
    thresh_str_ph = f"{amp_5sigma_ph:.2e} cm⁻² s⁻¹ TeV⁻¹"
except Exception:
    amp_5sigma_ph = None; thresh_str_ph = "N/A"
try:
    interp_fn_ho  = interp1d(sigma_det_amp_ho, HALO_AMPLITUDES, kind="linear", fill_value="extrapolate")
    amp_5sigma_ho = float(interp_fn_ho(5.0))
    thresh_str_ho = f"{amp_5sigma_ho:.2e} cm⁻² s⁻¹ TeV⁻¹"
except Exception:
    amp_5sigma_ho = None; thresh_str_ho = "N/A"

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(HALO_AMPLITUDES, sigma_det_amp_ho, "o-", color="steelblue", lw=2, ms=7,
            label="Halo-only (H₀=bkg, H₁=halo+bkg)")
ax.semilogx(HALO_AMPLITUDES, sigma_det_amp, "s--", color="darkorange", lw=2, ms=7,
            label="PWN+Halo (H₀=bkg only, H₁=PWN+halo+bkg)")
ax.axhline(5, ls="--", c="firebrick", lw=2, label="5σ threshold")
ax.axvline(HALO_AMPLITUDE.to_value(u.Unit("cm-2 s-1 TeV-1")), ls=":", c="grey", lw=1.5,
           label="Locked amplitude")
ax.set_xlabel("Halo Amplitude [cm⁻² s⁻¹ TeV⁻¹]", fontsize=12)
ax.set_ylabel("Detection significance [σ]", fontsize=12)
ax.set_title("Halo Detection Significance vs Amplitude\nHalo-only vs PWN+Halo (CTA)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10); ax.grid(which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_amp_scan_combined.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n5σ threshold — PWN+Halo: {thresh_str_ph}  |  Halo-only: {thresh_str_ho}")

from IPython.utils.capture import capture_output as _capture
_off_key       = "offset_scan"
_cache_loaded  = False

if CACHE_FILE.exists() and not FORCE_RERUN:
    try:
        _c = np.load(CACHE_FILE, allow_pickle=True)
        if _off_key + "_ts" in _c and _off_key + "_ts_ho" in _c:
            ts_offset     = list(_c[_off_key + "_ts"])
            sig_offset    = list(_c[_off_key + "_sigma"])
            ts_offset_ho  = list(_c[_off_key + "_ts_ho"])
            sig_offset_ho = list(_c[_off_key + "_sigma_ho"])
            offsets_deg   = list(_c[_off_key + "_offsets_deg"])
            OFFSET_FRACS  = list(_c[_off_key + "_fracs"])
            _cache_loaded = True
            print(f"✓ Offset scan loaded from cache ({CACHE_FILE})")
    except Exception as e:
        print(f"  Cache read failed ({e}), re-running ...")

if not _cache_loaded:
    center_icrs   = SOURCE_CENTER.icrs
    OFFSET_FRACS  = np.linspace(0.0, 5.0, num=10).tolist()
    ts_offset     = []
    sig_offset    = []
    ts_offset_ho  = []
    sig_offset_ho = []
    offsets_deg   = []

    print(f"{'Offset/θ_d0':>16}  {'Offset[deg]':>11}  {'TS_ph':>8}  {'σ_ph':>6}  {'TS_ho':>8}  {'σ_ho':>6}")
    print("-" * 70)

    for i_off, frac in enumerate(OFFSET_FRACS):
        off_d   = frac * HALO_THETA_D0.to_value(u.deg)
        off_ra  = off_d / np.cos(np.radians(center_icrs.dec.deg))
        src_off = SkyCoord(center_icrs.ra.deg + off_ra, center_icrs.dec.deg,
                           unit="deg", frame="icrs")
        offsets_deg.append(center_icrs.separation(src_off).deg)

        phi_off, _ = build_scan_halo_template(
            geom, src_off, D100_cm2_s=HALO_D100_CALIBRATED_CM2_S)

        # PWN and halo are both at the offset position (whole system shifted)
        ds_off = dataset.copy(name=f"off_{i_off}")
        ec_off = _make_pwn_model(f"pwn-off-{i_off}", geom)
        # Override PWN position to match offset
        ec_off.spatial_model.lon_0.value = src_off.ra.deg
        ec_off.spatial_model.lat_0.value = src_off.dec.deg
        sky_ho_mdl = SkyModel(
            spatial_model=TemplateSpatialModel(phi_off, normalize=True),
            spectral_model=PowerLawSpectralModel(
                amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-off-{i_off}")
        diff_off_roi  = diffuse_emission_map.interp_to_geom(ds_off.exposure.geom)
        diff_off_tmpl = TemplateSpatialModel(diff_off_roi, normalize=False)
        diff_off_mdl  = SkyModel(PowerLawNormSpectralModel(), diff_off_tmpl, name=f"diff-off-{i_off}")
        bkg_off       = FoVBackgroundModel(dataset_name=ds_off.name)
        with _capture():
            ds_off.models = Models([ec_off, sky_ho_mdl, bkg_off, diff_off_mdl])
        ds_off.fake(random_state=SEED + 20 + i_off)
        ts_h = run_pwn_halo(ds_off, phi_off, seed_offset=i_off*300)
        sig_h = np.sqrt(ts_h)
        ts_offset.append(ts_h); sig_offset.append(sig_h)

        ds_off_ho = dataset.copy(name=f"off_ho_{i_off}")
        sky_off_ho = SkyModel(
            spatial_model=TemplateSpatialModel(phi_off, normalize=True),
            spectral_model=PowerLawSpectralModel(
                amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-off-ho-{i_off}")
        bkg_off_ho = FoVBackgroundModel(dataset_name=ds_off_ho.name)
        with _capture():
            ds_off_ho.models = Models([sky_off_ho, bkg_off_ho])
        ds_off_ho.fake(random_state=SEED + 820 + i_off)
        ts_ho = run_halo_only(ds_off_ho, phi_off, seed_offset=i_off*300 + 800)
        sig_ho = np.sqrt(ts_ho)
        ts_offset_ho.append(ts_ho); sig_offset_ho.append(sig_ho)

        print(f"{frac:>16.2f}  {offsets_deg[-1]:>11.3f}  {ts_h:8.2f}  {sig_h:6.2f}σ  {ts_ho:8.2f}  {sig_ho:6.2f}σ")

    _ex = dict(np.load(CACHE_FILE, allow_pickle=True)) if CACHE_FILE.exists() else {}
    _ex[_off_key+"_ts"]          = np.array(ts_offset)
    _ex[_off_key+"_sigma"]       = np.array(sig_offset)
    _ex[_off_key+"_ts_ho"]       = np.array(ts_offset_ho)
    _ex[_off_key+"_sigma_ho"]    = np.array(sig_offset_ho)
    _ex[_off_key+"_offsets_deg"] = np.array(offsets_deg)
    _ex[_off_key+"_fracs"]       = np.array(OFFSET_FRACS)
    np.savez(CACHE_FILE, **_ex)
    print(f"\n✓ Offset scan saved to {CACHE_FILE}")

ts_offset     = np.array(ts_offset)
sig_offset    = np.array(sig_offset)
ts_offset_ho  = np.array(ts_offset_ho)
sig_offset_ho = np.array(sig_offset_ho)
offsets_deg   = np.array(offsets_deg)
_x_axis = offsets_deg / HALO_THETA_D0.to_value(u.deg)

_params_off = (f"θ_d0={HALO_THETA_D0.value:.2f}°  |  d={DIST_KPC} kpc  |  "
               f"t={T_AGE_YR/1e3:.0f} kyr  |  E_dat={E_DAT_TEV:.0f} TeV  [FIXED]\n"
               f"(Both halo template and PWN shifted together with offset)")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(_x_axis, sig_offset_ho, "o-", color="steelblue", lw=2, ms=8,
        label="Halo-only (H₀=bkg, H₁=halo+bkg)")
ax.plot(_x_axis, sig_offset, "s--", color="darkorange", lw=2, ms=8,
        label="PWN+Halo (H₀=bkg only, H₁=PWN+halo+bkg)")
ax.axvline(0, ls="--", c="green", lw=1.8, label="HESS J1837-069 (as simulated, zero offset)")
ax.axhline(5, ls="--", c="firebrick", lw=2, label="5σ threshold")
ax.set_xlabel("Halo offset from FoV centre  [× θ_d0]", fontsize=13)
ax.set_ylabel("Detection significance σ", fontsize=13)
ax.set_title("Halo detectability vs offset (whole source system shifted)\nHalo-only vs PWN+Halo — CTA",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=11); ax.grid(alpha=0.3)
ax.text(0.98, 0.97, _params_off, transform=ax.transAxes, fontsize=9,
        ha='right', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.85))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_offset_scan_combined.png", dpi=150, bbox_inches="tight")
plt.show()


# For each bkg scale, scan halo amplitudes and record significance (halo-only).
from IPython.utils.capture import capture_output as _capture
import warnings
warnings.filterwarnings("ignore", message="The filename is not defined")

_ampbkg_key    = "amp_bkg_scan"
_cache_loaded  = False
BKG_SCALES_AMP = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]
HALO_AMPS_FINE = np.logspace(-13, -11, 9)
BKG_AMP_COLS   = plt.cm.RdYlGn(np.linspace(0.15, 0.9, len(BKG_SCALES_AMP)))

if CACHE_FILE.exists() and not FORCE_RERUN:
    try:
        _c = np.load(CACHE_FILE, allow_pickle=True)
        if _ampbkg_key + "_sigma_grid" in _c:
            sigma_amp_bkg_grid = _c[_ampbkg_key + "_sigma_grid"]
            HALO_AMPS_FINE     = _c[_ampbkg_key + "_amplitudes"]
            BKG_SCALES_AMP     = list(_c[_ampbkg_key + "_bkg_scales"])
            _cache_loaded      = True
            print("✓ Amplitude vs background scan loaded from cache")
    except Exception as e:
        print(f"  Cache read failed ({e}), re-running ...")

if not _cache_loaded:
    n_bkg = len(BKG_SCALES_AMP)
    n_amp = len(HALO_AMPS_FINE)
    sigma_amp_bkg_grid = np.zeros((n_bkg, n_amp))

    for i_bkg, bkg_s in enumerate(BKG_SCALES_AMP):
        print(f"\n── bkg_scale = {bkg_s:.2f} ──")
        for i_amp, amp in enumerate(HALO_AMPS_FINE):
            amp_q = amp * u.Unit("cm-2 s-1 TeV-1")
            ds_ab = dataset.copy(name=f"ab_{i_bkg}_{i_amp}")
            sky_ab = SkyModel(
                spatial_model=TemplateSpatialModel(phi_map, normalize=True),
                spectral_model=PowerLawSpectralModel(
                    amplitude=amp_q, index=HALO_ALPHA, reference=HALO_E0),
                name=f"halo-ab-{i_bkg}-{i_amp}")
            bkg_ab = FoVBackgroundModel(dataset_name=ds_ab.name)
            bkg_ab.spectral_model.norm.value  = bkg_s
            bkg_ab.spectral_model.norm.frozen = False
            with _capture():
                ds_ab.models = Models([sky_ab, bkg_ab])
            ds_ab.fake(random_state=SEED + 3000 + i_bkg * 100 + i_amp)
            ts_ab = run_halo_only(ds_ab, phi_map, seed_offset=i_bkg * 100 + i_amp + 3000)
            sigma_amp_bkg_grid[i_bkg, i_amp] = np.sqrt(ts_ab)
        print(f"  σ = {np.round(sigma_amp_bkg_grid[i_bkg], 1)}")

    _ex = dict(np.load(CACHE_FILE, allow_pickle=True)) if CACHE_FILE.exists() else {}
    _ex[_ampbkg_key + "_sigma_grid"]  = sigma_amp_bkg_grid
    _ex[_ampbkg_key + "_amplitudes"]  = np.array(HALO_AMPS_FINE)
    _ex[_ampbkg_key + "_bkg_scales"]  = np.array(BKG_SCALES_AMP)
    np.savez(CACHE_FILE, **_ex)
    print("\n✓ Amplitude vs background scan saved")

sigma_amp_bkg_grid = np.array(sigma_amp_bkg_grid)
HALO_AMPS_FINE     = np.array(HALO_AMPS_FINE)
BKG_AMP_COLS       = plt.cm.RdYlGn(np.linspace(0.15, 0.9, len(BKG_SCALES_AMP)))

from scipy.interpolate import interp1d
thresh_amps = []
fig, ax = plt.subplots(figsize=(9, 5))

# σ vs amplitude for each bkg scale
for i_bkg, (bkg_s, col) in enumerate(zip(BKG_SCALES_AMP, BKG_AMP_COLS)):
    sigmas = sigma_amp_bkg_grid[i_bkg]
    ax.semilogx(HALO_AMPS_FINE, sigmas, "o-", color=col, lw=2, ms=6,
                label=f"bkg = {bkg_s:.2f}×")
    try:
        f5 = interp1d(sigmas, HALO_AMPS_FINE, kind="linear", fill_value="extrapolate")
        thresh_amps.append(float(f5(5.0)))
    except Exception:
        thresh_amps.append(np.nan)
ax.axhline(5, ls="--", c="firebrick", lw=2, label="5σ threshold")
ax.axvline(HALO_AMPLITUDE.to_value(u.Unit("cm-2 s-1 TeV-1")), ls=":", c="grey", lw=1.5,
           label="Locked amplitude")
ax.set_xlabel("Halo Amplitude  [cm⁻² s⁻¹ TeV⁻¹]", fontsize=13)
ax.set_ylabel("Detection significance σ  (halo-only)", fontsize=13)
ax.set_title("Amplitude Sensitivity vs Background Scale\nHalo-only scenario  —  CTA", fontsize=12, fontweight="bold")
ax.legend(fontsize=10, ncol=2); ax.grid(which="both", alpha=0.3)
ax.yaxis.set_major_locator(plt.MultipleLocator(10.0))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5.0))
ax.tick_params(axis='y', which='minor', length=3)
_ptxt_ab = (f"θ_d0={HALO_THETA_D0.value:.2f}°  |  d={DIST_KPC} kpc  |  "
            f"t={T_AGE_YR/1e3:.0f} kyr  |  E_dat={E_DAT_TEV:.0f} TeV  [FIXED]")
ax.text(0.02, 0.97, _ptxt_ab, transform=ax.transAxes, fontsize=9,
        ha='left', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.85))

plt.suptitle("CTA | HESS J1837-069 — Amplitude vs Background Sensitivity", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_amp_vs_bkg.png", dpi=150, bbox_inches="tight")
plt.show()


from IPython.utils.capture import capture_output as _capture
import warnings
warnings.filterwarnings("ignore", message="The filename is not defined")

_offbkg_key  = "offset_bkg_2d"
_cache_loaded = False

OFFSET_OB   = np.linspace(0.0, 3.0, 7)          # offset [× θ_d0]
BKG_OB      = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]  # bkg scale factors
E_DAT_OB    = 5.0   # TeV — local E_dat for this scan (< global {E_DAT_TEV} TeV)
# Recompute theta_d0 at E_DAT_OB for the offset template
THETA_OB    = diffusion_angle_deg(E0_TeV, D0_cm2_s, DELTA_DIFF, t_age_s, dist_cm, E_DAT_OB) * u.deg
print(f"Offset×Bkg scan: E_dat={E_DAT_OB} TeV → θ_d0 = {THETA_OB.value:.3f}°  (global = {HALO_THETA_D0.value:.3f}°)")

if CACHE_FILE.exists() and not FORCE_RERUN:
    try:
        _c = np.load(CACHE_FILE, allow_pickle=True)
        if _offbkg_key + "_sig_grid" in _c:
            sig_ob_grid = _c[_offbkg_key + "_sig_grid"]
            OFFSET_OB   = _c[_offbkg_key + "_offsets"]
            BKG_OB      = list(_c[_offbkg_key + "_bkg_scales"])
            _cache_loaded = True
            print("✓ Offset × Background scan loaded from cache")
    except Exception as e:
        print(f"  Cache read failed ({e}), re-running ...")

if not _cache_loaded:
    center_icrs = SOURCE_CENTER.icrs
    n_off = len(OFFSET_OB)
    n_bkg = len(BKG_OB)
    sig_ob_grid = np.zeros((n_bkg, n_off))

    # Size is fixed at THETA_OB for the whole scan (only offset/bkg vary) --
    # calibrate D100 once and reuse it at every grid point.
    D100_OB = calibrate_physical_D100_cm2_s(
        target_sigma_deg=THETA_OB.to_value(u.deg),
        target_energy_TeV=HALO_THETA_E0.to_value(u.TeV),
        age_kyr=T_AGE_YR / 1e3, distance_kpc=DIST_KPC, tau0_kyr=HALO_TAU0_KYR,
        injection_index=HALO_INJECTION_INDEX, cutoff_TeV=HALO_CUTOFF_TEV,
        diffusion_index=DELTA_DIFF, u_total_eV_cm3=HALO_U_TOTAL_EV_CM3,
        n_time_slices=HALO_N_TIME_SLICES,
    )

    for i_off, frac in enumerate(OFFSET_OB):
        off_d  = frac * THETA_OB.to_value(u.deg)
        off_ra = off_d / np.cos(np.radians(center_icrs.dec.deg))
        src_ob = SkyCoord(center_icrs.ra.deg + off_ra, center_icrs.dec.deg,
                          unit="deg", frame="icrs")
        phi_ob, _ = build_scan_halo_template(geom, src_ob, D100_cm2_s=D100_OB)

        print(f"\n── offset = {frac:.1f}×θ_d0 ({off_d:.2f}°) ──")
        for i_bkg, bkg_s in enumerate(BKG_OB):
            ds_ob = dataset.copy(name=f"ob_{i_off}_{i_bkg}")
            sky_ob = SkyModel(
                spatial_model=TemplateSpatialModel(phi_ob, normalize=True),
                spectral_model=PowerLawSpectralModel(
                    amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
                name=f"halo-ob-{i_off}-{i_bkg}")
            bkg_ob = FoVBackgroundModel(dataset_name=ds_ob.name)
            bkg_ob.spectral_model.norm.value  = bkg_s
            bkg_ob.spectral_model.norm.frozen = False
            with _capture():
                ds_ob.models = Models([sky_ob, bkg_ob])
            ds_ob.fake(random_state=SEED + 4000 + i_off * 100 + i_bkg)
            ts_ob = run_halo_only(ds_ob, phi_ob, seed_offset=i_off * 100 + i_bkg + 4000)
            sig_ob_grid[i_bkg, i_off] = np.sqrt(ts_ob)
            print(f"  bkg={bkg_s:.2f} → σ={sig_ob_grid[i_bkg, i_off]:.1f}")

    _ex = dict(np.load(CACHE_FILE, allow_pickle=True)) if CACHE_FILE.exists() else {}
    _ex[_offbkg_key + "_sig_grid"]   = sig_ob_grid
    _ex[_offbkg_key + "_offsets"]    = np.array(OFFSET_OB)
    _ex[_offbkg_key + "_bkg_scales"] = np.array(BKG_OB)
    np.savez(CACHE_FILE, **_ex)
    print("\n✓ Offset × Background scan saved")

OFFSET_OB   = np.array(OFFSET_OB)
BKG_OB      = np.array(BKG_OB)
sig_ob_grid = np.array(sig_ob_grid)
_off_deg_ob = OFFSET_OB * THETA_OB.to_value(u.deg)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
im = ax.pcolormesh(OFFSET_OB, BKG_OB, sig_ob_grid, cmap="YlOrRd", shading="auto", vmin=0)
fig.colorbar(im, ax=ax).set_label("Detection significance σ", fontsize=11)
try:
    cs = ax.contour(OFFSET_OB, BKG_OB, sig_ob_grid, levels=[5],
                    colors="white", linewidths=2.5, linestyles="--")
    ax.clabel(cs, fmt="5σ", fontsize=11, colors="white")
except Exception:
    pass
ax.scatter([0], [1.0], marker="*", s=350, color="cyan", edgecolors="black",
           linewidths=1.3, zorder=10, label="HESS J1837-069 (assumed)")
ax.legend(fontsize=9, loc="upper right")
ax.set_xlabel("Halo offset  [× θ_d0]", fontsize=13)
ax.set_ylabel("Background scale factor", fontsize=13)
ax.set_title("Offset × Background: Halo-only σ  (CTA)\nHESS J1837-069", fontsize=12, fontweight="bold")

# Right: σ vs offset curves, one per bkg scale
ax = axes[1]
bkg_cols_ob = plt.cm.RdYlGn(np.linspace(0.15, 0.9, len(BKG_OB)))
for i_bkg, (bkg_s, col) in enumerate(zip(BKG_OB, bkg_cols_ob)):
    ax.plot(_off_deg_ob, sig_ob_grid[i_bkg], "o-", color=col,
            lw=2, ms=6, label=f"bkg={bkg_s:.2f}×")
ax.axvline(0, ls="--", c="green", lw=1.8, label="HESS J1837-069 (zero offset)")
ax.axhline(5, ls="--", c="firebrick", lw=2, label="5σ")
ax.set_xlabel("Halo offset  [°]", fontsize=13)
ax.set_ylabel("Detection significance σ", fontsize=13)
ax.set_title("Significance vs Offset (by background scale)\nHalo-only  —  CTA", fontsize=12, fontweight="bold")
ax.legend(fontsize=10, ncol=2); ax.grid(alpha=0.3)
_ptxt_ob = (f"θ_d0={THETA_OB.value:.3f}°  |  d={DIST_KPC} kpc  |  "
            f"t={T_AGE_YR/1e3:.0f} kyr  |  E_dat={E_DAT_OB:.0f} TeV (scan)  |  global E_dat={E_DAT_TEV:.0f} TeV  [FIXED]")
axes[0].text(0.02, 0.02, _ptxt_ob, transform=axes[0].transAxes, fontsize=9,
             ha='left', va='bottom', bbox=dict(boxstyle='round', fc='white', alpha=0.85))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_offset_bkg_2d.png", dpi=150, bbox_inches="tight")
plt.show()


_halo_key     = "halo_size_scan"
_cache_loaded = False

if CACHE_FILE.exists() and not FORCE_RERUN:
    try:
        _c = np.load(CACHE_FILE, allow_pickle=True)
        if _halo_key + "_ts_list" in _c and _halo_key + "_ts_list_ph" in _c:
            SIGMA_LIST      = list(_c[_halo_key + "_sigma_list"])
            ts_var_list     = list(_c[_halo_key + "_ts_list"])
            ts_var_list_ph  = list(_c[_halo_key + "_ts_list_ph"])
            _cache_loaded = True
            print(f"✓ Halo size scan loaded from cache ({CACHE_FILE})")
    except Exception as e:
        print(f"  Cache read failed ({e}), re-running ...")

if not _cache_loaded:
    from IPython.utils.capture import capture_output as _capture
    SIGMA_LIST     = np.linspace(0.2, 2.0, num=10).tolist()
    ts_var_list    = []
    ts_var_list_ph = []

    print(f"{'θ_d0 [deg]':>12}  {'TS_halo_only':>13}  {'σ_ho':>6}  {'TS_pwn_halo':>12}  {'σ_ph':>6}")
    print("-" * 60)

    for i_var, th_d0 in enumerate(SIGMA_LIST):
        phi_var, _ = build_scan_halo_template(geom, SOURCE_CENTER.icrs, theta_d0_deg=th_d0)

        ds_var = dataset.copy(name=f"var_{i_var}")
        sky_var = SkyModel(
            spatial_model=TemplateSpatialModel(phi_var, normalize=True),
            spectral_model=PowerLawSpectralModel(
                amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-var-{i_var}")
        bkg_var = FoVBackgroundModel(dataset_name=ds_var.name)
        with _capture():
            ds_var.models = Models([sky_var, bkg_var])
        ds_var.fake(random_state=SEED + 30 + i_var)
        ts_h = run_halo_only(ds_var, phi_var, seed_offset=i_var*700)
        ts_var_list.append(ts_h)

        ds_var_ph = dataset.copy(name=f"var_ph_{i_var}")
        ec_var = _make_pwn_model(f"pwn-var-{i_var}", geom)
        sky_var_ph = SkyModel(
            spatial_model=TemplateSpatialModel(phi_var, normalize=True),
            spectral_model=PowerLawSpectralModel(
                amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-var-ph-{i_var}")
        bkg_var_ph = FoVBackgroundModel(dataset_name=ds_var_ph.name)
        with _capture():
            ds_var_ph.models = Models([ec_var, sky_var_ph, bkg_var_ph])
        ds_var_ph.fake(random_state=SEED + 830 + i_var)
        ts_ph = run_pwn_halo(ds_var_ph, phi_var, seed_offset=i_var*700 + 800)
        ts_var_list_ph.append(ts_ph)

        print(f"{th_d0:>12.2f}  {ts_h:>13.2f}  {np.sqrt(ts_h):>6.1f}σ  {ts_ph:>12.2f}  {np.sqrt(ts_ph):>6.1f}σ")

    _ex = dict(np.load(CACHE_FILE, allow_pickle=True)) if CACHE_FILE.exists() else {}
    _ex[_halo_key + "_sigma_list"]  = np.array(SIGMA_LIST)
    _ex[_halo_key + "_ts_list"]     = np.array(ts_var_list)
    _ex[_halo_key + "_ts_list_ph"]  = np.array(ts_var_list_ph)
    np.savez(CACHE_FILE, **_ex)
    print(f"\n✓ Halo size scan saved to {CACHE_FILE}")

SIGMA_LIST     = np.array(SIGMA_LIST)
ts_var_list    = np.array(ts_var_list)
ts_var_list_ph = np.array(ts_var_list_ph)
sig_var        = np.sqrt(ts_var_list)
sig_var_ph     = np.sqrt(ts_var_list_ph)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(SIGMA_LIST, sig_var, "o-", color="steelblue", lw=2, ms=8,
        label="Halo-only (H₀=bkg, H₁=halo+bkg)")
ax.plot(SIGMA_LIST, sig_var_ph, "s--", color="darkorange", lw=2, ms=8,
        label="PWN+Halo (H₀=bkg only, H₁=PWN+halo+bkg)")
ax.axhline(5, ls="--", c="firebrick", lw=2, label="5σ threshold")
ax.axvline(HALO_THETA_D0.to_value(u.deg), ls=":", c="black", lw=1.8, alpha=0.8,
           label=f"Locked θ_d0 = {HALO_THETA_D0.value:.3f}°")
ax.set_xlabel("Halo size θ_d0 [deg]", fontsize=12)
ax.set_ylabel("Detection significance [σ]", fontsize=12)
ax.set_title(
    "Halo Size vs Detectability\nHalo-only vs PWN+Halo (CTA)",
    fontsize=11, fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
_ptxt = (f"E_dat={E_DAT_TEV:.0f} TeV  |  D₀={D0_cm2_s:.0e} cm²/s\n"
         f"d={DIST_KPC} kpc  |  age={T_AGE_YR/1e3:.0f} kyr")
ax.text(0.98, 0.05, _ptxt, transform=ax.transAxes, fontsize=8,
        ha='right', va='bottom', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_halo_size_scan_combined.png", dpi=150, bbox_inches="tight")
plt.show()

# -- Background scaling scan -- PWN + HALO (was MISSING; sig_p10 etc. below --
# were referenced by the combined plot cell but never actually computed) --
from IPython.utils.capture import capture_output as _capture
_bkg_pwnhalo_key = "bkg_pwn_halo"
_cache_loaded = False

if CACHE_FILE.exists() and not FORCE_RERUN:
    try:
        _c = np.load(CACHE_FILE, allow_pickle=True)
        if _bkg_pwnhalo_key + "_results" in _c:
            bkg_results = dict(_c[_bkg_pwnhalo_key + "_results"].item())
            _cache_loaded = True
            print("✓ Bkg scan (PWN+halo) loaded from cache")
    except Exception as e:
        print(f"  Cache read failed ({e}), re-running ...")

if not _cache_loaded:
    _bkg_scenarios = {
        "bkg_p10": [s * 1.10 for s in BKG_SCALE_FACTORS],
        "bkg_m10": [s * 0.90 for s in BKG_SCALE_FACTORS],
        "bkg_p20": [s * 1.20 for s in BKG_SCALE_FACTORS],
        "bkg_m20": [s * 0.80 for s in BKG_SCALE_FACTORS],
    }
    bkg_results = {}

    for _bkg_key, scale_factors in _bkg_scenarios.items():
        ts_list  = []
        sig_list = []
        print(f"\n-- Scenario: {_bkg_key} --")
        print(f"{'Bkg scale':>10}  {'TS':>10}  {'sigma':>8}  {'Detected':>10}")
        print("-" * 46)
        for i_bkg, bkg_scale in enumerate(scale_factors):
            ds_bph = dataset.copy(name=f"bkgph_{_bkg_key}_{i_bkg}")
            ec_bph = _make_pwn_model(f"pwn-bkg-{_bkg_key}-{i_bkg}", ds_bph.counts.geom)
            sky_bph = SkyModel(
                spatial_model=TemplateSpatialModel(phi_map, normalize=True),
                spectral_model=PowerLawSpectralModel(
                    amplitude=HALO_AMPLITUDE,
                    index=HALO_ALPHA, reference=HALO_E0),
                name=f"halo-bkg-{_bkg_key}-{i_bkg}")
            bkg_bph = FoVBackgroundModel(dataset_name=ds_bph.name)
            bkg_bph.spectral_model.norm.value = bkg_scale
            with _capture():
                ds_bph.models = Models([ec_bph, sky_bph, bkg_bph])
            ds_bph.fake(random_state=SEED + 850 + i_bkg)
            ts_b = run_pwn_halo(ds_bph, phi_map, seed_offset=i_bkg*900 + 500)
            sig_b = np.sqrt(max(ts_b, 0.0))
            ts_list.append(ts_b); sig_list.append(sig_b)
            print(f"{bkg_scale:>10.2f}  {ts_b:>10.2f}  {sig_b:>8.2f}  {'✓ YES' if ts_b>=25 else '✗ NO':>10}")
        bkg_results[_bkg_key] = {
            "scales": scale_factors, "ts": ts_list, "sigma": sig_list}

    _ex = dict(np.load(CACHE_FILE, allow_pickle=True)) if CACHE_FILE.exists() else {}
    _ex[_bkg_pwnhalo_key + "_results"] = np.array(bkg_results, dtype=object)
    for _bkg_key, _payload in bkg_results.items():
        _ex[f"{_bkg_key}_scale_factors"] = np.array(_payload["scales"])
        _ex[f"{_bkg_key}_sigma"]         = np.array(_payload["sigma"])
    np.savez(CACHE_FILE, **_ex)
    print("\n✓ Bkg scan (PWN+halo) saved")

# -- Derive band arrays (these are what the next cell's plot needs) --------
sig_p10 = np.array(bkg_results["bkg_p10"]["sigma"])
sig_m10 = np.array(bkg_results["bkg_m10"]["sigma"])
sig_p20 = np.array(bkg_results["bkg_p20"]["sigma"])
sig_m20 = np.array(bkg_results["bkg_m20"]["sigma"])
sig_mid = (sig_p10 + sig_m10) / 2


fig, ax = plt.subplots(figsize=(8, 6))

# Halo-only
ax.fill_between(BKG_SCALE_FACTORS, sig_halo_m20, sig_halo_p20,
                alpha=0.15, color="steelblue", label="Halo-only ±20% bkg")
ax.fill_between(BKG_SCALE_FACTORS, sig_halo_m10, sig_halo_p10,
                alpha=0.40, color="steelblue", label="Halo-only ±10% bkg")
ax.plot(BKG_SCALE_FACTORS, sig_halo_mid, "o-", color="steelblue", lw=2.5, ms=8,
        label="Halo-only mean σ")

# PWN + halo joint
ax.fill_between(BKG_SCALE_FACTORS, sig_m20, sig_p20,
                alpha=0.18, color="green", label="PWN+Halo ±20% bkg")
ax.fill_between(BKG_SCALE_FACTORS, sig_m10, sig_p10,
                alpha=0.40, color="green", label="PWN+Halo ±10% bkg")
ax.plot(BKG_SCALE_FACTORS, sig_mid, "s-", color="green", lw=2.5, ms=8,
        label="PWN+Halo mean σ")

ax.axhline(5, ls="--", c="firebrick", lw=2, label="5σ threshold")

ax.set_xlabel("Background scale factor", fontsize=12)
ax.set_ylabel("Detection significance [σ]", fontsize=12)
ax.set_title("Background Systematics: Halo-Only vs PWN+Halo (Overlaid)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, ncol=2)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_bkg_combined.png", dpi=150, bbox_inches="tight")
plt.show()

# Using locked values from halo-only scans: E_DAT_TEV, D0_cm2_s, DIST_KPC

_age_key      = "age_scan"
_cache_loaded = False

E0_TeV = HALO_E0.to_value(u.TeV)

if CACHE_FILE.exists() and not FORCE_RERUN:
    try:
        _c = np.load(CACHE_FILE, allow_pickle=True)
        if _age_key + "_ages" in _c:
            AGE_VALUES     = list(_c[_age_key + "_ages"])
            ts_age_halo    = list(_c[_age_key + "_ts_halo"])
            theta_age_vals = list(_c[_age_key + "_theta_d0"])
            _cache_loaded  = True
            print(f"✓ Age scan loaded from cache ({CACHE_FILE})")
    except Exception as e:
        print(f"  Cache read failed ({e}), re-running ...")

if not _cache_loaded:
    from IPython.utils.capture import capture_output as _capture
    center_icrs = SOURCE_CENTER.icrs
    AGE_VALUES  = np.linspace(10e3, 350e3, 10)  # yr
    ts_age_halo    = []
    theta_age_vals = []

    print(f"{'Age [kyr]':>10}  {'theta_d0':>9}  {'TS_halo':>9}  {'sig_h':>7}")
    print("-" * 48)

    for i_age, t_yr in enumerate(AGE_VALUES):
        t_s_age    = t_yr * YR_TO_S
        theta_this = diffusion_angle_deg(
            E0_TeV, D0_cm2_s, DELTA_DIFF, t_s_age, dist_cm, E0_TeV)
        theta_age_vals.append(theta_this)

        phi_age, _ = build_scan_halo_template(geom, center_icrs, theta_d0_deg=theta_this)

        ds_age = dataset.copy(name=f"age_sim_{i_age}")
        ec_age = _make_pwn_model(f"pwn-age-{i_age}", geom)
        sky_ha = SkyModel(
            spatial_model=TemplateSpatialModel(phi_age, normalize=True),
            spectral_model=PowerLawSpectralModel(
                amplitude=HALO_AMPLITUDE,
                index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-age-{i_age}")
        bkg_age = FoVBackgroundModel(dataset_name=ds_age.name)
        with _capture():
            ds_age.models = Models([ec_age, sky_ha, bkg_age])
        ds_age.fake(random_state=SEED + 50 + i_age)

        ts_h = run_pwn_halo(ds_age, phi_age, seed_offset=i_age*500)
        ts_age_halo.append(ts_h)
        print(f"{t_yr/1e3:>10.1f}  {theta_this:>9.3f}  {ts_h:>9.2f}  {np.sqrt(ts_h):>7.1f}σ")

    _ex = dict(np.load(CACHE_FILE, allow_pickle=True)) \
          if CACHE_FILE.exists() else {}
    _ex[_age_key + "_ages"]     = np.array(AGE_VALUES)
    _ex[_age_key + "_ts_halo"]  = np.array(ts_age_halo)
    _ex[_age_key + "_theta_d0"] = np.array(theta_age_vals)
    np.savez(CACHE_FILE, **_ex)
    print(f"\n✓ Age scan saved to {CACHE_FILE}")

AGE_VALUES     = np.array(AGE_VALUES)
ts_age_halo    = np.array(ts_age_halo)
theta_age_vals = np.array(theta_age_vals)

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(AGE_VALUES/1e3, np.sqrt(ts_age_halo), "o-", color="pink",
        lw=2, ms=8, label="Halo detection σ")
ax.axhline(5, ls="--", c="firebrick", lw=2, label="5σ threshold")
ax.axvline(T_AGE_YR/1e3, ls=":", c="black", lw=1.8, alpha=0.7,
           label=f"Reference age = {T_AGE_YR/1e3:.0f} kyr")
ax.set_xlabel("Pulsar age [kyr]", fontsize=12)
ax.set_ylabel("Detection significance [σ]", fontsize=12)
ax.set_title("Halo detection significance vs pulsar age (CTA)\nPWN + Halo scenario", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_age_scan_pwn_halo.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Age range: {AGE_VALUES[0]/1e3:.0f} – {AGE_VALUES[-1]/1e3:.0f} kyr")
print(f"θ_d0 range: {theta_age_vals.min():.3f}° – {theta_age_vals.max():.3f}°")


fig, axes_ca = plt.subplots(1, 2, figsize=(15, 5))

ax_ca = axes_ca[0]
ax_ca.plot(AGE_VALUES_HALOONLY/1e3, np.sqrt(ts_age_halo_only), "o-",
           color="steelblue", lw=2.5, ms=8, label="Halo-only (H₀=bkg, H₁=halo+bkg)")
ax_ca.plot(AGE_VALUES/1e3, np.sqrt(ts_age_halo), "s--",
           color="darkorange", lw=2.5, ms=8, label="PWN+Halo (H₀=bkg only, H₁=PWN+halo+bkg)")
ax_ca.axhline(5, ls="--", c="firebrick", lw=2, label="5σ threshold")
ax_ca.axvline(T_AGE_YR/1e3, ls=":", c="grey", lw=1.8, alpha=0.7,
              label=f"Reference age = {T_AGE_YR/1e3:.0f} kyr")
ax_ca.set_xlabel("Pulsar age [kyr]", fontsize=12)
ax_ca.set_ylabel("Detection significance [σ]", fontsize=12)
ax_ca.set_title("Halo detection σ vs pulsar age\n(halo-only vs PWN+halo joint)",
                 fontsize=11, fontweight="bold")
ax_ca.legend(fontsize=9); ax_ca.grid(alpha=0.3)

ax_ca2 = axes_ca[1]
ax_ca2.plot(AGE_VALUES_HALOONLY/1e3, theta_age_only, "o-",
            color="steelblue", lw=2, ms=7, label="θ_d0 (halo-only scan)")
ax_ca2.plot(AGE_VALUES/1e3, theta_age_vals, "s--",
            color="darkorange", lw=2, ms=7, label="θ_d0 (PWN+halo scan)")
ax_ca2.axvline(T_AGE_YR/1e3, ls=":", c="grey", lw=1.5, label=f"Reference age={T_AGE_YR/1e3:.0f} kyr")
ax_ca2.axhline(HALO_THETA_D0.to_value(u.deg), ls="--", c="green", lw=1.5,
               label=f"HESS J1837-069 θ_d0={HALO_THETA_D0.value:.2f}°")
ax_ca2.set_xlabel("Pulsar age [kyr]", fontsize=12)
ax_ca2.set_ylabel("θ_d0 [°]", fontsize=12)
ax_ca2.set_title("Physical extension vs pulsar age", fontsize=11, fontweight="bold")
ax_ca2.legend(fontsize=9); ax_ca2.grid(alpha=0.3)
_ptxt = (f"D₀={D0_cm2_s:.0e} cm²/s  |  d={DIST_KPC} kpc  |  E_dat={E_DAT_TEV:.0f} TeV")
axes_ca[0].text(0.98, 0.03, _ptxt, transform=axes_ca[0].transAxes, fontsize=8,
               ha="right", va="bottom", bbox=dict(boxstyle="round", fc="white", alpha=0.8))

plt.suptitle("Age Scan — Halo-Only vs PWN+Halo (CTA)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_age_scan_combined.png", dpi=150, bbox_inches="tight")
plt.show()

from IPython.utils.capture import capture_output as _capture
import warnings
_off_all_key  = "offset_scan_all"
_cache_loaded = False

center_icrs = SOURCE_CENTER.icrs
E0_TeV      = HALO_E0.to_value(u.TeV)

_off_d   = 1.0 * HALO_THETA_D0.to_value(u.deg)
_off_ra  = _off_d / np.cos(np.radians(center_icrs.dec.deg))

if CACHE_FILE.exists() and not FORCE_RERUN:
    try:
        _c = np.load(CACHE_FILE, allow_pickle=True)
        if _off_all_key + "_ts_halo" in _c:
            OFFSET_FRACS_ALL = list(_c[_off_all_key + "_fracs"])
            offsets_deg_all  = list(_c[_off_all_key + "_offsets_deg"])
            ts_all_halo      = list(_c[_off_all_key + "_ts_halo"])
            ts_all_pev       = list(_c[_off_all_key + "_ts_pev"])
            _cache_loaded    = True
            print(f"✓ Offset scan (all) loaded from cache ({CACHE_FILE})")
    except Exception as e:
        print(f"  Cache read failed ({e}), re-running ...")

if not _cache_loaded:
    OFFSET_FRACS_ALL = np.linspace(0.0, 5.0, 10).tolist()
    offsets_deg_all  = []
    ts_all_halo      = []
    ts_all_pev       = []

    print(f"{'Offset[×θ_d0]':>14}  {'Offset[°]':>10}  "
          f"{'TS_halo':>8}  {'sig_h':>6}  "
          f"{'TS_pwn':>8}  {'sig_pwn':>7}")
    print("-" * 62)

    for i_off, frac in enumerate(OFFSET_FRACS_ALL):
       
        sys_off_d  = frac * HALO_THETA_D0.to_value(u.deg)
        sys_off_ra = sys_off_d / np.cos(np.radians(center_icrs.dec.deg))
        src_pos    = SkyCoord(center_icrs.ra.deg + sys_off_ra, center_icrs.dec.deg,
                              unit="deg", frame="icrs")
        offsets_deg_all.append(center_icrs.separation(src_pos).deg)

        phi_all, _ = build_scan_halo_template(
            geom, src_pos, D100_cm2_s=HALO_D100_CALIBRATED_CM2_S)

        _sp_all = PointSpatialModel(lon_0=src_pos.ra, lat_0=src_pos.dec, frame="icrs")
        _sp_all.lon_0.frozen = True; _sp_all.lat_0.frozen = True
        ec_all = SkyModel(
            spatial_model=_sp_all,
            spectral_model=PowerLawSpectralModel(
                amplitude=TEST_FLUX, index=SPECTRAL_INDEX, reference=E_REF),
            name=f"pwn-all-{i_off}")

        sky_hall = SkyModel(
            spatial_model=TemplateSpatialModel(phi_all, normalize=True),
            spectral_model=PowerLawSpectralModel(
                amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-all-{i_off}")
        bkg_all = FoVBackgroundModel(dataset_name=f"all_sim_{i_off}")
        ds_all  = dataset.copy(name=f"all_sim_{i_off}")
        with _capture():
            ds_all.models = Models([ec_all, sky_hall, bkg_all])
        ds_all.fake(random_state=SEED + 40 + i_off)

        ts_h = run_pwn_halo(ds_all, phi_all, seed_offset=i_off * 610)
        ts_all_halo.append(ts_h)

        ds_null = ds_all.copy(name=f"all_null_{i_off}")
        h_null  = SkyModel(
            spatial_model=TemplateSpatialModel(phi_all, normalize=True),
            spectral_model=PowerLawSpectralModel(
                amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-null-all-{i_off}")
        ds_null.models = Models([h_null, FoVBackgroundModel(dataset_name=ds_null.name)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_null = Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds_null])

        _sp_h0p = PointSpatialModel(lon_0=src_pos.ra, lat_0=src_pos.dec, frame="icrs")
        _sp_h0p.lon_0.frozen = True; _sp_h0p.lat_0.frozen = True
        ec_h0p = SkyModel(
            spatial_model=_sp_h0p,
            spectral_model=PowerLawSpectralModel(
                amplitude=TEST_FLUX, index=SPECTRAL_INDEX, reference=E_REF),
            name=f"pwn-all-h0p-{i_off}")
        h_h0p  = SkyModel(
            spatial_model=TemplateSpatialModel(phi_all, normalize=True),
            spectral_model=PowerLawSpectralModel(
                amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
            name=f"halo-all-h0p-{i_off}")
        ds_h0p = ds_all.copy(name=f"all_h0p_{i_off}")
        ds_h0p.models = Models([ec_h0p, h_h0p, FoVBackgroundModel(dataset_name=ds_h0p.name)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_h0p = Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds_h0p])
            r_h0p = Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds_h0p])

        ts_pev = max(0.0, r_null.total_stat - r_h0p.total_stat)
        ts_all_pev.append(ts_pev)

        print(f"{frac:>14.2f}  {offsets_deg_all[-1]:>10.3f}  "
              f"{ts_h:>8.2f}  {np.sqrt(ts_h):>6.1f}σ  "
              f"{ts_pev:>8.2f}  {np.sqrt(ts_pev):>7.1f}σ")

    _ex = dict(np.load(CACHE_FILE, allow_pickle=True)) if CACHE_FILE.exists() else {}
    _ex[_off_all_key + "_fracs"]       = np.array(OFFSET_FRACS_ALL)
    _ex[_off_all_key + "_offsets_deg"] = np.array(offsets_deg_all)
    _ex[_off_all_key + "_ts_halo"]     = np.array(ts_all_halo)
    _ex[_off_all_key + "_ts_pev"]      = np.array(ts_all_pev)
    np.savez(CACHE_FILE, **_ex)
    print(f"\n✓ Offset scan (all) saved to {CACHE_FILE}")

OFFSET_FRACS_ALL = np.array(OFFSET_FRACS_ALL)
offsets_deg_all  = np.array(offsets_deg_all)
ts_all_halo      = np.array(ts_all_halo)
ts_all_pev       = np.array(ts_all_pev)
sigma_all_halo   = np.sqrt(ts_all_halo)
sigma_all_pev    = np.sqrt(ts_all_pev)
print("✓ Offset scan (all) complete")

_2d_proper_key = "halo_offset_2d_proper"
_cache_loaded  = False

THETA_GRID_2D  = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0])  # halo sizes [°]
OFFSET_GRID_2D = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])        # offsets [× θ_d0]

if CACHE_FILE.exists() and not FORCE_RERUN:
    try:
        _c = np.load(CACHE_FILE, allow_pickle=True)
        if _2d_proper_key + "_ts_grid" in _c:
            TS_GRID_PROPER    = _c[_2d_proper_key + "_ts_grid"]
            SIG_GRID_PROPER   = _c[_2d_proper_key + "_sig_grid"]
            _cache_loaded = True
            print(f"✓ 2D proper scan loaded from cache")
    except Exception as e:
        print(f"  Cache read failed ({e}), re-running ...")

if not _cache_loaded:
    from IPython.utils.capture import capture_output as _capture
    center_icrs = SOURCE_CENTER.icrs
    TS_GRID_PROPER  = np.zeros((len(THETA_GRID_2D), len(OFFSET_GRID_2D)))
    SIG_GRID_PROPER = np.zeros_like(TS_GRID_PROPER)

    print(f"Running 2D scan: {len(THETA_GRID_2D)} θ × {len(OFFSET_GRID_2D)} offsets = "
          f"{len(THETA_GRID_2D)*len(OFFSET_GRID_2D)} sims")

    for i_th, theta_val in enumerate(THETA_GRID_2D):
        # D100 depends only on theta_val -- calibrate once per row, reuse
        # across every offset column in that row.
        D100_2d_row = calibrate_physical_D100_cm2_s(
            target_sigma_deg=float(theta_val),
            target_energy_TeV=HALO_THETA_E0.to_value(u.TeV),
            age_kyr=T_AGE_YR / 1e3, distance_kpc=DIST_KPC, tau0_kyr=HALO_TAU0_KYR,
            injection_index=HALO_INJECTION_INDEX, cutoff_TeV=HALO_CUTOFF_TEV,
            diffusion_index=DELTA_DIFF, u_total_eV_cm3=HALO_U_TOTAL_EV_CM3,
            n_time_slices=HALO_N_TIME_SLICES,
        )
        for i_off, off_frac in enumerate(OFFSET_GRID_2D):
            # Shift source by off_frac × theta_val
            sys_off_d  = off_frac * theta_val
            sys_off_ra = sys_off_d / np.cos(np.radians(center_icrs.dec.deg))
            src_pos = SkyCoord(
                center_icrs.ra.deg + sys_off_ra,
                center_icrs.dec.deg, unit="deg", frame="icrs")

            phi_2d_sim, _ = build_scan_halo_template(
                geom, src_pos, D100_cm2_s=D100_2d_row)

            ds_2d = dataset.copy(name=f"2d_{i_th}_{i_off}")
            sky_2d = SkyModel(
                spatial_model=TemplateSpatialModel(phi_2d_sim, normalize=True),
                spectral_model=PowerLawSpectralModel(
                    amplitude=HALO_AMPLITUDE,
                    index=HALO_ALPHA, reference=HALO_E0),
                name=f"halo-2d-{i_th}-{i_off}")
            with _capture():
                ds_2d.models = Models([sky_2d,
                                       FoVBackgroundModel(dataset_name=ds_2d.name)])
            ds_2d.fake(random_state=SEED + i_th*100 + i_off*10)

            ts_2d = run_halo_only(ds_2d, phi_2d_sim,
                                   seed_offset=i_th*200 + i_off*20)
            TS_GRID_PROPER[i_th, i_off]  = ts_2d
            SIG_GRID_PROPER[i_th, i_off] = np.sqrt(ts_2d)
            print(f"  θ={theta_val:.2f}°  off={off_frac:.1f}×θ → TS={ts_2d:.1f} "
                  f"({np.sqrt(ts_2d):.1f}σ)")

    _ex = dict(np.load(CACHE_FILE, allow_pickle=True)) if CACHE_FILE.exists() else {}
    _ex[_2d_proper_key + "_ts_grid"]  = TS_GRID_PROPER
    _ex[_2d_proper_key + "_sig_grid"] = SIG_GRID_PROPER
    np.savez(CACHE_FILE, **_ex)
    print("\n✓ 2D proper scan saved")

print(f"2D grid shape: {TS_GRID_PROPER.shape}  "
      f"(θ axis: {THETA_GRID_2D}, offset axis: {OFFSET_GRID_2D})")

# Uses TS_GRID_PROPER from the scan above — halo TS only (no outer product).
fig, axes_2d = plt.subplots(1, 2, figsize=(16, 6))

# Left: significance map
ax_s = axes_2d[0]
im_s = ax_s.pcolormesh(OFFSET_GRID_2D, THETA_GRID_2D, SIG_GRID_PROPER,
                        cmap="YlOrRd", shading="auto", vmin=0)
cb_s = fig.colorbar(im_s, ax=ax_s)
cb_s.set_label("Halo detection significance [σ]", fontsize=10)
cs_s = ax_s.contour(OFFSET_GRID_2D, THETA_GRID_2D, SIG_GRID_PROPER,
                     levels=[5], colors="white", linewidths=2, linestyles="--")
ax_s.clabel(cs_s, fmt="5σ", fontsize=10, colors="white")
ax_s.set_xlabel("Halo offset  [× θ_d0]", fontsize=12)
ax_s.set_ylabel("Halo size θ_d0 [°]", fontsize=12)
ax_s.set_title("Halo significance  (actual halo TS)", fontsize=11, fontweight="bold")

# Right: TS map
ax_t = axes_2d[1]
im_t = ax_t.pcolormesh(OFFSET_GRID_2D, THETA_GRID_2D, TS_GRID_PROPER,
                        cmap="YlOrRd", shading="auto", vmin=0)
cb_t = fig.colorbar(im_t, ax=ax_t)
cb_t.set_label("Halo TS", fontsize=10)
cs_t = ax_t.contour(OFFSET_GRID_2D, THETA_GRID_2D, TS_GRID_PROPER,
                     levels=[25], colors="white", linewidths=2, linestyles="--")
ax_t.clabel(cs_t, fmt="TS=25", fontsize=10, colors="white")
ax_t.set_xlabel("Halo offset  [× θ_d0]", fontsize=12)
ax_t.set_ylabel("Halo size θ_d0 [°]", fontsize=12)
ax_t.set_title("Halo TS  (actual halo TS, not outer product)", fontsize=11, fontweight="bold")

for _ax in (ax_s, ax_t):
    _ax.scatter([0], [HALO_THETA_D0.to_value(u.deg)], marker="*", s=350,
                color="cyan", edgecolors="black", linewidths=1.3, zorder=10,
                label="HESS J1837-069 (assumed)")
    _ax.legend(fontsize=9, loc="upper right")

plt.suptitle(
    f"2D Detectability Map: Halo Size vs Offset (CTA)\n"
    f"D₀={D0_cm2_s:.0e} cm²/s  |  δ={DELTA_DIFF}  |  d={DIST_KPC} kpc  |  "
    f"E_dat={E_DAT_TEV:.0f} TeV  |  HESS J1837-069",
    fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_2d_halo_offset.png", dpi=150, bbox_inches="tight")
plt.show()

i_peak = np.unravel_index(TS_GRID_PROPER.argmax(), TS_GRID_PROPER.shape)
print(f"Peak TS  : {TS_GRID_PROPER.max():.1f}  ({SIG_GRID_PROPER.max():.1f}σ)")
print(f"  at θ_d0={THETA_GRID_2D[i_peak[0]]:.2f}°, "
      f"offset={OFFSET_GRID_2D[i_peak[1]]:.1f}×θ_d0")

_2d_pwn_key    = "pwn_halo_offset_2d"
_pwn_2d_loaded = False

if CACHE_FILE.exists() and not FORCE_RERUN:
    try:
        _c = np.load(CACHE_FILE, allow_pickle=True)
        if _2d_pwn_key + "_ts" in _c:
            TS_PWN_2D  = _c[_2d_pwn_key + "_ts"]
            SIG_PWN_2D = _c[_2d_pwn_key + "_sig"]
            _pwn_2d_loaded = True
            print("✓ PWN+halo 2D scan loaded from cache")
    except Exception as e:
        print(f"  Cache read failed ({e}), re-running ...")

if not _pwn_2d_loaded:
    from IPython.utils.capture import capture_output as _capture
    center_icrs = SOURCE_CENTER.icrs
    TS_PWN_2D  = np.zeros((len(THETA_GRID_2D), len(OFFSET_GRID_2D)))
    SIG_PWN_2D = np.zeros_like(TS_PWN_2D)
    print(f"Running PWN+halo 2D scan: "
          f"{len(THETA_GRID_2D)} × {len(OFFSET_GRID_2D)} = "
          f"{len(THETA_GRID_2D)*len(OFFSET_GRID_2D)} sims")

    for i_th, theta_val in enumerate(THETA_GRID_2D):
        D100_2d_row = calibrate_physical_D100_cm2_s(
            target_sigma_deg=float(theta_val),
            target_energy_TeV=HALO_THETA_E0.to_value(u.TeV),
            age_kyr=T_AGE_YR / 1e3, distance_kpc=DIST_KPC, tau0_kyr=HALO_TAU0_KYR,
            injection_index=HALO_INJECTION_INDEX, cutoff_TeV=HALO_CUTOFF_TEV,
            diffusion_index=DELTA_DIFF, u_total_eV_cm3=HALO_U_TOTAL_EV_CM3,
            n_time_slices=HALO_N_TIME_SLICES,
        )
        for i_off, off_frac in enumerate(OFFSET_GRID_2D):
            sys_off_d  = off_frac * theta_val
            sys_off_ra = sys_off_d / np.cos(np.radians(center_icrs.dec.deg))
            src_pos = SkyCoord(
                center_icrs.ra.deg + sys_off_ra,
                center_icrs.dec.deg, unit="deg", frame="icrs")

            phi_2d, _ = build_scan_halo_template(
                geom, src_pos, D100_cm2_s=D100_2d_row)

            ds_2d = dataset.copy(name=f"pwn2d_{i_th}_{i_off}")
            sky_2d = SkyModel(
                spatial_model=TemplateSpatialModel(phi_2d, normalize=True),
                spectral_model=PowerLawSpectralModel(
                    amplitude=HALO_AMPLITUDE,
                    index=HALO_ALPHA, reference=HALO_E0),
                name=f"halo-2d-pwn-{i_th}-{i_off}")
            with _capture():
                ds_2d.models = Models([sky_2d,
                                       FoVBackgroundModel(dataset_name=ds_2d.name)])
            ds_2d.fake(random_state=SEED + i_th * 300 + i_off * 30)

            ts_val = run_pwn_halo(ds_2d, phi_2d,
                                   seed_offset=i_th * 200 + i_off * 20)
            TS_PWN_2D[i_th, i_off]  = ts_val
            SIG_PWN_2D[i_th, i_off] = np.sqrt(max(ts_val, 0.0))
            print(f"  θ={theta_val:.2f}°  off={off_frac:.1f}×θ → "
                  f"TS={ts_val:.1f} ({SIG_PWN_2D[i_th, i_off]:.1f}σ)")

    _ex = dict(np.load(CACHE_FILE, allow_pickle=True)) if CACHE_FILE.exists() else {}
    _ex[_2d_pwn_key + "_ts"]  = TS_PWN_2D
    _ex[_2d_pwn_key + "_sig"] = SIG_PWN_2D
    np.savez(CACHE_FILE, **_ex)
    print("\n✓ PWN+halo 2D scan saved")

print(f"PWN+halo 2D grid: {TS_PWN_2D.shape}")


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, SIG, title, cmap in zip(
        axes,
        [SIG_GRID_PROPER, SIG_PWN_2D],
        ["Halo-only", "PWN + Halo"],
        ["YlOrRd", "YlGn"]):
    im = ax.pcolormesh(OFFSET_GRID_2D, THETA_GRID_2D, SIG,
                       cmap=cmap, shading="auto", vmin=0)
    fig.colorbar(im, ax=ax).set_label("Detection significance [σ]", fontsize=10)
    try:
        cs = ax.contour(OFFSET_GRID_2D, THETA_GRID_2D, SIG,
                        levels=[5], colors="white", linewidths=2, linestyles="--")
        ax.clabel(cs, fmt="5σ", fontsize=10, colors="white")
    except Exception:
        pass
    ax.set_xlabel("Halo offset  [× θ_d0]", fontsize=12)
    ax.set_ylabel("Halo size θ_d0 [°]", fontsize=12)
    ax.set_title(f"Detectability — {title}", fontsize=11, fontweight="bold")
    ax.scatter([0], [HALO_THETA_D0.to_value(u.deg)], marker="*", s=350,
               color="cyan", edgecolors="black", linewidths=1.3, zorder=10,
               label="HESS J1837-069 (assumed)")
    ax.legend(fontsize=9, loc="upper right")

plt.suptitle(
    f"2D Detectability: Halo-only vs PWN+Halo  (CTA)\n"
    f"HESS J1837-069  |  D₀={D0_cm2_s:.0e} cm²/s  |  "
    f"d={DIST_KPC} kpc  |  t_age={T_AGE_YR/1e3:.0f} kyr",
    fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_2d_halo_vs_pwn.png", dpi=150, bbox_inches="tight")
plt.show()


res_det_H0['sqrt_ts'].write(OUTPUT_DIR / 'ct2_sig_map_halo_only.fits.gz', overwrite=True)
print('✓ CT2 significance map saved')


from gammapy.estimators import FluxPointsEstimator


KM_TO_CM = 1e5
KPC_CM   = 3.0857e21
YR_S     = 3.1557e7

v_kick_list = [0, 100, 200, 500]   # km/s -- cases to visualise
T_AGE_PM_KYR = 20.0                # kyr -- illustrative, not the benchmark's 23 kyr
DIST_PM_KPC  = 0.25                # kpc -- illustrative, not the benchmark's 6.6 kpc
MOTION_PA_PM_DEG = 0.0             # deg, East of North -- fixed direction across panels

fig, axes = plt.subplots(1, len(v_kick_list), figsize=(18, 4.5))
_pm_phi_2d_list = []
_pm_off_deg_list = []

for ax, v_kick in zip(axes, v_kick_list):
    disp_cm = v_kick * KM_TO_CM * (T_AGE_PM_KYR * 1e3) * YR_S
    off_deg = np.degrees(disp_cm / (DIST_PM_KPC * KPC_CM))

    phi_map_pm, _ = build_physical_halo_template(
        geom, SOURCE_CENTER.icrs,
        age_kyr=T_AGE_PM_KYR, distance_kpc=DIST_PM_KPC, tau0_kyr=HALO_TAU0_KYR,
        injection_index=HALO_INJECTION_INDEX, cutoff_TeV=HALO_CUTOFF_TEV,
        D100_cm2_s=HALO_D100_CALIBRATED_CM2_S, diffusion_index=DELTA_DIFF,
        u_total_eV_cm3=HALO_U_TOTAL_EV_CM3,
        spectral_index=HALO_ALPHA, spectral_e0_TeV=HALO_E0.to_value(u.TeV),
        v_transverse_kms=v_kick, motion_angle_deg=MOTION_PA_PM_DEG,
        axis_ratio=HALO_AXIS_RATIO, position_angle_deg=HALO_POSITION_ANGLE,   # real HESS-2006 ellipse, always
        offset_deg=0.0, offset_pa_deg=None,
        n_time_slices=HALO_N_TIME_SLICES,
    )

    phi_2d = phi_map_pm.data
    half = phi_map_pm.geom.width[0].to_value(u.deg) / 2
    _pm_phi_2d_list.append(phi_2d)
    _pm_off_deg_list.append(off_deg)

    ax.imshow(phi_2d, origin="lower", cmap="inferno", extent=[-half, half, -half, half])
    ax.scatter(0, 0, marker="x", s=250, c="cyan", linewidths=2.5, zorder=5,
               label="Halo centre (birth pos)")
    pa_rad = np.radians(MOTION_PA_PM_DEG)
    pwn_x = off_deg * np.sin(pa_rad)
    pwn_y = off_deg * np.cos(pa_rad)
    ax.scatter(pwn_x, pwn_y, marker="*", s=300, c="yellow", zorder=5,
               label=f"PWN now (Delta={off_deg:.2f} deg)")
    if v_kick > 0:
        ax.annotate("", xy=(pwn_x, pwn_y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="white", lw=2))
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_title(f"v = {v_kick} km/s\nDelta_theta = {off_deg:.2f} deg", fontsize=10, fontweight="bold")
    ax.set_xlabel("Delta lon [deg]")
    if ax is axes[0]:
        ax.set_ylabel("Delta lat [deg]")
    ax.legend(fontsize=7, loc="upper right")

plt.suptitle(
    f"Proper Motion -- real physical template (CTA-South geometry)\n"
    f"t_age={T_AGE_PM_KYR:.0f} kyr, d={DIST_PM_KPC} kpc, "
    f"D100={HALO_D100_CALIBRATED_CM2_S:.2e} cm2/s",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_proper_motion_physical_template.png", dpi=150, bbox_inches="tight")
plt.show()
np.savez(
    OUTPUT_DIR / "cta_proper_motion_physical_template_results.npz",
    v_kick_list=np.array(v_kick_list),
    off_deg_list=np.array(_pm_off_deg_list),
    phi_2d_stack=np.array(_pm_phi_2d_list),
    half_width_deg=half,
    t_age_kyr=T_AGE_PM_KYR, distance_kpc=DIST_PM_KPC, motion_pa_deg=MOTION_PA_PM_DEG,
)
print("Part A done: cta_proper_motion_physical_template.png / .npz")


energy_axis_true = dataset.exposure.geom.axes["energy_true"]
geom_img_true    = dataset.exposure.geom.to_image()
geom_cube_true   = geom_img_true.to_cube([energy_axis_true])

coords_true = geom_img_true.get_coord()
halo_center_true = SOURCE_CENTER.icrs
if HALO_OFFSET_DEG:
    _pa = HALO_OFFSET_PA_DEG if HALO_OFFSET_PA_DEG is not None else HALO_POSITION_ANGLE
    halo_center_true = halo_center_true.directional_offset_by(_pa * u.deg, HALO_OFFSET_DEG * u.deg)

dlon_true, dlat_true = halo_center_true.spherical_offsets_to(coords_true.skycoord)
xx_true_deg = dlon_true.to_value(u.deg)
yy_true_deg = dlat_true.to_value(u.deg)
pixel_size_true_deg = geom_img_true.pixel_scales.mean().to_value(u.deg)

e_centers_true = energy_axis_true.center.to_value(u.TeV)
motion_angle_rad = np.radians(HALO_MOTION_PA_DEG)

cube_true = np.zeros((len(e_centers_true),) + xx_true_deg.shape)
for ie, Eg in enumerate(e_centers_true):
    cube_true[ie] = halo_spatial_template_physical(
        Eg, xx_true_deg, yy_true_deg,
        age_kyr=T_AGE_YR / 1e3, distance_kpc=DIST_KPC, tau0_kyr=HALO_TAU0_KYR,
        injection_index=HALO_INJECTION_INDEX, cutoff_TeV=HALO_CUTOFF_TEV,
        D100_cm2_s=HALO_D100_CALIBRATED_CM2_S, diffusion_index=DELTA_DIFF,
        u_total_eV_cm3=HALO_U_TOTAL_EV_CM3,
        v_transverse_kms=HALO_V_TRANSVERSE_KMS, motion_angle_rad=motion_angle_rad,
        axis_ratio=HALO_AXIS_RATIO, position_angle_deg=HALO_POSITION_ANGLE,
        n_time_slices=HALO_N_TIME_SLICES, pixel_size_deg=pixel_size_true_deg,
    )

halo_cube_map = Map.from_geom(geom_cube_true)
halo_cube_map.quantity = cube_true * u.Unit("")
halo_cube_map.write(OUTPUT_DIR / "cta_halo_cube_template_v2.fits", overwrite=True)

print(f"{'E [TeV]':>10}  {'RMS size [deg]':>15}")
r2 = xx_true_deg**2 + yy_true_deg**2
for ie, Eg in enumerate(e_centers_true):
    rms = np.sqrt(np.sum(cube_true[ie] * r2))
    print(f"{Eg:10.2f}  {rms:15.4f}")

sky_halo_true_v2 = SkyModel(
    spatial_model=TemplateSpatialModel(halo_cube_map, normalize=True),
    spectral_model=PowerLawSpectralModel(
        amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
    name="halo-true-3d")

dataset_sim_halo_v2 = dataset.copy(name="sim-halo-3d")
dataset_sim_halo_v2.models = Models([sky_halo_true_v2, bkg_sim, diffuse_sim])
dataset_sim_halo_v2.fake(random_state=SEED)
print("Part B done: corrected 3D template built, saved, and re-simulated "
      "(dataset_sim_halo_v2 / halo_cube_map now available)")


ENERGY_EDGES_EXT = np.geomspace(1.0, 300.0, 5) * u.TeV
theta_fit_ext, theta_fit_err_ext, e_band_centers = [], [], []

print(f"{'Band [TeV]':>16}  {'Fitted sigma [deg]':>19}")
for i in range(len(ENERGY_EDGES_EXT) - 1):
    emin, emax = ENERGY_EDGES_EXT[i], ENERGY_EDGES_EXT[i + 1]
    ds_band = dataset_sim_halo_v2.slice_by_energy(emin, emax, name=f"cta_ext_band_{i}")

    sp_band = GaussianSpatialModel(
        lon_0=SOURCE_CENTER.icrs.ra, lat_0=SOURCE_CENTER.icrs.dec,
        sigma=0.1 * u.deg, frame="icrs")
    sp_band.lon_0.frozen = True
    sp_band.lat_0.frozen = True
    halo_band = SkyModel(
        spatial_model=sp_band,
        spectral_model=PowerLawSpectralModel(
            amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
        name=f"halo-ext-band-{i}")
    ds_band.models = Models([halo_band, FoVBackgroundModel(dataset_name=ds_band.name)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds_band])
        res_band = Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds_band])

    sigma_fit = sp_band.sigma.value
    sigma_err = sp_band.sigma.error if res_band.success else np.nan
    e_center = np.sqrt(emin.to_value(u.TeV) * emax.to_value(u.TeV))

    theta_fit_ext.append(sigma_fit)
    theta_fit_err_ext.append(sigma_err)
    e_band_centers.append(e_center)
    print(f"{emin.value:6.1f}-{emax.value:6.1f}   {sigma_fit:8.4f} +/- {sigma_err:.4f}")

theta_true_ext = [
    _physical_template_effective_sigma_deg(
        Eg, HALO_D100_CALIBRATED_CM2_S,
        age_kyr=T_AGE_YR / 1e3, distance_kpc=DIST_KPC, tau0_kyr=HALO_TAU0_KYR,
        injection_index=HALO_INJECTION_INDEX, cutoff_TeV=HALO_CUTOFF_TEV,
        diffusion_index=DELTA_DIFF, u_total_eV_cm3=HALO_U_TOTAL_EV_CM3,
        n_time_slices=HALO_N_TIME_SLICES,
    )
    for Eg in e_band_centers
]

fig, ax = plt.subplots(figsize=(6, 5))
ax.errorbar(e_band_centers, theta_fit_ext, yerr=theta_fit_err_ext, fmt="o", color="crimson",
            label="Fitted (CTA-South, 100h)")
ax.plot(e_band_centers, theta_true_ext, "k--", label="Injected physical template")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Energy [TeV]"); ax.set_ylabel(r"Fitted extension $\sigma$ [deg]")
ax.set_title("Energy-resolved halo extension, CTA-South")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_energy_resolved_extension.png", dpi=150, bbox_inches="tight")
plt.show()

_valid = np.isfinite(theta_fit_ext) & (np.array(theta_fit_ext) > 0)
_slope = np.nan
if _valid.sum() >= 2:
    _slope, _intercept = np.polyfit(
        np.log(np.array(e_band_centers)[_valid]), np.log(np.array(theta_fit_ext)[_valid]), 1)
    print(f"Empirical fitted scaling: sigma ~ E^{_slope:.3f}  "
          f"(expected -delta/2 = {-DELTA_DIFF/2:.3f} for diffusion-limited, "
          f"steeper if cooling-limited)")

np.savez(
    OUTPUT_DIR / "cta_energy_resolved_extension_results.npz",
    e_band_centers=np.array(e_band_centers),
    theta_fit=np.array(theta_fit_ext),
    theta_fit_err=np.array(theta_fit_err_ext),
    theta_true=np.array(theta_true_ext),
    slope=_slope,
)
import csv
with open(OUTPUT_DIR / "cta_energy_resolved_extension_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["E_band_center_TeV", "theta_fit_deg", "theta_fit_err_deg", "theta_true_deg"])
    for row in zip(e_band_centers, theta_fit_ext, theta_fit_err_ext, theta_true_ext):
        writer.writerow(row)
print("Part C done: cta_energy_resolved_extension.png / .npz / .csv")


tmpl_shape_v2 = TemplateSpatialModel(halo_cube_map, normalize=True)
ds_tmpl_fit_v2, sky_tmpl_fit_v2, res_tmpl_fit_v2 = fit_shape(
    dataset_sim_halo_v2, tmpl_shape_v2, "template-v2")

gauss_shape_v2 = GaussianSpatialModel(
    lon_0=SOURCE_CENTER.icrs.ra, lat_0=SOURCE_CENTER.icrs.dec,
    sigma=HALO_THETA_D0, frame="icrs")
gauss_shape_v2.lon_0.frozen = True; gauss_shape_v2.lat_0.frozen = True
ds_gauss_fit_v2, sky_gauss_fit_v2, res_gauss_fit_v2 = fit_shape(
    dataset_sim_halo_v2, gauss_shape_v2, "gaussian-v2")

k_tmpl_v2  = sum(not p.frozen for p in ds_tmpl_fit_v2.models.parameters)
k_gauss_v2 = sum(not p.frozen for p in ds_gauss_fit_v2.models.parameters)
aic_tmpl_v2   = 2 * k_tmpl_v2  + res_tmpl_fit_v2.total_stat
aic_gauss_v2  = 2 * k_gauss_v2 + res_gauss_fit_v2.total_stat
delta_stat_v2 = res_gauss_fit_v2.total_stat - res_tmpl_fit_v2.total_stat
delta_aic_v2  = aic_gauss_v2 - aic_tmpl_v2

print("=" * 74)
print("CORRECTED HALO SHAPE COMPARISON -- physical template vs 2D Gaussian")
print(f"{'Model':<24}  {'Stat (-2lnL)':>13}  {'k':>4}  {'AIC':>10}")
print("-" * 74)
print(f"{'Physical template (v2)':<24}  {res_tmpl_fit_v2.total_stat:>13.2f}  {k_tmpl_v2:>4}  {aic_tmpl_v2:>10.2f}")
print(f"{'2D Gaussian':<24}  {res_gauss_fit_v2.total_stat:>13.2f}  {k_gauss_v2:>4}  {aic_gauss_v2:>10.2f}")
print("=" * 74)
print(f"  Delta_AIC  (Gauss - template) = {delta_aic_v2:.2f}   "
      f"(compare against the thesis's original ΔAIC=48587 for CTA)")

np.savez(
    OUTPUT_DIR / "cta_aic_comparison_v2.npz",
    stat_template=res_tmpl_fit_v2.total_stat, k_template=k_tmpl_v2, aic_template=aic_tmpl_v2,
    stat_gaussian=res_gauss_fit_v2.total_stat, k_gaussian=k_gauss_v2, aic_gaussian=aic_gauss_v2,
    delta_stat=delta_stat_v2, delta_aic=delta_aic_v2, original_delta_aic=48587,
)
with open(OUTPUT_DIR / "cta_aic_comparison_v2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "stat", "k", "AIC"])
    writer.writerow(["physical_template_v2", res_tmpl_fit_v2.total_stat, k_tmpl_v2, aic_tmpl_v2])
    writer.writerow(["gaussian", res_gauss_fit_v2.total_stat, k_gauss_v2, aic_gauss_v2])
    writer.writerow(["delta_aic_gauss_minus_template", delta_aic_v2, "", ""])
    writer.writerow(["original_delta_aic_for_comparison", 48587, "", ""])
print("Part D done: cta_aic_comparison_v2.npz / .csv")


ds_H1_v2 = dataset_sim.copy(name="H1-v2")
if ds_H1_v2.mask_fit is None:
    ds_H1_v2.mask_fit = ds_H1_v2.mask_safe.copy()

pwn_H1_v2 = pwn_H1_src.copy(name="pwn-H1")
pwn_H1_v2.spatial_model.freeze()

halo_H1_v2 = SkyModel(
    spatial_model=TemplateSpatialModel(halo_cube_map, normalize=True),
    spectral_model=PowerLawSpectralModel(
        amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0),
    name="halo-H1")

bkg_H1_v2 = FoVBackgroundModel(dataset_name=ds_H1_v2.name)
ds_H1_v2.models = Models([pwn_H1_v2, halo_H1_v2, bkg_H1_v2])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds_H1_v2])
    Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds_H1_v2])

for model in ds_H1_v2.models:
    if isinstance(model, FoVBackgroundModel):
        model.spectral_model.norm.frozen = True

energy_edges_fp_v2 = np.geomspace(E_MIN.to_value(u.TeV), E_MAX.to_value(u.TeV), 14) * u.TeV

fpe_pwn_v2 = FluxPointsEstimator(energy_edges=energy_edges_fp_v2,
                                  source="pwn-H1", selection_optional=["ul"])
fp_pwn_v2 = fpe_pwn_v2.run([ds_H1_v2])

fpe_halo_v2 = FluxPointsEstimator(energy_edges=energy_edges_fp_v2,
                                   source="halo-H1", selection_optional=["ul"])
fp_halo_v2 = fpe_halo_v2.run([ds_H1_v2])

fp_pwn_v2.write(OUTPUT_DIR / "cta_fp_pwn_1_v2.fits", overwrite=True)
fp_halo_v2.write(OUTPUT_DIR / "cta_fp_halo_1_v2.fits", overwrite=True)
print(f"Part E done: saved {OUTPUT_DIR / 'cta_fp_pwn_1_v2.fits'} and "
      f"{OUTPUT_DIR / 'cta_fp_halo_1_v2.fits'}")
print("Next step: point the joint Fermi+HESS+CTA ECPL notebook at these "
      "_v2 files instead of cta_fp_pwn_1.fits / cta_fp_halo_1.fits and "
      "re-run the joint fit to see whether E_cutoff shifts away from 4.1 TeV.")

fig, ax = plt.subplots(figsize=(8, 6))
fp_halo.plot(ax=ax, sed_type="e2dnde", color="gray", marker="s", markersize=4,
             elinewidth=0.8, capsize=2, label="Halo flux points (original, flat template)")
fp_halo_v2.plot(ax=ax, sed_type="e2dnde", color="green", marker="o", markersize=5,
                elinewidth=0.9, capsize=2.5, label="Halo flux points (corrected, energy-dependent template)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.legend()
ax.set_title("CTA halo flux points: original vs corrected template")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cta_flux_points_original_vs_corrected.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: cta_flux_points_original_vs_corrected.png")


disk_shape_v2 = DiskSpatialModel(
    lon_0=SOURCE_CENTER.icrs.ra, lat_0=SOURCE_CENTER.icrs.dec,
    r_0=HALO_THETA_D0, frame="icrs")
disk_shape_v2.lon_0.frozen = True; disk_shape_v2.lat_0.frozen = True
ds_disk_fit_v2, sky_disk_fit_v2, res_disk_fit_v2 = fit_shape(
    dataset_sim_halo_v2, disk_shape_v2, "disk-v2")

k_disk_v2  = sum(not p.frozen for p in ds_disk_fit_v2.models.parameters)
aic_disk_v2 = 2 * k_disk_v2 + res_disk_fit_v2.total_stat
delta_aic_disk_gauss_v2 = aic_disk_v2 - aic_gauss_v2

print("=" * 74)
print("CORRECTED HALO SHAPE COMPARISON -- full three-way (Gaussian / Disk / Template)")
print(f"{'Model':<24}  {'Stat (-2lnL)':>13}  {'k':>4}  {'AIC':>10}")
print("-" * 74)
print(f"{'Physical template (v2)':<24}  {res_tmpl_fit_v2.total_stat:>13.2f}  {k_tmpl_v2:>4}  {aic_tmpl_v2:>10.2f}")
print(f"{'2D Gaussian':<24}  {res_gauss_fit_v2.total_stat:>13.2f}  {k_gauss_v2:>4}  {aic_gauss_v2:>10.2f}")
print(f"{'Disk':<24}  {res_disk_fit_v2.total_stat:>13.2f}  {k_disk_v2:>4}  {aic_disk_v2:>10.2f}")
print("=" * 74)
print(f"  Delta_AIC (disk - Gaussian)     = {delta_aic_disk_gauss_v2:.2f}   "
      f"(compare against the thesis's original ΔAIC=4989 for CTA)")
print(f"  Delta_AIC (Gauss - template)    = {delta_aic_v2:.2f}   "
      f"(compare against the thesis's original ΔAIC=48587 for CTA)")

np.savez(
    OUTPUT_DIR / "cta_aic_comparison_threeway_v2.npz",
    stat_template=res_tmpl_fit_v2.total_stat, k_template=k_tmpl_v2, aic_template=aic_tmpl_v2,
    stat_gaussian=res_gauss_fit_v2.total_stat, k_gaussian=k_gauss_v2, aic_gaussian=aic_gauss_v2,
    stat_disk=res_disk_fit_v2.total_stat, k_disk=k_disk_v2, aic_disk=aic_disk_v2,
    delta_aic_disk_gauss=delta_aic_disk_gauss_v2, delta_aic_gauss_template=delta_aic_v2,
    original_delta_aic_disk_gauss=4989, original_delta_aic_gauss_template=48587,
)
print("Part F done: cta_aic_comparison_threeway_v2.npz")


def fit_halo_spectrum_v2(dataset_sim_halo, halo_spectral_model, tag):
    ds = dataset_sim_halo.copy(name=f"halo_spec_{tag}")
    halo_test = SkyModel(
        spatial_model=TemplateSpatialModel(halo_cube_map, normalize=True),
        spectral_model=halo_spectral_model,
        name=f"halo-{tag}")
    ds.models = Models([halo_test, FoVBackgroundModel(dataset_name=ds.name)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds])
        res = Fit(optimize_opts={"print_level": 0, "tol": 6, "strategy": 2}).run([ds])
    return ds, halo_test, res

halo_pl_v2 = PowerLawSpectralModel(amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0)
ds_PL_v2, sky_PL_v2, res_PL_v2 = fit_halo_spectrum_v2(dataset_sim_halo_v2, halo_pl_v2, "PL-v2")

halo_ecpl_v2 = ExpCutoffPowerLawSpectralModel(
    amplitude=HALO_AMPLITUDE, index=HALO_ALPHA, reference=HALO_E0,
    lambda_=1 / (200.0 * u.TeV))
ds_ECPL_v2, sky_ECPL_v2, res_ECPL_v2 = fit_halo_spectrum_v2(dataset_sim_halo_v2, halo_ecpl_v2, "ECPL-v2")

k_PL_v2   = sum(not p.frozen for p in ds_PL_v2.models.parameters)
k_ECPL_v2 = sum(not p.frozen for p in ds_ECPL_v2.models.parameters)
ts_cutoff_v2 = res_PL_v2.total_stat - res_ECPL_v2.total_stat
delta_aic_pl_ecpl_v2 = (2 * k_PL_v2 + res_PL_v2.total_stat) - (2 * k_ECPL_v2 + res_ECPL_v2.total_stat)
sigma_cutoff_v2 = np.sqrt(max(ts_cutoff_v2, 0.0))

print("=" * 74)
print("CORRECTED HALO SPECTRAL SHAPE TEST -- PL vs ECPL")
print(f"  C_stat(PL)   = {res_PL_v2.total_stat:.2f}  (k={k_PL_v2})")
print(f"  C_stat(ECPL) = {res_ECPL_v2.total_stat:.2f}  (k={k_ECPL_v2})")
print(f"  TS_cutoff = {ts_cutoff_v2:.2f} ({sigma_cutoff_v2:.2f} sigma)")
print(f"  Delta_AIC(PL-ECPL) = {delta_aic_pl_ecpl_v2:.2f}   "
      f"(compare against the thesis's original TS_cutoff=0.59, dAIC=-1.41 for CTA)")
print("=" * 74)

np.savez(
    OUTPUT_DIR / "cta_halo_pl_vs_ecpl_v2.npz",
    stat_PL=res_PL_v2.total_stat, k_PL=k_PL_v2,
    stat_ECPL=res_ECPL_v2.total_stat, k_ECPL=k_ECPL_v2,
    ts_cutoff=ts_cutoff_v2, sigma_cutoff=sigma_cutoff_v2,
    delta_aic_pl_ecpl=delta_aic_pl_ecpl_v2,
    original_ts_cutoff=0.59, original_delta_aic=-1.41,
)
print("Part G done: cta_halo_pl_vs_ecpl_v2.npz")
