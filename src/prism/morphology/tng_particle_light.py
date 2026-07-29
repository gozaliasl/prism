"""TNG particle-driven physical morphology.

Procedurally constructs a galaxy's per-band surface-brightness map from its
TNG star/gas particle cutout (a locally-downloaded
``TNG_100_snap_{snap}_subhalo_{id}.h5``, see
``src.tng_galaxy_selector.local_particle_path``), then exposes it as a
drop-in lenstronomy ``INTERPOL`` light-profile builder --
``build_tng_particle_interpol_kwargs`` -- with the same kwargs shape as
``src.galaxygenius_stamps.build_*_interpol_kwargs`` (``image`` in
Jy/arcsec^2-equivalent surface-brightness units, ``center_x``, ``center_y``,
``phi_G``, ``scale``, ``magnitude``).

Unlike the GalaxyGenius/SKIRT stamps (a fixed pre-rendered cutout, pasted in
and rescaled), the image here is generated fresh for each call from the
particle distribution at the requested angular size/orientation -- i.e. this
is a forward model from physical particle data, not image compositing.

Pipeline, per subhalo (cached so all 4 bands reuse one pass):
1. Load star (PartType4) and gas (PartType0) particles.
2. Recenter on the stellar center of mass; rescale raw (comoving, h^-1)
   coordinates to physical kpc by anchoring the particles' own measured
   stellar half-mass radius to ``halfmassrad_stars_kpc`` from the TNG
   catalog (avoids needing exact a/h unit bookkeeping).
3. Apply one random 3D rotation (inclination + position angle) and project
   to the sky plane.
4. Bin star particles (mass-weighted by an approximate per-band SSP
   luminosity, from particle age + metallicity) and gas particles (mass) to
   a 2D grid sized to ``~6 * halfmassrad_stars_kpc``.
5. Apply a Calzetti-like wavelength-dependent dust attenuation derived from
   the projected gas surface density and metallicity.

This is an approximate stellar-population model, not a full SPS/SED code.
``_stellar_band_luminosity`` (Phase 6) interpolates a 15-age x 4-metallicity
L/M grid per JWST band -- a denser, metallicity-dependent successor to the
original 5-age-bin table (see ``src/galaxy_morphology/README.md``); a true
FSPS/BPASS-based replacement remains possible behind the same interface but
needs the separate SPS_HOME isochrone/spectral-library data
(github.com/cconroy20/fsps), not bundled with the pip package.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM
from scipy.ndimage import gaussian_filter

# Same cosmology used elsewhere (galaxygenius_stamps.py, jwst_lens_simulator.py).
_COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

# Grid size for the procedurally-binned image (pixels per side).
# Higher than the original 150 so adaptive SPH smoothing has enough samples
# before INTERPOL stretch to Euclid/JWST angular sizes.
NPIX = 256

# Full grid width = this * halfmassrad_stars_kpc.
# Kept large enough for extended stellar envelopes, but star particles used
# for the *light* map are hard-clipped at LIGHT_RADIUS_HALFMASS_FACTOR below
# so bound satellites / FoF companions do not appear as discrete clumps
# inside the primary galaxy stamp.
EXTENT_HALFMASS_FACTOR = 12.0

# Only deposit stellar light from particles within this multiple of R_half.
# Satellites beyond ~5–6 R_half should be separate field/group members, not
# bead-like clumps in the primary light profile.
LIGHT_RADIUS_HALFMASS_FACTOR = 5.5

# --- Approximate stellar-population luminosity-per-mass table -------------
# Phase 6 upgrade: a denser (15 age bins x 4 metallicity bins) literature-
# informed L/M grid per JWST band, replacing the original 5-age-bin,
# metallicity-agnostic (Z^-0.2) table. `python-fsps` was evaluated for a true
# SPS lookup but requires a separate SPS_HOME data download (isochrone/
# spectral libraries, github.com/cconroy20/fsps, not bundled with the pip
# package) -- out of scope for this offline build, so this remains a
# parametric approximation, just at finer resolution and with a per-band
# metallicity dependence (qualitatively: metal-rich old populations are
# redder -- suppressed blue-band, enhanced red-band L/M -- via BC03/Padova
# SSP trends).
#
# `AGE_GRID_GYR` interpolates the original 5-bin-center anchor values
# (log-age) onto 15 points; `METAL_GRID_ZREL` (in units of Z/Zsun) modulates
# each band's L/M via `(Z/Zsun) ** (alpha(band) * age_factor(age))`, with
# `alpha` more negative for bluer bands (high-Z dims them) and `age_factor`
# ramping from 0 (young, ~Z-independent) to 1 (old, full BC03-like Z trend).
_ANCHOR_AGE_CENTERS_GYR = np.array([0.005, 0.05, 0.5, 2.0, 8.5])

# Calibration anchors (F115W/F150W/F277W/F444W) plus F070W/F090W/F200W/F356W,
# obtained by linear extrapolation/interpolation of log10(L/M) vs.
# log10(wavelength) across the 4 calibration bands for each age bin (and
# similarly for `_BAND_METAL_ALPHA` below) -- same parametric-approximation
# spirit as the Phase 6 note above, just extended to JWST's other NIRCam
# wide/medium filters.
    # Phase: F277W/F444W anchors for the two oldest age bins (2.0, 8.5 Gyr)
    # were recalibrated against the real COSMOS-Web lens sample (see
    # analysis/sim_obs_comparison/qc_reports/per_band_v7). The original
    # table made old (quiescent) populations *redder with increasing
    # wavelength* (F444W/F115W L/M ratio = 4.67 at 8.5 Gyr), which inverted
    # the empirical peak-SB shape of real lenses (F444W/F115W peak ratio
    # ~0.29, i.e. flux *declines* from F115W to F444W for this
    # quiescent-dominated, z~0.5 sample). The two oldest bins for F150W/
    # F277W/F444W are rescaled so the F277W/F115W and F444W/F115W ratios
    # decline smoothly with age (1.33->0.67->0.50 and 1.5->0.42->0.29)
    # instead of rising, matching that empirical color. Young/intermediate
    # bins (0.005-0.5 Gyr), which dominate star-forming source galaxies,
    # are left unchanged.
_ANCHOR_BAND_LUMINOSITY = {
    "F070W": np.array([758.64, 60.878, 2.659, 0.450, 0.088]),
    "F090W": np.array([614.29, 55.104, 2.827, 0.520, 0.115]),
    "F115W": np.array([500.0, 50.0, 3.0, 0.6, 0.15]),
    "F150W": np.array([400.0, 45.0, 3.2, 0.6, 0.144]),
    "F200W": np.array([252.509, 37.207, 3.553, 0.901, 0.293]),
    "F277W": np.array([150.0, 30.0, 4.0, 0.4, 0.075]),
    "F356W": np.array([107.376, 24.181, 4.259, 1.398, 0.569]),
    "F444W": np.array([80.0, 20.0, 4.5, 0.25, 0.0435]),
    # Euclid VIS (broad optical, ~550-900nm) + NISP Y/J/H, same
    # extrapolation/interpolation scheme as the JWST bands above.
    "EUCLID_VIS": np.array([745.251, 60.369, 2.673, 0.455, 0.090]),
    "EUCLID_Y": np.array([525.852, 51.204, 2.957, 0.579, 0.141]),
    "EUCLID_J": np.array([431.378, 46.634, 3.131, 0.664, 0.181]),
    "EUCLID_H": np.array([305.879, 40.276, 3.401, 0.811, 0.250]),
}

AGE_GRID_GYR = np.array([
    0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0, 2.0,
    3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 14.0,
])
METAL_GRID_ZREL = np.array([0.2, 0.5, 1.0, 2.5])  # Z / Z_sun

# Per-band metallicity exponent (more negative -> high-Z dims this band).
_BAND_METAL_ALPHA = {
    "F070W": -0.730, "F090W": -0.588, "F115W": -0.45, "F150W": -0.30,
    "F200W": -0.183, "F277W": -0.05, "F356W": 0.056, "F444W": 0.15,
    "EUCLID_VIS": -0.718, "EUCLID_Y": -0.484, "EUCLID_J": -0.351, "EUCLID_H": -0.232,
}


def _age_metal_factor(age_gyr: np.ndarray) -> np.ndarray:
    """0 for young (<=0.1 Gyr, ~Z-independent) ramping to 1 for old
    (>=10 Gyr, full per-band metallicity trend)."""
    return np.clip((age_gyr - 0.1) / (10.0 - 0.1), 0.0, 1.0)


def _build_band_age_metal_grid() -> dict[str, np.ndarray]:
    """Precompute the (n_age, n_metal) L/M grid for each band by
    interpolating the anchor table onto `AGE_GRID_GYR` (log-age) and
    applying the per-band metallicity modulation."""
    log_anchor_age = np.log10(_ANCHOR_AGE_CENTERS_GYR)
    log_age_grid = np.log10(AGE_GRID_GYR)
    age_factor = _age_metal_factor(AGE_GRID_GYR)

    grids = {}
    for band, anchor_lum in _ANCHOR_BAND_LUMINOSITY.items():
        log_lum_anchor = np.log10(anchor_lum)
        base_lum = 10.0 ** np.interp(log_age_grid, log_anchor_age, log_lum_anchor,
                                      left=log_lum_anchor[0], right=log_lum_anchor[-1])
        alpha = _BAND_METAL_ALPHA[band]
        # grid[i, j] = base_lum[i] * (Z_rel[j]) ** (alpha * age_factor[i])
        z_pow = METAL_GRID_ZREL[None, :] ** (alpha * age_factor[:, None])
        grids[band] = base_lum[:, None] * z_pow
    return grids


BAND_AGE_METAL_LUMINOSITY = _build_band_age_metal_grid()

# Approximate observed-frame central wavelengths (nm) used only for the
# relative dust-attenuation power law between bands.
BAND_WAVELENGTH_NM = {
    "F070W": 700.0, "F090W": 900.0, "F115W": 1150.0, "F150W": 1500.0,
    "F200W": 2000.0, "F277W": 2770.0, "F356W": 3560.0, "F444W": 4440.0,
    "EUCLID_VIS": 715.0, "EUCLID_Y": 1083.0, "EUCLID_J": 1371.0, "EUCLID_H": 1774.0,
}
_DUST_REF_WAVELENGTH_NM = 550.0  # V-band reference
_DUST_POWER = 1.3
_DUST_K = 0.5  # tunable: dust-attenuation strength per unit normalized gas surface density

_SOLAR_METALLICITY = 0.0127

_PARTICLE_CACHE: dict[str, dict] = {}
_PROJECTION_CACHE: dict[str, dict] = {}
_CACHE_MAX = 256  # evict oldest entries beyond this to cap RAM usage


def _cache_set(cache: dict, key: str, value) -> None:
    if len(cache) >= _CACHE_MAX:
        oldest = next(iter(cache))
        del cache[oldest]
    cache[key] = value


def _load_particles(path: Path) -> dict:
    key = str(path)
    if key in _PARTICLE_CACHE:
        return _PARTICLE_CACHE[key]

    with h5py.File(path, "r") as f:
        header = dict(f["Header"].attrs)
        star = f["PartType4"]
        star_coords = np.asarray(star["Coordinates"], dtype=np.float64)
        star_init_mass = np.asarray(star["GFM_InitialMass"], dtype=np.float64)
        star_sft = np.asarray(star["GFM_StellarFormationTime"], dtype=np.float64)
        star_metal = np.asarray(star["GFM_Metallicity"], dtype=np.float64)

        gas_coords = gas_mass = gas_metal = None
        if "PartType0" in f:
            gas = f["PartType0"]
            gas_coords = np.asarray(gas["Coordinates"], dtype=np.float64)
            gas_mass = np.asarray(gas["Masses"], dtype=np.float64)
            gas_metal = np.asarray(gas["GFM_Metallicity"], dtype=np.float64)

    # GFM_StellarFormationTime <= 0 marks wind particles, not real stars.
    real = star_sft > 0
    star_coords = star_coords[real]
    star_init_mass = star_init_mass[real]
    star_sft = star_sft[real]
    star_metal = star_metal[real]

    snap_redshift = float(header.get("Redshift", 0.0))

    data = dict(
        header=header,
        snap_redshift=snap_redshift,
        star_coords=star_coords,
        star_init_mass=star_init_mass,
        star_sft=star_sft,
        star_metal=star_metal,
        gas_coords=gas_coords,
        gas_mass=gas_mass,
        gas_metal=gas_metal,
    )
    _cache_set(_PARTICLE_CACHE, key, data)
    return data


def _stellar_ages_gyr(star_sft: np.ndarray, snap_redshift: float) -> np.ndarray:
    """Convert GFM_StellarFormationTime (scale factor at formation) to ages
    (Gyr) at the snapshot's redshift."""
    a_form = np.clip(star_sft, 1e-3, 1.0)
    z_form = 1.0 / a_form - 1.0
    age_universe_at_snap = _COSMO.age(snap_redshift).value
    age_universe_at_form = _COSMO.age(z_form).value
    ages = age_universe_at_snap - age_universe_at_form
    return np.clip(ages, 0.0, None)


def _stellar_band_luminosity(ages_gyr: np.ndarray, metallicity: np.ndarray, band: str) -> np.ndarray:
    """Per-particle relative luminosity-per-unit-mass for ``band``, via
    bilinear interpolation (in log-age, log-metallicity) over the
    `BAND_AGE_METAL_LUMINOSITY` grid (see Phase 6 note above)."""
    grid = BAND_AGE_METAL_LUMINOSITY[band]

    log_age = np.log10(np.clip(ages_gyr, AGE_GRID_GYR[0], AGE_GRID_GYR[-1]))
    log_age_grid = np.log10(AGE_GRID_GYR)
    i = np.clip(np.searchsorted(log_age_grid, log_age) - 1, 0, len(log_age_grid) - 2)
    age_frac = (log_age - log_age_grid[i]) / (log_age_grid[i + 1] - log_age_grid[i])

    z_rel = np.clip(metallicity, 1e-5, None) / _SOLAR_METALLICITY
    log_z = np.log10(np.clip(z_rel, METAL_GRID_ZREL[0], METAL_GRID_ZREL[-1]))
    log_z_grid = np.log10(METAL_GRID_ZREL)
    j = np.clip(np.searchsorted(log_z_grid, log_z) - 1, 0, len(log_z_grid) - 2)
    z_frac = (log_z - log_z_grid[j]) / (log_z_grid[j + 1] - log_z_grid[j])

    g00 = grid[i, j]
    g01 = grid[i, j + 1]
    g10 = grid[i + 1, j]
    g11 = grid[i + 1, j + 1]
    lum_per_mass = (
        g00 * (1 - age_frac) * (1 - z_frac)
        + g01 * (1 - age_frac) * z_frac
        + g10 * age_frac * (1 - z_frac)
        + g11 * age_frac * z_frac
    )
    return lum_per_mass


def _project_particles(data: dict, halfmassrad_stars_kpc: float, rng) -> dict:
    """Recenter, rescale to physical kpc, apply one random 3D rotation, and
    project star/gas particles to the sky plane. Cached per particle file."""
    star_coords = data["star_coords"]
    star_mass = data["star_init_mass"]

    com = np.average(star_coords, axis=0, weights=star_mass)
    rel = star_coords - com

    r = np.linalg.norm(rel, axis=1)
    order = np.argsort(r)
    cum_mass = np.cumsum(star_mass[order])
    half_idx = np.searchsorted(cum_mass, 0.5 * cum_mass[-1])
    r_half_raw = max(r[order][half_idx], 1e-6)

    kpc_per_raw = float(halfmassrad_stars_kpc) / r_half_raw
    rel_kpc = rel * kpc_per_raw

    # Random viewing rotation. Quiescent/old populations are rounder than
    # thin disks -- bias inclination toward face-on so projected light is a
    # smooth elliptical rather than an edge-on bead chain.
    median_age = float(np.median(_stellar_ages_gyr(data["star_sft"], data["snap_redshift"])))
    if median_age > 2.0:
        incl = float(rng.beta(2.0, 5.0) * (np.pi / 2.0))  # prefer face-on
    else:
        incl = float(rng.uniform(0.0, np.pi / 2.0))
    pa = float(rng.uniform(0.0, 2.0 * np.pi))
    cosi, sini = np.cos(incl), np.sin(incl)
    cospa, sinpa = np.cos(pa), np.sin(pa)

    def rotate(xyz):
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        y2 = y * cosi - z * sini
        x2 = x
        x3 = x2 * cospa - y2 * sinpa
        y3 = x2 * sinpa + y2 * cospa
        return x3, y3

    star_x, star_y = rotate(rel_kpc)

    gas_x = gas_y = gas_mass = gas_metal = None
    if data["gas_coords"] is not None and len(data["gas_coords"]) > 0:
        gas_rel_kpc = (data["gas_coords"] - com) * kpc_per_raw
        gas_x, gas_y = rotate(gas_rel_kpc)
        gas_mass = data["gas_mass"]
        gas_metal = data["gas_metal"]

    return dict(
        star_x=star_x, star_y=star_y,
        star_mass=star_mass,
        star_ages_gyr=_stellar_ages_gyr(data["star_sft"], data["snap_redshift"]),
        star_metal=data["star_metal"],
        gas_x=gas_x, gas_y=gas_y, gas_mass=gas_mass, gas_metal=gas_metal,
        extent_kpc=EXTENT_HALFMASS_FACTOR * float(halfmassrad_stars_kpc),
        inclination_rad=float(incl),
        position_angle_rad=float(pa),
    )


def _band_images(particle_file: Path, halfmassrad_stars_kpc: float, rng,
                  use_cache: bool = True) -> dict[str, np.ndarray]:
    """Per-band (NPIX, NPIX) relative-luminosity images, dust-attenuated.

    Cached per particle file by default (one projection/binning pass for all
    bands, reused across calls). Pass ``use_cache=False`` to recompute a
    fresh random viewing projection (inclination + position angle) each
    call -- used for multi-projection dataset augmentation."""
    key = str(particle_file)
    if use_cache and key in _PROJECTION_CACHE:
        return _PROJECTION_CACHE[key]["band_images"]

    data = _load_particles(particle_file)
    if use_cache:
        # Deterministic projection (inclination/PA) seeded from the cutout's
        # own filename, not the shared pipeline RNG stream -- so a given
        # subhalo's rendered image is stable across pipeline runs/code
        # changes that alter how many RNG draws happen upstream (otherwise
        # every code change reshuffles which lens gets which projection,
        # making before/after comparisons impossible).
        seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 32)
        proj_rng = np.random.default_rng(seed)
    else:
        proj_rng = rng
    proj = _project_particles(data, halfmassrad_stars_kpc, proj_rng)

    extent = proj["extent_kpc"]
    edges = np.linspace(-extent / 2.0, extent / 2.0, NPIX + 1)

    # Gas surface-density map (for dust), normalized to its own median
    # nonzero pixel so the dust strength is independent of absolute units.
    tau_v_map = np.zeros((NPIX, NPIX))
    if proj["gas_x"] is not None:
        gas_map, _, _ = np.histogram2d(proj["gas_y"], proj["gas_x"], bins=[edges, edges],
                                        weights=proj["gas_mass"])
        gas_z_map, _, _ = np.histogram2d(proj["gas_y"], proj["gas_x"], bins=[edges, edges],
                                          weights=proj["gas_mass"] * proj["gas_metal"])
        with np.errstate(divide="ignore", invalid="ignore"):
            gas_z_avg = np.where(gas_map > 0, gas_z_map / np.maximum(gas_map, 1e-30), 0.0)
        nonzero = gas_map[gas_map > 0]
        norm = np.median(nonzero) if len(nonzero) else 1.0
        gas_map_norm = gas_map / max(norm, 1e-30)
        tau_v_map = _DUST_K * gas_map_norm * np.clip(gas_z_avg / _SOLAR_METALLICITY, 0.0, 5.0)
        tau_v_map = gaussian_filter(tau_v_map, sigma=1.0)

    # Smooth radial edge taper so the binned grid fades out before its hard
    # boundary, instead of being cut off abruptly. A separable per-axis
    # (square) taper leaves the diffuse halo at full brightness inside an
    # 80%-width square and then cuts it over a narrow band -- visible as a
    # sharp square edge around the galaxy. A radially-symmetric cosine taper
    # avoids the square shape, but a narrow transition band (e.g. 65%-100%)
    # still cuts non-negligible halo flux over too few pixels, leaving a
    # visible circular edge. Use a wide transition (flat out to 45% of the
    # half-grid-width, cosine falloff to zero at 100%, i.e. the inscribed
    # circle) so the falloff is gradual relative to the halo's own decay.
    axis = np.arange(NPIX)
    center = (NPIX - 1) / 2.0
    yy, xx = np.meshgrid(axis - center, axis - center, indexing="ij")
    r_norm = np.sqrt(xx ** 2 + yy ** 2) / center  # 0 at center, 1 at inscribed-circle edge
    taper = np.where(
        r_norm <= 0.45, 1.0,
        0.5 * (1.0 + np.cos(np.pi * np.clip((r_norm - 0.45) / 0.55, 0.0, 1.0)))
    )

    # Clip to the primary light aperture so FoF satellites do not render as
    # discrete "bead" clumps inside the galaxy stamp.
    r_star = np.hypot(proj["star_x"], proj["star_y"])
    light_r_max = LIGHT_RADIUS_HALFMASS_FACTOR * float(halfmassrad_stars_kpc)
    star_mask = r_star <= light_r_max
    star_x = proj["star_x"][star_mask]
    star_y = proj["star_y"][star_mask]
    star_mass = proj["star_mass"][star_mask]
    star_ages = proj["star_ages_gyr"][star_mask]
    star_metal = proj["star_metal"][star_mask]

    # Density-adaptive SPH-like smoothing: kernel width tracks local mean
    # inter-particle separation so sparse outskirts never show as dots.
    count_map, _, _ = np.histogram2d(star_y, star_x, bins=[edges, edges])
    density = gaussian_filter(count_map.astype(np.float64), sigma=1.5)
    peak = float(density.max())
    halo_frac = 1.0 - np.clip(density / max(peak * 0.08, 1e-30), 0.0, 1.0)

    n_particles = max(int(star_mask.sum()), 1)
    # Mean projected separation in grid pixels over the light aperture.
    light_area_frac = min(1.0, (light_r_max / max(extent * 0.5, 1e-6)) ** 2)
    n_eff = max(n_particles * light_area_frac, 1.0)
    mean_sep_pix = float(NPIX / np.sqrt(max(n_eff, 1.0)))
    # Floor raised vs older 0.8 so TNG100 / low-N cutouts blend into a
    # continuous elliptical/spiral profile at Euclid 0.1"/pix stretch.
    sigma_core = float(np.clip(1.25 * mean_sep_pix, 1.8, 6.0))
    sigma_halo = float(np.clip(sigma_core * 2.8, sigma_core + 2.0, 12.0))

    band_images = {}
    for band, wavelength_nm in BAND_WAVELENGTH_NM.items():
        lum = _stellar_band_luminosity(star_ages, star_metal, band)
        weights = star_mass * lum
        img_raw, _, _ = np.histogram2d(star_y, star_x, bins=[edges, edges], weights=weights)
        img_core = gaussian_filter(img_raw, sigma=sigma_core)
        img_halo = gaussian_filter(img_raw, sigma=sigma_halo)
        img = img_core * (1.0 - halo_frac) + img_halo * halo_frac
        # Extra mild isotropic smooth removes residual grain after adaptive mix.
        img = gaussian_filter(img, sigma=max(0.8, 0.35 * sigma_core))
        img = img * taper

        attenuation = np.exp(-tau_v_map * (_DUST_REF_WAVELENGTH_NM / wavelength_nm) ** _DUST_POWER)
        img = img * attenuation
        band_images[band] = np.clip(img, 0.0, None)

    if use_cache:
        _cache_set(_PROJECTION_CACHE, key, dict(
            band_images=band_images, extent_kpc=extent,
            inclination_rad=proj["inclination_rad"],
            position_angle_rad=proj["position_angle_rad"],
        ))
    return band_images


def get_projection_orientation(particle_file: Path, halfmassrad_stars_kpc: float, rng=None) -> tuple[float, float]:
    """Return ``(axis_ratio, position_angle_rad)`` for the (cached) sky
    projection of ``particle_file``, as used by `_band_images`.

    Approximates the projected axis ratio as ``cos(inclination)`` (thin-disk
    flattening), clipped to ``[0.3, 1.0]`` since real galaxies -- including
    ellipticals -- are not infinitely thin. Intended for aligning a lens
    mass model's ellipticity/PA with its TNG-particle-rendered light when
    both come from the same subhalo (mass-light orientation consistency)."""
    _band_images(Path(particle_file), halfmassrad_stars_kpc, rng if rng is not None else np.random.default_rng())
    cached = _PROJECTION_CACHE[str(particle_file)]
    axis_ratio = float(np.clip(np.cos(cached["inclination_rad"]), 0.3, 1.0))
    return axis_ratio, float(cached["position_angle_rad"])


def build_tng_particle_interpol_kwargs(
    band: str,
    particle_file: Path,
    halfmassrad_stars_kpc: float,
    magnitude_ref: float,
    ref_band: str = "F150W",
    center_x: float = 0.0,
    center_y: float = 0.0,
    phi_G: float = 0.0,
    target_size_arcsec: float = 1.2,
    rng=None,
    smooth_sigma: float = 0.0,
) -> dict:
    """Build lenstronomy INTERPOL kwargs (with 'magnitude') for ``band``,
    from a TNG particle cutout.

    Same kwargs shape/normalization convention as
    ``galaxygenius_stamps.build_source_interpol_kwargs`` /
    ``build_lens_light_interpol_kwargs``: the image's own per-band relative
    flux sets the color (anchored to ``magnitude_ref`` in ``ref_band``), and
    ``target_size_arcsec`` sets the full angular width of the procedurally
    binned grid (``NPIX`` pixels).

    ``rng`` is only consumed the first time a given ``particle_file`` is
    projected (the resulting projection/binning is cached and reused for all
    bands and subsequent calls in this process).
    """
    if rng is None:
        rng = np.random.default_rng()

    band_images = _band_images(Path(particle_file), halfmassrad_stars_kpc, rng)
    image_per_pixel = band_images[band]
    npix = image_per_pixel.shape[-1]
    # Enforce minimum physical stamp size (≥ PSF FWHM at F277W ≈ 0.09") so the
    # stamp is always resolved; sub-PSF stamps produce PSF diffraction spike artifacts.
    MIN_SIZE_ARCSEC = 0.18
    target_size_arcsec = max(float(target_size_arcsec), MIN_SIZE_ARCSEC)
    scale = float(target_size_arcsec) / npix

    image_sb = image_per_pixel / scale ** 2

    # Post-bin smooth in INTERPOL pixel units. Callers (lens/source/field)
    # should pass role-specific sigma; if unset, apply a conservative floor
    # so sparse cutouts never leave visible stellar "beads".
    n_star = int(np.sum(image_per_pixel > 0))
    auto_floor = float(np.clip(2.5 * np.sqrt(max(NPIX * NPIX / max(n_star, 1), 1.0)) / 8.0, 1.2, 4.0))
    applied_sigma = float(smooth_sigma) if smooth_sigma and smooth_sigma > 0.0 else auto_floor
    applied_sigma = max(applied_sigma, auto_floor * 0.75)
    image_sb = gaussian_filter(image_sb.astype(np.float64), sigma=applied_sigma).astype(np.float32)

    flux_band = float(np.sum(image_per_pixel))
    flux_ref = float(np.sum(band_images[ref_band]))
    color_offset = -2.5 * np.log10(max(flux_band, 1e-30) / max(flux_ref, 1e-30))
    magnitude = magnitude_ref + color_offset

    return dict(
        image=image_sb,
        center_x=float(center_x),
        center_y=float(center_y),
        phi_G=float(phi_G),
        scale=float(scale),
        magnitude=float(magnitude),
    )
