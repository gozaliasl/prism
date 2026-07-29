"""GalaxyGenius/SKIRT mock-galaxy stamps as lenstronomy INTERPOL light profiles.

Stamps are produced offline by running SKIRT9 radiative transfer through
GalaxyGenius on TNG/EAGLE particle data (see
``/Volumes/extHD/galaxygenius_build/``), with instrument noise and PSF
convolution disabled and ``imageUnit='flux'`` so the saved FITS contains
clean, noiseless per-band surface-brightness data in Jy/pixel.

Each stamp set is a ``galaxy_images.fits`` with one ImageHDU per JWST band
(F115W, F150W, F277W, F444W), each holding a stack of views (different
inclinations of the same subhalo). ``AVAILABLE_STAMP_SETS`` lists the TNG
subhalos rendered so far (a late-type/disky galaxy and a massive
early-type/elliptical); ``random_stamp_set`` picks one for morphology
diversity each time a stamp is used.

Three render modes are wired into ``jwst_lens_simulator.py``, all gated by
``config['galaxygenius_stamps']`` (off by default):

- **Field galaxies** (``enabled``/``fraction``): ``build_field_galaxy_interpol_kwargs``
  renders the stamp in the image plane, rescaled via
  ``angular_size_for_redshift(field_redshift)`` so its angular size reflects
  the field galaxy's own assigned redshift (not the stamp's native z=0.06).
- **Lensed source** (``source_enabled``/``source_fraction``/``source_size_scale_factor``):
  ``build_source_interpol_kwargs`` rescales the stamp via
  ``angular_size_for_redshift(source_redshift) * source_size_scale_factor``
  in the source plane before ray-tracing through the lens equation.
- **Foreground lens light** (``lens_enabled``/``lens_fraction``/``lens_target_size_factor``):
  ``build_lens_light_interpol_kwargs`` rescales the stamp to a multiple of
  the lens galaxy's own ``R_sersic`` (already physically derived for the
  lens's assigned redshift/mass via the Fundamental Plane) and renders it in
  the image plane, centered on the lens.

In all three cases, the stamp's own per-band flux ratios (its GalaxyGenius/
SKIRT SED) set the inter-band color, anchored to a single reference-band
magnitude computed the same way as for the Sersic-based models (see
``color_offset`` in each ``build_*_interpol_kwargs`` function).

Note: only the stamp's *morphology and SED shape* come from the TNG
subhalo at its native redshift (z=0.06, 10.59 dex stellar mass) -- the
*total brightness* (per-band magnitudes) and *angular size* of the rendered
galaxy are set by the simulated galaxy's own assigned redshift/magnitude via
``angular_size_for_redshift`` and the existing photometric model. The TNG
galaxy is not "the same object" as the simulated one; it supplies a
realistic resolved texture/color template, rescaled to be physically
consistent with the simulated galaxy's redshift.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits

STAMP_DIR = Path(__file__).resolve().parents[1] / "data" / "galaxygenius_stamps"

# The full batch-rendered TNG/GalaxyGenius output (see scripts/render_tng_batch.py),
# growing continuously as the background render job progresses. Each
# `{sim_prefix}_{subhalo_id}` directory here is registered as an additional
# stamp set alongside the two curated ones in STAMP_DIR, with native
# redshift/galaxy-type metadata looked up from the TNG local catalogs via the
# render manifest -- see _build_stamp_registry().
EXT_STAMP_DIR = Path("/Volumes/extHD/galaxygenius_build/workspace/mock_JWST")
RENDER_MANIFEST_PATH = Path("/Volumes/extHD/tng_local_catalog/render_manifest.csv")
TNG_LOCAL_CATALOGS = {
    "TNG100-1": Path("/Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet"),
    "TNG50-1": Path("/Volumes/extHD/tng_local_catalog/tng50-1_local_catalog.parquet"),
}
TNG_QUENCHED_SSFR_THRESHOLD = 1e-11  # same convention as jwst_lens_simulator.py

BAND_HDU_INDEX = {"F115W": 1, "F150W": 2, "F277W": 3, "F444W": 4}
BAND_PIXEL_SCALE_ARCSEC = {"F115W": 0.031, "F150W": 0.031, "F277W": 0.063, "F444W": 0.063}

# View index -> inclination (deg), shared by all stamp sets below.
VIEW_INCLINATIONS_DEG = {0: 0, 1: 30, 2: 60, 3: 90}

# The two originally curated stamps: TNG100-1 snapshot 94 (z=0.06) subhalos
# rendered through GalaxyGenius/SKIRT (NoMedium, 18" FOV, same 4
# inclinations). subhalo_31 (logM*~10.6, rhalf~5 kpc) is a late-type/disky,
# star-forming galaxy; subhalo_253881 (logM*~11.6, rhalf~20 kpc) is a massive,
# quiescent early-type/elliptical, framed on its central ~24 kpc core.
NATIVE_FOV_ARCSEC = 18.0  # shared by every stamp rendered via this pipeline

_CURATED_STAMPS = {
    "subhalo_31": {"dir": STAMP_DIR / "subhalo_31", "native_redshift": 0.06, "galaxy_type": "star_forming"},
    "subhalo_253881": {"dir": STAMP_DIR / "subhalo_253881", "native_redshift": 0.06, "galaxy_type": "quiescent"},
}

# A quiescent, smooth, massive elliptical is not a physically plausible
# analog for an object above z~2.5 (the universe is too young for a fully
# assembled, quenched massive galaxy) -- stamp sets classified 'quiescent'
# are excluded above this redshift, leaving only 'star_forming' templates.
HIGH_Z_THRESHOLD = 2.5

_STAMP_REGISTRY: dict[str, dict] | None = None


# Below this stellar mass, SKIRT's fixed photon-packet budget (1e6,
# NoMedium) gives low per-pixel S/N -- empirically, ~50% of (band, view)
# flux sums in the batch-rendered registry came out negative (unphysical)
# for the full mass range, vs. 0% for the two hand-picked massive curated
# stamps. Restricting the registry to massive galaxies avoids relying on
# the _load_stamp_set/_band_total_flux non-negative clipping to paper over
# genuinely noise-dominated renders, instead picking stamps that are
# intrinsically clean.
MIN_STAMP_STELLAR_MASS_LOGMSUN = 10.0

# Minimum radial flux concentration (flux within 0.3*Rmax / flux within
# 0.9*Rmax, both clipped non-negative) a rendered stamp must have in F150W to
# be registered. This is the defense-in-depth quality gate behind the mass
# filter and the render_tng_batch.py viewRedshift fix: a stamp that is pure
# SKIRT shot noise inside the radial edge-taper window (no real galaxy signal
# at all -- a "moon"-looking disc, no resolved structure) has a concentration
# matching the taper window's own shape, measured empirically at ~0.17,
# because a flat noise field inherits exactly the taper's radial envelope.
# Every real galaxy checked (the 2 curated stamps plus several freshly
# re-rendered batch stamps) measured 0.53-0.80 -- a wide margin above the
# noise floor. 0.30 sits safely between the two populations.
MIN_STAMP_CONCENTRATION = 0.30


def _stamp_concentration(image: np.ndarray) -> float:
    """Flux within the inner 30% of the stamp's radius, divided by flux
    within the inner 90% (both non-negative-clipped). Low for a noise-only
    render (matches the edge-taper window's own ~0.17), high (0.5-0.8) for a
    real, centrally-concentrated galaxy light profile."""
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    r = np.hypot(xx - cx, yy - cy)
    rmax = min(ny, nx) / 2.0
    img = np.clip(image, 0.0, None)
    total = img[r < 0.9 * rmax].sum()
    inner = img[r < 0.3 * rmax].sum()
    return float(inner / max(total, 1e-12))


def _stamp_quality_ok(stamp_dir: Path, band: str = "F150W",
                       min_concentration: float = MIN_STAMP_CONCENTRATION) -> bool:
    """True if this stamp's rendered images show real galaxy structure
    rather than being SKIRT shot noise inside the edge-taper window, checked
    across all views (a stamp only needs to pass on its best view, since a
    caller can request any view via random_stamp_set/available_views).

    Applies the same edge-taper window as _load_stamp_set (not the raw FITS
    pixels) since MIN_STAMP_CONCENTRATION was calibrated against tapered
    images, and reads directly from disk (bypassing the _STAMP_CACHE) since
    this runs once per candidate during registry construction, before a
    stamp set name even exists to cache against.
    """
    try:
        with fits.open(stamp_dir / "galaxy_images.fits") as hdul:
            stack = np.asarray(hdul[BAND_HDU_INDEX[band]].data, dtype=np.float64)
    except Exception:
        return False
    window = _edge_taper_window(stack.shape[-2:])
    return any(_stamp_concentration(view * window) >= min_concentration for view in stack)


def _build_stamp_registry(min_stellar_mass_logmsun: float = MIN_STAMP_STELLAR_MASS_LOGMSUN) -> dict[str, dict]:
    """Registry of every available stamp set: the two curated ones plus the
    full batch-rendered TNG/GalaxyGenius output in EXT_STAMP_DIR (see
    scripts/render_tng_batch.py) with stellar_mass_logmsun >=
    ``min_stellar_mass_logmsun``, each with native_redshift/galaxy_type/mass
    metadata looked up from the TNG local catalogs via the render manifest.

    Falls back to just the two curated stamps if the external drive, render
    manifest, or local catalogs aren't available (e.g. running without the
    extHD mounted) -- this keeps the original 2-stamp behavior as a graceful
    degradation rather than an error.
    """
    registry = dict(_CURATED_STAMPS)
    try:
        manifest = pd.read_csv(RENDER_MANIFEST_PATH)
        manifest = manifest[manifest["status"] == "ok"]
        catalogs = {sim: pd.read_parquet(path) for sim, path in TNG_LOCAL_CATALOGS.items() if path.exists()}

        n_skipped_faint = 0
        n_skipped_noisy = 0
        for _, row in manifest.iterrows():
            sim_prefix, sim, snap, subhalo = row["sim_prefix"], row["sim"], int(row["snapshot"]), int(row["subhalo_id"])
            name = f"{sim_prefix}_{subhalo}"
            stamp_dir = EXT_STAMP_DIR / name
            if not (stamp_dir / "galaxy_images.fits").exists():
                continue
            cat = catalogs.get(sim)
            if cat is None:
                continue
            match = cat[(cat["snapshot"] == snap) & (cat["subhalo_id"] == subhalo)]
            if match.empty:
                continue
            m = match.iloc[0]
            stellar_mass = float(m.get("stellar_mass_logmsun", np.nan))
            if not np.isfinite(stellar_mass) or stellar_mass < min_stellar_mass_logmsun:
                n_skipped_faint += 1
                continue
            # Defense-in-depth quality gate: even a massive subhalo can still
            # render as pure shot noise (a "moon"-looking disc with no
            # resolved structure) if it happened to be compact/dusty/far
            # enough that little flux reached the detector within the fixed
            # photon budget. Measured directly on the rendered pixels rather
            # than trusting the catalog's mass proxy.
            if not _stamp_quality_ok(stamp_dir):
                n_skipped_noisy += 1
                continue
            ssfr = float(m.get("ssfr_per_yr", np.nan))
            galaxy_type = "quiescent" if np.isfinite(ssfr) and ssfr < TNG_QUENCHED_SSFR_THRESHOLD else "star_forming"
            registry[name] = {
                "dir": stamp_dir,
                # NOT m["snapshot_redshift"] (the subhalo's true cosmic epoch):
                # render_tng_batch.py renders every batch stamp's mock
                # observation at a fixed nearby viewRedshift=0.06 regardless of
                # the subhalo's actual snapshot, for SKIRT photon-count S/N
                # (see run_inclinations_generic.py) -- this is the angular
                # scale the rendered image actually has, which is what
                # angular_size_for_redshift needs as its native_redshift.
                "native_redshift": 0.06,
                "galaxy_type": galaxy_type,
                "stellar_mass_logmsun": stellar_mass,
            }
        print(f"[INFO] galaxygenius_stamps: excluded {n_skipped_faint} batch-rendered stamps below "
              f"logM*={min_stellar_mass_logmsun}, {n_skipped_noisy} more for failing the "
              f"concentration>={MIN_STAMP_CONCENTRATION} quality gate (noise-dominated SKIRT renders)")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] galaxygenius_stamps: could not build extended registry ({exc}); "
              f"using only the {len(_CURATED_STAMPS)} curated stamps")
    return registry


def _stamp_registry() -> dict[str, dict]:
    global _STAMP_REGISTRY
    if _STAMP_REGISTRY is None:
        _STAMP_REGISTRY = _build_stamp_registry()
        print(f"[INFO] galaxygenius_stamps: {len(_STAMP_REGISTRY)} stamp sets available "
              f"({sum(1 for v in _STAMP_REGISTRY.values() if v['galaxy_type'] == 'star_forming')} star-forming, "
              f"{sum(1 for v in _STAMP_REGISTRY.values() if v['galaxy_type'] == 'quiescent')} quiescent)")
    return _STAMP_REGISTRY


def available_stamp_sets() -> list[str]:
    """All registered stamp set names (curated + batch-rendered, if available).
    Lazily builds the registry on first call (see _stamp_registry)."""
    return list(_stamp_registry().keys())


def random_stamp_set(rng, redshift: float | None = None) -> str:
    """Pick a random registered stamp set, restricted to physically
    plausible templates for ``redshift`` (if given): above
    ``HIGH_Z_THRESHOLD``, 'quiescent' stamp sets are excluded.
    """
    registry = _stamp_registry()
    if redshift is not None and redshift > HIGH_Z_THRESHOLD:
        eligible = [name for name, meta in registry.items() if meta["galaxy_type"] == "star_forming"]
    else:
        eligible = list(registry.keys())
    if not eligible:
        eligible = list(_CURATED_STAMPS.keys())
    return str(rng.choice(eligible))


def native_redshift_for_stamp_set(stamp_set: str) -> float:
    """Native redshift the given stamp set was rendered at (for
    angular_size_for_redshift). Defaults to 0.06 (the curated stamps'
    native redshift) if the stamp set isn't found in the registry.
    """
    return float(_stamp_registry().get(stamp_set, {}).get("native_redshift", 0.06))


# Default native redshift used when no per-stamp value is available
# (the curated stamps' native redshift).
NATIVE_REDSHIFT = 0.06

# Same cosmology used elsewhere in jwst_lens_simulator.py (COSMO).
_COSMO = FlatLambdaCDM(H0=70, Om0=0.3)


def angular_size_for_redshift(
    target_redshift: float,
    fov_native_arcsec: float = NATIVE_FOV_ARCSEC,
    native_redshift: float = NATIVE_REDSHIFT,
) -> float:
    """Angular size (arcsec) of the stamp's fixed physical extent, as seen
    at ``target_redshift``, given its native angular size at
    ``native_redshift``.

    Uses the angular-diameter-distance ratio
    ``D_A(native_redshift) / D_A(target_redshift)``, so this correctly
    reproduces the angular-size turnover at high redshift (a fixed-size
    object appears *larger* again beyond z ~ 1.5 in a FlatLambdaCDM
    cosmology).
    """
    da_native = _COSMO.angular_diameter_distance(native_redshift).value
    da_target = _COSMO.angular_diameter_distance(max(target_redshift, 1e-3)).value
    return float(fov_native_arcsec) * da_native / da_target

_STAMP_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _edge_taper_window(shape: tuple[int, int], frac: float = 0.45) -> np.ndarray:
    """Radially-symmetric raised-cosine window: 1.0 within ``frac`` of the
    half-width from center, cosine falloff to 0.0 at the inscribed circle
    (corners are also 0). Used to remove the SKIRT cutout's faint-but-nonzero
    flat background, which otherwise shows up as a hard edge once an
    INTERPOL stamp is rescaled/brightened.

    A separable per-axis (square) taper leaves the stamp's flux at full
    brightness inside an inner square and then cuts it over a narrow band at
    the square's own edges -- visible as a sharp square/diamond edge once
    rescaled to fill most of the frame (same artifact as the TNG-particle
    renderer's original taper, see ``galaxy_morphology/tng_particle_light.py``).
    A wide radial taper instead follows the stamp's own (roughly radial)
    surface-brightness falloff.
    """
    ny, nx = shape
    yy, xx = np.meshgrid(np.arange(ny) - (ny - 1) / 2.0, np.arange(nx) - (nx - 1) / 2.0, indexing="ij")
    center = min(ny, nx) / 2.0
    r_norm = np.sqrt((xx / center) ** 2 + (yy / center) ** 2)
    return np.where(
        r_norm <= frac, 1.0,
        0.5 * (1.0 + np.cos(np.pi * np.clip((r_norm - frac) / (1.0 - frac), 0.0, 1.0)))
    )


def _load_stamp_set(stamp_set: str = "subhalo_31") -> dict[str, np.ndarray]:
    """Load and cache per-band image stacks (views, ny, nx) in Jy/pixel,
    with a radial edge taper applied to remove the SKIRT background edge,
    and clipped to non-negative.

    SKIRT's NoMedium Monte Carlo radiative transfer with a finite photon
    packet count introduces shot-noise-like fluctuations that go negative
    for faint/low-surface-brightness subhalos -- physically meaningless
    (flux can't be negative). Left unclipped, these negative pixels pass
    straight through into the lenstronomy INTERPOL render and remain in the
    final 'sources_only'/'lens_only' images, where the pipeline's
    output.skip_empty_images quality gate sums raw pixel values and can get
    a net-negative "total flux" even though the genuine positive source
    signal is clearly visible -- this was silently dropping roughly half of
    all systems using a batch-rendered stamp (see _band_total_flux, which
    has the same clipping for the color-magnitude calculation specifically;
    clipping here as well ensures the rendered *image* is consistent with
    the magnitude actually assigned to it, not just the color).
    """
    if stamp_set not in _STAMP_CACHE:
        meta = _stamp_registry().get(stamp_set)
        stamp_dir = meta["dir"] if meta is not None else STAMP_DIR / stamp_set
        path = stamp_dir / "galaxy_images.fits"
        with fits.open(path) as hdul:
            data = {
                band: np.asarray(hdul[idx].data, dtype=np.float64)
                for band, idx in BAND_HDU_INDEX.items()
            }
        for band, stack in data.items():
            window = _edge_taper_window(stack.shape[-2:])
            data[band] = np.clip(stack * window, 0.0, None)
        _STAMP_CACHE[stamp_set] = data
    return _STAMP_CACHE[stamp_set]


def available_views(stamp_set: str = "subhalo_31") -> list[int]:
    """View indices available in a stamp set (e.g. different inclinations)."""
    return sorted(VIEW_INCLINATIONS_DEG)


def _band_total_flux(stamp_set: str, band: str, view_idx: int) -> float:
    """Total flux in this band/view, summing only non-negative pixels.

    SKIRT's NoMedium Monte Carlo radiative transfer with a finite photon
    packet count (1e6) introduces shot-noise-like pixel-to-pixel
    fluctuations that go negative for faint/low-surface-brightness subhalos
    -- physically meaningless (flux can't be negative) but, left unclipped,
    summed to a net-negative "total flux" for ~50% of (stamp, band, view)
    combinations in the batch-rendered registry (verified empirically; the
    two hand-picked curated stamps happened to be bright enough to never hit
    this). That negative sum then fed a log() in the color_offset
    calculation below, producing NaN/garbage magnitudes and, downstream,
    spuriously negative total image flux that tripped the pipeline's
    output.skip_empty_images quality gate for roughly half of all systems
    using a stamp. Clipping to non-negative before summing removes the
    artifact while preserving the genuine positive signal.
    """
    image = _load_stamp_set(stamp_set)[band][view_idx]
    return float(np.sum(np.clip(image, 0.0, None)))


def _color_offset(flux_band: float, flux_ref: float) -> float:
    """-2.5*log10(flux_band/flux_ref), floored to a tiny positive epsilon on
    both inputs so a residual zero-flux band/view (a genuinely undetectable
    stamp in that band, even after the non-negative clipping in
    _band_total_flux) can't produce a NaN/-inf color_offset."""
    eps = 1e-30
    return -2.5 * np.log10(max(flux_band, eps) / max(flux_ref, eps))


def build_field_galaxy_interpol_kwargs(
    band: str,
    view_idx: int,
    magnitude_ref: float,
    ref_band: str = "F150W",
    center_x: float = 0.0,
    center_y: float = 0.0,
    phi_G: float = 0.0,
    target_size_arcsec: float | None = None,
    stamp_set: str = "subhalo_31",
) -> dict:
    """Build lenstronomy INTERPOL kwargs (with 'magnitude') for one band.

    The stamp's own per-band flux ratios (its GalaxyGenius/SKIRT SED) set the
    color relative to ``ref_band``, anchored to the field galaxy's assigned
    ``magnitude_ref`` in that reference band. ``magnitude2amplitude`` then
    renormalizes each band's INTERPOL amplitude independently, so passing the
    derived per-band magnitude (rather than ``magnitude_ref`` for every band)
    is what preserves the stamp's color.

    ``target_size_arcsec``: the full angular width the stamp's
    ``NATIVE_FOV_ARCSEC`` (18") should be rescaled to, accounting for the
    field galaxy's actual redshift differing from the stamp's native
    ``NATIVE_REDSHIFT`` (z=0.06) -- typically
    ``angular_size_for_redshift(field_redshift)``. If ``None``, the stamp's
    native pixel scale is used (i.e. rendered as if at its native redshift,
    only appropriate when the field galaxy's redshift ~ NATIVE_REDSHIFT).
    """
    data = _load_stamp_set(stamp_set)
    image_per_pixel = data[band][view_idx]
    npix = image_per_pixel.shape[-1]

    MIN_SIZE_ARCSEC = 0.18  # prevent sub-PSF stamps → PSF spike artifacts
    if target_size_arcsec is None:
        scale = BAND_PIXEL_SCALE_ARCSEC[band]
    else:
        scale = max(float(target_size_arcsec), MIN_SIZE_ARCSEC) / npix

    # INTERPOL expects surface brightness (flux per arcsec^2), not flux per pixel.
    image_sb = image_per_pixel / scale**2

    flux_band = _band_total_flux(stamp_set, band, view_idx)
    flux_ref = _band_total_flux(stamp_set, ref_band, view_idx)
    color_offset = _color_offset(flux_band, flux_ref)
    magnitude = magnitude_ref + color_offset

    return dict(
        image=image_sb,
        center_x=float(center_x),
        center_y=float(center_y),
        phi_G=float(phi_G),
        scale=float(scale),
        magnitude=float(magnitude),
    )


def build_lens_light_interpol_kwargs(
    band: str,
    view_idx: int,
    magnitude_ref: float,
    ref_band: str = "F150W",
    center_x: float = 0.0,
    center_y: float = 0.0,
    phi_G: float = 0.0,
    target_size_arcsec: float = 2.0,
    stamp_set: str = "subhalo_31",
) -> dict:
    """Build lenstronomy INTERPOL kwargs (with 'magnitude') for the
    foreground lens galaxy's own light.

    Like the field-galaxy stamps, this is rendered directly in the image
    plane (not ray-traced), centered on the lens galaxy. ``target_size_arcsec``
    sets the stamp's full angular width and should be scaled to the lens
    galaxy's own light extent (e.g. a multiple of its Sersic ``R_sersic``) by
    the caller, since the stamp's native pixel scale corresponds to a
    different (TNG) galaxy at a different redshift.
    """
    data = _load_stamp_set(stamp_set)
    image_per_pixel = data[band][view_idx]
    npix = image_per_pixel.shape[-1]
    scale = float(target_size_arcsec) / npix

    image_sb = image_per_pixel / scale**2

    flux_band = _band_total_flux(stamp_set, band, view_idx)
    flux_ref = _band_total_flux(stamp_set, ref_band, view_idx)
    color_offset = _color_offset(flux_band, flux_ref)
    magnitude = magnitude_ref + color_offset

    return dict(
        image=image_sb,
        center_x=float(center_x),
        center_y=float(center_y),
        phi_G=float(phi_G),
        scale=float(scale),
        magnitude=float(magnitude),
    )


def build_source_interpol_kwargs(
    band: str,
    view_idx: int,
    magnitude_ref: float,
    ref_band: str = "F150W",
    center_x: float = 0.0,
    center_y: float = 0.0,
    phi_G: float = 0.0,
    target_size_arcsec: float = 1.2,
    stamp_set: str = "subhalo_31",
) -> dict:
    """Build lenstronomy INTERPOL kwargs (with 'magnitude') for a lensed
    source-plane galaxy.

    Unlike the field-galaxy stamps (rendered directly in the image plane at
    their native pixel scale), source-plane stamps are ray-traced through
    the lens equation, so their *angular size on the sky* is set by the lens
    magnification, not by the stamp's pixel scale. We therefore rescale the
    stamp to span ``target_size_arcsec`` across its full width in the source
    plane (independent of the per-band native pixel count), leaving the
    total flux -- and hence the color-preservation math -- unchanged, since
    ``total_flux = sum(image_sb) * scale**2`` is invariant under this
    rescaling.
    """
    data = _load_stamp_set(stamp_set)
    image_per_pixel = data[band][view_idx]
    npix = image_per_pixel.shape[-1]
    scale = float(target_size_arcsec) / npix

    image_sb = image_per_pixel / scale**2

    flux_band = _band_total_flux(stamp_set, band, view_idx)
    flux_ref = _band_total_flux(stamp_set, ref_band, view_idx)
    color_offset = _color_offset(flux_band, flux_ref)
    magnitude = magnitude_ref + color_offset

    return dict(
        image=image_sb,
        center_x=float(center_x),
        center_y=float(center_y),
        phi_G=float(phi_G),
        scale=float(scale),
        magnitude=float(magnitude),
    )
