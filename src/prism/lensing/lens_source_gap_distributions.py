"""Empirically-grounded lens-source physical-property GAP distributions.

Replaces ad hoc heuristics (flat uniform magnitude gaps, arbitrary Sersic-n
ranges) with parametric distributions fit to real strong-lens survey
statistics, so that generating N mock lenses draws physically realistic
lens/source property combinations instead of independently-guessed values.

Sources (see docstrings per function for exact numbers/derivation):
  - SLACS V  (Bolton et al. 2008, arXiv:0805.1931)      -- z_lens, z_source,
    sigma_v, axis ratio, R_eff: derived directly from the paper's Table 4
    (n=117 lenses).
  - SLACS XI (Newton et al. 2011, arXiv:1104.2608)       -- source Sersic
    index (n=46), source size, size-luminosity/size-mass relations,
    source stellar mass extremes.
  - BELLS I  (Brownstein et al. 2012, arXiv:1112.3683)   -- deeper-survey
    z_lens, z_source, sigma_v, R_eff, i-band lens magnitude (n=45).
  - SL2S III/IV (Sonnenfeld et al. 2013, arXiv:1307.4764, 1307.4759) --
    lens stellar mass (median 10^11.53, scatter 0.3 dex), deeper z_lens/
    z_source range, de Vaucouleurs (n=4) lens light.
  - Collett (2015, arXiv:1507.01034)                      -- physically
    self-consistent SIE mock-catalog recipe (Choi+2007 velocity dispersion
    function, axis-ratio-sigma correlation, source size-redshift scaling);
    used where no direct measurement exists (flagged per-function).

Every distribution below documents its literature support level in its
docstring: "measured" (derived from a real per-object table), "quoted"
(reported directly by a paper but not independently re-derived here), or
"heuristic (Collett recipe)" / "heuristic (extrapolated)" for the weaker
cases -- see the research report this module implements for full detail.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Real local Euclid Q1 modeling posteriors (335 systems, this project's own
# data/euclid_q1_psf/modeling_*.csv -- lens+source MGE-model photometry,
# lens Sersic fits, lens mass models). This is BETTER GROUNDED than the
# literature parametrics above for two reasons: (1) it's paired per-object
# lens/source data from the same joint model fit, not a combination of two
# separate papers' marginal statistics, and (2) it's Euclid-instrument-
# specific (VIS band), matching this project's own Euclid rendering path
# directly. Use `sample_from_q1_empirical` to bootstrap-resample real
# measured (mag_gap, lens_sersic_n, theta_E, ...) tuples instead of drawing
# from independent parametric marginals -- this preserves real correlations
# between these quantities that independent draws would destroy.
# ---------------------------------------------------------------------------

_Q1_GAPS_CSV = Path(__file__).resolve().parent.parent.parent.parent / "data" / "euclid_q1_psf" / "derived_lens_source_gaps.csv"
_q1_gaps_cache: "pd.DataFrame | None" = None


def load_q1_empirical_gaps() -> pd.DataFrame:
    """Load the 322-system real Euclid Q1 lens/source magnitude-gap table
    (derived from data/euclid_q1_psf/modeling_mge_magnitude.csv +
    modeling_lens_sersic.csv + modeling_lens_mass.csv, joined on id_str).

    Columns: lens_mag_vis, source_mag_vis_demag, mag_gap (source-lens,
    demagnified), lens_sersic_n, lens_reff_arcsec, theta_E_arcsec,
    magnification.

    Measured, n=322 (Euclid Q1 modeling sample). VIS-band mag_gap:
    mean 1.18, median 1.29, std 1.98, with a real negative tail (~15-25% of
    systems have the demagnified source AS BRIGHT OR BRIGHTER than the
    lens even in a careful joint model fit) -- i.e. a lens-dominant image
    is the *typical* case, not a universal one; genuine strong lenses do
    sometimes show a comparably-bright source. lens_sersic_n: mean 3.84,
    median 4.38, std 1.24 (broader than the literature de-Vaucouleurs-only
    assumption). theta_E: mean 0.97", median 0.88", std 0.43".
    """
    global _q1_gaps_cache
    if _q1_gaps_cache is None:
        _q1_gaps_cache = pd.read_csv(_Q1_GAPS_CSV)
    return _q1_gaps_cache


def sample_from_q1_empirical(rng, n: int, min_mag_gap: float | None = None) -> pd.DataFrame:
    """Bootstrap-resample n rows from the real Q1 (mag_gap, sersic_n,
    theta_E, reff, magnification) joint table, preserving real
    correlations between these quantities (e.g. compact high-theta_E
    systems tend to co-occur with particular gap/magnification values) --
    unlike drawing each quantity from an independent parametric marginal.

    min_mag_gap: if set, resample only from rows with mag_gap >= this value
    (e.g. 0.5) to bias toward the lens-dominant regime for a showcase/
    display use case, while still using only REAL measured combinations
    (not synthetic minimum-enforcement via np.maximum, which can decouple
    the total-magnitude gap from the source's real compactness/size that
    co-determines peak surface brightness).
    """
    df = load_q1_empirical_gaps()
    if min_mag_gap is not None:
        df = df[df["mag_gap"] >= min_mag_gap]
    idx = rng.integers(0, len(df), size=n)
    return df.iloc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Redshift gap
# ---------------------------------------------------------------------------

def sample_redshift_gap(rng, n: int, depth: str = "jwst") -> np.ndarray:
    """Delta_z = z_source - z_lens, Gamma-distributed (bounded at 0, right-skewed).

    depth="shallow" -> SLACS-like (SDSS+ACS depth): measured, mean 0.370,
        median 0.340, std 0.176, n=117 (Bolton+2008 Table 4, derived).
    depth="medium"  -> BELLS-like (BOSS spectroscopy depth): measured,
        mean 0.666, median 0.655, std 0.151, n=45 (Brownstein+2012 Table 2,
        derived).
    depth="jwst"/"deep" -> heuristic (extrapolated): no JWST-depth survey
        gap statistic exists yet; SL2S (deeper CFHTLS imaging) shows Delta_z
        often 1-2.5, so we extrapolate toward that regime. NOT a measured
        JWST statistic -- treat as a placeholder pending real JWST/Euclid
        deep-lens spectroscopic samples.
    """
    params = {
        "shallow": (4.4, 0.084),   # Gamma(shape, scale): mean=shape*scale=0.370, std=sqrt(shape)*scale=0.176
        "medium": (19.6, 0.034),   # mean=0.666, std=0.151
        "jwst": (5.5, 0.22),       # heuristic: mean~1.2, std~0.51 (SL2S-like extrapolation)
        "deep": (5.5, 0.22),
    }
    shape, scale = params.get(depth, params["jwst"])
    return rng.gamma(shape, scale, size=n)


# ---------------------------------------------------------------------------
# Sersic index: lens (early-type, ~de Vaucouleurs) vs source (star-forming)
# ---------------------------------------------------------------------------

def sample_lens_sersic_n(rng, n: int) -> np.ndarray:
    """n_lens ~ N(3.84, 1.24) truncated [0.8, 5.0].

    Measured: real Euclid Q1 modeling posteriors (this project's own
    data/euclid_q1_psf/modeling_lens_sersic.csv, n=322, free Sersic-index
    fit, not fixed de Vaucouleurs): mean 3.84, median 4.38, std 1.24,
    min 0.80. This supersedes the earlier literature-only N(4.0, 0.3)
    estimate (SLACS/SL2S fit FIXED de Vaucouleurs n=4, so their "n=4"
    reflects a modeling choice, not a measured scatter) -- the real
    Euclid Q1 free-fit shows substantially broader scatter, including a
    meaningful population of lower-n (less concentrated / later-type)
    lens galaxies down to n~0.8.
    """
    vals = rng.normal(3.84, 1.24, size=n)
    return np.clip(vals, 0.8, 5.0)


def sample_source_sersic_n(rng, n: int) -> np.ndarray:
    """n_source ~ Gamma(shape=4, scale=0.3) -- median ~1.1.

    Measured: SLACS XI (Newton et al. 2011) directly measured Sersic
    indices for 46 lensed source galaxies: "median n~1 ... The median for
    both SLACS and HUDF is 1.1". This is the best-supported statistic in
    the whole gap-distribution set -- a direct measurement of *lensed
    sources specifically*, not a field-galaxy proxy. Collett (2015)'s
    independent mock-recipe assumption (source light = n=1 exponential)
    matches this almost exactly, cross-validating the value.
    """
    return rng.gamma(4.0, 0.3, size=n)


def sersic_n_gap(n_lens: np.ndarray, n_source: np.ndarray) -> np.ndarray:
    """n_lens - n_source; literature-typical value ~2.9 (4.0 - 1.1)."""
    return n_lens - n_source


# ---------------------------------------------------------------------------
# Velocity dispersion (drives theta_E via the SIE relation -- see
# theta_E_from_sigma below; do NOT sample theta_E independently)
# ---------------------------------------------------------------------------

def sample_velocity_dispersion(rng, n: int, method: str = "gaussian") -> np.ndarray:
    """sigma_v [km/s] for the lens (SIE) deflector.

    method="gaussian": measured -- N(220, 45) truncated [120, 400], fit to
        combined SLACS (mean 230.4, std 43.8, n=117, derived from Bolton+
        2008 Table 4) and BELLS (mean 209.6, std 52.9, n=44, derived from
        Brownstein+2012 Table 2).
    method="choi_vdf": heuristic (Collett 2015 recipe) -- physically
        motivated Choi et al. 2007 Schechter-type velocity dispersion
        function (phi*=8.0e-3 h^3 Mpc^-3, sigma*=161 km/s, alpha=2.32,
        beta=2.67), sampled via rejection sampling on the differential
        number density d(phi)/d(sigma) rather than a simple Gaussian --
        gives the correct high-sigma tail for population-synthesis use.
    """
    if method == "gaussian":
        vals = rng.normal(220.0, 45.0, size=n)
        return np.clip(vals, 120.0, 400.0)

    # Choi+2007 modified-Schechter VDF, rejection sampling.
    sigma_star, alpha, beta = 161.0, 2.32, 2.67
    out = np.empty(n)
    lo, hi = 50.0, 500.0
    # envelope: the VDF peaks near sigma_star; use a broad uniform proposal
    # with a generous max-density estimate for rejection.
    x_grid = np.linspace(lo, hi, 2000)
    phi_grid = (x_grid / sigma_star) ** alpha * np.exp(-((x_grid / sigma_star) ** beta))
    m = phi_grid.max()
    filled = 0
    while filled < n:
        batch = rng.uniform(lo, hi, size=max(64, (n - filled) * 4))
        phi = (batch / sigma_star) ** alpha * np.exp(-((batch / sigma_star) ** beta))
        u = rng.uniform(0, m, size=batch.size)
        accept = batch[phi > u]
        take = min(len(accept), n - filled)
        out[filled:filled + take] = accept[:take]
        filled += take
    return out


def theta_E_from_sigma(sigma_kms: np.ndarray, d_l_mpc, d_s_mpc, d_ls_mpc) -> np.ndarray:
    """SIE Einstein radius [arcsec] from velocity dispersion (exact relation).

    theta_E = 4*pi*(sigma/c)^2 * D_LS/D_S  [radians] -> arcsec.

    This is the standard relation used by SLACS/SL2S/Collett(2015) --
    theta_E should be DERIVED this way rather than sampled as an
    independent nuisance parameter, since it's an exact consequence of the
    SIE lens model given sigma_v and the lens/source angular diameter
    distances.
    """
    c_kms = 299792.458
    theta_rad = 4.0 * np.pi * (np.asarray(sigma_kms) / c_kms) ** 2 * (d_ls_mpc / d_s_mpc)
    return theta_rad * 206265.0


# ---------------------------------------------------------------------------
# Magnitude gap (weakest-supported -- see module docstring)
# ---------------------------------------------------------------------------

def sample_magnitude_gap(rng, n: int, band_depth: str = "jwst") -> np.ndarray:
    """m_source - m_lens (apparent, unlensed/unmagnified source mag).

    band_depth="slacs": quoted/combined (NOT a single joint statistic) --
        Gaussian(mean=7.7, std=2.5), derived by combining SLACS V lens
        I814 photometry (mean 16.6, Bolton+2008 Table 4) with SLACS XI
        source I814 photometry (mean 24.3, "22 < F814W < 26" typical
        range, Newton et al. 2011) -- i.e. two separate papers/samples,
        not one paper's joint measurement. Treat with caution.
    band_depth="jwst"/other: heuristic (extrapolated) -- JWST-depth
        surveys detect much fainter/higher-z sources routinely, so the
        gap is expected to be smaller (source less suppressed relative to
        lens) than the SLACS/SDSS-depth number above. No direct JWST
        measurement exists; using Gaussian(mean=3.5, std=1.5) as a
        physically-motivated placeholder, tighter than SLACS's 7.7 mag
        (which reflects a shallow, SDSS-selected massive-ETG lens against
        marginally-detected emission-line sources -- not representative of
        deep JWST/Euclid imaging where sources are far more secure
        detections).
    """
    if band_depth == "slacs":
        mean, std = 7.7, 2.5
    else:
        mean, std = 3.5, 1.5
    vals = rng.normal(mean, std, size=n)
    return np.clip(vals, 0.3, 12.0)


# ---------------------------------------------------------------------------
# Stellar mass gap
# ---------------------------------------------------------------------------

def sample_lens_log_stellar_mass(rng, n: int) -> np.ndarray:
    """log10(M*_lens/Msun) ~ N(11.3, 0.3) truncated [10.5, 11.8].

    Measured/quoted: SL2S III median (grade A) 10^11.53, scatter 0.3 dex
    (Sonnenfeld+2013, arXiv:1307.4764); SLACS IX/X range 10^10.5-11.8
    (Auger et al. 2009/2010).
    """
    vals = rng.normal(11.3, 0.3, size=n)
    return np.clip(vals, 10.5, 11.8)


def sample_source_log_stellar_mass(rng, n: int, depth: str = "jwst") -> np.ndarray:
    """log10(M*_source/Msun).

    depth="slacs": heuristic tail-risk warning -- SLACS XI found some
        SLACS sources with M* below 10^8 (three extreme cases quoted:
        10^7.03, 10^6.99, 10^6.86 Msun), but these are magnification-
        boosted detections of a small-number tail (3 of 46), NOT
        representative of a typical source population. If you must match
        SLACS depth, use N(7.5, 1.0) truncated [6.5, 9.5] but treat the
        low end as an outlier regime, not the bulk.
    depth="jwst"/other: heuristic (extrapolated) -- JWST-depth surveys
        detect ordinary star-forming sources at similar z, expected to be
        more typically 10^9-10^10.5 Msun rather than the ultra-compact
        dwarfs SLACS's shallower selection preferentially detects.
        N(9.7, 0.6) truncated [8.5, 10.8] used as a physically-motivated
        placeholder.
    """
    if depth == "slacs":
        vals = rng.normal(7.5, 1.0, size=n)
        return np.clip(vals, 6.5, 9.5)
    vals = rng.normal(9.7, 0.6, size=n)
    return np.clip(vals, 8.5, 10.8)


def stellar_mass_gap_dex(log_m_lens: np.ndarray, log_m_source: np.ndarray) -> np.ndarray:
    return log_m_lens - log_m_source


# ---------------------------------------------------------------------------
# Axis ratio (q = b/a)
# ---------------------------------------------------------------------------

def sample_lens_axis_ratio(rng, n: int) -> np.ndarray:
    """q_lens ~ N(0.71, 0.15) truncated [0.3, 0.98].

    Measured: derived directly from SLACS V (Bolton+2008) Table 4 B/A
    column, n=117: mean 0.714, median 0.750, std 0.152.

    Note: Collett (2015)'s mock-catalog recipe instead uses a
    sigma_v-correlated Rayleigh distribution (q = A + B*sigma_v scale,
    A=0.38, B=5.7e-4) which gives a broader/lower-q population than this
    SLACS-measured Gaussian -- the two are not fully consistent; this
    function uses the direct measurement, not Collett's forward-model
    assumption.
    """
    vals = rng.normal(0.71, 0.15, size=n)
    return np.clip(vals, 0.3, 0.98)


def sample_source_axis_ratio(rng, n: int) -> np.ndarray:
    """q_source ~ Rayleigh(scale=0.3) truncated q > 0.2, reflected to <=1.

    Heuristic (Collett 2015 recipe only) -- no direct measured source
    axis-ratio distribution from SLACS/BELLS/SL2S source-plane
    reconstructions was found in the literature search; this is Collett's
    forward-model assumption (sources more elongated/irregular than
    lenses, consistent with clumpy/star-forming morphology), not an
    independently confirmed measurement.
    """
    vals = rng.rayleigh(0.3, size=n)
    q = 1.0 - np.clip(vals, 0.0, 0.8)
    return np.clip(q, 0.2, 1.0)


# ---------------------------------------------------------------------------
# Effective radius (size) ratio
# ---------------------------------------------------------------------------

def sample_lens_reff_arcsec(rng, n: int, depth: str = "shallow") -> np.ndarray:
    """R_eff,lens [arcsec].

    depth="shallow": measured -- SLACS-derived (Bolton+2008 Table 4,
        n=117): mean 2.23", median 1.95", std 1.56".
    depth="medium": measured -- BELLS-derived (Brownstein+2012 Table 2,
        n=45): mean 1.43", median 1.24", std 0.97" (BOSS de Vaucouleurs
        Reff, uncorrected for aperture).
    """
    if depth == "medium":
        vals = rng.normal(1.43, 0.97, size=n)
    else:
        vals = rng.normal(2.23, 1.56, size=n)
    return np.clip(vals, 0.2, 8.0)


def source_reff_from_mass_kpc(log_m_source: np.ndarray) -> np.ndarray:
    """R_eff,source [kpc] from the SLACS XI size-mass relation (quoted directly):

    log10(R_eff/kpc) = 0.24 * log10(M*/Msun) - 2.20

    Measured/quoted: Newton et al. 2011 (SLACS XI) explicit fitted
    relation for the lensed-source population. Median SLACS source size
    directly quoted: 0.14" (~0.8 kpc).
    """
    log_r = 0.24 * np.asarray(log_m_source) - 2.20
    return 10.0 ** log_r


# ---------------------------------------------------------------------------
# Convenience: draw a full self-consistent lens-source property set
# ---------------------------------------------------------------------------

def sample_lens_source_population(rng, n: int, depth: str = "jwst") -> dict:
    """Draw N physically-consistent lens-source property sets from the
    empirical gap distributions above.

    Returns a dict of arrays (length n): z_lens is NOT drawn here (still
    supplied by the caller's own redshift/mass config) -- this focuses on
    the GAP quantities the user asked for: delta_z, sersic n (lens &
    source & gap), sigma_v, magnitude gap, stellar mass gap, axis ratios,
    and R_eff for both lens and source.

    `depth` selects which literature calibration to use for the
    depth-sensitive gaps (redshift gap, magnitude gap, stellar mass):
    "shallow" (SLACS/SDSS-depth), "medium" (BELLS/BOSS-depth), or
    "jwst"/"deep" (heuristic extrapolation -- see per-function docstrings).
    """
    n_lens = sample_lens_sersic_n(rng, n)
    n_source = sample_source_sersic_n(rng, n)
    log_m_lens = sample_lens_log_stellar_mass(rng, n)
    log_m_source = sample_source_log_stellar_mass(rng, n, depth=depth)
    return {
        "delta_z": sample_redshift_gap(rng, n, depth=depth),
        "n_lens": n_lens,
        "n_source": n_source,
        "n_gap": sersic_n_gap(n_lens, n_source),
        "sigma_v_kms": sample_velocity_dispersion(rng, n, method="gaussian"),
        "mag_gap": sample_magnitude_gap(rng, n, band_depth="slacs" if depth == "shallow" else "jwst"),
        "log_m_lens": log_m_lens,
        "log_m_source": log_m_source,
        "mass_gap_dex": stellar_mass_gap_dex(log_m_lens, log_m_source),
        "q_lens": sample_lens_axis_ratio(rng, n),
        "q_source": sample_source_axis_ratio(rng, n),
        "reff_lens_arcsec": sample_lens_reff_arcsec(rng, n, depth="medium" if depth != "shallow" else "shallow"),
        "reff_source_kpc": source_reff_from_mass_kpc(log_m_source),
    }
