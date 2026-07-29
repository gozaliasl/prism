"""
Fundamental Plane and Faber-Jackson relations for early-type lens galaxies.

Calibrations used:
  - Local FP:      Bernardi et al. (2003), SDSS r-band,  astro-ph/0301626
  - z-evolution:   Singh, Yu & Seljak (2021), BOSS LOWZ/CMASS, arXiv:2001.07700
  - Lensing bias:  Sonnenfeld (2023), arXiv:2301.13230
  - High-z:        Horizon Run 5 extrapolation (Yoon+2025, A&A 2025/12)
  - DESI DR1:      Ross et al. (2025), arXiv:2512.03226
  - Mass-sigma:    Cappellari+2013 (ATLAS3D), Zahid+2016
"""

import numpy as np

# ---------------------------------------------------------------------------
# Fundamental Plane coefficients in the surface-brightness form:
#   log Re [kpc] = a · log σ [km/s]  +  b · <μe> [mag/arcsec²]  +  c
#
# Bernardi+2003 / Hyde & Bernardi (2009), SDSS r-band.
# b_μe = −b_Ie / 2.5  where b_Ie=0.75  →  b_μe = −0.300  (Hyde uses −0.315)
# ---------------------------------------------------------------------------
FP_A = 1.49      # log σ coefficient
FP_B = -0.315    # <μe> coefficient  (Hyde & Bernardi 2009, SDSS r-band)
FP_C = 4.20      # zero-point: σ=200, μe=22 → Re≈5 kpc

# Intrinsic scatter of the FP (dex in log Re)
FP_SCATTER_DEX = 0.08   # Bernardi+2003 rms residual

# ---------------------------------------------------------------------------
# Redshift evolution of FP zero-point  (Singh+2021, Eq.7)
# Δc(z) = FP_EVO_ALPHA · z  [galaxies more compact at higher z]
# ---------------------------------------------------------------------------
FP_EVO_ALPHA = -0.30

# ---------------------------------------------------------------------------
# Lensing selection bias  (Sonnenfeld 2023, arXiv:2301.13230)
# Lenses are biased small: δlog Re = -0.10 dex at fixed σ
# ---------------------------------------------------------------------------
LENS_BIAS_LOG_RE = -0.10

# ---------------------------------------------------------------------------
# Faber-Jackson  (Bernardi+2003 Table 5 / Hyde & Bernardi 2009)
# L ∝ σ^4  →  σ ∝ L^{1/4}
# Reference: σ★ = 225 km/s at M_r★ = −22.0 (AB)
# ---------------------------------------------------------------------------
FJ_SIGMA_STAR_KMS = 225.0
FJ_M_STAR_AB      = -22.0
FJ_SLOPE          = 0.25   # σ ∝ L^{0.25}

# ---------------------------------------------------------------------------
# Mass-sigma calibration  (Cappellari+2013 ATLAS3D; Zahid+2016 for high-z)
#   log σ = MASS_SIGMA_SLOPE · (log M★ − 11) + log σ_ref
# ATLAS3D slope 0.32 better reproduces σ~300 km/s for logM~12 ETG lenses
# than the Fall & Romanowsky (2013) slope of 0.27.
# ---------------------------------------------------------------------------
MASS_SIGMA_SLOPE = 0.32
MASS_SIGMA_REF   = 200.0   # km/s at log M★ = 11

# SIS Einstein radius conversion
_C_KMS = 2.998e5   # speed of light in km/s


def faber_jackson_sigma(abs_mag_r: float, rng=None, scatter_dex: float = 0.04) -> float:
    """
    Estimate velocity dispersion from r-band absolute magnitude via Faber-Jackson.

    Parameters
    ----------
    abs_mag_r : float
        Absolute r-band magnitude (AB).  For high-z use K-corrected value.
    rng : np.random.Generator or RandomState, optional
    scatter_dex : float
        Intrinsic scatter in log10(σ).

    Returns
    -------
    sigma_kms : float, clipped to [50, 500] km/s.
    """
    delta_mag = abs_mag_r - FJ_M_STAR_AB   # positive → fainter than L*
    log_sigma = np.log10(FJ_SIGMA_STAR_KMS) - FJ_SLOPE * delta_mag / 2.5
    if rng is not None:
        noise = rng.normal(0.0, scatter_dex) if hasattr(rng, 'normal') else float(np.random.normal(0.0, scatter_dex))
        log_sigma += noise
    return float(np.clip(10.0 ** log_sigma, 50.0, 500.0))


def sigma_from_mass(mass_log10: float, rng=None, scatter_dex: float = 0.05) -> float:
    """
    Estimate velocity dispersion from stellar mass.

    Uses Cappellari+2013 (ATLAS3D) calibration, slope 0.32, which better
    reproduces observed σ for massive ETG lens galaxies (logM ~ 11–12)
    than the shallower Fall & Romanowsky (2013) slope of 0.27.

    Parameters
    ----------
    mass_log10 : float
        log10(M★/M_sun).
    """
    log_sigma = MASS_SIGMA_SLOPE * (mass_log10 - 11.0) + np.log10(MASS_SIGMA_REF)
    if rng is not None:
        noise = rng.normal(0.0, scatter_dex) if hasattr(rng, 'normal') else float(np.random.normal(0.0, scatter_dex))
        log_sigma += noise
    return float(np.clip(10.0 ** log_sigma, 50.0, 500.0))


def _mu_e_from_mag_re(mag_total: float, re_arcsec: float) -> float:
    """
    Mean effective surface brightness from total magnitude and angular Re.

    <μe> = mag_total + 2.5 log10(2π Re_arcsec²)

    Valid for a de Vaucouleurs (n=4) profile; approximation for other n.
    Returns NaN if inputs are non-finite or Re <= 0.
    """
    if not (np.isfinite(mag_total) and np.isfinite(re_arcsec) and re_arcsec > 0):
        return float('nan')
    return float(mag_total + 2.5 * np.log10(2.0 * np.pi * re_arcsec ** 2))


def fundamental_plane_re_kpc(
    sigma_kms: float,
    surface_brightness_r: float,
    z: float,
    apply_lensing_bias: bool = True,
    rng=None,
    add_scatter: bool = True,
) -> float:
    """
    Predict effective radius (kpc) from the Fundamental Plane.

    Uses Bernardi+2003 calibration with Singh+2021 redshift evolution.

    Parameters
    ----------
    sigma_kms : float
        Velocity dispersion [km/s].
    surface_brightness_r : float
        Mean effective surface brightness <μe> in mag/arcsec² (r-band AB).
    z : float
        Lens redshift (for evolution correction).
    apply_lensing_bias : bool
        Apply Sonnenfeld (2023) -0.10 dex bias for lens-selected samples.
    rng : optional
        RNG for FP intrinsic scatter.
    add_scatter : bool
        Whether to add FP intrinsic scatter (disable when iterating μe).

    Returns
    -------
    re_kpc : float, clipped to [0.3, 30] kpc.
    """
    c_z = FP_C + FP_EVO_ALPHA * z
    log_re = FP_A * np.log10(sigma_kms) + FP_B * surface_brightness_r + c_z
    if apply_lensing_bias:
        log_re += LENS_BIAS_LOG_RE
    if add_scatter and rng is not None:
        noise = rng.normal(0.0, FP_SCATTER_DEX) if hasattr(rng, 'normal') else float(np.random.normal(0.0, FP_SCATTER_DEX))
        log_re += noise
    return float(np.clip(10.0 ** log_re, 0.3, 30.0))


def einstein_radius_from_sigma(sigma_kms: float, d_ls_over_ds: float) -> float:
    """
    SIS Einstein radius in arcseconds.

    θ_E = 4π (σ/c)² · (D_ls / D_s)  [radians] × 206265
    """
    theta_e_rad = 4.0 * np.pi * (sigma_kms / _C_KMS) ** 2 * d_ls_over_ds
    return float(np.clip(theta_e_rad * 206265.0, 0.05, 15.0))


def fp_consistent_lens_params(
    row: dict,
    lens_z: float,
    source_z: float,
    theta_E_catalog: float,
    lens_mass_log10: float,
    da_ls_over_ds: float,
    rng=None,
    enforce_fp: bool = True,
    da_l_mpc: float = None,
    theta_E_max: float = 5.0,
    catalog_theta_E_weight: float = 0.60,
) -> dict:
    """
    Return a self-consistent set of lens physical parameters using FP + FJ.

    Steps:
    1. Derive σ from catalog absolute magnitude (Faber-Jackson) if available,
       else from stellar mass (Cappellari+2013 mass-sigma).
    2. Compute FP-predicted Re iteratively: initial Re from ETG prior μe,
       then update μe from Re (one iteration removes the circular prior).
    3. Compute SIS θ_E from σ; blend 60% catalog / 40% FP.

    Parameters
    ----------
    row : dict
        Catalog row.  Used for 'abs_mag_r' / 'mag_r_abs' / 'mag_abs_r' /
        'lens_mag_total' columns if present.
    lens_z, source_z : float
    theta_E_catalog : float  [arcsec]
    lens_mass_log10 : float  log10(M★/M_sun)
    da_ls_over_ds : float    D_ls/D_s
    rng : optional
    enforce_fp : bool
    da_l_mpc : float, optional
        Angular diameter distance to lens [Mpc].  Used to convert kpc → arcsec
        for the μe self-consistency iteration.  If None, iteration is skipped.
    theta_E_max : float
        Hard upper clip on blended θ_E [arcsec]. Default 5″; paper/cluster
        configs may pass a larger value (e.g. 10″).

    Returns
    -------
    dict: sigma_kms, re_kpc, theta_E, fp_used, mu_e_used
    """
    # ------------------------------------------------------------------
    # Step 1: velocity dispersion
    # Prefer Faber-Jackson from catalog absolute magnitude when available,
    # fall back to mass-sigma relation.
    # ------------------------------------------------------------------
    abs_mag_r = None
    for col in ('abs_mag_r', 'mag_r_abs', 'mag_abs_r', 'abs_r'):
        val = row.get(col)
        if val is not None:
            try:
                v = float(val)
                if np.isfinite(v) and v < 0:   # magnitudes are negative
                    abs_mag_r = v
                    break
            except (TypeError, ValueError):
                pass

    if abs_mag_r is not None:
        sigma = faber_jackson_sigma(abs_mag_r, rng=rng, scatter_dex=0.04)
    else:
        sigma = sigma_from_mass(lens_mass_log10, rng=rng, scatter_dex=0.05)

    if not enforce_fp:
        return {
            'sigma_kms': sigma,
            're_kpc': None,
            'theta_E': theta_E_catalog,
            'fp_used': False,
            'mu_e_used': float('nan'),
        }

    # ------------------------------------------------------------------
    # Step 2: FP-predicted Re with one μe self-consistency iteration.
    #
    # The FP needs <μe> but <μe> depends on Re.  Standard approach:
    #   (a) Compute initial Re from the empirical ETG μe prior.
    #   (b) From Re (kpc → arcsec) + total magnitude, compute actual <μe>.
    #   (c) Recompute Re from the FP with the updated <μe>.
    # One iteration is sufficient; scatter added only on the final call.
    # ------------------------------------------------------------------
    mu_e_prior = 22.0 - 0.5 * (lens_z - 0.5)   # Hyde & Bernardi ETG prior

    # (a) Initial Re — no scatter yet
    re_kpc_init = fundamental_plane_re_kpc(
        sigma_kms=sigma,
        surface_brightness_r=mu_e_prior,
        z=lens_z,
        apply_lensing_bias=True,
        rng=None,
        add_scatter=False,
    )

    # (b) Refine μe if we have angular diameter distance and a total magnitude
    mu_e_used = mu_e_prior
    if da_l_mpc is not None and da_l_mpc > 0:
        re_arcsec_init = (re_kpc_init / (da_l_mpc * 1e3)) * 206265.0
        mag_total = None
        for col in ('lens_mag_total', 'mag_lens', 'mag_r', 'lens_magnitude'):
            val = row.get(col)
            if val is not None:
                try:
                    v = float(val)
                    if np.isfinite(v) and 12 < v < 32:
                        mag_total = v
                        break
                except (TypeError, ValueError):
                    pass
        if mag_total is not None:
            mu_e_iter = _mu_e_from_mag_re(mag_total, re_arcsec_init)
            if np.isfinite(mu_e_iter) and 16 < mu_e_iter < 28:
                mu_e_used = mu_e_iter

    # (c) Final Re with scatter
    re_kpc = fundamental_plane_re_kpc(
        sigma_kms=sigma,
        surface_brightness_r=mu_e_used,
        z=lens_z,
        apply_lensing_bias=True,
        rng=rng,
        add_scatter=True,
    )

    # ------------------------------------------------------------------
    # Step 3: SIS θ_E from σ; blend with catalog value
    # ------------------------------------------------------------------
    theta_E_fp = einstein_radius_from_sigma(sigma, da_ls_over_ds)
    w_cat = float(np.clip(catalog_theta_E_weight, 0.0, 1.0))
    theta_E_blended = w_cat * theta_E_catalog + (1.0 - w_cat) * theta_E_fp
    theta_E_blended = float(np.clip(theta_E_blended, 0.3, float(theta_E_max)))

    return {
        'sigma_kms': sigma,
        're_kpc': re_kpc,
        'theta_E': theta_E_blended,
        'theta_E_fp': float(theta_E_fp),
        'fp_used': True,
        'mu_e_used': mu_e_used,
    }
