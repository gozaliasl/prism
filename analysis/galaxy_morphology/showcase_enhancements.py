"""
Showcase-quality galaxy models: analytic bulge+disk+arms+bar built at native
resolution, then PSF-convolved once.  Texture is applied *before* blurring so
spiral arms and bars stay sharp and continuous (not blobby post-PSF overlays).
"""
import numpy as np
from scipy.ndimage import gaussian_filter

ZPT = 28.0
_EXP_RE_OVER_H = 1.678


def _disk_coords(shape, cx, cy, q, pa_deg):
    y, x = np.indices(shape)
    rel_x = x - cx
    rel_y = y - cy
    pa = np.radians(pa_deg)
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = rel_x * cos_pa + rel_y * sin_pa
    y_rot = -rel_x * sin_pa + rel_y * cos_pa
    q = float(np.clip(q, 0.12, 1.0))
    r = np.sqrt(x_rot ** 2 + (y_rot / q) ** 2)
    theta = np.arctan2(y_rot / q, x_rot + 1e-8)
    return x_rot, y_rot, r, theta, cos_pa, sin_pa, q


def _sersic_r(r, re, n):
    re = max(float(re), 0.5)
    n = max(float(n), 0.3)
    bn = 1.9992 * n - 0.3271
    return np.exp(-bn * ((np.maximum(r, 0.0) / re) ** (1.0 / n) - 1.0))


def _mag_to_sum_flux(mag, pixel_scale, numpix):
    """Total integrated flux in image pixels for AB mag at given zeropoint."""
    flux_per_pix = 10 ** (-0.4 * (mag - ZPT)) * (pixel_scale ** 2)
    return flux_per_pix * numpix * numpix


def convolve_gaussian_psf(image, fwhm_arcsec, pixel_scale):
    sigma_pix = (fwhm_arcsec / pixel_scale) / 2.355
    return gaussian_filter(np.asarray(image, dtype=np.float64), sigma_pix, mode='constant')


def _spiral_arm_factor(r, theta, n_arms, pitch_deg, r0, arm_width, arm_contrast, interarm_frac=0.30):
    pitch = np.radians(np.clip(pitch_deg, 6.0, 38.0))
    r_safe = np.maximum(r, 0.5)
    arms = np.zeros_like(r)
    for m in range(n_arms):
        offset = m * 2 * np.pi / n_arms
        phase = theta - offset - np.log(r_safe / r0) * np.tan(pitch)
        wrapped = np.arctan2(np.sin(phase), np.cos(phase))
        width = arm_width * (1.0 + 0.35 * np.clip(r / (r0 * 4.0), 0, 1))
        arms += np.exp(-0.5 * (wrapped / width) ** 2)
    arms /= max(np.max(arms), 1e-12)
    return 1.0 + arm_contrast * arms - arm_contrast * interarm_frac * (1.0 - arms)


def _radial_disk_mask(r, r_inner, r_outer):
    inner = 1.0 - np.exp(-((r - r_inner) / max(0.12 * r_inner, 0.8)) ** 2)
    outer = np.exp(-((np.maximum(r - r_outer, 0.0)) / max(0.18 * r_outer, 1.5)) ** 2)
    return np.clip(inner * outer, 0.0, 1.0)


def build_analytic_disk_galaxy(
    numpix,
    pixel_scale,
    total_mag,
    *,
    q=0.7,
    pa_deg=0.0,
    r_eff_arcsec=1.0,
    bt=0.22,
    n_bulge=4.0,
    re_bulge_frac=0.18,
    h_disk_frac=1.05,
    n_arms=2,
    pitch_deg=14.0,
    arm_contrast=0.55,
    arm_width=0.28,
    bar=None,
    dust_lane=None,
    seed=42,
):
    """
    Build bulge + exponential disk with optional log-spiral modulation, bar,
    and edge-on dust lane.  Returns unconvolved model (apply PSF separately).
    """
    rng = np.random.RandomState(seed)
    cx = cy = numpix / 2.0
    r_eff_pix = r_eff_arcsec / pixel_scale
    h_pix = h_disk_frac * r_eff_pix / _EXP_RE_OVER_H
    re_bulge_pix = re_bulge_frac * r_eff_pix

    _, _, r, theta, _, _, q_val = _disk_coords(
        (numpix, numpix), cx, cy, q, pa_deg,
    )

    total_flux = _mag_to_sum_flux(total_mag, pixel_scale, numpix)
    bulge_flux = total_flux * bt
    disk_flux = total_flux * (1.0 - bt)

    bulge = bulge_flux * _sersic_r(r, re_bulge_pix, n_bulge)
    bulge /= max(np.sum(bulge), 1e-12)

    disk = np.exp(-r / max(h_pix, 1.0))
    r_inner = max(0.32 * re_bulge_pix, 1.5)
    r_outer = min(numpix * 0.46, 4.2 * h_pix)
    r0 = max(r_inner * 1.2, 0.12 * r_outer)
    disk_mask = _radial_disk_mask(r, r_inner, r_outer)

    if n_arms > 0:
        arm_phases = bar.get('arm_phase_offsets') if bar else None
        if arm_phases is None:
            arm_factor = _spiral_arm_factor(
                r, theta, n_arms, pitch_deg, r0, arm_width, arm_contrast,
            )
        else:
            # Arms anchored to bar ends
            pitch = np.radians(pitch_deg)
            r_safe = np.maximum(r, 0.5)
            arms = np.zeros_like(r)
            for offset in arm_phases:
                phase = theta - offset - np.log(r_safe / r0) * np.tan(pitch)
                wrapped = np.arctan2(np.sin(phase), np.cos(phase))
                arms += np.exp(-0.5 * (wrapped / arm_width) ** 2)
            arms /= max(np.max(arms), 1e-12)
            arm_factor = 1.0 + arm_contrast * arms - arm_contrast * 0.28 * (1.0 - arms)
        disk *= arm_factor

    disk *= disk_mask
    disk = disk_flux * disk / max(np.sum(disk), 1e-12)

    image = bulge + disk

    if bar is not None:
        x_rot, y_rot, r_d, _, _, _, _ = _disk_coords(
            (numpix, numpix), cx, cy, q, bar.get('pa_deg', pa_deg),
        )
        bar_a = bar.get('length_pix', 0.46 * h_pix * _EXP_RE_OVER_H)
        bar_b = max(bar_a * bar.get('axis_ratio', 0.19), 1.5)
        rb = np.sqrt((x_rot / bar_a) ** 2 + (y_rot / bar_b) ** 2)
        bar_prof = np.zeros_like(r_d)
        inside = rb < 1.0
        bar_prof[inside] = (1.0 - rb[inside] ** 2) ** 1.6
        bar_prof *= np.exp(-r_d / (1.3 * h_pix))
        bar_flux = total_flux * bar.get('flux_frac', 0.12)
        bar_prof = bar_flux * bar_prof / max(np.sum(bar_prof), 1e-12)
        image += bar_prof

    if dust_lane is not None:
        _, y_rot, r_d, _, _, _, _ = _disk_coords(
            (numpix, numpix), cx, cy, q, dust_lane.get('pa_deg', pa_deg),
        )
        depth = dust_lane.get('depth', 0.42)
        width = dust_lane.get('width_pix', max(1.2, 0.06 * numpix))
        lane = np.exp(-0.5 * (y_rot / width) ** 2)
        disk_atten = np.exp(-r_d / (1.1 * h_pix))
        atten = 1.0 - depth * lane * (disk_atten / (np.max(disk_atten) + 1e-12))
        image *= np.clip(atten, 0.25, 1.0)
        image = image / max(np.sum(image), 1e-12) * total_flux

    # Sparse HII knots along arms (pre-PSF, small and faint)
    if n_arms > 0 and arm_contrast > 0.2:
        y, x = np.indices((numpix, numpix))
        peak = np.max(image)
        offsets = bar.get('arm_phase_offsets') if bar else [
            m * 2 * np.pi / n_arms for m in range(n_arms)
        ]
        pitch_rad = np.radians(pitch_deg)
        for offset in offsets:
            for frac in np.linspace(0.45, 0.90, n_arms * 2 + 2):
                r_k = r_inner + frac * (r_outer - r_inner)
                th_k = offset + np.log(r_k / r0) * np.tan(pitch_rad)
                th_k += rng.uniform(-0.05, 0.05)
                xr = r_k * np.cos(th_k)
                yr = q_val * r_k * np.sin(th_k)
                pa = np.radians(pa_deg)
                xk = cx + xr * np.cos(pa) - yr * np.sin(pa)
                yk = cy + xr * np.sin(pa) + yr * np.cos(pa)
                sigma = rng.uniform(0.7, 1.3)
                image += peak * rng.uniform(0.012, 0.028) * np.exp(
                    -((x - xk) ** 2 + (y - yk) ** 2) / (2 * sigma ** 2)
                )
        image = image / max(np.sum(image), 1e-12) * total_flux

    return np.clip(image, 0, None)


# GALFIT-style presets per showcase morph type (analytic path)
ANALYTIC_MORPH = {
    'spiral': dict(
        bt=0.22, n_arms=2, pitch_deg=11.0, arm_contrast=0.62, arm_width=0.22,
        h_disk_frac=1.05, re_bulge_frac=0.16,
    ),
    'late_spiral': dict(
        bt=0.10, n_arms=3, pitch_deg=24.0, arm_contrast=0.78, arm_width=0.30,
        h_disk_frac=1.35, re_bulge_frac=0.11, q_override=0.55,
    ),
    'barred_spiral': dict(
        bt=0.18, n_arms=2, pitch_deg=16.0, arm_contrast=0.50, arm_width=0.20,
        h_disk_frac=1.0, re_bulge_frac=0.14, bar_pa=28.0,
    ),
    'edge_on': dict(
        bt=0.28, n_arms=0, h_disk_frac=1.25, re_bulge_frac=0.15,
        q_override=0.18, dust_depth=0.48,
    ),
}


def build_analytic_showcase(morph_type, params, numpix, pixel_scale, total_mag, seed=42):
    """Return unconvolved analytic model for disk-dominated showcase types."""
    preset = dict(ANALYTIC_MORPH[morph_type])
    q = preset.pop('q_override', params['q'])
    bar_pa = preset.pop('bar_pa', 0.0)

    bar = None
    dust = None
    n_arms = preset.get('n_arms', 2)

    if morph_type == 'barred_spiral':
        r_eff_pix = params['R_eff'] / pixel_scale
        h_pix = preset['h_disk_frac'] * r_eff_pix / _EXP_RE_OVER_H
        bar = {
            'pa_deg': bar_pa,
            'length_pix': 0.52 * h_pix * _EXP_RE_OVER_H,
            'axis_ratio': 0.17,
            'flux_frac': 0.14,
            'arm_phase_offsets': [np.radians(bar_pa), np.radians(bar_pa) + np.pi],
        }

    if morph_type == 'edge_on':
        dust = {'pa_deg': 0.0, 'depth': preset.pop('dust_depth', 0.45), 'width_pix': 1.0}
        n_arms = 0

    return build_analytic_disk_galaxy(
        numpix, pixel_scale, total_mag,
        q=q, pa_deg=0.0, r_eff_arcsec=params['R_eff'],
        bt=preset.get('bt', 0.2),
        re_bulge_frac=preset.get('re_bulge_frac', 0.18),
        h_disk_frac=preset.get('h_disk_frac', 1.0),
        n_arms=n_arms,
        pitch_deg=preset.get('pitch_deg', 15.0),
        arm_contrast=preset.get('arm_contrast', 0.5),
        arm_width=preset.get('arm_width', 0.25),
        bar=bar, dust_lane=dust, seed=seed,
    )


def enhance_showcase_model(image, morph_type, params, idx, numpix, pixel_scale):
    """
    Post-process non-analytic types (clumps, ring) on an unconvolved base image.
    Disk spirals use build_analytic_showcase instead — no enhancement here.
    """
    center = numpix // 2
    seed = 700 + idx
    img = np.asarray(image, dtype=np.float64)
    r_eff_pix = float(params['R_eff']) / pixel_scale

    if morph_type in ANALYTIC_MORPH:
        return img

    if morph_type in ('irregular', 'clumpy', 'starburst', 'primordial'):
        from prism.core.jwst_lens_simulator import add_clumpy_structure_to_image
        n_clumps = {'irregular': 12, 'clumpy': 14, 'starburst': 16, 'primordial': 18}[morph_type]
        strength = {'irregular': 0.45, 'clumpy': 0.55, 'starburst': 0.65, 'primordial': 0.60}[morph_type]
        return add_clumpy_structure_to_image(
            img, center, center, pixel_scale,
            n_clumps=n_clumps, clump_strength=strength, seed=seed,
        )

    if morph_type == 'ring':
        from prism.core.jwst_lens_simulator import add_ring_structure_to_image
        ring_radius = float(np.clip(2.4 * r_eff_pix, 0.12 * numpix, 0.42 * numpix))
        return add_ring_structure_to_image(
            img, center, center, pixel_scale,
            ring_radius=ring_radius, ring_width=max(0.07 * ring_radius, 1.5),
            n_knots=14, seed=seed,
        )

    return img
