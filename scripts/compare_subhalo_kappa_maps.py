"""
Paired kappa-map comparison: SIE+shear macromodel alone vs. macromodel + NFW
dark-matter subhalo perturbers, for the same underlying lens.

Produces, for each representative system, a 4-panel figure:
  [smooth kappa] [perturbed kappa] [Delta-kappa residual] [perturber positions]

This isolates the effect of substructure for paper figures -- the macromodel
(SIE+shear) is identical in both panels; only the subhalo population differs.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prism.core.simulator import CONFIG, COSMO, generate_subhalo_population  # noqa: E402
from src.prism_kappa_output import compute_kappa_products  # noqa: E402
from lenstronomy.LensModel.lens_model import LensModel  # noqa: E402
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions  # noqa: E402

OUT_DIR = PROJECT_ROOT / "docs" / "figures" / "subhalo_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Force subhalo generation on regardless of the active config file
CONFIG.setdefault("subhalos", {})
CONFIG["subhalos"].update({
    "enabled": True,
    "count_range": [3, 12],
    "mass_function_slope": 1.9,
    "mass_range_log10_host": [-4.5, -2.0],
    "mass_fraction_of_host": 0.01,
    "concentration_relation": "duffy2008",
    "max_radius_einstein_units": 3.0,
    "profile": "NFW",
})

NUM_PIX = 300
DELTA_PIX = 0.031

# Representative systems spanning the typical theta_E / redshift range of the
# galaxy-scale lens population (values drawn from catalog-typical ranges).
SYSTEMS = [
    dict(name="compact_lowz", theta_E=0.71, lens_z=0.6, source_z=2.0,
         lens_mass_log10=11.3, q=0.75, phi_deg=25.0, gamma_ext=0.04, phi_ext_deg=70.0, seed=101),
    dict(name="typical_midz", theta_E=0.90, lens_z=1.0, source_z=2.8,
         lens_mass_log10=11.6, q=0.65, phi_deg=-40.0, gamma_ext=0.06, phi_ext_deg=10.0, seed=202),
    dict(name="large_highz", theta_E=1.25, lens_z=1.4, source_z=3.5,
         lens_mass_log10=11.9, q=0.55, phi_deg=60.0, gamma_ext=0.08, phi_ext_deg=-30.0, seed=303),
]


def sie_shear_kwargs(theta_E, q, phi_deg, gamma_ext, phi_ext_deg):
    phi = np.deg2rad(phi_deg)
    e1 = (1 - q) / (1 + q) * np.cos(2 * phi)
    e2 = (1 - q) / (1 + q) * np.sin(2 * phi)
    phi_ext = np.deg2rad(phi_ext_deg)
    gamma1 = gamma_ext * np.cos(2 * phi_ext)
    gamma2 = gamma_ext * np.sin(2 * phi_ext)
    return (
        ["SIE", "SHEAR"],
        [
            dict(theta_E=theta_E, e1=e1, e2=e2, center_x=0.0, center_y=0.0),
            dict(gamma1=gamma1, gamma2=gamma2, ra_0=0.0, dec_0=0.0),
        ],
    )


def critical_curves_caustics(lens_model, kwargs_lens, compute_window, grid_scale):
    """Return (ra_crit, dec_crit, ra_caustic, dec_caustic) point-cloud arrays."""
    ext = LensModelExtensions(lens_model)
    ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = ext.critical_curve_caustics(
        kwargs_lens, compute_window=compute_window, grid_scale=grid_scale,
        center_x=0.0, center_y=0.0,
    )
    return ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list


def overlay_curves(ax, ra_crit, dec_crit, ra_caustic, dec_caustic, crit_label=True):
    for i, (rc, dc) in enumerate(zip(ra_crit, dec_crit)):
        ax.plot(rc, dc, color="red", lw=1.3, alpha=0.9,
                label="critical curve" if (crit_label and i == 0) else None)
    for i, (rc, dc) in enumerate(zip(ra_caustic, dec_caustic)):
        ax.plot(rc, dc, color="lime", lw=1.3, alpha=0.9, ls="--",
                label="caustic" if (crit_label and i == 0) else None)


def make_panel(sysdef):
    name = sysdef["name"]
    theta_E = sysdef["theta_E"]

    model_list, kwargs_lens = sie_shear_kwargs(
        theta_E, sysdef["q"], sysdef["phi_deg"], sysdef["gamma_ext"], sysdef["phi_ext_deg"]
    )

    lens_model_smooth = LensModel(model_list)
    smooth = compute_kappa_products(lens_model_smooth, kwargs_lens, num_pix=NUM_PIX, delta_pix=DELTA_PIX)

    rng = np.random.default_rng(sysdef["seed"])
    sub_names, sub_kwargs = generate_subhalo_population(
        host_mass_log10=sysdef["lens_mass_log10"],
        host_redshift=sysdef["lens_z"],
        source_redshift=sysdef["source_z"],
        theta_E=theta_E,
        rng=rng,
        cosmo=COSMO,
    )
    print(f"[{name}] generated {len(sub_names)} NFW perturbers")

    full_model_list = model_list + sub_names
    full_kwargs = kwargs_lens + sub_kwargs
    lens_model_full = LensModel(full_model_list)
    perturbed = compute_kappa_products(lens_model_full, full_kwargs, num_pix=NUM_PIX, delta_pix=DELTA_PIX)

    delta_kappa = perturbed["kappa"] - smooth["kappa"]
    extent = smooth["extent"]
    half_fov = extent[1]

    # Critical curves & caustics of the FULL (perturbed) lens model -- these are
    # what an observer would actually compare data against.
    ra_crit, dec_crit, ra_caustic, dec_caustic = critical_curves_caustics(
        lens_model_full, full_kwargs, compute_window=2 * half_fov, grid_scale=DELTA_PIX
    )
    # Critical curves & caustics of the smooth macromodel only (for reference / panel 1)
    ra_crit_s, dec_crit_s, ra_caustic_s, dec_caustic_s = critical_curves_caustics(
        lens_model_smooth, kwargs_lens, compute_window=2 * half_fov, grid_scale=DELTA_PIX
    )

    # Display kappa on a clipped linear scale so the extended galaxy profile
    # (not just the singular core) is visible -- SIE kappa diverges at r->0,
    # so an unclipped log/linear scale washes out the extended structure.
    # vmax=1.5 (rather than the global kappa_max) keeps contrast around the
    # kappa=1 critical line, where the lensing structure actually matters.
    kappa_vmax = 1.5
    norm_kappa = plt.Normalize(vmin=0.0, vmax=kappa_vmax)
    cmap_kappa = "inferno"
    dmax = np.max(np.abs(delta_kappa))
    norm_delta = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)

    # Zoom window: tight crop around the Einstein ring, but wide enough to
    # keep the nearby perturbers (the ones that actually matter for anomalies)
    # in frame -- driven by whichever is larger: the ring itself, or the
    # radius enclosing the perturbers nearest to the critical curve.
    sub_radii = [np.hypot(kw["center_x"], kw["center_y"]) for kw in sub_kwargs]
    zoom_half = max(1.6 * theta_E, 1.1 * max(sub_radii, default=theta_E))
    xylim = (-zoom_half, zoom_half)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))

    # Marker size proportional to perturber mass -- shared between panels 0/1/3
    # so the reader can immediately locate *where* the (subtle, sub-percent)
    # kappa perturbations sit, even though they're invisible against the
    # galaxy's much larger overall convergence amplitude on a shared colour scale.
    max_alpha_rs = max((kw["alpha_Rs"] for kw in sub_kwargs), default=1.0)

    def mark_perturbers(ax, edgecolor="lime"):
        for kw in sub_kwargs:
            size = 50 + 250 * kw["alpha_Rs"] / max_alpha_rs
            ax.scatter(kw["center_x"], kw["center_y"], s=size, facecolors="none",
                       edgecolors=edgecolor, linewidths=1.6, zorder=5)

    # Effective Einstein radii (enclosed-mean-kappa = 1, see the corrected
    # _compute_theta_E_effective) of the smooth vs. perturbed models. Unlike
    # the *input* theta_E -- a fixed macromodel parameter that is, by
    # construction, identical in both cases -- theta_E_eff is measured
    # directly from each kappa map and genuinely shifts (typically by much
    # less than a pixel) once the subhalo population is added on top; this is
    # the actual physical signature the user is asking to see.
    theta_E_eff_smooth = smooth["theta_E_eff"]
    theta_E_eff_perturbed = perturbed["theta_E_eff"]

    def mark_effective_theta_E(ax):
        ax.add_patch(plt.Circle((0, 0), theta_E_eff_smooth, fill=False, color="dodgerblue",
                                ls="--", lw=1.4,
                                label=r"$\theta_{E,\rm eff}$ smooth = %.3f$''$" % theta_E_eff_smooth,
                                zorder=6))
        ax.add_patch(plt.Circle((0, 0), theta_E_eff_perturbed, fill=False, color="orange",
                                ls="-.", lw=1.4,
                                label=r"$\theta_{E,\rm eff}$ perturbed = %.3f$''$" % theta_E_eff_perturbed,
                                zorder=6))

    im0 = axes[0].imshow(np.clip(smooth["kappa"], 0, kappa_vmax), origin="lower", extent=extent,
                         cmap=cmap_kappa, norm=norm_kappa)
    axes[0].contour(smooth["kappa"], levels=[1.0], colors="cyan", linewidths=1.1, linestyles=":",
                    extent=extent, origin="lower")
    axes[0].plot([], [], color="cyan", lw=1.1, ls=":", label=r"$\kappa = 1$ contour")
    overlay_curves(axes[0], ra_crit_s, dec_crit_s, ra_caustic_s, dec_caustic_s)
    axes[0].set_title("SIE + shear only\n(smooth macromodel)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label=r"$\kappa$ (clipped at %.1f)" % kappa_vmax)

    im1 = axes[1].imshow(np.clip(perturbed["kappa"], 0, kappa_vmax), origin="lower", extent=extent,
                         cmap=cmap_kappa, norm=norm_kappa)
    axes[1].contour(perturbed["kappa"], levels=[1.0], colors="cyan", linewidths=1.1, linestyles=":",
                    extent=extent, origin="lower")
    axes[1].plot([], [], color="cyan", lw=1.1, ls=":", label=r"$\kappa = 1$ (perturbed)")
    # Show, in addition, the SMOOTH model's kappa=1 contour as a white dashed
    # line: any visible offset between the cyan (perturbed) and white (smooth)
    # critical-convergence contours is the direct fingerprint of substructure.
    axes[1].contour(smooth["kappa"], levels=[1.0], colors="white", linewidths=1.0,
                    linestyles="--", extent=extent, origin="lower", alpha=0.8)
    axes[1].plot([], [], color="white", lw=1.0, ls="--", alpha=0.8, label=r"$\kappa = 1$ (smooth)")
    overlay_curves(axes[1], ra_crit, dec_crit, ra_caustic, dec_caustic)
    mark_perturbers(axes[1])
    axes[1].set_title(f"SIE + shear + {len(sub_names)} NFW subhalos\n(perturbed model; "
                      "lime circles = perturber locations)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label=r"$\kappa$ (clipped at %.1f)" % kappa_vmax)

    im2 = axes[2].imshow(delta_kappa, origin="lower", extent=extent, cmap="RdBu_r", norm=norm_delta)
    overlay_curves(axes[2], ra_crit, dec_crit, ra_caustic, dec_caustic, crit_label=False)
    mark_effective_theta_E(axes[2])
    mark_perturbers(axes[2], edgecolor="crimson")
    axes[2].set_title(r"$\Delta\kappa$ = perturbed $-$ smooth"
                      "\n(substructure signature; crimson circles = perturber locations,"
                      "\nsize $\\propto$ mass)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label=r"$\Delta\kappa$")

    for ax in axes:
        ax.set_xlim(*xylim)
        ax.set_ylim(*xylim)
        ax.set_aspect("equal")
        ax.set_xlabel("arcsec")
        ax.set_ylabel("arcsec")
        ax.legend(loc="lower right", fontsize=7, framealpha=0.6)

    fig.suptitle(
        f"System '{name}':  $\\theta_E$ = {theta_E:.2f}\"   $z_l$ = {sysdef['lens_z']:.2f}   "
        f"$z_s$ = {sysdef['source_z']:.2f}   $\\log_{{10}}(M_{{host}}/M_\\odot)$ = {sysdef['lens_mass_log10']:.1f}"
        f"   |   zoomed on Einstein ring ($\\pm${zoom_half:.2f}\")",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = OUT_DIR / f"subhalo_comparison_{name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{name}] saved {out_path}")
    return out_path


if __name__ == "__main__":
    paths = [make_panel(s) for s in SYSTEMS]
    print("\nDone. Figures written to:")
    for p in paths:
        print(" ", p)
