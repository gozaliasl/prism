#!/usr/bin/env python3
"""Generate ~30 diverse 1′ Euclid paper-gallery strong lenses.

Morphology modes:
  sersic  — multicomponent de Vaucouleurs elliptical (no TNG particles)
  hybrid  — selective TNG particles + Sersic (smoothed)
  tng     — full TNG particle morphology (heavy SPH smoothing)

Lens classes (forced per batch):
  field   — single_field elliptical
  pair    — binary SIE+SIE
  group   — BGG / group

Usage:
  python scripts/local/run_paper_gallery_30.py
  python scripts/local/run_paper_gallery_30.py --n-per-cell 3   # 3×3×3=27
  python scripts/local/run_paper_gallery_30.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "outputs" / "euclid_paper_gallery_30"

# Shared Euclid 1′ settings with brighter arcs so sources stand out from the lens.
BASE = {
    "n_lenses": 4,
    "n_non_lenses": 0,
    "variations_per_base": 1,
    "n_field_max": 40,
    "telescope": "euclid",
    "euclid_q1": {
        "enabled": True,
        "data_dir": "data/euclid_q1_psf",
        "use_empirical_psf": True,
        "use_population_priors": True,
        "use_q1_photometry": True,
        "use_q1_shear": True,
        "resample_theta_E": True,
        "shear_range": {"min": 0.05, "max": 0.16},
    },
    "multi_resolution": {"enabled": False},
    "bands": ["EUCLID_VIS", "EUCLID_Y", "EUCLID_J", "EUCLID_H"],
    "redshifts": {
        "lens_min": 0.35,
        "lens_max": 0.80,
        "source_min": 1.0,
        "source_max": 2.4,
        "enforce_source_behind_lens": True,
        "min_delta_z": 0.45,
    },
    "mass": {
        "lens_mass_min": 11.1,
        "lens_mass_max": 12.3,
        "use_mass_theta_E_correlation": True,
    },
    "geometry": {
        "visible_lensed_arcs": True,
        "force_caustic_source_position": True,
        "min_magnification": 6.0,
        "max_magnification": 28.0,
        "source_fraction_mean": 0.30,
        "source_fraction_sigma": 0.28,
        "source_fraction_min": 0.12,
        "source_fraction_max": 0.48,
        "lens_radius_min": 0.55,
        "lens_radius_max": 2.2,
        "enforce_source_smaller_than_lens": True,
        "max_source_to_lens_ratio": 0.75,
    },
    "field": {
        "density_mag_limit": 23.5,
        "avoid_center_arcsec": 1.2,
        "min_pair_separation_arcsec": 0.30,
        "max_fraction_of_half_size": 0.92,
        "size_max_arcsec": 1.5,
        "size_max_fraction_of_lens": 0.7,
        "min_count": 0,
    },
    # Brighter arcs relative to lens (easier to distinguish)
    "photometry": {
        "lens_mag_min": 19.0,
        "lens_mag_max": 22.2,
        "source_mag_min": 18.0,
        "source_mag_max": 20.5,
        "min_source_fainter_than_lens_mag": 0.15,
        "max_source_fainter_than_lens_mag": 0.85,
        "visible_arc_mag_diff_max": 0.85,
        "visible_arc_source_mag_max": 20.0,
        "source_mag_diff_min": 0.15,
        "source_mag_diff_max": 0.90,
        "source_base_mag": 19.1,
        "field_mag_limit": 24.0,
    },
    "morphology": {
        "multicomponent_enabled": True,
        "apply_to_lens": True,
        "apply_to_source": True,
        "native_ring_merger_geometry": False,
        "bt_fractions": {"elliptical": 0.95, "s0": 0.90, "spiral": 0.22},
        "components": {
            "bulge": {"n_sersic": 4.0, "r_frac": 0.30},
            "disk": {"n_sersic": 1.0, "r_frac": 0.90},
        },
    },
    "lens_morphology": {
        "force_morph_type": "elliptical",
        "min_lens_sersic": 3.8,
    },
    "galaxygenius_stamps": {"enabled": False},
    "time_delays": {"enabled": False},
    "fundamental_plane": {"enabled": True},
    "output": {
        "unified_storage": True,
        "save_intermediate_rgb_jpg": True,
        "extended_image_fov_arcmin": 1.0,
        "extended_lensing_fov_arcmin": 1.0,
        "stacked_npy": {"enabled": False},
        "rgb": {
            "arc_boost": 0.85,
            "color_enhance": 1.45,
            "sat_percent": 0.025,
        },
    },
    "detector_chain": {
        "enabled": True,
        "effects": {
            "ipc": True,
            "charge_diffusion": True,
            "brighter_fatter": True,
            "nonlinearity": True,
            "dark_current": True,
            "poisson_shot_noise": True,
            "read_noise": True,
            "one_f_noise": True,
            "saturation": True,
            "prnu": True,
            "persistence": False,
            "gain_adc": True,
        },
    },
    "telescope_configs": {
        "euclid": {
            "bands": ["EUCLID_VIS", "EUCLID_Y", "EUCLID_J", "EUCLID_H"],
            "pixel_scale": 0.10,
            "exposure_time": 565.0,
            "magnitude_zero_point": 25.58,
            "image_size": 600,
            "lens_population": {
                "redshifts": {
                    "lens_min": 0.35,
                    "lens_max": 0.80,
                    "source_min": 1.0,
                    "source_max": 2.4,
                    "min_delta_z": 0.45,
                },
                "theta_E": {
                    "min": 0.70,
                    "max": 2.4,
                    "median": 1.20,
                    "log_scatter": 0.28,
                },
            },
        }
    },
    "add_artifacts": True,
    "add_noise": True,
    "save_intermediate_images": True,
    "save_npy": True,
    "save_jpg": True,
}

TNG_COMMON = {
    "enabled": True,
    "sim_mode": "tng_mixed",
    "local_catalog_path": "/Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet",
    "local_catalog_path_tng50": "/Volumes/extHD/tng_local_catalog/tng50-1_local_catalog.parquet",
    "logM_tol": 0.45,
    "max_attempts": 30,
    "require_local_particles": False,
    "lens_sfr_class": "quiescent",
    "source_logM_default": 9.8,
    "source_logM_scatter": 0.45,
    "environment_drives_field_count": True,
}

MORPH = {
    "sersic": {
        "tng_mode": {
            **TNG_COMMON,
            "field_enabled": False,
            "particle_morphology": {"enabled": False},
        }
    },
    "hybrid": {
        "tng_mode": {
            **TNG_COMMON,
            "field_enabled": True,
            "field_fraction": 0.25,
            "field_logM_default": 9.0,
            "field_logM_scatter": 0.75,
            "particle_morphology": {
                "enabled": True,
                "min_particles": 400,
                "min_particles_field": 150,
                "lens_enabled": True,
                "lens_fraction": 0.70,
                "lens_smooth_sigma": 3.5,
                "source_enabled": True,
                "source_fraction": 0.50,
                "source_smooth_sigma": 2.5,
                "field_enabled": True,
                "field_fraction": 0.15,
                "field_smooth_sigma": 2.8,
                "companion_enabled": True,
                "companion_fraction": 0.50,
                "companion_smooth_sigma": 3.2,
            },
        }
    },
    "tng": {
        "tng_mode": {
            **TNG_COMMON,
            "field_enabled": True,
            "field_fraction": 0.85,
            "field_logM_default": 9.0,
            "field_logM_scatter": 0.75,
            "particle_morphology": {
                "enabled": True,
                "min_particles": 500,
                "min_particles_field": 200,
                "lens_enabled": True,
                "lens_fraction": 1.0,
                "lens_smooth_sigma": 4.0,
                "source_enabled": True,
                "source_fraction": 1.0,
                "source_smooth_sigma": 3.0,
                "field_enabled": True,
                "field_fraction": 0.80,
                "field_smooth_sigma": 3.0,
                "companion_enabled": True,
                "companion_fraction": 1.0,
                "companion_smooth_sigma": 3.5,
            },
        }
    },
}

CLASS = {
    "field": {
        "lens_environment": "isolated",
        "lens_class_distribution": {
            "enabled": True,
            "single_field": {"fraction": 1.0},
            "group": {"fraction": 0.0},
            "binary_sie_sie": {"fraction": 0.0},
            "binary_nfw_nfw": {"fraction": 0.0},
            "binary_shear_only": {"fraction": 0.0},
        },
        "environment": {
            "types": {
                "isolated_field": {
                    "fraction": 1.0,
                    "galaxy_count_mean": 2.8,
                    "galaxy_count_std": 0.8,
                    "min_galaxies": 1,
                    "max_galaxies": 5,
                    "separation_mean": 2.0,
                    "avoid_factor": 1.0,
                    "max_radius_arcsec": 28.0,
                    "shear_min": 0.03,
                    "shear_max": 0.10,
                }
            }
        },
    },
    "pair": {
        "lens_environment": "pair",
        "lens_class_distribution": {
            "enabled": True,
            "single_field": {"fraction": 0.0},
            "group": {"fraction": 0.0},
            "binary_sie_sie": {"fraction": 1.0},
            "binary_nfw_nfw": {"fraction": 0.0},
            "binary_shear_only": {"fraction": 0.0},
        },
        "environment": {
            "types": {
                "galaxy_pair": {
                    "fraction": 1.0,
                    "galaxy_count_mean": 3.2,
                    "galaxy_count_std": 0.9,
                    "min_galaxies": 2,
                    "max_galaxies": 6,
                    "separation_mean": 1.8,
                    "avoid_factor": 1.0,
                    "max_radius_arcsec": 28.0,
                    "shear_min": 0.04,
                    "shear_max": 0.12,
                }
            }
        },
    },
    "group": {
        "lens_environment": "group",
        "lens_class_distribution": {
            "enabled": True,
            "single_field": {"fraction": 0.0},
            "group": {"fraction": 1.0},
            "binary_sie_sie": {"fraction": 0.0},
            "binary_nfw_nfw": {"fraction": 0.0},
            "binary_shear_only": {"fraction": 0.0},
        },
        "environment": {
            "types": {
                "group": {
                    "fraction": 1.0,
                    "galaxy_count_mean": 3.5,
                    "galaxy_count_std": 1.0,
                    "min_galaxies": 2,
                    "max_galaxies": 8,
                    "separation_mean": 1.6,
                    "avoid_factor": 1.0,
                    "max_radius_arcsec": 28.0,
                    "shear_min": 0.05,
                    "shear_max": 0.15,
                }
            }
        },
    },
}

# Distinct seeds so each cell explores different Einstein radii / arcs
SEEDS = {
    ("sersic", "field"): 101,
    ("sersic", "pair"): 102,
    ("sersic", "group"): 103,
    ("hybrid", "field"): 201,
    ("hybrid", "pair"): 202,
    ("hybrid", "group"): 203,
    ("tng", "field"): 301,
    ("tng", "pair"): 302,
    ("tng", "group"): 303,
}


def build_config(morph: str, klass: str, n_lenses: int) -> dict:
    cfg = json.loads(json.dumps(BASE))  # deep copy
    cfg["n_lenses"] = n_lenses
    cfg.update(MORPH[morph])
    class_cfg = CLASS[klass]
    cfg["lens_class_distribution"] = class_cfg["lens_class_distribution"]
    cfg["environment"] = class_cfg["environment"]
    cfg["tng_mode"]["lens_environment"] = class_cfg["lens_environment"]
    return cfg


def run_cell(morph: str, klass: str, n_lenses: int, dry_run: bool) -> Path:
    out_dir = OUT_ROOT / f"{morph}_{klass}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_config(morph, klass, n_lenses)
    cfg_path = out_dir / "run_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    seed = SEEDS[(morph, klass)]
    cmd = [
        sys.executable, "-u", str(REPO / "src" / "jwst_lens_simulator.py"),
        "--config", str(cfg_path),
        "--cosmos_catalog", "data/cosmos_web_lens_structural_properties.csv",
        "--lens_analysis_catalog", "data/lens_analysis_catalog.csv",
        "--merged_field_catalog", "data/merged_lens_field_catalog.csv",
        "--output_dir", str(out_dir),
        "--n_lenses", str(n_lenses),
        "--n_non_lenses", "0",
        "--n_field_max", "40",
        "--variations_per_base", "1",
        "--seed", str(seed),
        "--add_artifacts",
        "--save_intermediate",
        "--no_date_suffix",
    ]
    print(f"\n{'='*70}\n[RUN] {morph} / {klass}  n={n_lenses}  seed={seed}\n  → {out_dir}\n{'='*70}")
    if dry_run:
        print(" ".join(cmd))
        return out_dir

    log_path = out_dir / "run.log"
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[ERROR] {morph}/{klass} failed (exit {proc.returncode}); see {log_path}")
    else:
        print(f"[OK] {morph}/{klass}")
    return out_dir


def build_chooser_gallery() -> Path:
    """Copy all panel JPGs into a flat chooser folder with descriptive names."""
    chooser = OUT_ROOT / "CHOOSER"
    chooser.mkdir(parents=True, exist_ok=True)
    index = []
    for morph in ("sersic", "hybrid", "tng"):
        for klass in ("field", "pair", "group"):
            cell = OUT_ROOT / f"{morph}_{klass}"
            jpg_dir = cell / "jpg_rgb"
            if not jpg_dir.is_dir():
                continue
            for jpg in sorted(jpg_dir.glob("PRISM_lens_*.jpg")):
                # PRISM_lens_SF_000004.jpg → sersic_field_SF_000004.jpg
                short = jpg.name.replace("PRISM_lens_", "")
                dest_name = f"{morph}_{klass}_{short}"
                dest = chooser / dest_name
                shutil.copy2(jpg, dest)
                # also copy kappa/flexion if present
                lid = short.split("_")[-1].replace(".jpg", "")
                kappa = cell / "kappa_maps" / f"{lid}_kappa_panel.jpg"
                if kappa.exists():
                    shutil.copy2(kappa, chooser / f"{morph}_{klass}_{lid}_kappa_panel.jpg")
                flex = cell / "kappa_maps" / f"{lid}_flexion_panel.jpg"
                if flex.exists():
                    shutil.copy2(flex, chooser / f"{morph}_{klass}_{lid}_flexion_panel.jpg")
                index.append({
                    "file": dest_name,
                    "morphology": morph,
                    "lens_class": klass,
                    "source_jpg": str(jpg),
                    "lens_id": lid,
                })
    (chooser / "INDEX.json").write_text(json.dumps(index, indent=2))
    # Simple HTML contact sheet
    rows = []
    for item in index:
        rows.append(
            f'<div style="display:inline-block;margin:8px;text-align:center;width:320px">'
            f'<a href="{item["file"]}"><img src="{item["file"]}" style="width:300px;border:1px solid #444"/></a>'
            f'<div style="font:12px monospace;color:#ccc">{item["morphology"]} | {item["lens_class"]} | {item["lens_id"]}</div>'
            f'</div>'
        )
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Paper Gallery 30</title>"
        "<style>body{background:#111;color:#eee;font-family:sans-serif}</style></head><body>"
        "<h1>Euclid 1′ paper gallery — pick your hero</h1>"
        "<p>morphology × class × examples. Click any panel to open full size.</p>"
        + "\n".join(rows)
        + "</body></html>"
    )
    (chooser / "index.html").write_text(html)
    print(f"[GALLERY] {len(index)} panels → {chooser}")
    print(f"          open {chooser / 'index.html'}")
    return chooser


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-cell", type=int, default=4,
                        help="Lenses per morph×class cell (default 4 → 36 total)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gallery-only", action="store_true",
                        help="Only rebuild CHOOSER from existing outputs")
    parser.add_argument("--morph", nargs="*", default=["sersic", "hybrid", "tng"])
    parser.add_argument("--class", dest="classes", nargs="*",
                        default=["field", "pair", "group"])
    args = parser.parse_args()

    if not args.gallery_only:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        for morph in args.morph:
            for klass in args.classes:
                run_cell(morph, klass, args.n_per_cell, args.dry_run)

    if not args.dry_run:
        build_chooser_gallery()


if __name__ == "__main__":
    main()
