#!/usr/bin/env python3
"""Side-by-side real-vs-simulated comparison panels for matched lenses.

The simulator's `lens_id` (0-based index into the run) corresponds row-for-
row to `data/cosmos_web_lens_structural_properties.csv` (ASSOC_ID), which is
the catalog used for each lens's structural/photometric parameters. This
script, for a given list of lens_ids and a sim output directory, finds the
matching real-lens FITS cutouts in data/real_lenses/<ASSOC_ID>_<band>.fits
and renders a per-band + RGB comparison figure (real on top, sim below).

Note: theta_E/lens_redshift for a given lens_id are drawn from
lens_analysis_catalog.csv row lens_id (a *different* catalog, not
necessarily the same real lens as ASSOC_ID) -- so this is a structural/
photometric-morphology match, not a full physical match. Good enough for
"does our rendering of this kind of galaxy look like the real thing"
comparisons.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval

BANDS = ["F115W", "F150W", "F277W", "F444W"]
REPO = Path(__file__).resolve().parents[3]
REAL_DIR = REPO / "data" / "real_lenses"
STRUCT_CSV = REPO / "data" / "cosmos_web_lens_structural_properties.csv"


def norm_for(data):
    return ImageNormalize(data, interval=PercentileInterval(99.5), stretch=AsinhStretch())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", required=True, help="e.g. /Volumes/extHD/jwst_lens_outputs/tng100_particle_test100")
    ap.add_argument("--lens-ids", required=True, nargs="+", type=int)
    ap.add_argument("--out", required=True, help="output directory for figures")
    args = ap.parse_args()

    import pandas as pd
    struct = pd.read_csv(STRUCT_CSV)
    sim_dir = Path(args.sim_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_dir = sim_dir / "unified_npz"

    for lens_id in args.lens_ids:
        assoc_id = struct["ASSOC_ID"].iloc[lens_id]
        real_files = {b: REAL_DIR / f"{assoc_id}_{b}.fits" for b in BANDS}
        if not all(f.exists() for f in real_files.values()):
            print(f"[skip] lens_id={lens_id} ({assoc_id}): missing real FITS")
            continue

        # find sim npz for this lens_id (any prefix, no epoch)
        candidates = sorted(npz_dir.glob(f"PRISM_lens_*_{lens_id:06d}.npz"))
        candidates = [c for c in candidates if "epoch" not in c.name]
        if not candidates:
            print(f"[skip] lens_id={lens_id} ({assoc_id}): no sim npz")
            continue
        sim_file = candidates[0]
        d = np.load(sim_file, allow_pickle=True)
        meta = json.loads(str(d["metadata"]))
        sim_bands = meta.get("bands", BANDS)
        cube = d["image_final"]

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for j, band in enumerate(BANDS):
            real_data = fits.getdata(real_files[band])
            axes[0, j].imshow(real_data, origin="lower", cmap="gray", norm=norm_for(real_data))
            axes[0, j].set_title(f"REAL {band}\n{assoc_id}")
            axes[0, j].axis("off")

            if band in sim_bands:
                i = sim_bands.index(band)
                sim_data = cube[i]
                axes[1, j].imshow(sim_data, origin="lower", cmap="gray", norm=norm_for(sim_data))
            axes[1, j].set_title(f"SIM {band}\nlens_id={lens_id}")
            axes[1, j].axis("off")

        # RGB panels (col 4)
        rgb_sim = d["rgb_visualization"]
        axes[1, 4].imshow(rgb_sim, origin="lower")
        axes[1, 4].set_title(f"SIM RGB\n{sim_file.stem}")
        axes[1, 4].axis("off")

        rgb_path = REAL_DIR / f"{assoc_id}_trilogy_rgb.png"
        if rgb_path.exists():
            real_rgb = plt.imread(rgb_path)
            axes[0, 4].imshow(real_rgb, origin="lower")
        axes[0, 4].set_title(f"REAL RGB\n{assoc_id}")
        axes[0, 4].axis("off")

        fig.suptitle(
            f"lens_id={lens_id}  z_lens={meta.get('lens_redshift'):.2f}  "
            f"z_source={meta.get('source_redshift'):.2f}  theta_E={meta.get('theta_E'):.2f}\"",
            fontsize=12,
        )
        fig.tight_layout()
        out_path = out_dir / f"compare_{lens_id:06d}_{assoc_id}.png"
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
