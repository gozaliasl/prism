"""Single-row RGB comparison of the same lens across all 5 telescopes,
zoomed in tightly on the lens itself (small/native-FOV renders already
crop close to the deflector, so this crops a fixed physical arcsec window
centered on the frame for a consistent, tight view).

Run: python scripts/local/build_5telescope_zoom_row.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/Volumes/exthd-prism/prism-lensing")

TEL_DIRS = {
    "JWST": "outputs_test/q1test_jwst_date_20260802_223350",
    "Roman": "outputs_test/q1test_roman_date_20260802_223508",
    "Euclid": "outputs_test/q1test_euclid_matched_date_20260802_223538",
    "Subaru": "outputs_test/q1test_subaru_date_20260802_223610",
    "Rubin": "outputs_test/q1test_lsst_date_20260802_223610".replace("223610", "223638"),
}
LENS_ID = 2
ZOOM_ARCSEC = 2.5  # half-width of the crop window (i.e. 5" x 5" total)

# metadata['delta_pix'] is not actually saved by the pipeline (always None),
# so relying on it silently made every telescope crop using JWST's 0.031"/px
# fallback -- correct for JWST/Euclid by coincidence-adjacent values, but for
# Subaru (0.168"/px) and Rubin (0.20"/px) this grabbed ~13-16" of real sky
# while labeling it "5 arcsec". Use each telescope's real pixel scale
# explicitly instead.
PIXEL_SCALES = {"JWST": 0.031, "Roman": 0.11, "Euclid": 0.10, "Subaru": 0.168, "Rubin": 0.20}

OUT_PATH = BASE / "outputs_test/q1_lens2_5telescope_zoom_row.png"


def npz_path(rel_dir: str) -> Path:
    return BASE / rel_dir / "unified_npz" / f"PRISM_lens_SF_{LENS_ID:06d}.npz"


def load(rel_dir: str):
    d = np.load(npz_path(rel_dir), allow_pickle=True)
    im = d["image_final"]
    meta = json.loads(str(d["metadata"]))
    return im, meta


def stretch(x, black_pct=50.0, vmax_pct=99.7, soft_frac=0.08):
    black = np.percentile(x, black_pct)
    sub = np.clip(x - black, 0, None)
    vmax = np.percentile(sub, vmax_pct)
    if vmax <= 0:
        vmax = sub.max() or 1.0
    soft = vmax * soft_frac
    return np.clip(np.arcsinh(sub / soft) / np.arcsinh(vmax / soft), 0, 1)


def rgb_from_bands(im, bands):
    stretched = [stretch(im[bi]) for bi in range(len(bands))]
    return np.stack([stretched[-1], 0.5 * (stretched[-2] + stretched[1]), stretched[0]], axis=-1)


def center_crop(rgb, pixel_scale, half_arcsec):
    h, w = rgb.shape[:2]
    cy, cx = h // 2, w // 2
    half_px = max(1, int(round(half_arcsec / pixel_scale)))
    y0, y1 = max(0, cy - half_px), min(h, cy + half_px)
    x0, x1 = max(0, cx - half_px), min(w, cx + half_px)
    return rgb[y0:y1, x0:x1]


def main():
    fig, axes = plt.subplots(1, len(TEL_DIRS), figsize=(3.0 * len(TEL_DIRS), 3.4), facecolor="black")

    for ax, (tname, rel_dir) in zip(axes, TEL_DIRS.items()):
        im, meta = load(rel_dir)
        bands = meta["bands"]
        pixel_scale = PIXEL_SCALES[tname]
        rgb = rgb_from_bands(im, bands)
        crop = center_crop(rgb, pixel_scale, ZOOM_ARCSEC)
        ax.imshow(crop, origin="lower")
        ax.set_title(tname, color="white", fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.patch.set_facecolor("black")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, facecolor="black", bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
