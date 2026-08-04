"""Per-lens figure: big 1-arcmin panel on the left (JWST, reference full-field
view), and a stacked column on the right with 5 zoomed-in crops centered on
the lens (JWST -> Roman -> Euclid -> Subaru -> Rubin, top to bottom), one
figure per selected lens.

Run: python scripts/local/build_lens_zoom_panels.py
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
    "JWST": "jwst", "Roman": "roman", "Euclid": "euclid_matched",
    "Subaru": "subaru", "Rubin": "lsst",
}
TEL_ORDER = ["JWST", "Roman", "Euclid", "Subaru", "Rubin"]
PIXEL_SCALES = {"JWST": 0.031, "Roman": 0.11, "Euclid": 0.10, "Subaru": 0.168, "Rubin": 0.20}

LENS_IDS = [2, 4, 8, 12, 16, 19, 21, 33]
ZOOM_ARCSEC = 9.0  # crop half-width*2 centered on the lens for the zoom column

OUT_DIR = BASE / "outputs_test/lens_zoom_panels"


def find_dir(tel_key: str, size: str) -> Path:
    suffix = "" if size == "small" else "_1arcmin"
    pattern = f"multi_tel_deep_40_{TEL_DIRS[tel_key]}{suffix}_date_*"
    matches = sorted(BASE.glob(f"outputs_test/{pattern}"))
    if not matches:
        raise FileNotFoundError(f"No output dir matching {pattern}")
    return matches[-1]


def npz_path(tel_dir: Path, lens_id: int) -> Path:
    return tel_dir / "unified_npz" / f"PRISM_lens_SF_{lens_id:06d}.npz"


def load(tel_dir: Path, lens_id: int):
    d = np.load(npz_path(tel_dir, lens_id), allow_pickle=True)
    im = d["image_final"]
    meta = json.loads(str(d["metadata"]))
    return im, meta["bands"]


def stretch_rgb(im, bands, black_pct=50.0, vmax_pct=99.7, softening_frac=0.08):
    def ch(bi):
        b = bands[bi]
        x = im[bi].astype(np.float64)
        black = np.percentile(x, black_pct)
        sub = np.clip(x - black, 0, None)
        vmax = np.percentile(sub, vmax_pct)
        if vmax <= 0:
            vmax = sub.max() or 1.0
        soft = vmax * softening_frac
        return np.clip(np.arcsinh(sub / soft) / np.arcsinh(vmax / soft), 0, 1)

    n = len(bands)
    r = ch(n - 1)
    g = 0.5 * (ch(n - 2) + ch(1)) if n >= 4 else ch(min(1, n - 1))
    b = ch(0)
    return np.stack([r, g, b], axis=-1)


def center_crop(rgb, pixel_scale, half_width_arcsec):
    h, w = rgb.shape[:2]
    cy, cx = h // 2, w // 2
    half_px = max(1, int(round(half_width_arcsec / pixel_scale)))
    y0, y1 = max(0, cy - half_px), min(h, cy + half_px)
    x0, x1 = max(0, cx - half_px), min(w, cx + half_px)
    return rgb[y0:y1, x0:x1]


def build_lens_figure(lens_id: int):
    jwst_dir_1am = find_dir("JWST", "1arcmin")
    im_big, bands_big = load(jwst_dir_1am, lens_id)
    rgb_big = stretch_rgb(im_big, bands_big)

    zoom_rgbs = []
    for tname in TEL_ORDER:
        # Only JWST needs the 1-arcmin render (used above for the big panel);
        # the other 4 telescopes only need their small/native-FOV render for
        # the zoom-in crop, since we don't render 1-arcmin for them.
        tdir = find_dir(tname, "1arcmin" if tname == "JWST" else "small")
        im, bands = load(tdir, lens_id)
        rgb = stretch_rgb(im, bands)
        crop = center_crop(rgb, PIXEL_SCALES[tname], ZOOM_ARCSEC / 2.0)
        zoom_rgbs.append((tname, crop))

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(5, 2, width_ratios=[2.2, 1], wspace=0.08, hspace=0.12)

    ax_big = fig.add_subplot(gs[:, 0])
    ax_big.imshow(rgb_big, origin="lower")
    ax_big.set_xticks([]); ax_big.set_yticks([])
    ax_big.set_title(f"lens {lens_id} — JWST, 1 arcmin FOV", fontsize=12)

    for i, (tname, crop) in enumerate(zoom_rgbs):
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(crop, origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(tname, fontsize=11, rotation=0, labelpad=35, va="center")

    fig.suptitle(f"Lens {lens_id}: 1 arcmin field (left) + {ZOOM_ARCSEC:.0f}\" zoom on lens, "
                 "all 5 telescopes (right, top to bottom)", fontsize=11)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    out_path = OUT_DIR / f"lens_{lens_id:06d}_zoom_panel.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    for lid in LENS_IDS:
        build_lens_figure(lid)


if __name__ == "__main__":
    main()
