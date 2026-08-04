"""Grid figure: rows = selected lens IDs, columns = telescopes.
Each cell is an RGB composite built from that telescope's own 4 bands,
using a per-telescope (not per-cell) fixed stretch so brightness differences
across telescopes remain visually meaningful.

Run: python scripts/local/build_lens_telescope_grid.py [small|1arcmin|both]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/Volumes/exthd-prism/prism-lensing")

TEL_DIRS = {
    "jwst": "jwst", "roman": "roman", "euclid": "euclid_matched",
    "subaru": "subaru", "lsst": "lsst",
}
TEL_LABELS = {"jwst": "JWST", "roman": "Roman", "euclid": "Euclid",
              "subaru": "Subaru", "lsst": "Rubin"}

LENS_IDS = [2, 4, 8, 12, 16, 19, 21, 33]  # 36 excluded per user request


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


def compute_stretch(tel_dir: Path, lens_ids, black_pct=50.0, vmax_pct=99.7, softening_frac=0.08):
    pooled = {}
    bands = None
    for lid in lens_ids:
        im, bands = load(tel_dir, lid)
        for bi, b in enumerate(bands):
            pooled.setdefault(b, []).append(im[bi].ravel())
    stretch = {}
    for b in bands:
        allpix = np.concatenate(pooled[b])
        black = np.percentile(allpix, black_pct)
        sub = np.clip(allpix - black, 0, None)
        vmax = np.percentile(sub, vmax_pct)
        if vmax <= 0:
            vmax = sub.max() or 1.0
        stretch[b] = dict(black=float(black), vmax=float(vmax), soft=float(vmax * softening_frac))
    return stretch, bands


def apply_rgb(im, bands, stretch):
    def ch(bi):
        b = bands[bi]
        s = stretch[b]
        sub = np.clip(im[bi].astype(np.float64) - s["black"], 0, None)
        return np.clip(np.arcsinh(sub / s["soft"]) / np.arcsinh(s["vmax"] / s["soft"]), 0, 1)

    n = len(bands)
    r = ch(n - 1)
    g = 0.5 * (ch(n - 2) + ch(1)) if n >= 4 else ch(min(1, n - 1))
    b = ch(0)
    return np.stack([r, g, b], axis=-1)


def build(size: str):
    tel_dirs = {TEL_LABELS[k]: find_dir(k, size) for k in TEL_DIRS}
    out_path = BASE / f"outputs_test/grid_8lenses_5telescopes_{size}.png"

    nrows, ncols = len(LENS_IDS), len(tel_dirs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))

    tel_stretch = {}
    for tname, tdir in tel_dirs.items():
        stretch, bands = compute_stretch(tdir, LENS_IDS)
        tel_stretch[tname] = (stretch, bands)
        print(f"[{size}] {tname}: dir={tdir.name} bands={bands}")

    for i, lid in enumerate(LENS_IDS):
        for j, (tname, tdir) in enumerate(tel_dirs.items()):
            ax = axes[i, j]
            im, bands = load(tdir, lid)
            stretch, _ = tel_stretch[tname]
            rgb = apply_rgb(im, bands, stretch)
            ax.imshow(rgb, origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(tname, fontsize=12)
            if j == 0:
                ax.set_ylabel(f"lens {lid}", fontsize=10)

    fov_label = "~small FOV (per-telescope native)" if size == "small" else "1 arcmin FOV"
    fig.suptitle(f"Same 8 lens systems across 5 telescopes ({fov_label}, shared real "
                  "COSMOS-Web field, N-exposure-stacked detector chain)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    sizes = ["small", "1arcmin"] if which == "both" else [which]
    for size in sizes:
        build(size)


if __name__ == "__main__":
    main()
