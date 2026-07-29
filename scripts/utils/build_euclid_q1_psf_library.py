#!/usr/bin/env python3
"""
Build a curated empirical Euclid Q1 PSF library from external cutouts.

Extracts VIS/Y/J/H PSF stamps from lens.zip, pads to 101×101, and writes
kernels under data/euclid_q1_psf/tiles/ in the same layout as psf_v5_30mas.

Source data (default): /Volumes/extHD/Euclid_lens_q1/lens.zip

Usage:
    python scripts/utils/build_euclid_q1_psf_library.py
    python scripts/utils/build_euclid_q1_psf_library.py --n-tiles 50
    python scripts/utils/build_euclid_q1_psf_library.py --source /path/to/lens.zip
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

BANDS = {
    'EUCLID_VIS': 'VIS_PSF',
    'EUCLID_Y': 'NIR_Y_PSF',
    'EUCLID_J': 'NIR_J_PSF',
    'EUCLID_H': 'NIR_H_PSF',
}
KERNEL_SIZE = 101
PIXEL_SCALE = 0.10


def psf_fwhm_pix(arr: np.ndarray) -> float:
    d = np.maximum(arr.astype(float), 0)
    if d.sum() <= 0:
        return float('nan')
    w = d / d.sum()
    ny, nx = w.shape
    y, x = np.mgrid[:ny, :nx]
    xcm = (w * x).sum()
    ycm = (w * y).sum()
    var = (w * ((x - xcm) ** 2 + (y - ycm) ** 2)).sum()
    return float(2.355 * np.sqrt(var))


def pad_kernel(arr: np.ndarray, size: int = KERNEL_SIZE) -> np.ndarray:
    arr = np.maximum(arr.astype(np.float64), 0)
    if arr.sum() > 0:
        arr = arr / arr.sum()
    ny, nx = arr.shape
    out = np.zeros((size, size), dtype=np.float64)
    y0 = (size - ny) // 2
    x0 = (size - nx) // 2
    out[y0:y0 + ny, x0:x0 + nx] = arr
    if out.sum() > 0:
        out /= out.sum()
    return out


def select_systems(src_dir: Path, n_tiles: int) -> pd.DataFrame:
    disc = pd.read_csv(src_dir / 'q1_discovery_engine_lens_catalog.csv')
    mass = pd.read_csv(src_dir / 'modeling_lens_mass.csv')

    records = []
    with zipfile.ZipFile(src_dir / 'lens.zip') as z:
        fits_names = sorted(
            n for n in z.namelist()
            if n.endswith('.fits') and n.count('/') == 2 and 'mask' not in n
        )
        for fn in fits_names:
            id_str = fn.split('/')[1]
            tile = id_str.split('_')[0]
            with fits.open(io.BytesIO(z.read(fn))) as hdul:
                fwhm_vis = psf_fwhm_pix(hdul['VIS_PSF'].data)
            grade_row = disc.loc[disc.id_str == id_str, 'grade']
            grade = grade_row.iloc[0] if len(grade_row) else 'unknown'
            te_row = mass.loc[mass.id_str == id_str, 'einstein_radius_median_pdf']
            theta_e = float(te_row.iloc[0]) if len(te_row) and pd.notna(te_row.iloc[0]) else np.nan
            records.append({
                'id_str': id_str,
                'tile_index': tile,
                'grade': grade,
                'vis_fwhm_pix': fwhm_vis,
                'vis_fwhm_arcsec': fwhm_vis * PIXEL_SCALE,
                'theta_E': theta_e,
                'fits_path': fn,
            })

    df = pd.DataFrame(records)
    candidates = df[df.grade.isin(['A', 'B'])].copy()
    if len(candidates) < n_tiles:
        candidates = df.copy()

    candidates['fwhm_bin'] = pd.qcut(candidates.vis_fwhm_pix, q=5, duplicates='drop')
    selected = []
    used_tiles: set[str] = set()
    for _, grp in candidates.groupby('fwhm_bin', observed=True):
        for _, row in grp.sort_values('grade').iterrows():
            if row.tile_index in used_tiles:
                continue
            selected.append(row)
            used_tiles.add(row.tile_index)
            break

    selected_ids = {r.id_str for r in selected}
    for _, row in candidates.sort_values(['grade', 'vis_fwhm_pix']).iterrows():
        if len(selected) >= n_tiles:
            break
        if row.id_str in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(row.id_str)

    return pd.DataFrame(selected).reset_index(drop=True)


def build_library(src_dir: Path, out_dir: Path, n_tiles: int) -> pd.DataFrame:
    sel_df = select_systems(src_dir, n_tiles)
    tiles_dir = out_dir / 'tiles'
    tiles_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    with zipfile.ZipFile(src_dir / 'lens.zip') as z:
        for _, row in sel_df.iterrows():
            suffix = row.id_str.split('_', 1)[1][:12]
            tile_name = f"Q1_{row.tile_index}_{suffix}"
            tile_path = tiles_dir / tile_name
            tile_path.mkdir(exist_ok=True)

            with fits.open(io.BytesIO(z.read(row.fits_path))) as hdul:
                for band, hdu_name in BANDS.items():
                    kernel = pad_kernel(hdul[hdu_name].data, KERNEL_SIZE)
                    fits.writeto(
                        tile_path / f'{band}_kernel.fits',
                        kernel.astype(np.float32),
                        overwrite=True,
                    )
                    fwhm_padded = psf_fwhm_pix(kernel)

            meta_rows.append({
                'tile': tile_name,
                'id_str': row.id_str,
                'tile_index': row.tile_index,
                'grade': row.grade,
                'vis_fwhm_pix_native': row.vis_fwhm_pix,
                'vis_fwhm_arcsec_native': row.vis_fwhm_arcsec,
                'vis_fwhm_pix_padded': fwhm_padded,
                'theta_E_arcsec': row.theta_E,
                'kernel_size': KERNEL_SIZE,
                'pixel_scale_arcsec': PIXEL_SCALE,
                'source': str(src_dir / 'lens.zip'),
            })

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(out_dir / 'psf_catalog.csv', index=False)
    print(f"Wrote {len(meta)} tiles ({len(meta) * 4} kernels) → {out_dir}")
    return meta


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/Volumes/extHD/Euclid_lens_q1'))
    parser.add_argument('--output', type=Path, default=repo / 'data' / 'euclid_q1_psf')
    parser.add_argument('--n-tiles', type=int, default=30)
    args = parser.parse_args()
    build_library(args.source, args.output, args.n_tiles)


if __name__ == '__main__':
    main()
