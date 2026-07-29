"""Training-set assembly for the Phase 9 conditional VAE.

Builds 64x64 normalized grayscale stamps + condition vectors (morph class
one-hot, normalized log10(M*), redshift-bucket one-hot) from all locally
available TNG particle cutouts
(`src.tng_galaxy_selector.local_particle_path`), via
`galaxy_morphology.tng_particle_light._band_images`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from src.tng_galaxy_selector import load_local_catalog, local_particle_path
from src.galaxy_morphology.tng_particle_light import _band_images

STAMP_SIZE = 64

# Mirrors jwst_lens_simulator.tng_sed_galaxy_type's sSFR thresholds.
TNG_QUENCHED_SSFR_THRESHOLD = 1e-11
TNG_STARBURST_SSFR_THRESHOLD = 1e-9

MORPH_CLASSES = ["passive", "star_forming", "dusty_starburst"]
LOGM_MIN, LOGM_MAX = 8.0, 12.0
Z_BUCKET_EDGES = [0.0, 0.5, 1.0, 2.0, np.inf]
N_Z_BUCKETS = len(Z_BUCKET_EDGES) - 1

# morph one-hot + normalized log10(M*) + redshift-bucket one-hot
COND_DIM = len(MORPH_CLASSES) + 1 + N_Z_BUCKETS


def sed_galaxy_type(ssfr_per_yr: float) -> str:
    if ssfr_per_yr < TNG_QUENCHED_SSFR_THRESHOLD:
        return "passive"
    if ssfr_per_yr > TNG_STARBURST_SSFR_THRESHOLD:
        return "dusty_starburst"
    return "star_forming"


def redshift_bucket(redshift: float) -> int:
    return int(np.clip(np.digitize(redshift, Z_BUCKET_EDGES) - 1, 0, N_Z_BUCKETS - 1))


def condition_vector(morph_type: str, logM: float, redshift: float) -> np.ndarray:
    """Build the (COND_DIM,) condition vector used by the encoder/decoder."""
    vec = np.zeros(COND_DIM, dtype=np.float32)
    if morph_type in MORPH_CLASSES:
        vec[MORPH_CLASSES.index(morph_type)] = 1.0
    logm_norm = (np.clip(logM, LOGM_MIN, LOGM_MAX) - LOGM_MIN) / (LOGM_MAX - LOGM_MIN)
    vec[len(MORPH_CLASSES)] = logm_norm
    vec[len(MORPH_CLASSES) + 1 + redshift_bucket(redshift)] = 1.0
    return vec


def _resize_to_stamp(image: np.ndarray, size: int = STAMP_SIZE) -> np.ndarray:
    factor = size / image.shape[0]
    out = zoom(image, factor, order=1)
    out = out[:size, :size]
    if out.shape != (size, size):
        pad = ((0, size - out.shape[0]), (0, size - out.shape[1]))
        out = np.pad(out, pad)
    return out


def _normalize_stamp(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, None)
    positive = image[image > 0]
    if positive.size:
        image = np.log1p(image / (np.median(positive) + 1e-12))
    maxv = float(image.max())
    if maxv > 0:
        image = image / maxv
    return image.astype(np.float32)


def build_dataset(catalog_path, band: str = "F150W", max_n: int | None = None, seed: int = 42,
                   bands: list[str] | None = None, n_projections: int = 1,
                   extra_catalog_paths: list | None = None,
                   flip_augment: bool = True):
    """Assemble (images, conditions, meta) from local TNG particle cutouts.

    Args:
        catalog_path: path to the primary local catalog parquet (e.g. TNG100-1).
        band: JWST band used for the (single-band) training stamps when
            ``bands`` is not given.
        max_n: optional cap on the number of subhalos used.
        seed: RNG seed for subsampling and per-subhalo viewing projections.
        bands: optional list of JWST bands -- each cutout contributes one
            stamp per band (multi-band augmentation), all sharing the same
            projection(s). Defaults to ``[band]``.
        n_projections: number of independent random viewing projections
            (inclination + position angle) rendered per cutout
            (multi-orientation augmentation). Each projection is rendered
            fresh (``use_cache=False``) so orientations differ.
        extra_catalog_paths: optional list of additional catalog parquet paths
            (e.g. TNG50-1) whose rows are merged with the primary catalog
            before building stamps. The ``sim`` column of each row is used to
            resolve the correct local particle path.
        flip_augment: if True (default), each stamp is augmented with 3
            additional flips (horizontal, vertical, both), giving ×4 stamps
            per projection per band.

    Returns:
        images: float32 array, shape (N, 1, STAMP_SIZE, STAMP_SIZE), in [0, 1]
        conditions: float32 array, shape (N, COND_DIM)
        meta: list of dicts with snapshot/subhalo_id/morph_type/logM/redshift/projection
    """
    import pandas as pd

    cat = load_local_catalog(catalog_path)
    if extra_catalog_paths:
        parts = [cat]
        for ep in extra_catalog_paths:
            parts.append(load_local_catalog(ep))
        cat = pd.concat(parts, ignore_index=True)

    rng = np.random.default_rng(seed)
    bands = bands if bands is not None else [band]

    rows = []
    for _, row in cat.iterrows():
        sim_raw = row["sim"] if "sim" in row.index else None
        sim = str(sim_raw) if sim_raw and str(sim_raw) != "nan" else "TNG100-1"
        if local_particle_path(int(row["snapshot"]), int(row["subhalo_id"]), sim=sim) is not None:
            rows.append(row)

    print(f"  Found {len(rows)} cutouts with local particle files.")

    if max_n is not None and len(rows) > max_n:
        idx = rng.choice(len(rows), size=max_n, replace=False)
        rows = [rows[i] for i in idx]

    images, conditions, meta = [], [], []
    n_rows = len(rows)
    for _ri, row in enumerate(rows):
        if _ri % 500 == 0:
            print(f"  Rendering stamps {_ri}/{n_rows} ({100*_ri//n_rows}%) ...")
        snap, sub = int(row["snapshot"]), int(row["subhalo_id"])
        sim_raw = row["sim"] if "sim" in row.index else None
        sim = str(sim_raw) if sim_raw and str(sim_raw) != "nan" else "TNG100-1"
        path = local_particle_path(snap, sub, sim=sim)
        morph_type = sed_galaxy_type(float(row["ssfr_per_yr"]))
        logM = float(row["stellar_mass_logmsun"])
        redshift = float(row["snapshot_redshift"])
        cond = condition_vector(morph_type, logM, redshift)

        for proj_idx in range(n_projections):
            use_cache = (n_projections == 1)
            try:
                band_images = _band_images(Path(path), float(row["halfmassrad_stars_kpc"]), rng,
                                            use_cache=use_cache)
            except Exception:
                continue

            for b in bands:
                if b not in band_images:
                    continue
                stamp = _normalize_stamp(_resize_to_stamp(band_images[b]))
                # Base stamp + 3 flip variants (lr, ud, both) = ×4 augmentation
                variants = [stamp]
                if flip_augment:
                    variants += [
                        np.fliplr(stamp),
                        np.flipud(stamp),
                        np.fliplr(np.flipud(stamp)),
                    ]
                for flip_idx, s in enumerate(variants):
                    images.append(s)
                    conditions.append(cond)
                    meta.append(dict(snapshot=snap, subhalo_id=sub, sim=sim,
                                      morph_type=morph_type, logM=logM, redshift=redshift,
                                      band=b, projection=proj_idx, flip=flip_idx))

    images_arr = np.stack(images)[:, None, :, :].astype(np.float32)
    conditions_arr = np.stack(conditions).astype(np.float32)
    return images_arr, conditions_arr, meta
