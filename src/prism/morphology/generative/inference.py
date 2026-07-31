"""Phase 9 inference: sample the trained conditional VAE and package the
result as lenstronomy INTERPOL kwargs, matching the shape returned by
`tng_particle_light.build_tng_particle_interpol_kwargs` /
`galaxygenius_stamps` builders (`image, center_x, center_y, phi_G, scale,
magnitude`).

This is a single-band model: the generated 64x64 image is shared across
bands (like the Sersic geometry it substitutes for), with `magnitude` set
directly from the caller's per-band, K-corrected `magnitude_ref` -- color
variation is left to the caller's existing per-band magnitude logic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from prism.morphology.generative.dataset import COND_DIM, STAMP_SIZE, condition_vector
from prism.morphology.generative.model import ConditionalVAE

DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints" / "cvae_v2.pt"

_MODEL_CACHE: dict[str, ConditionalVAE] = {}


def _load_model(checkpoint_path) -> ConditionalVAE:
    key = str(checkpoint_path)
    if key not in _MODEL_CACHE:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        latent_dim = ckpt.get("latent_dim", 32)
        model = ConditionalVAE(cond_dim=ckpt.get("cond_dim", COND_DIM), latent_dim=latent_dim)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def is_available(checkpoint_path=DEFAULT_CHECKPOINT) -> bool:
    return Path(checkpoint_path).exists()


def build_generative_interpol_kwargs(
    morph_type: str,
    logM: float,
    redshift: float,
    magnitude_ref: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    phi_G: float = 0.0,
    target_size_arcsec: float = 1.2,
    rng=None,
    checkpoint_path=DEFAULT_CHECKPOINT,
) -> dict:
    """Sample the conditional VAE and return INTERPOL kwargs.

    Args:
        morph_type: one of `dataset.MORPH_CLASSES` ("passive",
            "star_forming", "dusty_starburst"); unrecognized values get an
            all-zero morph one-hot (network falls back to its prior).
        logM: log10(stellar mass / Msun).
        redshift: galaxy redshift (bucketed per `dataset.redshift_bucket`).
        magnitude_ref: target AB magnitude for the rendered band.
        center_x, center_y, phi_G: placement (arcsec, arcsec, radians).
        target_size_arcsec: full angular width of the rendered stamp.
        rng: optional `np.random.Generator` for the VAE's latent sample.
        checkpoint_path: trained checkpoint (see `train.py`).

    Returns:
        dict with keys `image, center_x, center_y, phi_G, scale, magnitude`.
    """
    model = _load_model(checkpoint_path)

    if rng is None:
        rng = np.random.default_rng()

    cond = condition_vector(morph_type, logM, redshift)
    cond_t = torch.from_numpy(cond[None, :])

    with torch.no_grad():
        z = torch.from_numpy(rng.standard_normal(model.latent_dim).astype(np.float32))[None, :]
        image = model.decoder(z, cond_t)[0, 0].numpy()

    image = np.clip(image, 0.0, None)

    # Circular apodization: apply a soft circular window so the stamp has no
    # rectangular boundary discontinuity when convolved with the PSF.
    # Rectangular tapers leave non-zero corners that produce Fourier ringing
    # (star-with-spikes artifact) in the wide PSF of F277W/F444W.
    npix = image.shape[-1]
    cx, cy = (npix - 1) / 2.0, (npix - 1) / 2.0
    yy, xx = np.mgrid[:npix, :npix]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_inner = npix * 0.32   # full weight inside 32% of half-width
    r_outer = npix * 0.46   # zero weight outside 46% of half-width
    circ_window = np.clip((r_outer - r) / (r_outer - r_inner), 0.0, 1.0).astype(np.float32)
    circ_window = 0.5 * (1 - np.cos(np.pi * circ_window))  # smooth transition
    image = image * circ_window

    # Enforce a minimum physical size so the stamp is always resolved (≥ PSF FWHM
    # at F277W ≈ 0.09"). Sub-PSF stamps produce point-source PSF spike artifacts.
    MIN_SIZE_ARCSEC = 0.18
    target_size_arcsec = max(float(target_size_arcsec), MIN_SIZE_ARCSEC)

    scale = float(target_size_arcsec) / npix
    image_sb = image / scale ** 2

    return dict(
        image=image_sb,
        center_x=float(center_x),
        center_y=float(center_y),
        phi_G=float(phi_G),
        scale=float(scale),
        magnitude=float(magnitude_ref),
    )
