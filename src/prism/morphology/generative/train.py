#!/usr/bin/env python3
"""Train the Phase 9 conditional VAE on locally-available TNG particle stamps.

Example:
    python -m src.galaxy_morphology.generative.train \\
        --catalog /Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet \\
        --epochs 30 --out src/galaxy_morphology/generative/checkpoints/cvae.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from prism.morphology.generative.dataset import COND_DIM, STAMP_SIZE, build_dataset
from prism.morphology.generative.model import ConditionalVAE, vae_loss

DEFAULT_CATALOG = "/Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", default=DEFAULT_CATALOG)
    p.add_argument("--catalog2", default=None,
                   help="Optional second catalog parquet (e.g. TNG50-1) merged before training.")
    p.add_argument("--band", default="F150W")
    p.add_argument("--bands", nargs="+", default=None,
                   help="Multi-band augmentation: one stamp per cutout per band "
                        "(default: single --band).")
    p.add_argument("--n-projections", type=int, default=1,
                   help="Random viewing projections (inclination+PA) per cutout "
                        "(multi-orientation augmentation).")
    p.add_argument("--max-n", type=int, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.5,
                   help="Target (final) KL weight after annealing.")
    p.add_argument("--beta-start", type=float, default=0.0,
                   help="Initial KL weight at epoch 1 (KL annealing).")
    p.add_argument("--anneal-epochs", type=int, default=None,
                   help="Number of epochs to linearly ramp beta from "
                        "--beta-start to --beta (default: all epochs).")
    p.add_argument("--latent-dim", type=int, default=64,
                   help="VAE latent dimension (default: 64).")
    p.add_argument("--free-bits", type=float, default=0.5,
                   help="Per-dimension KL floor (nats) to prevent posterior "
                        "collapse; 0 disables.")
    p.add_argument("--no-flip-augment", action="store_true",
                   help="Disable ×4 flip augmentation (lr, ud, both). On by default.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(Path(__file__).parent / "checkpoints" / "cvae.pt"))
    args = p.parse_args()

    extra_catalogs = [args.catalog2] if args.catalog2 else None
    print(f"Building dataset from {args.catalog} "
          f"(bands={args.bands or [args.band]}, n_projections={args.n_projections}"
          f"{', catalog2=' + args.catalog2 if args.catalog2 else ''}) ...")
    images, conditions, meta = build_dataset(
        args.catalog, band=args.band, max_n=args.max_n, seed=args.seed,
        bands=args.bands, n_projections=args.n_projections,
        extra_catalog_paths=extra_catalogs,
        flip_augment=not args.no_flip_augment,
    )
    print(f"Dataset: {images.shape[0]} stamps, shape {images.shape[1:]}, cond_dim={conditions.shape[1]}")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    ds = TensorDataset(torch.from_numpy(images), torch.from_numpy(conditions))
    # drop_last avoids a trailing batch of size 1, which BatchNorm can't
    # handle in train mode.
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = ConditionalVAE(cond_dim=COND_DIM, latent_dim=args.latent_dim).to(device)
    print(f"Model: latent_dim={args.latent_dim}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    anneal_epochs = args.anneal_epochs if args.anneal_epochs is not None else args.epochs

    history = []
    for epoch in range(args.epochs):
        if anneal_epochs > 1:
            frac = min(1.0, epoch / (anneal_epochs - 1))
        else:
            frac = 1.0
        beta = args.beta_start + frac * (args.beta - args.beta_start)

        total, total_recon, total_kld, n = 0.0, 0.0, 0.0, 0
        for x, c in dl:
            x, c = x.to(device), c.to(device)
            recon, mu, logvar = model(x, c)
            loss, recon_loss, kld = vae_loss(recon, x, mu, logvar, beta=beta, free_bits=args.free_bits)

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = x.shape[0]
            total += loss.item() * bs
            total_recon += recon_loss.item() * bs
            total_kld += kld.item() * bs
            n += bs

        avg, avg_recon, avg_kld = total / n, total_recon / n, total_kld / n
        history.append(avg)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  beta={beta:.4f}  loss={avg:.5f}  recon={avg_recon:.5f}  kld={avg_kld:.5f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "cond_dim": COND_DIM,
        "stamp_size": STAMP_SIZE,
        "latent_dim": args.latent_dim,
        "band": args.band,
        "loss_history": history,
        "beta_start": args.beta_start,
        "beta_end": args.beta,
        "anneal_epochs": anneal_epochs,
        "free_bits": args.free_bits,
    }, out_path)
    print(f"Saved checkpoint -> {out_path}")


if __name__ == "__main__":
    main()
