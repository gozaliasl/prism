"""Small conditional VAE for 64x64 grayscale morphology stamps (Phase 9).

Conv encoder/decoder, latent dim 32, condition vector
(`dataset.COND_DIM`) concatenated at the encoder bottleneck and decoder
input. CPU/MPS-trainable in reasonable time on a few thousand 64x64 stamps.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 32


class _Encoder(nn.Module):
    def __init__(self, cond_dim: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        # BatchNorm after each conv stabilizes the encoder's logvar output
        # early in training, which otherwise tends to collapse to ~0
        # (posterior == prior) before the decoder learns to use z.
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),    # 64 -> 32
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),   # 32 -> 16
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 16 -> 8
            nn.Conv2d(128, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 8 -> 4
        )
        self.fc_mu = nn.Linear(128 * 4 * 4 + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4 + cond_dim, latent_dim)

    def forward(self, x, c):
        h = self.conv(x).flatten(1)
        h = torch.cat([h, c], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    def __init__(self, cond_dim: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        # Halved channel widths vs. the original (256-base) decoder: a
        # lower-capacity decoder can't reconstruct the per-condition "mean
        # blob" from the condition vector alone, so it has more incentive
        # to actually use the latent code -- another standard collapse
        # mitigation alongside KL annealing / free bits.
        self.fc = nn.Linear(latent_dim + cond_dim, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),  # 4 -> 8
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),   # 8 -> 16
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),   # 16 -> 32
            nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid(),                                 # 32 -> 64
        )

    def forward(self, z, c):
        h = self.fc(torch.cat([z, c], dim=1))
        h = h.view(-1, 128, 4, 4)
        return self.deconv(h)


class ConditionalVAE(nn.Module):
    def __init__(self, cond_dim: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = _Encoder(cond_dim, latent_dim)
        self.decoder = _Decoder(cond_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, c)
        return recon, mu, logvar

    def sample(self, c):
        device = next(self.parameters()).device
        z = torch.randn(c.shape[0], self.latent_dim, device=device)
        return self.decoder(z, c)


def vae_loss(recon, x, mu, logvar, beta: float = 0.5, free_bits: float = 0.0):
    """Reconstruction + beta-weighted KL, with an optional free-bits floor.

    ``free_bits`` (nats per latent dimension) clamps each dimension's KL term
    from below before summing, so the optimizer has no gradient to push an
    already-collapsed dimension (KL < free_bits) further toward zero --
    a standard fix for posterior collapse (Kingma et al. 2016).
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld = torch.mean(torch.sum(kld_per_dim, dim=1))
    if free_bits > 0:
        kld_clamped = torch.clamp(kld_per_dim, min=free_bits)
        kld_for_loss = torch.mean(torch.sum(kld_clamped, dim=1))
    else:
        kld_for_loss = kld
    return recon_loss + beta * kld_for_loss, recon_loss, kld
