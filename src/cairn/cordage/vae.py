"""Modality-specific variational autoencoders.

Ref: Methods Sec. 4.4, Eq. (1) reconstruction objective with beta-weighted KL.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class VAEOutput(NamedTuple):
    latent: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor
    reconstruction: torch.Tensor


class ModalityVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, input_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> VAEOutput:
        stats = self.encoder(x)
        mean, logvar = stats[..., : self.latent_dim], stats[..., self.latent_dim :]
        logvar = torch.clamp(logvar, -8.0, 8.0)
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std) if self.training else torch.zeros_like(std)
        latent = mean + noise * std
        reconstruction = self.decoder(latent)
        return VAEOutput(
            latent=latent, mean=mean, logvar=logvar, reconstruction=reconstruction
        )


def gaussian_kl(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1.0 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
