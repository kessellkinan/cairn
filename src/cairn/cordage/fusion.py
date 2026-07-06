"""Encoder bank and cross-modal attention fusion.

Ref: Methods Sec. 4.4, Eq. (1)-(2). Absent modalities fall back to a learned
default latent (graceful degradation). The fused latent initializes the
physics-constrained dynamics.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple

import torch
from torch import nn

from cairn.cordage.vae import ModalityVAE, gaussian_kl
from cairn.field.modalities import ORDER, Modality
from cairn.kit.wiring import AblationFlags, DataConfig, ModelConfig


class FusionOutput(NamedTuple):
    latent: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor
    reconstruction_loss: torch.Tensor
    kl: torch.Tensor


class EncoderBank(nn.Module):
    def __init__(
        self, data: DataConfig, model: ModelConfig, ablation: AblationFlags
    ) -> None:
        super().__init__()
        self._modalities = list(ORDER)
        self._per = model.latent_dim // len(ORDER)
        self._ablation = ablation
        self.vaes = nn.ModuleDict(
            {
                m.name: ModalityVAE(spec.keep, self._per, model.encoder_hidden)
                for m, spec in zip(ORDER, data.modalities)
            }
        )
        self.defaults = nn.Parameter(torch.zeros(len(ORDER), self._per))
        nn.init.normal_(self.defaults, std=0.02)
        if ablation.cross_modal_attention and ablation.fusion:
            self.attention: nn.MultiheadAttention = nn.MultiheadAttention(
                self._per, model.heads, batch_first=True
            )
        self.residual = nn.Linear(model.latent_dim, model.latent_dim)
        self.single = nn.Linear(self._per, model.latent_dim)
        self.posterior_logvar = nn.Linear(model.latent_dim, model.latent_dim)

    def _encode(
        self, features: Dict[str, torch.Tensor], present: torch.Tensor
    ) -> tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        latents: List[torch.Tensor] = []
        anchor = features[Modality.TRANSCRIPTOMICS.name]
        recon = anchor.new_zeros(anchor.shape[0])
        kl = anchor.new_zeros(anchor.shape[0])
        for idx, modality in enumerate(self._modalities):
            x = features[modality.name]
            out = self.vaes[modality.name](x)
            weight = present[:, idx].to(x.dtype)
            mse = ((out.reconstruction - x) ** 2).flatten(1).sum(dim=-1)
            recon = recon + mse * weight
            kl = kl + gaussian_kl(out.mean, out.logvar) * weight
            mask = present[:, idx].view(-1, *([1] * (out.latent.dim() - 1))).bool()
            default = self.defaults[idx].expand_as(out.latent)
            latents.append(torch.where(mask, out.latent, default))
        return latents, recon, kl

    def forward(
        self, features: Dict[str, torch.Tensor], present: torch.Tensor
    ) -> FusionOutput:
        latents, recon, kl = self._encode(features, present)
        concatenated = torch.cat(latents, dim=-1)
        if not self._ablation.fusion:
            mean = self.single(latents[0])
        elif self._ablation.cross_modal_attention:
            tokens = torch.stack(latents, dim=-2)
            attended, _ = self.attention(tokens, tokens, tokens)
            mean = attended.reshape(*concatenated.shape) + self.residual(concatenated)
        else:
            mean = self.residual(concatenated)
        logvar = torch.clamp(self.posterior_logvar(mean), -8.0, 8.0)
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std) if self.training else torch.zeros_like(std)
        return FusionOutput(
            latent=mean + noise * std,
            mean=mean,
            logvar=logvar,
            reconstruction_loss=recon,
            kl=kl,
        )
