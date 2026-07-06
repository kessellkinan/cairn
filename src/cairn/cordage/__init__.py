"""Modality encoders and cross-modal fusion."""

from cairn.cordage.fusion import EncoderBank, FusionOutput
from cairn.cordage.vae import ModalityVAE, VAEOutput, gaussian_kl

__all__ = [
    "EncoderBank",
    "FusionOutput",
    "ModalityVAE",
    "VAEOutput",
    "gaussian_kl",
]
