#!/usr/bin/env python
"""LightningModule + LightningCLI entry point for multimodal fusion training.

Run with:
    python trainer.py fit --config config.yaml
"""

import sys
from pathlib import Path
from typing import Optional

import lightning as L
import torch
from lightning.pytorch.cli import LightningCLI
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent))

from model.fusion import MultimodalFusion
from data.datamodule import FusionDataModule


class MultimodalFusionModule(L.LightningModule):
    """Trains per-modality projectors via pairwise InfoNCE contrastive loss.

    The underlying encoders (image, spectrum, …) are **not** stored or trained
    here — this module receives precomputed CLS embeddings from the dataloader
    and only trains the lightweight projectors in MultimodalFusion.

    Modality registration
    ---------------------
    Pass a ``modalities`` dict in the constructor (or via the YAML config)::

        modalities:
          image:
            input_dim: 768
            hidden_dim: 512
          spectrum:
            input_dim: 128
            hidden_dim: 256

    Adding a new modality later (no architecture changes needed)::

        modalities:
          photometry:
            input_dim: 64
            hidden_dim: null  # → single linear layer

    Args:
        latent_dim:    Shared embedding dimension D (default 256).
        temperature:   InfoNCE softmax temperature (default 0.07).
        lr:            Base learning rate for AdamW (overridden per-step by
                       WarmupCosineLR callback when used).
        weight_decay:  AdamW weight decay.
        modalities:    ``{name: {input_dim: int, hidden_dim: int|null}}`` dict
                       used to register modality projectors at construction time.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        temperature: float = 0.07,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        modalities: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.fusion = MultimodalFusion(latent_dim=latent_dim, temperature=temperature)

        if modalities:
            for name, cfg in modalities.items():
                self.fusion.register_modality(
                    name=name,
                    input_dim=cfg["input_dim"],
                    hidden_dim=cfg.get("hidden_dim", None),
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_batch(self, batch: dict) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Split a flat batch dict into inputs and availability dicts.

        The dataset stores per-modality embeddings as ``{name}_emb`` and
        availability flags as ``{name}_avail``.  We reconstruct the two dicts
        expected by MultimodalFusion.forward().
        """
        inputs: dict[str, Tensor] = {}
        availability: dict[str, Tensor] = {}
        for name in self.fusion.projectors:
            emb_key = f"{name}_emb"
            avail_key = f"{name}_avail"
            if emb_key in batch:
                inputs[name] = batch[emb_key]           # (B, D_in)
                availability[name] = batch[avail_key]   # (B,) bool
        return inputs, availability

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        inputs, availability = self._parse_batch(batch)
        embeddings = self.fusion(inputs, availability)
        loss, loss_dict = self.fusion.compute_pairwise_loss(embeddings, availability)

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        for key, val in loss_dict.items():
            self.log(f"train_{key}", val, on_step=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        inputs, availability = self._parse_batch(batch)
        embeddings = self.fusion(inputs, availability)
        loss, loss_dict = self.fusion.compute_pairwise_loss(embeddings, availability)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        for key, val in loss_dict.items():
            self.log(f"val_{key}", val, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.fusion.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
        )


def main():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    cli = LightningCLI(
        MultimodalFusionModule,
        FusionDataModule,
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()
