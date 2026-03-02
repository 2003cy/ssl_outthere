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
    Pass a ``modalities`` dict in the constructor (or via the YAML config).
    Set ``input_dim: null`` to auto-detect from the datamodule's ``.npy`` files::

        modalities:
          image:
            input_dim: null   # auto-detected from image.npy shape
            hidden_dim: 512
          spectrum:
            input_dim: null   # auto-detected from spectrum.npy shape
            hidden_dim: 256

    Explicit dims are also supported (useful when no datamodule is available)::

        modalities:
          photometry:
            input_dim: 64
            hidden_dim: null  # → single linear layer

    Auto-detection flow
    -------------------
    - Modalities with ``input_dim: null`` are registered in ``setup()`` once
      ``self.trainer.datamodule.input_dims`` is available.
    - Modalities with an explicit ``input_dim`` are registered in ``__init__``.
    - ``configure_optimizers()`` is called after ``setup()``, so all projector
      parameters are present when the optimizer is built.

    Args:
        latent_dim:    Shared embedding dimension D (default 256).
        temperature:   InfoNCE softmax temperature (default 0.07).
        lr:            Base learning rate for AdamW (overridden per-step by
                       WarmupCosineLR callback when used).
        weight_decay:  AdamW weight decay.
        modalities:    ``{name: {input_dim: int|null, hidden_dim: int|null}}``
                       dict used to register modality projectors.
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
                if cfg.get("input_dim") is not None:
                    # Explicit dim provided — register immediately.
                    self.fusion.register_modality(
                        name=name,
                        input_dim=cfg["input_dim"],
                        hidden_dim=cfg.get("hidden_dim", None),
                    )
                # else: input_dim is null → deferred to setup()

    # ------------------------------------------------------------------
    # Setup (deferred modality registration)
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Register modalities whose input_dim was set to null in the config.

        Called by Lightning after datamodule.setup(), so
        ``self.trainer.datamodule.input_dims`` is already available.
        """
        modalities = self.hparams.modalities or {}
        for name, cfg in modalities.items():
            if name in self.fusion.projectors:
                continue  # already registered in __init__
            # Auto-detect from datamodule
            input_dim = self.trainer.datamodule.input_dims[name]
            self.fusion.register_modality(
                name=name,
                input_dim=input_dim,
                hidden_dim=cfg.get("hidden_dim", None),
            )
            print(f"[setup] Registered modality '{name}' with auto-detected input_dim={input_dim}")

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
