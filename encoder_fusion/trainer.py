#!/usr/bin/env python
"""LightningModule + LightningCLI entry point for multimodal fusion training.

Usage:
    python trainer.py fit --config config_dja.yaml
    python trainer.py fit --config config_dja.yaml --trainer.devices=[0,1]
"""

import math
import os
import sys
from itertools import combinations

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# Ensure local model/ and data/ are importable when running from this directory
sys.path.insert(0, os.path.dirname(__file__))

from typing import Optional

import lightning as L
import torch
from lightning.pytorch.cli import LightningCLI
from torch import Tensor

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
        lr:            Peak learning rate for AdamW (the single LR definition;
                       warmup + cosine decay are applied in configure_optimizers,
                       mirroring encoder_spectrum/LowResPT).
        weight_decay:  AdamW weight decay.
        betas:         AdamW betas.
        warmup_steps:  Linear warmup duration in optimizer steps.
        min_lr:        Floor the cosine schedule decays to.
        modalities:    ``{name: {input_dim: int|null, hidden_dim: int|null}}``
                       dict used to register modality projectors.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        temperature: float = 0.07,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.95),
        warmup_steps: int = 100,
        min_lr: float = 1e-5,
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
                        pool=cfg.get("pool", None),
                        num_heads=cfg.get("num_heads", 4),
                        stats_dim=cfg.get("stats_dim", None),
                        stats_scale=cfg.get("stats_scale", None),
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
            # stats_dim: explicit config value wins, else auto-detect from the
            # stats_keys dataset width (None → no stats injection).
            stats_dim = cfg.get("stats_dim", None)
            if stats_dim is None:
                stats_dim = self.trainer.datamodule.stats_dims.get(name)
            self.fusion.register_modality(
                name=name,
                input_dim=input_dim,
                hidden_dim=cfg.get("hidden_dim", None),
                pool=cfg.get("pool", None),
                num_heads=cfg.get("num_heads", 4),
                stats_dim=stats_dim,
                stats_scale=cfg.get("stats_scale", None),
            )
            pool = cfg.get("pool", None)
            print(f"[setup] Registered modality '{name}' with auto-detected "
                  f"input_dim={input_dim}, pool={pool}, stats_dim={stats_dim}, "
                  f"stats_scale={cfg.get('stats_scale', None)}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_batch(
        self, batch: dict
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        """Split a flat batch dict into inputs, availability, token-mask, stats dicts.

        The dataset stores per-modality embeddings as ``{name}_emb``, availability
        flags as ``{name}_avail``, valid-token masks as ``{name}_mask`` (token-
        sequence modalities), and per-sample side-features as ``{name}_stats``.
        We reconstruct the dicts expected by MultimodalFusion.forward().
        """
        inputs: dict[str, Tensor] = {}
        availability: dict[str, Tensor] = {}
        masks: dict[str, Tensor] = {}
        stats: dict[str, Tensor] = {}
        for name in self.fusion.projectors:
            emb_key = f"{name}_emb"
            if emb_key in batch:
                inputs[name] = batch[emb_key]               # (B, D_in) or (B, T, D_in)
                availability[name] = batch[f"{name}_avail"] # (B,) bool
                if f"{name}_mask" in batch:
                    masks[name] = batch[f"{name}_mask"]     # (B, T) bool
                if f"{name}_stats" in batch:
                    stats[name] = batch[f"{name}_stats"]    # (B, S) float
        return inputs, availability, masks, stats

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> Optional[Tensor]:
        # Surface the scheduler's current LR (replaces the old WarmupCosineLR
        # callback's logging; EpochPrinter reads the "lr" key).
        sched = self.lr_schedulers()
        if sched is not None:
            self.log("lr", sched.get_last_lr()[0], on_step=True, prog_bar=True)

        inputs, availability, masks, stats = self._parse_batch(batch)
        embeddings = self.fusion(inputs, availability, masks, stats)
        loss, loss_dict = self.fusion.compute_pairwise_loss(embeddings, availability)

        if not loss_dict:
            return None  # no valid pairs in this batch; skip optimizer step

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        for key, val in loss_dict.items():
            self.log(f"train_{key}", val, on_step=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        # Stash projected embeddings to compute retrieval over the FULL val set
        # at epoch end (gallery = all val samples, not per-batch — per-batch would
        # inflate Recall@k when the last batch is small).
        self._val_store: dict[str, list] = {}

    def validation_step(self, batch: dict, batch_idx: int) -> Optional[Tensor]:
        inputs, availability, masks, stats = self._parse_batch(batch)
        embeddings = self.fusion(inputs, availability, masks, stats)
        loss, loss_dict = self.fusion.compute_pairwise_loss(embeddings, availability)

        for name, emb in embeddings.items():
            self._val_store.setdefault(name, []).append(
                (emb.detach(), availability[name].detach())
            )

        if not loss_dict:
            return None  # no valid pairs in this batch; skip logging

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        for key, val in loss_dict.items():
            self.log(f"val_{key}", val, on_epoch=True, sync_dist=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        store = getattr(self, "_val_store", {})
        if len(store) < 2:
            return
        embs = {n: torch.cat([e for e, _ in v]) for n, v in store.items()}
        avs  = {n: torch.cat([a for _, a in v]) for n, v in store.items()}

        # Symmetric Recall@k over all valid modality pairs (both directions).
        # Embeddings are already L2-normalized, so emb_a @ emb_b.T is cosine.
        r1, r10 = [], []
        for a, b in combinations(embs.keys(), 2):
            valid = avs[a] & avs[b]
            if int(valid.sum()) < 2:
                continue
            ea, eb = embs[a][valid], embs[b][valid]      # (M, D), row i ↔ col i
            M = ea.shape[0]
            labels = torch.arange(M, device=ea.device)
            for sim in (ea @ eb.T, eb @ ea.T):           # a→b and b→a
                order = sim.argsort(dim=1, descending=True)
                rank = (order == labels[:, None]).float().argmax(dim=1)  # 0-based rank of true match
                r1.append((rank < 1).float().mean())
                r10.append((rank < min(10, M)).float().mean())
        if r1:
            self.log("val_R@1",  torch.stack(r1).mean(),  prog_bar=True, sync_dist=True)
            self.log("val_R@10", torch.stack(r10).mean(), prog_bar=True, sync_dist=True)
        self._val_store = {}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.fusion.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=tuple(self.hparams.betas),
        )

        total_steps  = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_steps
        min_lr_ratio = self.hparams.min_lr / self.hparams.lr

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer":    opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


def main():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    LightningCLI(
        model_class=MultimodalFusionModule,
        datamodule_class=FusionDataModule,
        save_config_callback=None,
        seed_everything_default=42,
    )


if __name__ == "__main__":
    main()
