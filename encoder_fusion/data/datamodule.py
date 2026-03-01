"""LightningDataModule for multimodal fusion training."""

from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split
import lightning as L

from .dataset import FusionEmbeddingDataset


def _fusion_collate(batch: list[dict]) -> dict[str, Tensor]:
    """Stack a list of per-sample dicts into a batched dict.

    All values are Tensors of fixed shape (D,) or () so plain stacking works.
    """
    keys = batch[0].keys()
    return {key: torch.stack([b[key] for b in batch]) for key in keys}


class FusionDataModule(L.LightningDataModule):
    """DataModule for multimodal contrastive learning with precomputed embeddings.

    Expects one ``.npy`` file per modality, each of shape ``(N, D)``.  Rows with
    all-NaN embeddings are treated as missing for that modality.

    Args:
        modality_paths:  ``{name: "/path/to/embeddings.npy"}`` mapping.
        batch_size:      Training / validation batch size.
        num_workers:     DataLoader worker processes.
        train_val_split: Fraction of data used for training (rest = validation).
    """

    def __init__(
        self,
        modality_paths: dict[str, str],
        batch_size: int = 256,
        num_workers: int = 4,
        train_val_split: float = 0.9,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset: Optional[FusionEmbeddingDataset] = None
        self.val_dataset: Optional[FusionEmbeddingDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return  # already set up

        full_dataset = FusionEmbeddingDataset(self.hparams.modality_paths)
        n_total = len(full_dataset)
        n_train = int(n_total * self.hparams.train_val_split)
        n_val = n_total - n_train

        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=_fusion_collate,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=_fusion_collate,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()
