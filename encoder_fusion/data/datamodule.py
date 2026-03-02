"""LightningDataModule for multimodal fusion training."""

import warnings
from typing import Optional

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
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

    Expects a single HDF5 file with one dataset per modality.  Rows with
    all-NaN embeddings are treated as missing for that modality.

    Args:
        h5_path:           Path to HDF5 file containing embedding arrays.
        modality_keys:     ``{modality_name: h5_key}`` mapping,
                           e.g. ``{"image": "image_embed", "spectrum": "spectrum_embed"}``.
        batch_size:        Training / validation batch size.
        num_workers:       DataLoader worker processes.
        train_val_split:   Fraction of data used for training (rest = validation).
        modality_mask_prob: Per-modality masking probability applied to the
                           *training* split only (0.0 = no masking).  See
                           FusionEmbeddingDataset for details.
        min_sn50:          Exclude samples with ``sn50 < min_sn50`` (requires
                           an ``sn50`` dataset in the H5 file, e.g. saved by
                           extract_embeddings_jda.py).  0.0 = no filter.
                           Analogous to encoder_spectrum's ``min_sn50``.
        min_n_valid:       Exclude samples with fewer than this many valid
                           spectral pixels (requires ``n_valid`` in H5).
                           Analogous to encoder_spectrum's ``min_length``.
        max_n_valid:       Exclude samples with more than this many valid
                           spectral pixels.  999999 = no upper limit.
                           Analogous to encoder_spectrum's ``max_length``.
    """

    def __init__(
        self,
        h5_path: str,
        modality_keys: dict[str, str],
        batch_size: int = 256,
        num_workers: int = 4,
        train_val_split: float = 0.9,
        modality_mask_prob: float = 0.0,
        min_sn50: float = 0.0,
        min_n_valid: int = 0,
        max_n_valid: int = 999999,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset: Optional[FusionEmbeddingDataset] = None
        self.val_dataset: Optional[FusionEmbeddingDataset] = None

    @property
    def input_dims(self) -> dict[str, int]:
        """Read embedding dims directly from HDF5 dataset shapes (cheap).

        Available before setup() is called — safe to use in model.setup().
        """
        with h5py.File(self.hparams.h5_path, "r") as f:
            return {
                name: f[key].shape[1]
                for name, key in self.hparams.modality_keys.items()
            }

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return  # already set up

        with h5py.File(self.hparams.h5_path, "r") as f:
            first_key = next(iter(self.hparams.modality_keys.values()))
            n_total = f[first_key].shape[0]

            # ── Data selection (mirrors encoder_spectrum min_sn50 / min_length) ──
            selection = np.ones(n_total, dtype=bool)

            if self.hparams.min_sn50 > 0.0:
                if "sn50" in f:
                    selection &= f["sn50"][:] >= self.hparams.min_sn50
                else:
                    warnings.warn(
                        f"min_sn50={self.hparams.min_sn50} requested but 'sn50' "
                        f"not found in {self.hparams.h5_path} — filter skipped."
                    )

            if self.hparams.min_n_valid > 0 or self.hparams.max_n_valid < 999999:
                if "n_valid" in f:
                    nv = f["n_valid"][:]
                    selection &= (nv >= self.hparams.min_n_valid) & (nv <= self.hparams.max_n_valid)
                else:
                    warnings.warn(
                        f"n_valid filter requested but 'n_valid' not found in "
                        f"{self.hparams.h5_path} — filter skipped."
                    )

        indices = np.where(selection)[0]
        n_kept = len(indices)
        if n_kept < n_total:
            print(f"[FusionDataModule] selection: {n_kept}/{n_total} samples kept")

        # Reproducible shuffle then split on the filtered set
        rng = np.random.default_rng(42)
        rng.shuffle(indices)
        n_train = int(n_kept * self.hparams.train_val_split)

        self.train_dataset = FusionEmbeddingDataset(
            h5_path=self.hparams.h5_path,
            modality_keys=self.hparams.modality_keys,
            indices=indices[:n_train],
            modality_mask_prob=self.hparams.modality_mask_prob,
        )
        self.val_dataset = FusionEmbeddingDataset(
            h5_path=self.hparams.h5_path,
            modality_keys=self.hparams.modality_keys,
            indices=indices[n_train:],
            modality_mask_prob=0.0,  # no masking during validation
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
