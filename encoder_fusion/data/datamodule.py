"""LightningDataModule for multimodal fusion training."""

import warnings
from typing import Optional, Union

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
                           extract_embeddings_dja.py).  0.0 = no filter.
                           Analogous to encoder_spectrum's ``min_sn50``.
        min_n_valid:       Exclude samples with fewer than this many valid
                           spectral pixels (requires ``n_valid`` in H5).
                           Analogous to encoder_spectrum's ``min_length``.
        max_n_valid:       Exclude samples with more than this many valid
                           spectral pixels.  999999 = no upper limit.
                           Analogous to encoder_spectrum's ``max_length``.
        frac_valid_pix:    Keep only samples whose fraction of valid tokens
                           (mean over the token axis of each ``token_mask_keys``
                           mask) is strictly greater than this.  0.0 = no filter.
                           Mirrors encoder_spectrum/LowResPT's ``frac_valid_pix``
                           (which thresholds ``valid.mean(axis=1)``) and drops the
                           near-empty spectra that otherwise pool to a zero vector.
        group_key:         H5 key of a per-row group id (e.g. ``id`` = dja_id).
                           When set, the train/val split is grouped so every row
                           sharing a group id lands on the SAME side. Required for
                           the multi-survey crossmatch, where one spectrum is
                           matched to several image cutouts (duplicate rows) — a
                           plain row split would leak a spectrum across the split
                           and inflate retrieval metrics. None = plain row split.
    """

    def __init__(
        self,
        h5_path: str,
        modality_keys: dict[str, Union[str, list[str]]],
        batch_size: int = 256,
        num_workers: int = 4,
        train_val_split: float = 0.9,
        modality_mask_prob: float = 0.0,
        min_sn50: float = 0.0,
        min_n_valid: int = 0,
        max_n_valid: int = 999999,
        frac_valid_pix: float = 0.0,
        token_mask_keys: Optional[dict[str, str]] = None,
        stats_keys: Optional[dict[str, str]] = None,
        group_key: Optional[str] = None,
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
            def _feat(key):
                # str → that key; list → first key (members share the feature dim).
                k = key if isinstance(key, str) else key[0]
                return f[k].shape[-1]
            return {
                name: _feat(key)
                for name, key in self.hparams.modality_keys.items()
            }

    @property
    def stats_dims(self) -> dict[str, int]:
        """{modality: side-feature width} from the stats_keys datasets.

        A 1-D stats dataset (N,) counts as width 1. Available before setup().
        """
        stats_keys = self.hparams.stats_keys or {}
        if not stats_keys:
            return {}
        with h5py.File(self.hparams.h5_path, "r") as f:
            return {
                name: (1 if f[key].ndim == 1 else f[key].shape[-1])
                for name, key in stats_keys.items()
            }

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return  # already set up

        with h5py.File(self.hparams.h5_path, "r") as f:
            first_key = next(iter(self.hparams.modality_keys.values()))
            # value may be a str or a list of keys (members share the row count)
            if not isinstance(first_key, str):
                first_key = first_key[0]
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

            # Window-coverage cut (mirrors LowResPT's frac_valid_pix): keep only
            # samples whose fraction of valid tokens exceeds the threshold. Drops
            # near-empty spectra that pool to a degenerate (zero) embedding.
            if self.hparams.frac_valid_pix > 0.0:
                mask_keys = self.hparams.token_mask_keys or {}
                if not mask_keys:
                    warnings.warn(
                        f"frac_valid_pix={self.hparams.frac_valid_pix} requested but "
                        f"no token_mask_keys configured — filter skipped."
                    )
                for name, mkey in mask_keys.items():
                    valid_frac = f[mkey][:].mean(axis=1)        # (N,) in [0, 1]
                    selection &= valid_frac > self.hparams.frac_valid_pix

            # Per-row group id (e.g. dja_id) for grouped splitting, read before close.
            gkey = self.hparams.group_key
            if gkey and gkey in f:
                group_all = f[gkey][:]
            else:
                group_all = None
                if gkey:
                    warnings.warn(
                        f"group_key='{gkey}' not found in {self.hparams.h5_path} — "
                        f"falling back to a plain row split."
                    )

        indices = np.where(selection)[0]
        n_kept = len(indices)
        if n_kept < n_total:
            print(f"[FusionDataModule] selection: {n_kept}/{n_total} samples kept")

        # Reproducible split on the filtered set. With group_key, split by unique
        # group so every row of a group (e.g. all survey cutouts of one spectrum)
        # stays on one side; otherwise a plain shuffled row split.
        rng = np.random.default_rng(42)
        if group_all is not None:
            groups = group_all[indices]
            uniq = np.unique(groups)
            rng.shuffle(uniq)
            n_train_groups = int(len(uniq) * self.hparams.train_val_split)
            train_groups = set(uniq[:n_train_groups].tolist())
            is_train = np.fromiter((g in train_groups for g in groups),
                                   dtype=bool, count=len(groups))
            train_idx, val_idx = indices[is_train], indices[~is_train]
            rng.shuffle(train_idx)
            rng.shuffle(val_idx)
            print(f"[FusionDataModule] grouped split on '{self.hparams.group_key}': "
                  f"{len(uniq)} groups → {len(train_idx)} train / {len(val_idx)} val rows")
            self._train_indices, self._val_indices = train_idx, val_idx
        else:
            rng.shuffle(indices)
            n_train = int(n_kept * self.hparams.train_val_split)
            self._train_indices, self._val_indices = indices[:n_train], indices[n_train:]

        self.train_dataset = FusionEmbeddingDataset(
            h5_path=self.hparams.h5_path,
            modality_keys=self.hparams.modality_keys,
            indices=self._train_indices,
            modality_mask_prob=self.hparams.modality_mask_prob,
            token_mask_keys=self.hparams.token_mask_keys,
            stats_keys=self.hparams.stats_keys,
        )
        self.val_dataset = FusionEmbeddingDataset(
            h5_path=self.hparams.h5_path,
            modality_keys=self.hparams.modality_keys,
            indices=self._val_indices,
            modality_mask_prob=0.0,  # no masking during validation
            token_mask_keys=self.hparams.token_mask_keys,
            stats_keys=self.hparams.stats_keys,
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
