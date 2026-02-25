"""DataModule for multi-band spectrum training."""

from typing import Callable, Dict, List, Optional

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split, Sampler

from .dataset import MASpectrumDataset, JDASpectrumDataset


class InfiniteLoopSampler(Sampler):
    """Sampler that cycles through the dataset infinitely with shuffling."""

    def __init__(self, data_source, shuffle: bool = True):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = 0

    def __iter__(self):
        while True:
            if self.shuffle:
                # Create shuffled indices with increasing seed for variety
                indices = torch.randperm(len(self.data_source), 
                                        generator=torch.Generator().manual_seed(self.seed))
                self.seed += 1
                yield from indices.tolist()
            else:
                yield from range(len(self.data_source))

    def __len__(self):
        return len(self.data_source)


class MASpectrumDataModule(L.LightningDataModule):
    """DataModule for single-band (or multi-band) spectrum data.
    
    Args:
        h5_path: Path to HDF5 file (e.g., images/OUTTHERE/spectrum/f115w.h5)
        batch_size: Batch size for training/validation
        num_workers: Number of workers for DataLoader
        train_val_split: Fraction of data for training (rest goes to validation)
        collate_fn: Custom collate function (optional)
        valid_flux_mask_fn: Custom function to compute valid_mask (optional)
        use_infinite_sampler: Use infinite loop sampler for training (default: False)
        min_snr: Minimum SNR threshold. Samples with SNR < min_snr are excluded.
    """

    def __init__(
        self,
        h5_path: str,
        batch_size: int = 128,
        num_workers: int = 4,
        train_val_split: float = 0.9,
        collate_fn: Optional[Callable] = None,
        valid_flux_mask_fn: Optional[Callable] = None,
        use_infinite_sampler: bool = False,
        min_snr: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None) -> None:
        """Load dataset and split into train/val."""
        if self.dataset is None:
            # Get valid_flux_mask_fn from hparams, default to None if not present
            valid_mask_fn = getattr(self.hparams, 'valid_flux_mask_fn', None)
            min_snr = getattr(self.hparams, 'min_snr', None)
            self.dataset = MASpectrumDataset(
                self.hparams.h5_path,
                valid_flux_mask_fn=valid_mask_fn,
                min_snr=min_snr,
            )

            # Train/val split
            n_total = len(self.dataset)
            n_train = int(n_total * self.hparams.train_val_split)
            n_val = n_total - n_train

            self.train_dataset, self.val_dataset = random_split(
                self.dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader:
        collate_fn = getattr(self.hparams, 'collate_fn', None) or self._default_collate
        
        # Use infinite loop sampler if enabled
        if getattr(self.hparams, 'use_infinite_sampler', False):
            sampler = InfiniteLoopSampler(self.train_dataset, shuffle=True)
            return DataLoader(
                self.train_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                sampler=sampler,  # Use custom sampler instead of shuffle
                drop_last=True,
                collate_fn=collate_fn,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn,
            )

    def val_dataloader(self) -> DataLoader:
        collate_fn = getattr(self.hparams, 'collate_fn', None) or self._default_collate
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader (same as validation)."""
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        """Predict dataloader (use validation set)."""
        return self.val_dataloader()

    @staticmethod
    def _default_collate(batch: List[dict]) -> dict:
        """Stack dictionaries into batched tensors."""
        keys = batch[0].keys()
        collated = {}
        for key in keys:
            collated[key] = torch.stack([b[key] for b in batch])
        return collated


class JDASpectrumDataModule(L.LightningDataModule):
    """DataModule for JDA spectrum data (jda_spectra.h5).

    Args:
        h5_path:              Path to the JDA HDF5 file.
        batch_size:           Batch size for training/validation.
        num_workers:          Number of DataLoader workers.
        train_val_split:      Fraction of data used for training.
        use_infinite_sampler: Cycle through data infinitely (for masked modelling).
        min_sn50:             Exclude spectra with sn50 < this value (None = no cut).
        min_length:           Exclude spectra whose actual length (# finite pixels)
                              is <= this value (None = no lower bound).
        max_length:           Exclude spectra whose actual length (# finite pixels)
                              exceeds this value (None = keep all lengths).
    """

    def __init__(
        self,
        h5_path: str,
        batch_size: int = 256,
        num_workers: int = 4,
        train_val_split: float = 0.9,
        use_infinite_sampler: bool = True,
        min_sn50: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None) -> None:
        if self.dataset is None:
            self.dataset = JDASpectrumDataset(
                self.hparams.h5_path,
                min_sn50=self.hparams.min_sn50,
                min_length=self.hparams.min_length,
                max_length=self.hparams.max_length,
            )
            n_total = len(self.dataset)
            n_train = int(n_total * self.hparams.train_val_split)
            n_val = n_total - n_train
            self.train_dataset, self.val_dataset = random_split(
                self.dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )

    @staticmethod
    def _pad_collate(batch: List[dict]) -> dict:
        """Pad variable-length spectra to the longest in the batch.

        JDA spectra are truncated to their actual finite-pixel length, so
        different samples may have different shapes.  We pad with zeros and
        extend valid_mask with False so padded positions are invisible to
        both attention and loss computation.
        """
        max_len = max(b["flux"].shape[0] for b in batch)
        collated: dict = {}
        for key in batch[0].keys():
            if key == "valid_mask":
                padded = torch.zeros(len(batch), max_len, dtype=torch.bool)
            else:
                padded = torch.zeros(len(batch), max_len, dtype=batch[0][key].dtype)
            for i, b in enumerate(batch):
                t = b[key]
                padded[i, : t.shape[0]] = t
            collated[key] = padded
        return collated

    def train_dataloader(self) -> DataLoader:
        if self.hparams.use_infinite_sampler:
            sampler = InfiniteLoopSampler(self.train_dataset, shuffle=True)
            return DataLoader(
                self.train_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                sampler=sampler,
                drop_last=True,
                collate_fn=self._pad_collate,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self._pad_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=self._pad_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()
