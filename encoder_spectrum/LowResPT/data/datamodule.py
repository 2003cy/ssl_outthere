"""DataModule for LowResPT DJA low-resolution spectrum training."""

from typing import List, Optional, Sequence

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Sampler, random_split

from .dataset import LowResDataset


class InfiniteLoopSampler(Sampler):
    """Sampler that cycles through the dataset infinitely with shuffling."""

    def __init__(self, data_source, shuffle: bool = True):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = 0

    def __iter__(self):
        while True:
            if self.shuffle:
                indices = torch.randperm(
                    len(self.data_source),
                    generator=torch.Generator().manual_seed(self.seed),
                )
                self.seed += 1
                yield from indices.tolist()
            else:
                yield from range(len(self.data_source))

    def __len__(self):
        return len(self.data_source)


class LowResDataModule(L.LightningDataModule):
    """DataModule for DJA low-res spectra (read from the merged FITS).

    Args:
        fits_path:            Path to DJA_spectra_v4.5.fits.
        batch_size:           Batch size for training/validation.
        num_workers:          Number of DataLoader workers.
        train_val_split:      Fraction of data used for training.
        use_infinite_sampler: Cycle through data infinitely (for MAE training).
        grades:               Keep only these quality grades (None = no grade cut).
        min_obs_frac:         Keep spectra with obs_365_frac > this (None = no cut).
        min_sn50:             Exclude spectra with sn50 < this value.
        min_redshift:         Exclude spectra with z_best <= this value.
        max_redshift:         Exclude spectra with z_best > this value (None = no cut).
        frac_valid_pix:       Keep spectra whose valid-pixel fraction inside the
                              [wl_ref_min, wl_ref_max] window is > this ([0,1];
                              None = no cut).
        use_jansky:           If True, deliver raw uJy (f_nu) flux; if False
                              (default), convert to f_lambda (∝ f_nu / λ²).
        err_column:           Per-pixel error column for inverse-variance loss
                              weighting ("full_err" or "err").
    """

    def __init__(
        self,
        fits_path: str,
        batch_size: int = 256,
        num_workers: int = 4,
        train_val_split: float = 0.9,
        use_infinite_sampler: bool = True,
        grades: Optional[Sequence[int]] = (1, 2, 3),
        min_obs_frac: Optional[float] = 0.,
        min_sn50: Optional[float] = None,
        min_redshift: Optional[float] = None,
        max_redshift: Optional[float] = None,
        frac_valid_pix: Optional[float] = None,
        use_jansky: bool = False,
        wl_ref_min: float = 1.0,
        wl_ref_max: float = 2.0,
        err_column: str = "full_err",
    ):
        """
        wl_ref_min / wl_ref_max: observed-frame wavelength range (µm). Used both
        to CUT each spectrum to that window at the dataset level AND by the
        model's positional encoding to globally normalise wavelengths. Linked to
        model.wl_ref_min/max via LightningCLI in trainer.py — define once here,
        picked up by the dataset and model automatically.
        """
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = None
        self.val_dataset = None
        # Assign a prebuilt LowResDataset here to skip the FITS read in setup().
        # The Optuna sweep uses this to share one dataset across every trial; note
        # that an injected dataset carries its own cuts, so the `grades` /
        # `frac_valid_pix` / ... hparams below are then ignored.
        self.dataset = None

    def setup(self, stage: str = None) -> None:
        if self.train_dataset is None:
            dataset = self.dataset if self.dataset is not None else LowResDataset(
                self.hparams.fits_path,
                grades=getattr(self.hparams, "grades", (1, 2, 3)),
                min_obs_frac=getattr(self.hparams, "min_obs_frac", 0.5),
                min_sn50=getattr(self.hparams, "min_sn50", None),
                min_redshift=getattr(self.hparams, "min_redshift", None),
                max_redshift=getattr(self.hparams, "max_redshift", None),
                frac_valid_pix=getattr(self.hparams, "frac_valid_pix", None),
                wl_ref_min=getattr(self.hparams, "wl_ref_min", 1.0),
                wl_ref_max=getattr(self.hparams, "wl_ref_max", 2.0),
                use_jansky=getattr(self.hparams, "use_jansky", False),
                err_column=getattr(self.hparams, "err_column", "full_err"),
            )
            n_total = len(dataset)
            n_train = int(n_total * self.hparams.train_val_split)
            n_val = n_total - n_train
            self.train_dataset, self.val_dataset = random_split(
                dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )

    @staticmethod
    def _pad_collate(batch: List[dict]) -> dict:
        """Pad variable-length spectra to the longest in the batch."""
        max_len = max(b["flux"].shape[0] for b in batch)
        collated: dict = {}
        for key in batch[0].keys():
            if batch[0][key].dim() == 0:
                collated[key] = torch.stack([b[key] for b in batch])
                continue
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
                pin_memory=torch.cuda.is_available(),
                persistent_workers=self.hparams.num_workers > 0,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self._pad_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=self._pad_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()
