"""Device-resident variant of LowResDataModule.

The spectra live on a fixed 56-pixel grid and the whole pre-training set is
~13 MB, so the DataLoader is pure overhead: at batch 1024 it runs ~1000
`__getitem__` calls per step, each doing two numpy copies, two divisions and a
per-sample copy of the shared wavelength row, then collates and ships the result
across a worker pipe. Holding the tensors on the device instead and slicing them
by index removes all of it.

Subclasses `LowResDataModule` so the FITS reading, the cuts and the 90/10 split
are shared; only the two dataloader methods are replaced. The split itself is
reused verbatim -- `random_split` returns `Subset`s, whose `.indices` are taken
directly -- so a run here sees exactly the same train/val partition as the
DataLoader path.
"""

import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .datamodule import LowResDataModule


class GPUBatches:
    """Batches sliced out of device-resident tensors; stands in for a DataLoader.

    `wavelength` is one shared row expanded to the batch rather than copied per
    sample. Shuffling draws from its own generator, so it never consumes the
    global RNG that drives token masking.
    """

    def __init__(self, tensors: dict, indices: Tensor, batch_size: int,
                 shuffle: bool, drop_last: bool, seed: int = 0):
        self.tensors = tensors
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        n = self.indices.numel()
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self):
        n = self.indices.numel()
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            order = self.indices[torch.randperm(n, generator=g).to(self.indices.device)]
            self.epoch += 1
        else:
            order = self.indices
        wave = self.tensors["wavelength"]
        for s in range(0, n, self.batch_size):
            idx = order[s:s + self.batch_size]
            if self.drop_last and idx.numel() < self.batch_size:
                break
            yield {
                "flux": self.tensors["flux"][idx],
                "err": self.tensors["err"][idx],
                "valid_mask": self.tensors["valid_mask"][idx],
                "redshift": self.tensors["redshift"][idx],
                "wavelength": wave.expand(idx.numel(), -1),
            }


class GPULowResDataModule(LowResDataModule):
    """LowResDataModule that keeps the whole sample on the training device."""

    def __init__(self, *args, device: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._device = device
        self._tensors = None

    def _resolve_device(self):
        if self._device is not None:
            return torch.device(self._device)
        if getattr(self, "trainer", None) is not None:
            return self.trainer.strategy.root_device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build(self):
        """Move flux/err/valid/redshift to the device, converting units once.

        `LowResDataset.__getitem__` divides by lambda^2 on every access; doing it
        here means it happens once for the whole sample instead of once per
        sample per epoch.
        """
        if self._tensors is not None:
            return self._tensors
        dev = self._resolve_device()
        ds = self.train_dataset.dataset          # unwrap the random_split Subset

        def _t(a, dtype=np.float32):
            # np.asarray also normalises the byte order: columns read straight
            # out of the FITS are big-endian, which torch cannot view.
            return torch.from_numpy(np.asarray(a, dtype=dtype)).to(dev)

        wave = _t(ds.wave)
        flux, err = _t(ds._flux), _t(ds._err)
        if not ds.use_jansky:
            flux = flux / wave ** 2
            err = err / wave ** 2
        self._tensors = {
            "flux": flux,
            "err": err,
            "valid_mask": _t(ds._valid, np.bool_),
            "redshift": _t(ds.z_best),
            "wavelength": wave,
        }
        return self._tensors

    def _indices(self, subset) -> Tensor:
        dev = self._resolve_device()
        return torch.as_tensor(subset.indices, dtype=torch.long, device=dev)

    def train_dataloader(self) -> GPUBatches:
        return GPUBatches(self._build(), self._indices(self.train_dataset),
                          batch_size=self.hparams.batch_size,
                          shuffle=True, drop_last=True)

    def val_dataloader(self) -> GPUBatches:
        return GPUBatches(self._build(), self._indices(self.val_dataset),
                          batch_size=self.hparams.batch_size,
                          shuffle=False, drop_last=False)
