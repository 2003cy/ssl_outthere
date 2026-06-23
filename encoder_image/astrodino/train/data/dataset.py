"""JWST image dataset backed by per-tile ``.npy`` shards + a slim index FITS.

Produced by ``images/cosmos_2025/cutout_export_npy.py``. Under ``root`` lives::

    image_index_<survey>_<filter>.fits        one row per cutout
    <filter>/nircam_<survey>_<filter>_<tile>.npy   (N, H, W) float16  (MJy/sr)

The index columns ``rel_path`` + ``local_idx`` locate every cutout: row ``i`` of
the index points at row ``local_idx[i]`` of the array at ``root/rel_path[i]``.
Arrays are memmapped lazily (per worker) so nothing is read until ``__getitem__``.

Pretraining uses *every* cutout in the index — no filtering. The catalogue is
shuffled once (seed 42) and split 90/5/5 into train/val/test. Training is
single-channel (image only); ``val`` returns the image as its own target.
"""

import logging
import os
import re
from enum import Enum
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from astropy.table import Table
from torchvision.datasets import VisionDataset

logger = logging.getLogger("astrodino")

# rel_path looks like ``<filter>/nircam_cosmos_<filter>_<tile>.npy`` -> tile alias.
_TILE_RE = re.compile(r"_([A-Za-z0-9]+)\.npy$")


def parse_tile(rel_path: str) -> str:
    """Extract the tile alias (e.g. 'A1') from a cutout rel_path."""
    m = _TILE_RE.search(str(rel_path))
    return m.group(1) if m else ""

# Reproducible 90 / 5 / 5 train-val-test split fractions.
_SPLIT_FRACTIONS = {
    "train": (0.00, 0.90),
    "val":   (0.90, 0.95),
    "test":  (0.95, 1.00),
}


class _Split(Enum):
    TRAIN = "train"
    VAL   = "val"
    TEST  = "test"


class JWST(VisionDataset):
    """JWST cutouts read from ``.npy`` shards via a slim image-index FITS.

    Args:
        split:            One of "train", "val", "test".
        root:             Directory holding ``image_index_<survey>_<filter>.fits``
                          and the ``<filter>/*.npy`` shards.
        filter:           JWST filter (default "f150w").
        survey:           Survey label in the index filename (default "cosmos").
        transforms:       Joint image+target transform (torchvision API).
        transform:        Image-only transform (the DINO augmentation).
        target_transform: Target-only transform.
    """

    Split = _Split

    def __init__(
        self,
        *,
        split: str,
        root: str,
        filter: str = "f150w",
        survey: str = "cosmos",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split

        index_path = os.path.join(root, f"image_index_{survey}_{filter}.fits")
        index = Table.read(index_path)
        self._rel_path = np.asarray(index["rel_path"]).astype(str)
        self._local_idx = np.asarray(index["local_idx"]).astype(np.int64)
        self._indices = self._assign_split(np.arange(len(index)), split)
        self._mmaps: dict[str, np.ndarray] = {}  # rel_path -> memmap, opened per worker

        logger.info(
            "JWST [%s] — %d / %d cutouts (%s)",
            split, len(self._indices), len(index), filter,
        )

    @staticmethod
    def _assign_split(indices: np.ndarray, split: str) -> np.ndarray:
        """Shuffle with a fixed seed and slice out the requested split."""
        if split not in _SPLIT_FRACTIONS:
            raise ValueError(f"Unknown split '{split}'. Choose from {list(_SPLIT_FRACTIONS)}")
        shuffled = np.random.default_rng(seed=42).permutation(indices)
        lo, hi = _SPLIT_FRACTIONS[split]
        n = len(shuffled)
        return shuffled[int(lo * n): int(hi * n)]

    def _shard(self, rel_path: str) -> np.ndarray:
        """Memmap a tile shard, caching the handle for this worker process."""
        arr = self._mmaps.get(rel_path)
        if arr is None:
            arr = np.load(os.path.join(self.root, rel_path), mmap_mode="r")
            self._mmaps[rel_path] = arr
        return arr

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        row = self._indices[index]
        cutout = self._shard(self._rel_path[row])[self._local_idx[row]]  # (H, W) float16

        # (H, W) -> (1, H, W) float32; zero out empty (NaN/inf) pixels.
        img = np.nan_to_num(cutout.astype(np.float32), copy=False)
        image = torch.from_numpy(img[np.newaxis])

        target = None
        if self.transform is not None:
            tile = parse_tile(self._rel_path[row])
            image = self.transform(image, tile=tile)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self._split == _Split.VAL.value:
            target = image
        return image, target
