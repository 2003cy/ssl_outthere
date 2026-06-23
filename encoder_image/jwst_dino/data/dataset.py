"""JWST image dataset: per-tile ``.npy`` shards indexed by a slim FITS table.

Layout under ``root`` (produced by images/cosmos_2025/cutout_export_npy.py)::

    image_index_<survey>_<filter>.fits             one row per cutout
    <filter>/nircam_<survey>_<filter>_<tile>.npy   (N, H, W) float16, MJy/sr

Each index row gives ``rel_path`` + ``local_idx``: cutout ``i`` is row
``local_idx[i]`` of the shard at ``root/rel_path[i]``. Shards are memmapped per
worker, so pixels are read only in ``__getitem__``.

Every cutout is used (no filtering). Multiple surveys can be combined by passing a
list (e.g. ``["cosmos", "ceers"]``); each survey's index is shuffled once (fixed
seed) and sliced 90/5/5 *independently*, then the per-survey slices are concatenated
— so every split holds the same survey proportions and no survey can land wholly in
one split. The per-survey sky_sigma is recovered downstream from the tile alias in
each row's rel_path (parse_tile), so no survey label needs to flow through here.
"""

import os
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
from astropy.table import Table
from torch.utils.data import Dataset

# (start, end) fractions of the shuffled index for each split.
SPLITS = {"train": (0.00, 0.90), "val": (0.90, 0.95), "test": (0.95, 1.00)}
SPLIT_SEED = 42


def parse_tile(rel_path: str) -> str:
    """Tile alias from a rel_path, e.g. 'nircam_cosmos_f150w_A1.npy' -> 'A1'."""
    return os.path.basename(rel_path).removesuffix(".npy").rsplit("_", 1)[-1]


class JWST(Dataset):
    def __init__(
        self,
        split: str,
        root: str,
        filter: str = "f150w",
        survey: Union[str, Sequence[str]] = "cosmos",
        transform: Optional[Callable] = None,
    ):
        self.root = os.path.expandvars(os.path.expanduser(root))
        self.split = split
        self.transform = transform

        surveys = [survey] if isinstance(survey, str) else list(survey)
        rel_paths, local_idxs, selected = [], [], []
        offset = 0  # running row offset into the concatenated arrays
        for s in surveys:
            index = Table.read(os.path.join(self.root, f"image_index_{s}_{filter}.fits"))
            rel_paths.append(np.asarray(index["rel_path"]).astype(str))
            local_idxs.append(np.asarray(index["local_idx"]).astype(np.int64))
            sel = self._split_indices(len(index), split)  # split within this survey
            selected.append(sel + offset)
            offset += len(index)
            print(f"JWST [{split}] {s} — {len(sel)} / {len(index)} cutouts ({filter})")

        self.rel_path = np.concatenate(rel_paths)
        self.local_idx = np.concatenate(local_idxs)
        self.indices = np.concatenate(selected)
        self.shards: dict[str, np.ndarray] = {}  # rel_path -> memmap, opened per worker

        print(f"JWST [{split}] — {len(self.indices)} / {offset} cutouts total "
              f"({', '.join(surveys)}; {filter})")

    @staticmethod
    def _split_indices(n: int, split: str) -> np.ndarray:
        order = np.random.default_rng(SPLIT_SEED).permutation(n)
        lo, hi = SPLITS[split]
        return order[int(lo * n): int(hi * n)]

    def _shard(self, rel_path: str) -> np.ndarray:
        if rel_path not in self.shards:
            self.shards[rel_path] = np.load(os.path.join(self.root, rel_path), mmap_mode="r")
        return self.shards[rel_path]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        row = self.indices[i]
        cutout = self._shard(self.rel_path[row])[self.local_idx[row]]  # (H, W) float16

        # (H, W) -> (1, H, W) float32; empty (NaN/inf) pixels -> 0.
        image = torch.from_numpy(np.nan_to_num(cutout.astype(np.float32))[None])
        if self.transform is not None:
            image = self.transform(image, tile=parse_tile(self.rel_path[row]))

        # Target is unused (collate keeps only the crops); kept for the (x, y) contract.
        return image, ()
