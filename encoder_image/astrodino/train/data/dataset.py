import logging
import os
from enum import Enum
from typing import Any, Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torchvision.datasets import VisionDataset

logger = logging.getLogger("astrodino")

# JWST NIRCam default pixel scale: 30 mas/pixel → degrees-to-pixels conversion.
# Used only as a fallback; the actual scale is read per-store from the
# `pixscale_mas` HDF5 attribute (see JWST._read_pixscale).
_JWST_DEFAULT_PIXSCALE_MAS: float = 30.0
_JWST_DEG_TO_PIX: float = 3600 * 1000 / _JWST_DEFAULT_PIXSCALE_MAS

# Reproducible 90 / 5 / 5 train-val-test split fractions
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
    """
    JWST dataset backed by per-tile HDF5 files under `root/<filter>/`.

    Each HDF5 store (one per COSMOS-Web tile + one CEERS file) contains:
      - "image"  (N, H, W) float32  -- single-channel cutouts, MJy/sr
      - "ra", "dec"  (N,)           -- sky coordinates (deg)
      - "id"     (N,) int64         -- per-store catalogue row id
      - "seg"    (N, H, W) uint8    -- 0/1 source segmentation mask
                                       (COSMOS tiles only; absent for CEERS)
      - any extra fields requested via `extra_returns`

    Store-level HDF5 attributes include `pixscale_mas`, `image_size`,
    `bunit`, `survey` (and `tile`/`seg_size` for COSMOS, `sky_sigma` for
    CEERS). The degrees→pixels scale used by range filters is read from
    `pixscale_mas` (fallback 30 mas).

    Training is single-channel (image only); `seg` is currently ignored.

    The full catalogue is shuffled once (seed=42) and split 90/5/5 into
    train / val / test. An optional range cut on a per-sample field (e.g. a
    morphology field added later) can restrict eligible samples before the
    split; the only wired cut is the Sérsic-radius cut, inactive unless the
    `radius_sersic` field is present and `re_min_pix`/`re_max_pix` are set.

    Args:
        split:          One of "train", "val", "test".
        root:           Root directory; h5 files are read from `root/filter/`.
        filter:         JWST filter subdirectory (default "f115w").
        transforms:     Joint image+target transform (torchvision API).
        transform:      Image-only transform.
        target_transform: Target-only transform.
        extra_returns:  List of h5 dataset keys to return as `target` in
                        addition to (or instead of) the image.
        re_min_pix:     Minimum effective radius in pixels (inclusive).
        re_max_pix:     Maximum effective radius in pixels (inclusive).
    """

    Split = _Split

    def __init__(
        self,
        *,
        split: str,
        root: str,
        filter: str = 'f115w',
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extra_returns: Optional[list] = None,
        re_min_pix: Optional[float] = None,
        re_max_pix: Optional[float] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self._extra_returns = extra_returns

        # Step 1: open all tile h5 files
        self._files = self._open_h5_files(root, filter)

        # Degrees→pixels scale from the store attrs (fallback to the 30 mas default).
        self._deg_to_pix = self._read_pixscale(self._files)

        # Step 2: build cumulative index boundaries across files
        #   _cum_lengths[i] = first global index belonging to file i
        self._cum_lengths = np.concatenate(
            [[0], np.cumsum([len(f["ra"]) for f in self._files])]
        )

        # Step 3: apply dataset filters → valid global indices
        #   To add a new filter: add a param above and a _range_filter() call below.
        valid_indices = self._build_valid_mask([
            self._range_filter("radius_sersic", scale=self._deg_to_pix, re_low=re_min_pix, re_high=re_max_pix),
        ])

        # Step 4: shuffle and slice out the requested split
        self._indices = self._assign_split(valid_indices, split)

        # Step 5: precompute (file_idx, local_idx) for every sample
        self._index_map = self._build_index_map(self._indices)

        logger.info(
            "JWST [%s] — %d samples from %d files",
            split, len(self._indices), len(self._files),
        )

    # ── private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _open_h5_files(root: str, filter: str) -> list:
        filter_dir = os.path.join(root, filter) if filter else root
        if not os.path.isdir(filter_dir):
            raise FileNotFoundError(f"JWST filter directory not found: {filter_dir}")
        files = []
        for fname in sorted(f for f in os.listdir(filter_dir) if f.endswith(".h5")):
            fpath = os.path.join(filter_dir, fname)
            try:
                files.append(h5py.File(fpath, 'r'))
            except (OSError, IOError) as exc:
                logger.warning("Skipping %s: %s", fpath, exc)
        if not files:
            raise RuntimeError(f"No readable h5 files found in {filter_dir}")
        return files

    @staticmethod
    def _read_pixscale(files: list) -> float:
        """Degrees→pixels factor from the `pixscale_mas` store attribute.

        Reads the first store's `pixscale_mas` attribute (all f150w stores share
        30 mas); falls back to the module default when the attribute is absent.
        """
        pix_mas = float(files[0].attrs.get("pixscale_mas", _JWST_DEFAULT_PIXSCALE_MAS))
        return 3600.0 * 1000.0 / pix_mas

    def _range_filter(
        self,
        field: str,
        scale: float = 1.0,
        re_low: Optional[float] = None,
        re_high: Optional[float] = None,
    ) -> np.ndarray:
        """Build a per-sample boolean mask for a single h5 field.

        Reads `field` from every tile file, multiplies by `scale`, then keeps
        samples where the value is finite and within [lo, hi] (either bound is
        optional).  Files that lack the field are treated as NaN (excluded when
        any bound is active).

        Args:
            field:  h5 dataset key (e.g. "radius_sersic", "snr_f150w").
            scale:  Multiplicative unit conversion applied before comparison.
            lo:     Minimum allowed value (inclusive).  None = no lower bound.
            hi:     Maximum allowed value (inclusive).  None = no upper bound.
        Returns:
            Boolean ndarray of shape (n_total,).
        """
        n_total = int(self._cum_lengths[-1])
        if re_low is None and re_high is None:
            return np.ones(n_total, dtype=bool)

        values = np.concatenate([
            f[field][:] * scale if field in f else np.full(len(f["ra"]), np.nan)
            for f in self._files
        ])
        mask = np.isfinite(values)
        if re_low is not None:
            mask &= values >= re_low
        if re_high is not None:
            mask &= values <= re_high
        return mask

    def _build_valid_mask(self, masks: list) -> np.ndarray:
        """AND-combine a list of boolean masks and return the surviving indices.

        Args:
            masks:  List of boolean ndarrays of shape (n_total,), one per filter.
        Returns:
            1-D integer ndarray of global indices that pass all filters.
        """
        n_total = int(self._cum_lengths[-1])
        combined = np.ones(n_total, dtype=bool)
        for mask in masks:
            combined &= mask
        n_kept = int(combined.sum())
        if n_kept < n_total:
            logger.info("Dataset filters: %d / %d samples kept", n_kept, n_total)
        return np.where(combined)[0]

    @staticmethod
    def _assign_split(indices: np.ndarray, split: str) -> np.ndarray:
        """Shuffle `indices` with a fixed seed and return the slice for `split`."""
        if split not in _SPLIT_FRACTIONS:
            raise ValueError(f"Unknown split '{split}'. Choose from {list(_SPLIT_FRACTIONS)}")
        shuffled = np.random.default_rng(seed=42).permutation(indices)
        n = len(shuffled)
        lo, hi = _SPLIT_FRACTIONS[split]
        return shuffled[int(lo * n): int(hi * n)]

    def _build_index_map(self, indices: np.ndarray) -> list:
        """Map each global index → (file_idx, local_idx) via binary search."""
        file_indices  = np.searchsorted(self._cum_lengths[1:], indices, side='right')
        local_indices = indices - self._cum_lengths[file_indices]
        return list(zip(file_indices.tolist(), local_indices.tolist()))

    # ── Dataset interface ─────────────────────────────────────────────────

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        file_idx, local_idx = self._index_map[index]
        f = self._files[file_idx]

        # (H, W) single-channel → (1, H, W) float32 tensor
        img_np = f["image"][local_idx].astype("float32")
        image  = torch.from_numpy(img_np[np.newaxis])

        target = None
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # Optional: return extra catalogue fields as target (e.g. for evaluation)
        if self._extra_returns is not None:
            target = []
            for key in self._extra_returns:
                if key not in f:
                    # e.g. `seg` is present in COSMOS tiles but not the CEERS store.
                    raise KeyError(
                        f"extra_returns key '{key}' is missing from tile "
                        f"'{os.path.basename(f.filename)}'. Available keys: "
                        f"{list(f.keys())}"
                    )
                target.append(f[key][local_idx].astype("float32"))

        # VAL mode: return image as its own reconstruction target
        if self._split == JWST.Split.VAL.value:
            target = image

        return image, target
