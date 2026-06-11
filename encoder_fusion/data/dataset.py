"""Dataset for multimodal fusion training using precomputed embeddings in HDF5 format.

Workflow
--------
1. Run each frozen encoder offline to produce a single HDF5 file::

       with h5py.File("embeddings/dja/embeddings.h5", "w") as f:
           f.create_dataset("image_embed",    data=image_cls)    # (N, D_img)
           f.create_dataset("spectrum_embed", data=spectrum_cls)  # (N, D_spec)

   Missing embeddings for a sample should be stored as all-NaN rows.

2. Pass the path + key mapping to FusionEmbeddingDataset::

       dataset = FusionEmbeddingDataset(
           h5_path="embeddings/dja/embeddings.h5",
           modality_keys={"image": "image_embed", "spectrum": "spectrum_embed"},
       )

3. The dataset returns per-sample dicts consumed by FusionDataModule.

Row alignment
-------------
Row i of every dataset key must correspond to the same astronomical object.
For modalities where some objects have no data, store NaN for all embedding
dimensions of that row.

Availability
------------
A sample is considered available for a modality if its embedding is finite
(i.e. no NaN values).  This is checked once at construction time via
``available_mask``, which is a bool array of shape (N,) per modality.

Masking
-------
Set ``modality_mask_prob > 0`` to randomly mask modality availability during
training (analogous to encoder_spectrum's ``train_mask``).  Each available
modality is independently masked with this probability per sample, forcing the
contrastive loss to train on partial-modality batches and improving robustness
to missing data at inference.  Set to 0.0 for validation (no masking).
"""

from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class FusionEmbeddingDataset(Dataset):
    """Multimodal dataset backed by precomputed embeddings in an HDF5 file.

    Args:
        h5_path:           Path to the HDF5 file containing embedding arrays.
        modality_keys:     Mapping from modality name to HDF5 dataset key,
                           e.g. ``{"image": "image_embed", "spectrum": "spectrum_embed"}``.
                           Each key must point to an ``(N, D)`` float32 array.
                           Rows with all-NaN values are treated as missing.
        indices:           Optional subset of row indices (e.g. for train/val
                           splits).  If None, all rows are used.  The array is
                           copied so external modifications do not affect this
                           dataset.
        modality_mask_prob: Probability of randomly masking each *available*
                           modality for a sample (training augmentation).
                           Analogous to ``mask_ratio`` in encoder_spectrum.
                           Set to 0.0 (default) during validation.
    """

    def __init__(
        self,
        h5_path: str,
        modality_keys: dict[str, Union[str, list[str]]],
        indices: Optional[np.ndarray] = None,
        modality_mask_prob: float = 0.0,
        token_mask_keys: Optional[dict[str, str]] = None,
        stats_keys: Optional[dict[str, str]] = None,
    ):
        if not modality_keys:
            raise ValueError("modality_keys must contain at least one entry.")

        h5_path = Path(h5_path)
        if not h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_path}")

        token_mask_keys = token_mask_keys or {}
        stats_keys = stats_keys or {}

        self._embeddings: dict[str, np.ndarray] = {}
        self._available: dict[str, np.ndarray] = {}   # (N,) bool per modality
        self._token_masks: dict[str, np.ndarray] = {} # (N, T) bool, token-seq modalities only
        self._stats: dict[str, np.ndarray] = {}       # (N, S) per-sample side-feature
        self._mask_prob = modality_mask_prob

        n_samples: Optional[int] = None
        with h5py.File(h5_path, "r") as f:
            available_keys = list(f.keys())
            for name, key in modality_keys.items():
                # A modality key may be a single H5 key (str) or a LIST of keys
                # concatenated along the token axis — e.g. prepend the image CLS
                # to the patch tokens: image: [image_patch_embed, image_cls_embed].
                # 2-D members (N, D) become one token (N, 1, D); 3-D members keep
                # their token axis. All members must share the last (feature) dim.
                # A single str key keeps its native rank (2-D vector or 3-D seq).
                keys = [key] if isinstance(key, str) else list(key)
                parts: list[np.ndarray] = []
                for k in keys:
                    if k not in f:
                        raise KeyError(
                            f"Key '{k}' (modality '{name}') not found in {h5_path}. "
                            f"Available keys: {available_keys}"
                        )
                    arr = f[k][:].astype(np.float32)
                    if arr.ndim not in (2, 3):
                        raise ValueError(
                            f"Array '{k}' (modality '{name}') must be 2-D (N, D) or "
                            f"3-D (N, T, D), got shape {arr.shape}"
                        )
                    if not isinstance(key, str) and arr.ndim == 2:
                        arr = arr[:, None, :]            # (N, D) -> (N, 1, D) single token
                    parts.append(arr)

                if len(parts) == 1:
                    emb = parts[0]
                else:
                    feat = parts[0].shape[-1]
                    if any(p.shape[-1] != feat for p in parts):
                        raise ValueError(
                            f"Modality '{name}' concatenates {keys} but feature dims "
                            f"differ: {[p.shape[-1] for p in parts]}"
                        )
                    emb = np.concatenate(parts, axis=1)  # (N, sum T, D) token sequence
                if n_samples is None:
                    n_samples = emb.shape[0]
                elif emb.shape[0] != n_samples:
                    raise ValueError(
                        f"Modality '{name}' (key='{key}') has {emb.shape[0]} rows, "
                        f"expected {n_samples} (same as other modalities)."
                    )
                self._embeddings[name] = emb
                # Available = no NaN anywhere in the row (works for 2-D and 3-D).
                self._available[name] = ~np.any(
                    np.isnan(emb).reshape(emb.shape[0], -1), axis=1
                )  # (N,)

                # Optional per-token validity mask (token-sequence modalities).
                mkey = token_mask_keys.get(name)
                if mkey is not None:
                    if mkey not in f:
                        raise KeyError(
                            f"token_mask key '{mkey}' for modality '{name}' not found "
                            f"in {h5_path}. Available keys: {available_keys}"
                        )
                    self._token_masks[name] = f[mkey][:].astype(bool)  # (N, T)

                # Optional per-sample side-feature stats (e.g. spectrum [mean, std]).
                skey = stats_keys.get(name)
                if skey is not None:
                    if skey not in f:
                        raise KeyError(
                            f"stats key '{skey}' for modality '{name}' not found "
                            f"in {h5_path}. Available keys: {available_keys}"
                        )
                    s = f[skey][:].astype(np.float32)               # (N,) or (N, S)
                    self._stats[name] = s[:, None] if s.ndim == 1 else s

        self._n_total = n_samples
        self._indices: np.ndarray = (
            indices.copy() if indices is not None else np.arange(n_samples)
        )

        modality_coverage = {
            name: self._available[name][self._indices].sum()
            for name in self._embeddings
        }
        print(
            f"FusionEmbeddingDataset: {len(self._indices)} samples, "
            f"mask_prob={modality_mask_prob:.2f}, "
            f"coverage: { {k: int(v) for k, v in modality_coverage.items()} }"
        )

    @property
    def modality_names(self) -> list[str]:
        return list(self._embeddings.keys())

    @property
    def input_dims(self) -> dict[str, int]:
        """Return {modality_name: feature_dim} inferred from loaded arrays.

        Feature dim is the last axis: D for (N, D) vectors, the per-token dim for
        (N, T, D) token sequences.
        """
        return {name: arr.shape[-1] for name, arr in self._embeddings.items()}

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        """Return embeddings and availability masks for one sample.

        Returns a flat dict with keys:
            ``"{name}_emb"``   – ``Tensor (D,)`` — the embedding (may contain NaN
                                  if the modality is missing; use avail flag).
            ``"{name}_avail"`` – ``bool Tensor ()`` — True if the modality is present
                                  *and* was not randomly masked this step.
        """
        real_idx = self._indices[idx]
        result: dict[str, Tensor] = {}
        for name in self._embeddings:
            emb = self._embeddings[name][real_idx]         # (D,) or (T, D) float32
            avail = bool(self._available[name][real_idx])  # scalar bool

            # Random modality masking (training augmentation).
            # Each available modality is independently masked with mask_prob,
            # analogous to encoder_spectrum's per-token train_mask.
            if avail and self._mask_prob > 0.0 and np.random.random() < self._mask_prob:
                avail = False

            result[f"{name}_emb"] = torch.from_numpy(emb.copy())
            result[f"{name}_avail"] = torch.tensor(avail, dtype=torch.bool)
            if name in self._token_masks:
                result[f"{name}_mask"] = torch.from_numpy(
                    self._token_masks[name][real_idx].copy()
                )                                          # (T,) bool
            if name in self._stats:
                result[f"{name}_stats"] = torch.from_numpy(
                    self._stats[name][real_idx].copy()
                )                                          # (S,) float32
        return result
