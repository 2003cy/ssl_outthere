"""Dataset for multimodal fusion training using precomputed embeddings.

Workflow
--------
1. Run each frozen encoder offline to produce per-modality embedding files::

       # Image encoder → save (N, 768) array
       np.save("embeddings/image.npy", image_cls_tokens)

       # Spectrum encoder → save (N, 128) array
       np.save("embeddings/spectrum.npy", spectrum_cls_tokens)

   Missing embeddings for a sample should be stored as all-NaN rows.

2. Pass the paths to FusionEmbeddingDataset::

       dataset = FusionEmbeddingDataset(
           modality_paths={"image": "embeddings/image.npy",
                           "spectrum": "embeddings/spectrum.npy"},
       )

3. The dataset returns per-sample dicts consumed by FusionDataModule.

Row alignment
-------------
Row i of every embedding file must correspond to the same astronomical object.
For modalities where some objects have no data (e.g. no spectrum), store NaN
for all embedding dimensions of that row.

Availability
------------
A sample is considered available for a modality if its embedding is finite
(i.e. no NaN values).  This is checked once at construction time via
``available_mask``, which is a bool array of shape (N,) per modality.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class FusionEmbeddingDataset(Dataset):
    """Multimodal dataset backed by precomputed embedding numpy files.

    Args:
        modality_paths: Mapping from modality name to path of a ``.npy`` file
                        containing an ``(N, D)`` float32 array.  Rows with all-NaN
                        values are treated as missing for that modality.
        indices:        Optional subset of row indices to use (e.g. for train/val
                        splits).  If None, all rows are used.
    """

    def __init__(
        self,
        modality_paths: dict[str, str],
        indices: Optional[np.ndarray] = None,
    ):
        if not modality_paths:
            raise ValueError("modality_paths must contain at least one entry.")

        self._embeddings: dict[str, np.ndarray] = {}
        self._available: dict[str, np.ndarray] = {}   # (N,) bool per modality

        n_samples: Optional[int] = None
        for name, path in modality_paths.items():
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Embedding file not found for modality '{name}': {path}")
            emb = np.load(path, allow_pickle=False).astype(np.float32)
            if emb.ndim != 2:
                raise ValueError(
                    f"Embedding array for '{name}' must be 2-D (N, D), got shape {emb.shape}"
                )
            if n_samples is None:
                n_samples = emb.shape[0]
            elif emb.shape[0] != n_samples:
                raise ValueError(
                    f"Modality '{name}' has {emb.shape[0]} rows, "
                    f"expected {n_samples} (same as other modalities)."
                )
            self._embeddings[name] = emb
            # Available = no NaN in the embedding row
            self._available[name] = ~np.any(np.isnan(emb), axis=1)  # (N,)

        self._n_total = n_samples
        self._indices: np.ndarray = (
            indices if indices is not None else np.arange(n_samples)
        )

        modality_coverage = {
            name: self._available[name][self._indices].sum()
            for name in self._embeddings
        }
        print(
            f"FusionEmbeddingDataset: {len(self._indices)} samples, "
            f"coverage: { {k: int(v) for k, v in modality_coverage.items()} }"
        )

    @property
    def modality_names(self) -> list[str]:
        return list(self._embeddings.keys())

    @property
    def input_dims(self) -> dict[str, int]:
        """Return {modality_name: embedding_dim} inferred from loaded arrays."""
        return {name: arr.shape[1] for name, arr in self._embeddings.items()}

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        """Return embeddings and availability masks for one sample.

        Returns a flat dict with keys:
            ``"{name}_emb"``   – ``Tensor (D,)`` — the embedding (may contain NaN
                                  if the modality is missing; masked by avail flag).
            ``"{name}_avail"`` – ``bool Tensor ()`` — True if the modality is present.
        """
        real_idx = self._indices[idx]
        result: dict[str, Tensor] = {}
        for name in self._embeddings:
            emb = self._embeddings[name][real_idx]          # (D,) numpy float32
            avail = bool(self._available[name][real_idx])   # scalar bool
            result[f"{name}_emb"] = torch.from_numpy(emb.copy())
            result[f"{name}_avail"] = torch.tensor(avail, dtype=torch.bool)
        return result
