"""Multi-band spectrum dataset for OUTTHERE.

Reads from per-filter HDF5 files (f115w.h5, f150w.h5, f200w.h5).
Each sample returns (flux, wavelength, valid_mask) where:
- flux: shape (L,), the spectrum values (NaN/Inf replaced with 0)
- wavelength: shape (L,), the wavelength grid
- valid_mask: shape (L,), True where flux is valid (not 0, -99, NaN, or Inf)

Processing:
1. valid_mask is computed BEFORE modifying flux (marks original bad values)
2. NaN and Inf values in flux are then replaced with 0
3. Training uses valid_mask to exclude bad positions from loss
"""

from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MASpectrumDataset(Dataset):
    """Single-band spectrum dataset from HDF5.
    
    Args:
        h5_path: Path to the filter HDF5 file (e.g., f115w.h5)
        valid_flux_mask_fn: Function to compute valid_mask from flux.
                            Default: exclude 0, -99, and non-finite values.
        min_snr: Minimum SNR threshold. Samples with SNR < min_snr are excluded.
                Default: None (no filtering).
    """

    FILL_VALUE = np.float32(-99.0)

    def __init__(
        self,
        h5_path: str,
        valid_flux_mask_fn=None,
        min_snr: Optional[float] = None,
    ):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        # Read metadata from HDF5 attrs
        with h5py.File(self.h5_path, "r") as f:
            total_samples = f["flux"].shape[0]
            self.spec_len = f["flux"].shape[1]
            self.filter_name = f.attrs.get("filter", "unknown")
            
            # SNR filtering
            if min_snr is not None and "spec_sn" in f:
                snr = f["spec_sn"][:]
                self.valid_indices = np.where(snr >= min_snr)[0]
                print(f"SNR filtering: {len(self.valid_indices)}/{total_samples} samples with SNR >= {min_snr}")
            else:
                self.valid_indices = np.arange(total_samples)
        
        self.n_samples = len(self.valid_indices)

        # Default valid mask: exclude 0, -99, and non-finite
        if valid_flux_mask_fn is None:
            self.valid_flux_mask_fn = self._default_valid_mask
        else:
            self.valid_flux_mask_fn = valid_flux_mask_fn

    @staticmethod
    def _default_valid_mask(flux: np.ndarray) -> np.ndarray:
        """Return True where flux is valid (not 0, -99, or NaN/Inf)."""
        return (
            (flux != 0)
            & ~np.isclose(flux, MASpectrumDataset.FILL_VALUE, atol=1e-5)
            & np.isfinite(flux)
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """Load a single spectrum.
        
        Returns:
            dict with keys:
                - flux: shape (L,), flux values (with NaN/Inf replaced)
                - wavelength: shape (L,), wavelength grid
                - valid_mask: shape (L,), bool where flux is valid
        """
        # Map to actual index in HDF5 file
        real_idx = self.valid_indices[idx]
        
        with h5py.File(self.h5_path, "r") as f:
            flux = f["flux"][real_idx].astype(np.float32)
            wavelength = f["wave"][real_idx].astype(np.float32)

        # Compute valid mask BEFORE modifying flux
        valid_mask = self.valid_flux_mask_fn(flux)

        # Handle NaN and Inf values: replace with 0
        flux = np.where(np.isfinite(flux), flux, 0.0)

        return {
            "flux": torch.from_numpy(flux),  # shape (L,)
            "wavelength": torch.from_numpy(wavelength),  # shape (L,)
            "valid_mask": torch.from_numpy(valid_mask),  # shape (L,) bool
        }
