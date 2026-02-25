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
            if min_snr is not None:
                if "spec_sn" not in f:
                    import warnings
                    warnings.warn(
                        f"min_snr={min_snr} was specified but the HDF5 file has no "
                        f"'spec_sn' dataset. SNR filtering is skipped and all "
                        f"{total_samples} samples will be used.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self.valid_indices = np.arange(total_samples)
                else:
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


class JDASpectrumDataset(Dataset):
    """JDA spectrum dataset from a single HDF5 file produced by export_jda_spectrum.py.

    The HDF5 file is expected to contain:
        wave             float32  (N, L)  wavelength in microns
        flux             float32  (N, L)  flux in uJy  (bad pixels already zeroed)
        sn50             float32  (N,)    median S/N from catalog
        z_best           float32  (N,)    best-fit redshift
        srcid            int64    (N,)    source ID
        objid            int64    (N,)    object ID
        ra               float32  (N,)    right ascension (deg)
        dec              float32  (N,)    declination (deg)
        phot_f150w_tot_1 float32  (N,)    F150W total photometry

    Valid mask: True where flux != 0  (bad pixels were zeroed during export).
    Each sample returns the same (flux, wavelength, valid_mask) dict as MASpectrumDataset.

    Args:
        h5_path:  Path to the JDA HDF5 file.
        min_sn50: Minimum sn50 threshold; samples below it are excluded.
                  Default: None (no filtering).
    """

    def __init__(
        self,
        h5_path: str,
        min_sn50: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        with h5py.File(self.h5_path, "r") as f:
            total_samples = f["flux"].shape[0]
            self.spec_len = f["flux"].shape[1]

            wave_all = f["wave"][:]  # (N, L), NaN-padded

        # Actual spectrum length = number of finite wavelength pixels per row
        lengths = np.sum(np.isfinite(wave_all), axis=1)  # (N,)

        mask = np.ones(total_samples, dtype=bool)

        if min_length is not None:
            mask &= lengths >= min_length
            print(
                f"min_length={min_length}: {mask.sum()}/{total_samples} "
                f"samples with length >= {min_length}"
            )

        if max_length is not None:
            mask &= lengths <= max_length
            print(
                f"max_length={max_length}: {mask.sum()}/{total_samples} "
                f"samples with length <= {max_length}"
            )

        if min_sn50 is not None:
            with h5py.File(self.h5_path, "r") as f:
                sn50 = f["sn50"][:]
            mask &= np.isfinite(sn50) & (sn50 >= min_sn50)
            print(
                f"sn50 filtering: {mask.sum()}/{total_samples} "
                f"samples with sn50 >= {min_sn50}"
            )

        self.valid_indices = np.where(mask)[0]
        self.n_samples = len(self.valid_indices)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """Load a single spectrum.

        Returns:
            dict with keys:
                flux       Tensor (L,)  flux in uJy
                wavelength Tensor (L,)  wavelength in microns
                valid_mask Tensor (L,)  bool, True where flux != 0
        """
        real_idx = self.valid_indices[idx]

        with h5py.File(self.h5_path, "r") as f:
            flux       = f["flux"][real_idx].astype(np.float32)
            wavelength = f["wave"][real_idx].astype(np.float32)

        # Truncate to the actual spectrum length (finite wavelength pixels are
        # stored contiguously at the start; the rest is NaN padding).
        # This avoids passing 3971-length tensors when only 56 pixels are real.
        actual_len = int(np.isfinite(wavelength).sum())
        flux       = flux[:actual_len]
        wavelength = wavelength[:actual_len]

        # valid: non-zero flux (wavelength is always finite after truncation)
        valid_mask = flux != 0.0

        return {
            "flux":       torch.from_numpy(flux),
            "wavelength": torch.from_numpy(wavelength),
            "valid_mask": torch.from_numpy(valid_mask),
        }
