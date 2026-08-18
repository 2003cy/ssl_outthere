"""DJA low-resolution spectrum dataset for LowResPT (reads the merged FITS).

Observed-frame wavelengths only — SSL pretraining sees observed-frame input.
Redshift is returned alongside for downstream analysis but never used to
transform the model input (would be label leakage).

Source file
-----------
`DJA_spectra_v4.5.fits` (built by data/survey/DJA/build_dja_spectra_catalog.py):

    HDU 'CATALOG' : one row per PRISM object, all CSV metadata columns plus
                    vector columns flux/err/full_err/valid_spec, each (473,)
    HDU 'WAVE'    : the shared wavelength grid (473,), micron

Unlike the old per-object HDF5 files, the FITS stores every object on the *same*
473-pixel wavelength grid, so a wavelength window selects the same pixel indices
for every spectrum. All selection that used to happen at HDF5-build time (grade
and obs-fraction pre-cuts) and at dataset time (S/N, redshift) is now performed
here, at the dataset-class level.

Flux units: stored as f_nu in microjansky. With use_jansky=False (default) the
flux is converted to f_lambda (∝ f_nu / λ²); with use_jansky=True the raw uJy
(f_nu) flux is returned unchanged.

Valid mask: the per-pixel `valid_spec` reduction flag (conversion-independent;
not inferred from flux != 0, since a valid pixel may legitimately have flux 0).

IO note
-------
A FITS BinTable is column-strided on disk, so reading a single object per
``__getitem__`` (``hdu.data['flux'][idx]``) would re-stride the whole 160 MB
column every call. Instead we read the needed columns once in ``__init__``,
slice to the wavelength window (→ a few MB), and keep them in RAM. This both
avoids per-sample IO and removes the per-worker file-handle / mmap juggling the
HDF5 version needed: forked DataLoader workers share the arrays copy-on-write.
"""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from astropy.io import fits
from torch.utils.data import Dataset

# Scalar metadata columns pulled from the CATALOG HDU and exposed as attributes
# (dataset order) for downstream notebooks (e.g. group-aware split by objid).
_META_COLS = (
    "sn50", "z_best", "grade", "obs_365_frac",
    "srcid", "objid", "ra", "dec", "phot_f150w_tot_1",
)


class LowResDataset(Dataset):
    """DJA low-res spectrum dataset (observed-frame), read from the merged FITS.

    Args:
        fits_path:    Path to DJA_spectra_v4.5.fits.
        grades:       Keep only these quality grades (None = no grade cut).
                      Default (1, 2, 3) reproduces the old HDF5 pre-selection.
        min_obs_frac: Keep spectra with obs_365_frac > this (None = no cut).
                      Default 0.5 reproduces the old HDF5 pre-selection.
        min_sn50:     Exclude spectra with sn50 < this value (None = no cut).
        min_redshift: Exclude spectra with z_best <= this value (None = no cut).
        max_redshift: Exclude spectra with z_best > this value (None = no cut).
        wl_ref_min:   Lower bound (µm) of the observed-frame wavelength window;
                      pixels with wave <= wl_ref_min are dropped (None = no cut).
        wl_ref_max:   Upper bound (µm); pixels with wave >= wl_ref_max are dropped
                      (None = no cut). Default window (1.0, 2.0) µm matches the
                      old low-res cutout coverage (56 pixels).
        frac_valid_pix: Keep only spectra whose fraction of valid pixels INSIDE
                      the [wl_ref_min, wl_ref_max] window is > this value, in
                      [0, 1] (None = no cut). Removes near-empty spectra that have
                      little/no usable data in the window and would collapse to a
                      degenerate (all-zero) embedding.
        use_jansky:   If True, return the raw uJy (f_nu) flux. If False (default),
                      convert to f_lambda (∝ f_nu / λ²).
        err_column:   Which per-pixel error column to load for inverse-variance
                      loss weighting ("full_err" includes systematic-error
                      inflation; "err" is the formal pipeline error). Same units
                      as flux, converted to f_lambda alongside it.
    """

    def __init__(
        self,
        fits_path: str,
        grades: Optional[Sequence[int]] = (1, 2, 3),
        min_obs_frac: Optional[float] = 0.,
        min_sn50: Optional[float] = None,
        min_redshift: Optional[float] = None,
        max_redshift: Optional[float] = None,
        wl_ref_min: Optional[float] = 1.0,
        wl_ref_max: Optional[float] = 2.0,
        frac_valid_pix: Optional[float] = None,
        use_jansky: bool = False,
        err_column: str = "full_err",
    ):
        self.use_jansky = use_jansky
        self.err_column = err_column
        if not self.use_jansky:
            print(
                "Converting flux from f_nu (uJy) to f_lambda (∝ f_nu / λ²) "
                "(use_jansky=True to disable this conversion)"
            )
        self.fits_path = Path(fits_path)
        if not self.fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {fits_path}")

        # ── Read the shared wavelength grid and select the wavelength window ──
        # memmap=False: read fully and close, so no file descriptor lingers
        # across the DataLoader worker fork.
        with fits.open(self.fits_path, memmap=False) as hdul:
            wave = np.asarray(hdul["WAVE"].data, dtype=np.float32)  # (L,)

            wl_keep = np.ones(wave.shape[0], dtype=bool)
            if wl_ref_min is not None:
                wl_keep &= wave > wl_ref_min
            if wl_ref_max is not None:
                wl_keep &= wave < wl_ref_max
            self.wave = wave[wl_keep]  # (Lwin,) shared across all objects

            cat = hdul["CATALOG"].data
            total_samples = cat.shape[0]

            # Flux: read the column once, slice to the window, NaN/Inf → 0.
            flux = np.asarray(cat["flux"], dtype=np.float32)[:, wl_keep]
            finite = np.isfinite(flux)
            flux[~finite] = 0.0

            # Per-pixel error for inverse-variance loss weighting. Non-finite or
            # non-positive errors are sentinels (bad pixels) — left as-is here
            # and excluded downstream (model._token_weights drops err <= 0 /
            # non-finite via the err_ok mask).
            err = np.asarray(cat[self.err_column], dtype=np.float32)[:, wl_keep]

            # Valid mask: the authoritative per-pixel validity flag from the data
            valid = np.asarray(cat["valid_spec"], dtype=bool)[:, wl_keep] & finite

            meta = {c: np.asarray(cat[c]) for c in _META_COLS}

        # ── Build the selection mask (all filtering at dataset level) ──
        mask = np.ones(total_samples, dtype=bool)

        def _report(name):
            print(f"{name} filtering: {mask.sum()}/{total_samples} samples kept")

        if grades is not None:
            mask &= np.isin(meta["grade"], np.asarray(grades))
            _report(f"grade in {tuple(grades)}")

        '''
        if min_obs_frac is not None:
            obsf = meta["obs_365_frac"].astype(np.float64)
            mask &= np.isfinite(obsf) & (obsf > min_obs_frac)
            _report(f"obs_365_frac > {min_obs_frac}")
        '''

        if min_sn50 is not None:
            sn50 = meta["sn50"].astype(np.float64)
            mask &= np.isfinite(sn50) & (sn50 >= min_sn50)
            _report(f"sn50 >= {min_sn50}")

        if min_redshift is not None:
            z = meta["z_best"].astype(np.float64)
            mask &= np.isfinite(z) & (z > min_redshift)
            _report(f"z_best > {min_redshift}")

        if max_redshift is not None:
            z = meta["z_best"].astype(np.float64)
            mask &= np.isfinite(z) & (z <= max_redshift)
            _report(f"z_best <= {max_redshift}")

        # Window-coverage cut: keep only spectra whose fraction of valid pixels
        # INSIDE the [wl_ref_min, wl_ref_max] window exceeds frac_valid_pix.
        # `valid` is already sliced to the window, so valid.mean(axis=1) is that
        # fraction per object. Drops near-empty spectra (e.g. zero valid pixels in
        # the window) that would otherwise collapse to an identical all-zero
        # embedding / patch sequence and pin a probe's predictions to one value.
        if frac_valid_pix is not None:
            valid_frac = valid.mean(axis=1)                  # (total_samples,) in [0,1]
            mask &= valid_frac > frac_valid_pix
            _report(f"valid_pix_frac > {frac_valid_pix}")

        self.valid_indices = np.where(mask)[0]
        self.n_samples = len(self.valid_indices)

        # ── Preload filtered, windowed arrays into RAM (dataset order) ──
        self._flux = np.ascontiguousarray(flux[self.valid_indices])    # (n, Lwin) raw f_nu
        self._err = np.ascontiguousarray(err[self.valid_indices])      # (n, Lwin) raw f_nu
        self._valid = np.ascontiguousarray(valid[self.valid_indices])  # (n, Lwin) bool
        for c in _META_COLS:
            setattr(self, c, meta[c][self.valid_indices])

        print(f"LowResDataset: {self.n_samples} spectra, {self.wave.shape[0]} pixels "
              f"({self.wave[0]:.3f}–{self.wave[-1]:.3f} µm)")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """Load a single spectrum (observed-frame wavelength).

        Returns:
            flux:       Tensor (Lwin,)  f_lambda (∝ f_nu/λ²) by default, or uJy
                                        (f_nu) if use_jansky=True
            wavelength: Tensor (Lwin,)  observed-frame wavelength in microns
            valid_mask: Tensor (Lwin,)  bool, the `valid_spec` reduction flag
            err:        Tensor (Lwin,)  per-pixel error in the same units as flux
            redshift:   scalar Tensor, z_best (downstream analysis only)
        """
        flux       = self._flux[idx].copy()       # raw f_nu (uJy)
        err        = self._err[idx].copy()        # raw f_nu (uJy)
        wavelength = self.wave                     # shared grid (Lwin,)

        # Authoritative per-pixel validity from the `valid_spec` column
        # (conversion-independent; not inferred from flux != 0).
        valid_mask = self._valid[idx].copy()

        if not self.use_jansky:
            # f_lambda ∝ f_nu / λ². Wavelength is observed-frame µm, finite & >0,
            # the 1/λ² shape matters — c folded to 1. err scales the same way.
            flux = flux / (wavelength ** 2)
            err = err / (wavelength ** 2)

        return {
            "flux":       torch.from_numpy(flux),
            "wavelength": torch.from_numpy(wavelength.copy()),
            "valid_mask": torch.from_numpy(valid_mask),
            "err":        torch.from_numpy(err),
            "redshift":   torch.tensor(np.float32(self.z_best[idx])),
        }
