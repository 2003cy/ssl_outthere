"""Line-aware masking for spectral MAE pre-training.

Provides two public symbols used by MASpecFormer._mask_input and the
benchmark / visualisation notebooks:

    find_line_peaks(flux_valid, prominence) -> np.ndarray
    mask_input(flux, valid_mask, *, mask_ratio, min_unmasked,
               max_line_blocks, line_block_size, line_prominence)
               -> (masked_flux, train_mask)

Algorithm
---------
Per spectrum:
  1. Detect emission-line peaks using scipy.signal.find_peaks with a
     prominence filter on the *normalised* flux (std ≈ 1).  Peaks are
     returned sorted by prominence descending.
  2. Place a centered block of `line_block_size` pixels around each of
     the top min(max_line_blocks, n_peaks) peaks.  Blocks are clipped to
     the valid-pixel index range and placement stops early if the mask
     budget is exhausted.
  3. Fill the remaining budget uniformly at random from non-line valid
     pixels.

Falls back to pure random masking when max_line_blocks=0 or when no
peaks pass the prominence threshold.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def find_line_peaks(flux_valid: np.ndarray, prominence: float) -> np.ndarray:
    """Detect emission-line peaks with a prominence filter.

    Args:
        flux_valid:  1-D numpy array of normalised flux at valid positions.
        prominence:  Minimum peak prominence in normalised-flux units.
                     After per-sample normalisation (std ≈ 1) real emission
                     lines are typically 2–10 σ; noise bumps are ≲ 1 σ.
                     prominence = 1.0 is a robust starting point.

    Returns:
        peak_indices: indices into *flux_valid*, sorted by prominence desc.
                      Empty array when no peaks pass the threshold.
    """
    from scipy.signal import find_peaks as _sp_find_peaks

    if len(flux_valid) < 3:
        return np.empty(0, dtype=np.intp)
    peaks, props = _sp_find_peaks(flux_valid, prominence=prominence)
    if len(peaks) == 0:
        return np.empty(0, dtype=np.intp)
    order = np.argsort(props["prominences"])[::-1]
    return peaks[order]


def mask_input(
    flux: Tensor,
    valid_mask: Tensor,
    *,
    mask_ratio: float,
    min_unmasked: int,
    max_line_blocks: int,
    line_block_size: int,
    line_prominence: float,
) -> Tuple[Tensor, Tensor]:
    """Apply line-aware masking to a batch of normalised spectra.

    Args:
        flux:             (B, T) normalised flux (already mean/std normalised).
        valid_mask:       (B, T) bool, True = valid pixel.
        mask_ratio:       Total fraction of valid pixels to mask.
        min_unmasked:     Skip masking if a spectrum has ≤ this many valid pixels.
        max_line_blocks:  Mask up to this many emission-line blocks (0 = pure random).
        line_block_size:  Width of each centered block around a detected peak.
                          Use odd values: 1 = peak only, 3 = peak±1, 5 = peak±2.
        line_prominence:  Prominence threshold passed to find_line_peaks.

    Returns:
        masked_flux:  (B, T), masked positions zeroed out.
        train_mask:   (B, T) bool, True where a pixel was masked.
    """
    B, T = flux.shape
    half = line_block_size // 2

    masked_flux = flux.clone()
    train_mask  = torch.zeros(B, T, dtype=torch.bool, device=flux.device)

    for i in range(B):
        valid_indices = valid_mask[i].nonzero(as_tuple=True)[0]
        num_valid     = len(valid_indices)

        if num_valid <= min_unmasked:
            continue

        num_to_mask = max(1, int(num_valid * mask_ratio))

        # ── 1. Detect emission-line peaks ─────────────────────────────────
        line_local_set: set = set()
        if max_line_blocks > 0:
            flux_np    = flux[i, valid_indices].detach().cpu().numpy()
            peak_local = find_line_peaks(flux_np, line_prominence)

            # ── 2. Place centered blocks around top peaks ─────────────────
            n_placed = 0
            for p_loc in peak_local:
                if n_placed >= max_line_blocks:
                    break
                if len(line_local_set) >= num_to_mask:
                    break
                for offset in range(-half, half + 1):
                    j = int(p_loc) + offset
                    if 0 <= j < num_valid and len(line_local_set) < num_to_mask:
                        line_local_set.add(j)
                n_placed += 1

        # ── 3. Random pool: fill remaining budget ─────────────────────────
        n_rand   = num_to_mask - len(line_local_set)
        non_line = [j for j in range(num_valid) if j not in line_local_set]
        if n_rand > 0 and len(non_line) > 0:
            perm = torch.randperm(len(non_line), device=flux.device)
            for k in perm[:n_rand].tolist():
                line_local_set.add(non_line[k])

        # ── 4. Apply mask ─────────────────────────────────────────────────
        local_idx = torch.tensor(sorted(line_local_set),
                                 dtype=torch.long, device=flux.device)
        mask_idx  = valid_indices[local_idx]
        masked_flux[i, mask_idx] = 0.0
        train_mask[i, mask_idx]  = True

    return masked_flux, train_mask
