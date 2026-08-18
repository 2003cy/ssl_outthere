"""Token-level, line-aware BLOCK masking for LowResPT patch-MAE pre-training.

Public symbols
--------------
    find_line_peaks(flux_valid, prominence) -> np.ndarray
        Detect emission-line peaks in pixel space.

    mask_patches(patches, flux_norm_pixel, valid_patches, token_valid_mask, *,
                 mask_ratio, min_unmasked, max_line_blocks, line_prominence,
                 patch_size, stride, continuous_patch_length=1)
        -> (masked_patches, train_mask_token, line_mask_token)

Per-spectrum algorithm
----------------------
  1. Emission-line peaks are found with a prominence filter on the normalised
     flux (std ≈ 1) and mapped to the token that best covers each peak.
  2. The top `num_line_detect` distinct valid line tokens (by prominence) form a
     candidate pool; `max_line_blocks` of them are chosen UNIFORMLY AT RANDOM and
     anchored as blocks — emission lines have PRIORITY over random masking, but
     which detected lines get masked varies per draw (augmentation). With
     num_line_detect == max_line_blocks this reduces to "mask the strongest lines".
     max_line_blocks == 0 → pure random masking.
  3. The remaining token budget, ceil(N_valid * mask_ratio), is filled with
     blocks anchored on random free valid tokens.
  4. Every masked token has its full patch vector zeroed.

Block geometry (continuous_patch_length = C)
--------------------------------------------
Each selected token is the *anchor* of a contiguous run of C valid tokens, with
the anchor at a uniformly-random position inside the run (C=3 → left/middle/
right; C=2 → an edge; C=1 → original single-token masking). For C>1, blocks never
share a token AND never abut: a ≥1-token gap separates any two blocks, so every
masked run is exactly C long (never merged into 2C/3C). For C=1 the gap is
dropped — it is plain random MAE where masked tokens may sit adjacent. Only the
line *anchor* token (where the peak lands) is flagged in `line_mask_token`; the
surrounding block tokens count as continuum. `line_mask_token` is used only for
diagnostic line-vs-continuum metrics (no loss weighting).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

# torch RNG for block placement lives on CPU: the index sets are tiny and the
# work is a Python loop anyway, so a host-side generator avoids per-block
# host<->device syncs.
_RNG_DEVICE = "cpu"


def find_line_peaks(flux_valid: np.ndarray, prominence: float) -> np.ndarray:
    """Detect emission-line peaks with a prominence filter.

    Args:
        flux_valid:  1-D numpy array of normalised flux at valid positions.
        prominence:  Minimum peak prominence in normalised-flux units.

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


def _peak_to_token(peak_pixel: int, patch_size: int, stride: int, n_tokens: int) -> int:
    """Map an emission-line peak pixel to the token that best covers it.

    Token t covers pixel p iff  t*stride <= p < t*stride + patch_size. Among the
    covering tokens we pick the one whose patch centre is closest to the peak,
    clamped to the valid token range.
    """

    t_min = max(0, (peak_pixel - patch_size + 1 + stride - 1) // stride)
    t_max = min(n_tokens - 1, peak_pixel // stride)
    t_center = (peak_pixel - patch_size // 2) // max(stride, 1)
    return max(t_min, min(t_max, t_center))


def _try_place_block(
    anchor: int,
    valid_tokens: set,
    occupied: set,
    n_tokens: int,
    block_len: int,
) -> bool:
    """Try to mask a length-`block_len` block containing `anchor`, growing
    `occupied` in place. Returns whether a block was placed.

    A candidate block is accepted only when (a) all of its tokens are valid and
    free → no OVERLAP, and (b) for block_len > 1, both flanking tokens
    (start-1, start+block_len) are free → no TOUCHING, which keeps a ≥1-token gap
    between blocks. The gap is skipped for block_len == 1 (single-token masking
    is plain random MAE: adjacent masked tokens are fine). The anchor is offered
    every in-bounds offset (random order), so it lands at a uniformly-random
    position inside the accepted run.
    """
    for offset in torch.randperm(block_len, device=_RNG_DEVICE).tolist():
        start = anchor - offset
        if start < 0 or start + block_len > n_tokens:
            continue
        block = range(start, start + block_len)
        if any((t not in valid_tokens) or (t in occupied) for t in block):
            continue                                       # overlap / invalid token
        if block_len > 1 and ((start - 1) in occupied or (start + block_len) in occupied):
            continue                                       # would touch a placed block
        occupied.update(block)
        return True
    return False


def mask_patches(
    patches: Tensor,
    flux_norm_pixel: Tensor,
    valid_patches: Tensor,
    token_valid_mask: Tensor,
    *,
    mask_ratio: float,
    min_unmasked: int,
    max_line_blocks: int,
    line_prominence: float,
    patch_size: int,
    stride: int,
    continuous_patch_length: int = 1,
    num_line_detect: int = 1,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Apply token-level line-aware block masking to a batch of patch sequences.

    Args:
        patches:          (B, N, P) normalised flux patches.
        flux_norm_pixel:  (B, L_used) pixel-level normalised flux (for line detection).
        valid_patches:    (B, N, P) bool — kept for API compatibility (unused here).
        token_valid_mask: (B, N) bool, True where the token is valid.
        mask_ratio:       Fraction of valid tokens to mask (target budget).
        min_unmasked:     Skip masking a spectrum with <= this many valid tokens.
        max_line_blocks:  Place up to this many emission-line blocks first (0 = pure random).
        line_prominence:  Prominence threshold for find_line_peaks.
        patch_size:       Number of pixels per patch (P).
        stride:           Stride between consecutive patches.
        continuous_patch_length: Block length C (1, 2, or 3); see module docstring.
        num_line_detect:  Size of the candidate pool of strongest detected lines
                          from which `max_line_blocks` are picked at random
                          (clamped to >= max_line_blocks).

    Returns:
        masked_patches:    (B, N, P) — masked tokens zeroed out.
        train_mask_token:  (B, N) bool — True where a token was masked.
        line_mask_token:   (B, N) bool — True for the emission-line anchor tokens
                                          (subset of train_mask_token; empty when
                                          max_line_blocks == 0).
    """
    B, N, _ = patches.shape
    C = max(1, int(continuous_patch_length))

    masked_patches   = patches.clone()
    train_mask_token = torch.zeros(B, N, dtype=torch.bool, device=patches.device)
    line_mask_token  = torch.zeros(B, N, dtype=torch.bool, device=patches.device)

    for i in range(B):
        valid_list = token_valid_mask[i].nonzero(as_tuple=True)[0].tolist()  # ascending
        if len(valid_list) <= min_unmasked:
            continue

        budget = max(1, math.ceil(len(valid_list) * mask_ratio))
        valid_tokens = set(valid_list)
        occupied: set = set()
        line_anchors: List[int] = []

        # 1. Emission-line blocks have priority. Detect the strongest lines, then
        #    mask a random subset of them.
        if max_line_blocks > 0:
            peak_pixels = find_line_peaks(
                flux_norm_pixel[i].detach().cpu().numpy(), line_prominence
            )
            # Candidate pool: the top `pool_size` distinct valid line tokens by
            # prominence (>= max_line_blocks so we can always fill the quota).
            pool_size = max(num_line_detect, max_line_blocks)
            candidates: List[int] = []
            seen: set = set()
            for peak_pixel in peak_pixels:
                if len(candidates) >= pool_size:
                    break
                token = _peak_to_token(int(peak_pixel), patch_size, stride, N)
                if token in seen or token not in valid_tokens:
                    continue
                seen.add(token)
                candidates.append(token)
            # Randomly choose which detected lines to actually mask.
            for k in torch.randperm(len(candidates), device=_RNG_DEVICE).tolist():
                if len(line_anchors) >= max_line_blocks or len(occupied) >= budget:
                    break
                token = candidates[k]
                if token in occupied:                  # swallowed by an earlier line block
                    continue
                if _try_place_block(token, valid_tokens, occupied, N, C):
                    line_anchors.append(token)

        # 2. Random blocks fill the remaining budget.
        free = [t for t in valid_list if t not in occupied]
        for k in torch.randperm(len(free), device=_RNG_DEVICE).tolist():
            if len(occupied) >= budget:
                break
            anchor = free[k]
            if anchor in occupied:                 # swallowed by an earlier block
                continue
            _try_place_block(anchor, valid_tokens, occupied, N, C)

        # 3. Commit this spectrum's masked tokens (and flag the line anchors).
        if occupied:
            idx = torch.tensor(sorted(occupied), dtype=torch.long, device=patches.device)
            masked_patches[i, idx] = 0.0
            train_mask_token[i, idx] = True
            if line_anchors:
                li = torch.tensor(sorted(line_anchors), dtype=torch.long,
                                  device=patches.device)
                line_mask_token[i, li] = True

    return masked_patches, train_mask_token, line_mask_token
