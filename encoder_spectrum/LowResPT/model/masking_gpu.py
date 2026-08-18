"""Vectorised block masking, a drop-in replacement for `masking.mask_patches`.

The reference implementation loops over the batch in Python and calls `.tolist()`
several times per spectrum, so at batch 1024 it forces ~3000 GPU
synchronisations per step and dominates training time.

Here the loop is over *blocks* instead of over samples: every iteration places
one block in all spectra at once, so the trip count is a small constant
(`N // C + 1`, i.e. 7 to 28 for the 27-token grid) rather than the batch size.
Each iteration is pure tensor work with no host round-trip.

Only pure random masking is vectorised. Line-aware masking depends on
`scipy.signal.find_peaks`, which is per-spectrum and CPU-only, so
`max_line_blocks > 0` falls straight through to the original implementation and
behaves exactly as before.

Constraints reproduced from `_try_place_block`:
  * every token of a block is valid and not already taken (no overlap),
  * for C > 1 both flanking tokens are free, keeping a >=1-token gap between
    blocks; that gap is not required for C == 1,
  * a spectrum with <= `min_unmasked` valid tokens is left untouched,
  * blocks are added until the number of masked tokens reaches the budget
    `max(1, ceil(n_valid * mask_ratio))`, so the last block may overshoot it.
"""

from typing import Tuple

import torch
from torch import Tensor

from .masking import mask_patches


def mask_patches_fast(
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
    """Same signature and return contract as `masking.mask_patches`."""
    if max_line_blocks > 0:
        return mask_patches(
            patches, flux_norm_pixel, valid_patches, token_valid_mask,
            mask_ratio=mask_ratio, min_unmasked=min_unmasked,
            max_line_blocks=max_line_blocks, line_prominence=line_prominence,
            patch_size=patch_size, stride=stride,
            continuous_patch_length=continuous_patch_length,
            num_line_detect=num_line_detect,
        )

    B, N, _ = patches.shape
    C = max(1, int(continuous_patch_length))
    dev = patches.device
    valid = token_valid_mask

    occupied = torch.zeros(B, N, dtype=torch.bool, device=dev)
    if C > N:
        return patches.clone(), occupied, occupied.clone()

    n_valid = valid.sum(1)
    budget = torch.ceil(n_valid.float() * mask_ratio).long().clamp(min=1)
    eligible = n_valid > min_unmasked

    W = N - C + 1                      # number of candidate block starts
    valid_run = valid.unfold(1, C, 1).all(-1)          # (B, W) all C tokens valid

    for _ in range(N // C + 1):
        free = ~occupied
        ok = valid_run & free.unfold(1, C, 1).all(-1)
        if C > 1:
            # Flanking tokens must be free; out of range counts as free.
            left = torch.ones(B, W, dtype=torch.bool, device=dev)
            left[:, 1:] = free[:, :W - 1]
            right = torch.ones(B, W, dtype=torch.bool, device=dev)
            if N - C > 0:
                right[:, :N - C] = free[:, C:N]
            ok = ok & left & right

        need = eligible & (occupied.sum(1) < budget) & ok.any(1)
        if not bool(need.any()):
            break

        # One uniformly-random legal start per spectrum.
        start = torch.rand(B, W, device=dev).masked_fill(~ok, -1.0).argmax(1)
        idx = start.unsqueeze(1) + torch.arange(C, device=dev)      # (B, C)
        place = torch.zeros_like(occupied).scatter_(1, idx, True)
        occupied |= place & need.unsqueeze(1)

    train_mask_token = occupied & valid
    masked_patches = patches * (~train_mask_token).unsqueeze(-1)
    line_mask_token = torch.zeros_like(train_mask_token)
    return masked_patches, train_mask_token, line_mask_token
