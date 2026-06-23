"""
preprocessing.py — image pre-processing for the AstroDINO benchmark.

Single-band only (in_chans=1). ``AsinhStretch`` is a sign-preserving arcsinh
stretch that matches the training pipeline (encoder_image/astrodino/train/data/
augmentations.py): ``out = sign(b)*arcsinh(Q*|b|)/sqrt(Q)`` with ``b = img*scale``,
no clipping at 0 so negative sky noise is preserved.

Factory:
  get_stretch(cfg) -> (stretch_callable, in_chans=1)
"""

import numpy as np


class AsinhStretch:
    """Sign-preserving arcsinh stretch for a single-band cutout.

    Mirrors the training stretch exactly. Input (1,H,W), (C,H,W) (uses channel 0)
    or (H,W); returns (1, H, W) float32 (centred at 0, negatives kept).

    Parameters
    ----------
    scale : float
        Multiplicative scale applied before the stretch (default 1.0).
    Q : float
        Arcsinh stretch factor (default 20.0).
    """

    def __init__(self, scale: float = 1.0, Q: float = 20.0):
        self.scale = scale
        self.Q = Q

    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        arr = np.asarray(imgs, dtype=np.float32)
        if arr.ndim == 3:
            base = arr[0]
        elif arr.ndim == 2:
            base = arr
        else:
            base = arr.squeeze()

        b = base * self.scale
        out = np.sign(b) * np.arcsinh(self.Q * np.abs(b)) / np.sqrt(self.Q + 1e-8)
        return out.astype(np.float32)[np.newaxis]   # (1, H, W)


def get_stretch(cfg=None):
    """Return (stretch, in_chans). Single-band only; in_chans is always 1."""
    return AsinhStretch(), 1
