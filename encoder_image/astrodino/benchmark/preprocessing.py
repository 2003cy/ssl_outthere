"""
preprocessing.py — image pre-processing utilities for the AstroDINO benchmark.

Two ToRGB implementations:

  ToRGB3Band  Legacy-Survey-style colour mapping for models trained with in_chans=3.
              Takes (3,H,W) or (H,W,C) input (pass 3 identical channels for
              single-band JWST data) and returns (3, H, W) float32 in [0, 1].

  ToRGB1Band  Arcsinh stretch for models trained with in_chans=1.
              Takes (1,H,W), (C,H,W) (uses channel 0), or (H,W) input
              and returns (1, H, W) float32 in [0, 1].

Factory:
  get_torgb(cfg) -> (to_rgb_callable, in_chans: int)
      Reads cfg.student.in_chans (defaults to 3 if absent).
      Pass an OmegaConf config, a plain int, or None.
"""

import numpy as np
from omegaconf import OmegaConf


class ToRGB3Band:
    """
    Legacy-Survey-style colour mapping (g/r/z → RGB).

    Matches the training preprocessing used for models built with in_chans=3.
    For single-band JWST data, repeat the image to 3 identical channels before
    calling; the mapping still produces a valid arcsinh-stretched result.

    Parameters
    ----------
    scales : dict, optional
        Override per-band (plane, scale) pairs.
    m : float
        Softening constant (default 0.03).
    Q : float
        Arcsinh stretch factor (default 20).
    bands : sequence of str
        Band names, must be keys in rgb_scales (default ["g", "r", "z"]).
    """

    _RGB_SCALES = {
        "u": (2, 1.5),
        "g": (2, 6.0),
        "r": (1, 3.4),
        "i": (0, 1.0),
        "z": (0, 2.2),
    }

    def __init__(self, scales=None, m=0.03, Q=20, bands=("g", "r", "z")):
        rgb_scales = dict(self._RGB_SCALES)
        if scales:
            rgb_scales.update(scales)
        self.rgb_scales = rgb_scales
        self.m = m
        self.Q = Q
        self.bands = list(bands)

    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        arr = np.asarray(imgs, dtype=np.float32)
        # Accept (C,H,W) or (H,W,C)
        if arr.shape[0] != len(self.bands):
            arr = arr.transpose(2, 0, 1)

        # Mean intensity for the arcsinh stretch
        I = np.zeros(arr.shape[1:], dtype=np.float32)
        for img, band in zip(arr, self.bands):
            _, scale = self.rgb_scales[band]
            I += np.maximum(0.0, img * scale + self.m)
        I /= len(self.bands)

        fI = np.arcsinh(self.Q * I) / (np.sqrt(self.Q) + 1e-8)
        I  = np.where(I == 0.0, 1e-6, I)

        H, W = arr.shape[1], arr.shape[2]
        rgb = np.zeros((3, H, W), dtype=np.float32)
        for img, band in zip(arr, self.bands):
            plane, scale = self.rgb_scales[band]
            rgb[plane] = (img * scale + self.m) * fI / I

        return np.clip(rgb, 0.0, 1.0)  # (3, H, W)


class ToRGB1Band:
    """
    Arcsinh stretch for a single-band cutout.

    Matches the training preprocessing used for models built with in_chans=1.

    Parameters
    ----------
    scale : float
        Multiplicative scale applied before the stretch (default 1.0).
    m : float
        Softening constant (default 0.03).
    Q : float
        Arcsinh stretch factor (default 20.0).
    """

    def __init__(self, scale: float = 1.0, m: float = 0.03, Q: float = 20.0):
        self.scale = scale
        self.m = m
        self.Q = Q

    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        arr = np.asarray(imgs, dtype=np.float32)
        if arr.ndim == 3:
            base = arr[0]           # take first channel of (C,H,W)
        elif arr.ndim == 2:
            base = arr
        else:
            base = arr.squeeze()

        base = np.maximum(0.0, base * self.scale + self.m)
        I    = np.where(base == 0.0, 1e-6, base)
        fI   = np.arcsinh(self.Q * I) / (np.sqrt(self.Q) + 1e-8)
        out  = np.clip(base * fI / I, 0.0, 1.0).astype(np.float32)
        return out[np.newaxis]      # (1, H, W)


def get_torgb(cfg):
    """
    Return (to_rgb, in_chans) for the given model config.

    Parameters
    ----------
    cfg : OmegaConf DictConfig, int, or None
        - OmegaConf config  : reads cfg.student.in_chans (default 3 if absent).
        - int               : used directly as in_chans.
        - None              : treated as in_chans=3.

    Returns
    -------
    to_rgb   : ToRGB1Band if in_chans == 1, else ToRGB3Band
    in_chans : int  (1 or 3)
    """
    if cfg is None:
        in_chans = 3
    elif isinstance(cfg, int):
        in_chans = cfg
    else:
        in_chans = int(OmegaConf.select(cfg, "student.in_chans", default=3) or 3)

    if in_chans == 1:
        return ToRGB1Band(), 1
    else:
        return ToRGB3Band(), 3