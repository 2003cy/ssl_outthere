"""DINO multi-crop augmentations for single-band JWST cutouts.

Noise model: a light robustness regularizer that does NOT homogenize depth.
Per cutout we add zero-mean Gaussian noise scaled by the cutout's own per-tile
background sigma (``SKY_SIGMA``, keyed by survey/tile), with a half-normal scale ``s=|N(0,w)|``
(mode at 0 → most views get ~no extra noise, a wide tail occasionally reaches
large noise). This preserves the intrinsic depth ordering across tiles. Noise is
added in the raw-flux domain (MJy/sr) before the asinh stretch; negative
excursions are kept (physically real for background-subtracted data).

``asinh_stretch`` is a sign-preserving arcsinh stretch (no clipping at 0).
"""

import numpy as np
import torch
from torchvision import transforms

# Measured background sky sigma per survey/field (MJy/sr), from
# images/cosmos_2025/measure_sky_sigma_per_tile.py. Keyed by the tile alias that
# parse_tile() pulls from each cutout's rel_path; tile names are unique across
# surveys, so a single flat lookup (_SKY_SIGMA_BY_TILE) suffices.
# Method: random 128px patches, masked twice before the std — coverage mask
# (finite & nonzero; edge/seam gaps are exact-0, not NaN, so seams are rejected)
# and source mask (segmap == 0; catalogued sources removed) — then median of the
# sigma-clipped background std.
#   cosmos — COSMOS-Web F150W per tile (remarkably uniform, 2.20e-2 .. 2.37e-2).
#   ceers  — single 'fullceers' EGS mosaic (no sub-tiles), ~3.4x deeper.
SKY_SIGMA = {
    "cosmos": {
        "A1": 2.2847e-02, "A2": 2.2922e-02, "A3": 2.3431e-02, "A4": 2.3138e-02,
        "A5": 2.3695e-02, "A6": 2.3294e-02, "A7": 2.3379e-02, "A8": 2.3041e-02,
        "A9": 2.3293e-02, "A10": 2.3317e-02,
        "B1": 2.3181e-02, "B2": 2.3374e-02, "B3": 2.1966e-02, "B4": 2.3229e-02,
        "B5": 2.2789e-02, "B6": 2.3150e-02, "B7": 2.3135e-02, "B8": 2.2926e-02,
        "B9": 2.2975e-02, "B10": 2.3121e-02,
    },
    "ceers": {
        "EGS": 6.7102e-03,
    },
}

# Flattened tile -> sigma for the per-tile lookup (tile names are globally unique).
_SKY_SIGMA_BY_TILE = {
    tile: sig for fields in SKY_SIGMA.values() for tile, sig in fields.items()
}
# Fallback for unknown tiles: COSMOS median (the shallower survey — conservative).
SKY_SIGMA_FALLBACK = float(np.median(list(SKY_SIGMA["cosmos"].values())))


class DataAugmentationJWSTDINO(object):
    def __init__(
        self,
        local_crops_number,
        global_crops_size=144,
        local_crops_size=60,
        center_crop_size: int = -1,
        # noise augmentation
        noise_w: float = 2,
        noise_s_max: float | None = None,
    ):
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.center_crop_size = center_crop_size

        if (
            self.center_crop_size is not None
            and self.center_crop_size >= 0
            and (
                self.center_crop_size < self.global_crops_size
                or self.center_crop_size < self.local_crops_size
            )
        ):
            raise ValueError(
                "center_crop_size must be >= both global_crops_size and local_crops_size"
            )

        print("###################################")
        print("Using data augmentation parameters:")
        print(f"local_crops_number: {local_crops_number}")
        print(f"global_crops_size: {global_crops_size}")
        print(f"local_crops_size: {local_crops_size}")
        print(f"center_crop_size: {center_crop_size}")
        print(f"noise_w: {noise_w}  noise_s_max: {noise_s_max}")
        print("###################################")
        print(f"Model input image: {global_crops_size}x{global_crops_size}")
        print(f"####################################\n")

        # Rotation before CenterCrop so fill artifacts are removed by the crop.
        rotation = transforms.RandomApply(
            [transforms.RandomRotation(degrees=45, fill=0)], p=0.5
        )

        center_crop = (
            [transforms.CenterCrop(self.center_crop_size)]
            if self.center_crop_size is not None and self.center_crop_size > 0
            else []
        )

        self.geometric_augmentation_global = transforms.Compose([
            rotation,
            *center_crop,
            transforms.RandomCrop(global_crops_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

        self.geometric_augmentation_local = transforms.Compose([
            rotation,
            *center_crop,
            transforms.RandomCrop(local_crops_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

        # Noise apply probability per view (DINO intent): global_1 strongly
        # augmented, global_2 near-clean (teacher view), local in between.
        self.p_noise_global1 = 1.0
        self.p_noise_global2 = 0.1
        self.p_noise_local = 0.5

        self.noise = GaussianNoise(w=noise_w, s_max=noise_s_max)
        self.stretch = AsinhStretch(return_channel_pos=2)

    def _photometric(self, base, p_noise, sigma_sky):
        """per-tile noise (with prob p_noise) -> asinh stretch."""
        x = base
        if np.random.random() < p_noise:
            x = self.noise(x, sigma_sky)
        return self.stretch(x)

    def __call__(self, image, tile: str):
        sigma_sky = _SKY_SIGMA_BY_TILE.get(tile, SKY_SIGMA_FALLBACK)
        output = {}

        im1 = np.array(self.geometric_augmentation_global(image))
        global_crop_1 = torch.tensor(
            self._photometric(im1, self.p_noise_global1, sigma_sky)
        ).permute(2, 0, 1)

        im2 = np.array(self.geometric_augmentation_global(image))
        global_crop_2 = torch.tensor(
            self._photometric(im2, self.p_noise_global2, sigma_sky)
        ).permute(2, 0, 1)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        local_crops = [
            torch.tensor(
                self._photometric(
                    np.array(self.geometric_augmentation_local(image)),
                    self.p_noise_local, sigma_sky,
                )
            ).permute(2, 0, 1)
            for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output


class AsinhStretch:
    """Sign-preserving arcsinh stretch of a single-band cutout.

    ``out = sign(b) * arcsinh(Q*|b|) / sqrt(Q)`` with ``b = img*scale``. Negatives
    are kept (no clipping at 0), so zero-mean sky/augmentation noise keeps its
    negative half — physically correct for background-subtracted data.

    Input: (1,H,W) or (H,W). Output: (H,W,1) when return_channel_pos=2 (training
    pipeline; caller does permute(2,0,1)), (1,H,W) when return_channel_pos=0.
    """

    def __init__(self, scale: float = 1.0, Q: float = 20.0, return_channel_pos: int = 0):
        self.scale = scale
        self.Q = Q
        self._return_channel_pos = return_channel_pos

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
        out = out.astype(np.float32)[:, :, np.newaxis]   # (H,W,1)

        if self._return_channel_pos == 0:
            return out.transpose(2, 0, 1)                # (1,H,W)
        return out                                       # (H,W,1)


class GaussianNoise:
    """Per-tile relative additive Gaussian noise (light robustness regularizer).

    ``sigma_add = s * sigma_sky`` with ``s = |N(0,w)|`` (half-normal, mode at 0):
    most calls add ~nothing, a wide tail occasionally reaches large noise. Scaled
    by the cutout's own per-tile background sigma so it preserves the intrinsic
    depth ordering (no homogenization, no quadrature target). Added to channel 0
    in the raw-flux domain; negative excursions kept.
    """

    def __init__(self, mean: float = 0.0, w: float = 1.5, s_max: float | None = None):
        self.mean = mean
        self.w = w
        self.s_max = s_max

    def __call__(self, image: np.ndarray, sigma_sky: float) -> np.ndarray:
        s = abs(np.random.normal(0.0, self.w))
        if self.s_max is not None:
            s = min(s, self.s_max)
        sigma_add = s * sigma_sky
        if sigma_add <= 0.0:
            return image
        h, w = image[0].shape
        image[0, :, :] += np.random.normal(self.mean, sigma_add, size=(h, w))
        return image
