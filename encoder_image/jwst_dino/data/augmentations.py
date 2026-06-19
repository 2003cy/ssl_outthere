"""DINO multi-crop augmentations for single-band JWST cutouts.

Adapted from encoder_image/astrodino/train/data/augmentations.py. The pipeline is
unchanged; the only edit is renaming the (mis-named) ``ToRGB`` stage to
``AsinhStretch`` — it has always done a single-band arcsinh stretch, not an RGB
conversion. The original RGB ``ToRGB`` (legacy-survey) has been removed.
"""

import logging

import numpy as np
import skimage.filters
import torch
from torchvision import transforms

logger = logging.getLogger("jwst_dino")


class DataAugmentationJWSTDINO(object):
    def __init__(
        self,
        local_crops_number,
        global_crops_size=144,
        local_crops_size=60,
        center_crop_size: int = -1,
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

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"center_crop_size: {center_crop_size}")
        logger.info("###################################")

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

        global_transfo1_extra = transforms.Compose([
            RandomGaussianBlur(p=1.0),
            RandomGaussianNoise(p=1.0, im_dim=global_crops_size),
        ])
        global_transfo2_extra = transforms.Compose([
            RandomGaussianBlur(p=0.1),
            RandomGaussianNoise(p=0.1, im_dim=global_crops_size),
        ])
        local_transfo_extra = transforms.Compose([
            RandomGaussianBlur(p=0.5),
            RandomGaussianNoise(p=0.5, im_dim=local_crops_size),
        ])

        asinh_stretch = AsinhStretch(return_channel_pos=2)

        self.global_transfo1 = transforms.Compose([global_transfo1_extra, asinh_stretch])
        self.global_transfo2 = transforms.Compose([global_transfo2_extra, asinh_stretch])
        self.local_transfo = transforms.Compose([local_transfo_extra, asinh_stretch])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = np.array(self.geometric_augmentation_global(image))
        global_crop_1 = torch.tensor(self.global_transfo1(im1_base)).permute(2, 0, 1)

        im2_base = np.array(self.geometric_augmentation_global(image))
        global_crop_2 = torch.tensor(self.global_transfo2(im2_base)).permute(2, 0, 1)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            torch.tensor(
                self.local_transfo(np.array(self.geometric_augmentation_local(image)))
            ).permute(2, 0, 1)
            for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output


class RandomGaussianBlur(transforms.RandomApply):
    """Randomly apply Gaussian blur to the image."""

    def __init__(self, *, p: float = 0.5):
        keep_p = 1 - p
        super().__init__([GaussianBlur()], p=keep_p)


class RandomGaussianNoise(transforms.RandomApply):
    """Randomly apply Gaussian noise to the image."""

    def __init__(self, *, im_dim=144, p: float = 0.5):
        keep_p = 1 - p
        super().__init__([GaussianNoise(im_dim=im_dim)], p=keep_p)


class AsinhStretch:
    """Arcsinh stretch of a single-band cutout, returning a single-channel array.

    Input: (1,H,W) or (H,W) numpy array of raw flux values.
    Output: (H,W,1) when return_channel_pos=2 (default for the training pipeline),
            (1,H,W) when return_channel_pos=0.
    DataAugmentationJWSTDINO.__call__ converts (H,W,1) -> (1,H,W) via permute(2,0,1).
    """

    def __init__(self, scale: float = 1.0, m: float = 0.03, Q: float = 20.0, return_channel_pos: int = 0):
        self.scale = scale
        self.m = m
        self.Q = Q
        self._return_channel_pos = return_channel_pos

    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        return_channel_pos = self._return_channel_pos
        # Accept HxW, 1xHxW, or 3xHxW inputs (already channel-first)
        arr = np.asarray(imgs, dtype=np.float32)
        if arr.ndim == 3:
            base = arr[0]
        elif arr.ndim == 2:
            base = arr
        else:
            base = arr.squeeze()

        base = np.maximum(0.0, base * self.scale + self.m)
        I = base
        I += (I == 0.0) * 1e-6
        fI = np.arcsinh(self.Q * I) / np.sqrt(self.Q + 1e-8)
        stretched = np.clip(base * fI / I, 0.0, 1.0).astype(np.float32)

        rgb = stretched[:, :, np.newaxis]
        if return_channel_pos == 0:
            return rgb.transpose(2, 0, 1)  # (1, H, W)
        elif return_channel_pos == 2:
            return rgb                      # (H, W, 1)


class GaussianNoise:
    """Adds Gaussian noise in quadrature to channel 0 to simulate shallower observations.

    sigma_sky is a fixed reference background noise level measured from JWST
    COSMOS-Web f150w mosaics (~0.021 MJy/sr). sigma_final is drawn uniformly from
    [0, k_max * sigma_sky]; noise is added in quadrature only when
    sigma_final > sigma_sky (~50% of calls). k_max = 2 covers up to 4x variance.
    """

    def __init__(
        self,
        mean: float = 0,
        im_dim: int = 144,
        sigma_sky: float = 2.1e-02,   # MJy/sr; JWST COSMOS-Web f150w median
        k_max: float = 2.0,
    ):
        self.mean = mean
        self.im_dim = im_dim
        self.sigma_sky = sigma_sky
        self.k_max = k_max

    def __call__(self, image: np.ndarray):
        sigma_final = np.random.uniform(0.0, self.k_max * self.sigma_sky)
        sigma_augment_sq = sigma_final**2 - self.sigma_sky**2
        if sigma_augment_sq <= 0.0:
            return image
        sigma_augment = np.sqrt(sigma_augment_sq)
        image[0, :, :] += np.random.normal(
            self.mean, sigma_augment, size=(self.im_dim, self.im_dim)
        )
        return image


class GaussianBlur:
    """Applies additional Gaussian blur to channel 0 to simulate a degraded PSF.

    JWST f150w PSF FWHM ~ 60 mas = 2 px at 30 mas/pix -> sigma_psf ~ 0.85 px.
    sigma_final ~ U[0, k_max * sigma_psf]; extra blur applied only when
    sigma_final > sigma_psf (quadrature). k_max = 1.5 -> FWHM range [60, 90] mas.
    """

    def __init__(
        self,
        sigma_psf: float = 0.85,   # pixels; JWST f150w at 30 mas/pix
        k_max: float = 1.5,
    ):
        self.sigma_psf = sigma_psf
        self.k_max = k_max

    def __call__(self, image: np.ndarray):
        sigma_final = np.random.uniform(0.0, self.k_max * self.sigma_psf)
        sigma_augment_sq = sigma_final**2 - self.sigma_psf**2
        if sigma_augment_sq <= 0.0:
            return image
        sigma_augment = np.sqrt(sigma_augment_sq)
        image[0, :, :] = skimage.filters.gaussian(
            image[0, :, :], sigma=sigma_augment, mode="reflect"
        )
        return image
