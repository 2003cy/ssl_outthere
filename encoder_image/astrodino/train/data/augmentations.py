# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import List

import numpy as np
import skimage.filters
import skimage.transform
import torch
from torchvision import transforms

logger = logging.getLogger("dinov2")


class DataAugmentationAstroDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=144,
        local_crops_size=60,
        center_crop_size: int = -1
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
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
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"center_crop_size: {center_crop_size}")
        logger.info("###################################")

        # random resized crop and flip
        geo_global = []
        geo_local = []
        if self.center_crop_size is not None and self.center_crop_size > 0:
            geo_global.append(transforms.CenterCrop(self.center_crop_size))
            geo_local.append(transforms.CenterCrop(self.center_crop_size))

        geo_global.extend(
            [
                transforms.RandomCrop(global_crops_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                #transforms.RandomRotation(degrees=(-15, 15), fill=0),
            ]
        )

        geo_local.extend(
            [
                transforms.RandomCrop(local_crops_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                #transforms.RandomRotation(degrees=(-15, 15), fill=0),
            ]
        )

        self.geometric_augmentation_global = transforms.Compose(geo_global)
        self.geometric_augmentation_local = transforms.Compose(geo_local)

        global_transfo1_extra = transforms.Compose(
            [
                RandomGaussianBlur(p=1.0),
                RandomGaussianNoise(p=1.0, im_dim=global_crops_size),
            ]
        )

        global_transfo2_extra = transforms.Compose(
            [
                RandomGaussianBlur(p=0.1),
                RandomGaussianNoise(p=0.1, im_dim=global_crops_size),
            ]
        )

        local_transfo_extra = transforms.Compose(
            [
                RandomGaussianBlur(p=0.5),
                RandomGaussianNoise(p=0.5, im_dim=local_crops_size),
            ]
        )

        to_rgb = ToRGB(return_channel_pos=2)

        self.global_transfo1 = transforms.Compose([global_transfo1_extra, to_rgb])
        self.global_transfo2 = transforms.Compose([global_transfo2_extra, to_rgb])
        self.local_transfo = transforms.Compose([local_transfo_extra, to_rgb])
        

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = np.array(self.geometric_augmentation_global(image))
        global_crop_1 = torch.tensor(self.global_transfo1(im1_base)).permute(2, 0, 1)

        im2_base = np.array(self.geometric_augmentation_global(image))
        global_crop_2 = torch.tensor(self.global_transfo2(im2_base)).permute(2, 0, 1)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
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
        transform = GaussianBlur()
        super().__init__([transform], p=keep_p)


class RandomGaussianNoise(transforms.RandomApply):
    """Randomly apply Gaussian noise to the image."""

    def __init__(self, *, im_dim=144, p: float = 0.5):
        keep_p = 1 - p
        transform = GaussianNoise(im_dim=im_dim)
        super().__init__([transform], p=keep_p)

'''
class ToRGB:
    """
    Transformation from raw image data (nanomaggies) to the rgb values displayed
    at the legacy viewer https://www.legacysurvey.org/viewer

    Code copied from
    https://github.com/legacysurvey/imagine/blob/master/map/views.py
    """

    def __init__(self, scales=None, m=0.03, Q=20, bands=["g", "r", "z"]):
        rgb_scales = {
            "u": (2, 1.5),
            "g": (2, 6.0),
            "r": (1, 3.4),
            "i": (0, 1.0),
            "z": (0, 2.2),
        }
        if scales is not None:
            rgb_scales.update(scales)

        self.rgb_scales = rgb_scales
        self.m = m
        self.Q = Q
        self.bands = bands
        self.axes, self.scales = zip(*[rgb_scales[bands[i]] for i in range(len(bands))])

        # rearange scales to correspond to image channels after swapping
        self.scales = [self.scales[i] for i in self.axes]

    def __call__(self, imgs):
        # Check image shape and set to C x H x W
        if imgs.shape[0] != len(self.bands):
            imgs = np.transpose(imgs, (2, 0, 1))

        I = 0
        for img, band in zip(imgs, self.bands):
            plane, scale = self.rgb_scales[band]
            img = np.maximum(0, img * scale + self.m)
            I = I + img
        I /= len(self.bands)

        Q = 20
        fI = np.arcsinh(Q * I) / np.sqrt(Q)
        I += (I == 0.0) * 1e-6
        H, W = I.shape
        rgb = np.zeros((H, W, 3), np.float32)
        for img, band in zip(imgs, self.bands):
            plane, scale = self.rgb_scales[band]
            rgb[:, :, plane] = (img * scale + self.m) * fI / I

        rgb = np.clip(rgb, 0, 1)
        return rgb
'''

class ToRGB:
    """
    Stretch a single-band cutout and broadcast it to RGB by cloning the channel
    after applying a simple arcsinh stretch (mirrors ToRGB logic but mono-band).
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

        # Repeat into three identical channels; DataAugmentation converts to CHW later.
        rgb = np.stack((stretched, stretched, stretched), axis=-1)
        if return_channel_pos == 0:
            rgb = np.transpose(rgb, (2, 0, 1))  # C x H x W
            return rgb
        elif return_channel_pos == 2:
            return rgb

class GaussianNoise:
    """
    Augmentations tuned to the Legacy Survey Data (with minor modifications).

    Code copied from
    https://github.com/georgestein/ssl-legacysurvey/blob/main/ssl_legacysurvey/data_loaders/decals_augmentations.py#L296
    """

    def __init__(
        self,
        scaling: List = [1.0],
        mean: float = 0,
        im_dim: int = 144,
        im_ch: int = 3,
        decals: bool = True,
        uniform: bool = False,
    ):
        self.mean = mean
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform

        # Log normal fit paramaters
        self.shape_dist = np.array([0.2264926, 0.2431146, 0.1334844])
        self.loc_dist = np.array([-0.0006735, -0.0023663, -0.0143416])
        self.scale_dist = np.array([0.0037602, 0.0067417, 0.0260779])

        self.sigma_dist = np.log(self.scale_dist)

        # noise in channels is uncorrelated, as images taken at dirrerent times/telescopes
        self.noise_ch_min = np.array([0.001094, 0.001094, 0.001094])
        self.noise_ch_max = np.array([0.013, 0.018, 0.061])

    def __call__(self, image: np.ndarray):
        # draw 'true' noise level of each channel from lognormal fits
        self.sigma_true = (
            np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
        )

        if self.uniform:
            # draw desired augmented noise level from uniform, to target tails more
            self.sigma_final = np.random.uniform(self.noise_ch_min, self.noise_ch_max)
        else:
            self.sigma_final = (
                np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
            )

        # Gaussian noise adds as c^2 = a^2 + b^2
        self.sigma_augment = self.sigma_final**2 - self.sigma_true**2
        self.sigma_augment[self.sigma_augment < 0.0] = 0.0
        self.sigma_augment = np.sqrt(self.sigma_augment)

        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.0:
                image[i, :, :] += np.random.normal(
                    self.mean, self.sigma_augment[i], size=(self.im_dim, self.im_dim)
                )

        return image


class GaussianBlur:
    """
    Augmentations tuned to the Legacy Survey Data (with minor modifications).

    Code copied from
    https://github.com/georgestein/ssl-legacysurvey/blob/main/ssl_legacysurvey/data_loaders/decals_augmentations.py#L296
    """

    def __init__(
        self,
        scaling: List = [1.0],
        im_dim: int = 144,
        im_ch: int = 3,
        decals: bool = True,
        uniform: bool = False,
    ):
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform

        # Log normal fit paramaters
        self.shape_dist = np.array([0.2109966, 0.3008485, 0.3471172])
        self.loc_dist = np.array([1.0807153, 1.2394326, 1.1928363])
        self.scale_dist = np.array([1.3153171, 0.9164757, 0.8233702])

        self.sigma_dist = np.log(self.scale_dist)

        self.psf_ch_min = np.array([1.3233109, 1.2667341, 1.2126263])
        self.psf_ch_max = np.array([5.0, 4.5, 4.25])

    def __call__(self, image: np.ndarray):
        # noise in channels is uncorrelated, as images taken at different times/telescopes
        # draw 'true' noise level of each channel from lognormal fits
        self.sigma_true = (
            np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
        )

        if self.uniform:
            # draw desired augmented noise level from uniform, to target tails more
            self.sigma_final = np.random.uniform(self.psf_ch_min, self.psf_ch_max)
        else:
            self.sigma_final = (
                np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
            )

        # Gaussian noise adds as c^2 = a^2 + b^2
        self.sigma_augment = self.sigma_final**2 - self.sigma_true**2
        self.sigma_augment[self.sigma_augment < 0.0] = 0.0
        self.sigma_augment = np.sqrt(self.sigma_augment)

        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.0:
                image[i, :, :] = skimage.filters.gaussian(
                    image[i, :, :], sigma=self.sigma_augment[i], mode="reflect"
                )

        return image
