# Dataset file for DESI Legacy Survey data
import logging
import os
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from PIL import Image as im
from torchvision.datasets import VisionDataset

logger = logging.getLogger("astrodino")
_Target = float

# Here I modify the dataset to split: train 1million, val 100k, test 400k


class _SplitFull(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            #_SplitFull.TRAIN: 74_500_000,
            #_SplitFull.VAL: 100_000,
            #_SplitFull.TEST: 400_000,
            _SplitFull.TRAIN: 1_000_000,
            _SplitFull.VAL: 100_000,
            _SplitFull.TEST: 100_000,
        }
        return split_lengths[self]

class JWST(VisionDataset):
    Target = Union[_Target]
    Split = Union[_SplitFull]

    def __init__(
        self,
        *,
        split: "JWST.Split",
        root: str,
        extra: str = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        filter: str = 'f115w',
        extra_returns: list = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split
        self._extra_returns = extra_returns

        # We start by opening every available .h5 file located under root/filter folder (different tile names)
        self._files = []
        filter_root = os.path.join(root, filter) if filter else root
        if not os.path.isdir(filter_root):
            raise FileNotFoundError(f"JWST filter directory not found: {filter_root}")
        h5_files = sorted(f for f in os.listdir(filter_root) if f.endswith(".h5"))
        for fname in h5_files:
            fpath = os.path.join(filter_root, fname)
            try:
                file = h5py.File(fpath)
                #TODO sort and clean up data                
                self._files.append(file)
            except (OSError, IOError) as exc:
                logger.warning("Skipping %s due to: %s", fpath, exc)
                
        self._sample_size = int(sum(len(f["ra"]) for f in self._files))  #TODO change this to variable size based on data
        # Create randomized array of indices
        rng = np.random.default_rng(seed=42)
        
        self._indices = rng.permutation(int(self._sample_size))
        if split == JWST.Split.TRAIN.value:
            #self._indices = self._indices[:700_000]
            #use 90% of data as training
            self._indices = self._indices[:int(0.9 * len(self._indices))]
        elif split == JWST.Split.VAL.value:
            #self._indices = self._indices[700_000:710_000]
            #use 5% of data as val
            self._indices = self._indices[int(0.9 * len(self._indices)):int(0.95 * len(self._indices))]
        else:
            #self._indices = self._indices[710_000:]
            #use 5% of data as test
            self._indices = self._indices[int(0.95 * len(self._indices)):]

        # Calculate the length of each file and cumulative lengths for index mapping
        self._file_lengths = [len(f["ra"]) for f in self._files]
        self._cum_lengths = [0]
        for length in self._file_lengths:
            self._cum_lengths.append(self._cum_lengths[-1] + length)

        # Precompute the file_index and local_index for each random shuffled index 
        self._index_map = []
        for idx in self._indices:
            file_idx = 0
            while file_idx < len(self._cum_lengths) - 1 and idx >= self._cum_lengths[file_idx + 1]:
                file_idx += 1
            local_idx = idx - self._cum_lengths[file_idx]
            self._index_map.append((file_idx, local_idx))

    @property
    def split(self) -> "JWST.Split":
        return self._split

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        file_idx, local_idx = self._index_map[index]
        image = self._files[file_idx]["image"][local_idx].astype("float32")
        #transform 1channel to 3 channel
        image = np.repeat(image[np.newaxis, :, :], 3, axis=0)
        image = torch.tensor(image)
        target = None

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        #if extra returning for test in extra_returns: list
        if self._extra_returns is not None:
            target = []
            for entry in self._extra_returns:
                target.append(self._files[file_idx][entry][local_idx].astype("float32"))
        #allocate return based on split
        if self._split == LegacySurvey.Split.VAL.value:
            target = image
            
        return image, target

    def __len__(self) -> int:
        return len(self._indices)

class LegacySurvey(VisionDataset):
    Target = Union[_Target]
    Split = Union[_SplitFull]

    def __init__(
        self,
        *,
        split: "LegacySurvey.Split",
        root: str,
        extra: str = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        channel: int = 2,
        return_flux: bool = False,  # whether to return fluxes along with images
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split
        self._channel = channel
        self._return_flux = return_flux
        # We start by opening the hdf5 files located at the root directory
        self._files = []
        for folder_path in [f'{root}/images/north', f'{root}/images/south']:
#            folder_path = os.path.join(root, folder)
            files = sorted([f for f in os.listdir(folder_path) if f.endswith('.h5')])
            for file in files:
                try:
                    self._files.append(h5py.File(os.path.join(folder_path, file)))
                except (OSError, IOError) as e:
                    continue

        # Create randomized array of indices
        rng = np.random.default_rng(seed=42)
        self._indices = rng.permutation(int(1.5e7))

        if split == LegacySurvey.Split.TRAIN.value:
            self._indices = self._indices[:700_000]
        elif split == LegacySurvey.Split.VAL.value:
            self._indices = self._indices[-200_000:-100_000]
        else:
            self._indices = self._indices[-100_000:]

    @property
    def split(self) -> "LegacySurvey.Split":
        return self._split

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        true_index = self._indices[index]
        image = self._files[true_index // int(1e6)]["images"][
            true_index % int(1e6)
        ].astype("float32")
        if self._channel is not None:
            image = image[self._channel:self._channel+1,:,:]
            image = np.repeat(image, 3, axis=0)  # convert to 3 channels mono image
        image = torch.tensor(image)
        target = None

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        #if returning fluxes
        if self._return_flux:
            fluxes = self._files[true_index // int(1e6)]["flux"][
                true_index % int(1e6)
            ].astype("float32")
            target = torch.tensor(fluxes[self._channel])
        #allocate return based on split
        if self._split == LegacySurvey.Split.VAL.value:
            return image, image
        elif self._split == LegacySurvey.Split.TRAIN.value:
            return image, target
        else:
            return image, target

    def __len__(self) -> int:
        return len(self._indices)


class _SplitNorth(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _SplitNorth.TRAIN: 13_500_000,
            _SplitNorth.VAL: 100_000,
            _SplitNorth.TEST: 400_000,
        }
        return split_lengths[self]


class LegacySurveyNorth(VisionDataset):
    Target = Union[_Target]
    Split = Union[_SplitNorth]

    def __init__(
        self,
        *,
        split: "LegacySurvey.Split",
        root: str,
        extra: str = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        # We start by opening the hdf5 files located at the root directory
        self._files = [
            h5py.File(
                os.path.join(
                    root, "north/images_npix152_0%02d000000_0%02d000000.h5" % (i, i + 1)
                )
            )
            for i in range(14)
        ]

        # Create randomized array of indices
        rng = np.random.default_rng(seed=42)
        self._indices = rng.permutation(int(1.4e7))
        if split == LegacySurvey.Split.TRAIN.value:
            self._indices = self._indices[:13_500_000]
        elif split == LegacySurvey.Split.VAL.value:
            self._indices = self._indices[13_500_000:-400_000]
        else:
            self._indices = self._indices[-400_000:]

    @property
    def split(self) -> "LegacySurvey.Split":
        return self._split

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        true_index = self._indices[index]
        image = self._files[true_index // int(1e6)]["images"][
            true_index % int(1e6)
        ].astype("float32")
        image = torch.tensor(image)
        target = None

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self._indices)
