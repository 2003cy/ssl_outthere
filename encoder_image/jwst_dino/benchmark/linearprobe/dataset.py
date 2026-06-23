"""COSMOS f150w morphology dataset for jwst_dino linear-probe evaluation.

Mirrors astrodino's benchmark/linearprobe/dataset.py (JWSTMorphDataset) but sources
images from the jwst_dino on-disk format (per-tile .npy shards + a slim image-index
FITS, see ../../data/dataset.py) instead of pre-baked h5, and assigns labels by
cross-matching the TRAINING catalog with the COSMOS-Web morphology catalog AT LOAD
TIME — nothing is precomputed into the cutout files.

Cross-match (exact, by catalog row id):
    image_index_cosmos_f150w.fits  ── column `id` = master-catalog row index
    COSMOSWeb_mastercatalog_v1_ml_morph.fits  ── row-aligned with photom/lephare
The three COSMOS-Web v1 catalogs share 784016 rows and `id` is the row index, so
`ml_morph[id]` is the label for cutout with that id (verified: index vs photom[id]
positional offset = 0.0"). No ra/dec matching needed for COSMOS; ra/dec are kept in
the index only as a sanity check.

Label = `morph_flag_f150w`  (0 spheroid, 1 disk, 2 irregular, 3 bulge+disk;
999999 = unclassified, dropped). `delta_f150w` is the classification confidence
(NaN where unclassified); keep `delta < delta_threshold`. Optional cuts:
`exclude_irregular` drops class 2; `effective_radius_min` drops unresolved sources
(sersic `radius_sersic` cross-matched from photom, deg->px at 30 mas).

__getitem__ returns (image, label): the cutout center-cropped to `crop_size` and
asinh-stretched (Q=20) — the same clean preprocessing the model's global crop uses,
with no noise / rotation (deterministic, for embedding extraction).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table
from torch.utils.data import Dataset
from torchvision import transforms

# Reuse the model's exact asinh stretch so probe inputs match training.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data.augmentations import AsinhStretch  # noqa: E402

CLASS_NAMES = {0: "spheroid", 1: "disk", 2: "irregular", 3: "bulge+disk"}
UNCLASSIFIED = 999999  # morph_flag sentinel for "no morphology classification"
DEG_TO_PIX = 3600 * 1000 / 30.0  # radius_sersic[deg] -> pixels at 30 mas/pix (as astrodino)


class CosmosMorphDataset(Dataset):
    """COSMOS-Web f150w cutouts paired with ml_morph labels (cross-matched by id).

    Args:
        root: jwst_dino data root holding ``image_index_cosmos_<filter>.fits`` and
            ``<filter>/*.npy`` shards (e.g. ``~/ssl_outthere/data/image``).
        morph_catalog: path to ``COSMOSWeb_mastercatalog_v1_ml_morph.fits``.
        filter: band (selects both the index file and the ``<flag>_<filter>`` columns).
        crop_size: center-crop size fed to the encoder (= model global_crops_size, 72).
        delta_threshold: keep ``delta_<filter> < this`` (None disables; astrodino: 0.5).
        max_samples: cap total samples (-1 = all), applied after balancing.
        balanced: undersample majority classes to the minority count.
        samples_per_class: fixed N per class (>0 overrides ``balanced``).
        Q, scale: asinh stretch params (match the model: Q=20, scale=1).
        seed: RNG for balancing / subsampling.
        verbose: print the label distribution.
    """

    def __init__(
        self,
        root: str,
        morph_catalog: str,
        filter: str = "f150w",
        crop_size: int = 72,
        delta_threshold: float | None = 0.5,
        effective_radius_min: float | None = None,
        exclude_irregular: bool = False,
        photom_catalog: str | None = None,
        max_samples: int = -1,
        balanced: bool = False,
        samples_per_class: int = -1,
        Q: float = 20.0,
        scale: float = 1.0,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.root = os.path.expandvars(os.path.expanduser(root))
        self.filter = filter
        self.center_crop = transforms.CenterCrop(crop_size)
        self.stretch = AsinhStretch(scale=scale, Q=Q, return_channel_pos=0)  # -> (1,H,W)
        self.rng = np.random.default_rng(seed)
        self.shards: dict[str, np.ndarray] = {}  # rel_path -> memmap (per worker)

        # ── training catalog (cutouts) ──────────────────────────────────────────
        index = Table.read(os.path.join(self.root, f"image_index_cosmos_{filter}.fits"))
        ids = np.asarray(index["id"], dtype=np.int64)
        rel_path = np.asarray(index["rel_path"]).astype(str)
        local_idx = np.asarray(index["local_idx"], dtype=np.int64)

        # ── morphology labels (cross-match by row id) ───────────────────────────
        mm = fits.open(os.path.expandvars(os.path.expanduser(morph_catalog)), memmap=True)[1].data
        flag = np.asarray(mm[f"morph_flag_{filter}"])[ids]
        delta = np.asarray(mm[f"delta_{filter}"])[ids]

        valid = (flag != UNCLASSIFIED) & np.isfinite(delta)
        if delta_threshold is not None:
            valid &= delta < delta_threshold
        if exclude_irregular:
            valid &= flag != 2  # drop the irregular class (keep spheroid/disk/bulge+disk)

        # Sersic effective-radius cut (cross-matched from photom by the same id): drop tiny
        # / unresolved sources whose morphology is unreliable (astrodino used reff>=3px).
        if effective_radius_min is not None:
            cat = photom_catalog or morph_catalog.replace("_ml_morph.fits", "_photom_primary.fits")
            ph = fits.open(os.path.expandvars(os.path.expanduser(cat)), memmap=True)[1].data
            re_pix = np.asarray(ph["radius_sersic"])[ids] * DEG_TO_PIX
            valid &= np.isfinite(re_pix) & (re_pix >= effective_radius_min)

        # (rel_path, local_idx, label) for each kept cutout
        self._samples = list(zip(rel_path[valid], local_idx[valid], flag[valid].astype(np.int64)))
        if verbose:
            cuts = [f"delta<{delta_threshold}"]
            if effective_radius_min is not None:
                cuts.append(f"reff>={effective_radius_min}px")
            if exclude_irregular:
                cuts.append("no-irr")
            print(f"CosmosMorph [{filter}] — {len(self._samples)} / {len(ids)} cutouts "
                  f"({', '.join(cuts)}, classified)")

        if samples_per_class > 0 or balanced:
            self._samples = self._balance(samples_per_class, balanced, verbose)
        if 0 < max_samples < len(self._samples):
            pick = self.rng.choice(len(self._samples), size=max_samples, replace=False)
            self._samples = [self._samples[i] for i in pick]

        if verbose:
            labels = np.array([s[2] for s in self._samples])
            print("Label distribution:")
            for u, c in zip(*np.unique(labels, return_counts=True)):
                print(f"  {u} {CLASS_NAMES.get(int(u), '?'):11s}: {c:6d} ({100*c/len(labels):.1f}%)")

    def _balance(self, samples_per_class: int, balanced: bool, verbose: bool):
        by_label: dict[int, list] = {}
        for s in self._samples:
            by_label.setdefault(s[2], []).append(s)
        counts = {k: len(v) for k, v in by_label.items()}
        n = samples_per_class if samples_per_class > 0 else min(counts.values())
        if verbose:
            print(f"Balancing to {n}/class (from {counts})")
        out = []
        for label, items in sorted(by_label.items()):
            if len(items) > n:
                items = [items[i] for i in self.rng.choice(len(items), size=n, replace=False)]
            out.extend(items)
        self.rng.shuffle(out)
        return out

    def _shard(self, rel_path: str) -> np.ndarray:
        if rel_path not in self.shards:
            self.shards[rel_path] = np.load(os.path.join(self.root, rel_path), mmap_mode="r")
        return self.shards[rel_path]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        rel_path, local_idx, label = self._samples[index]
        cutout = self._shard(rel_path)[local_idx]                       # (H, W) float16
        img = np.nan_to_num(cutout.astype(np.float32))[None]           # (1, H, W)
        img = self.center_crop(torch.from_numpy(img)).numpy()          # (1, crop, crop)
        tensor = torch.from_numpy(self.stretch(img))                   # (1, crop, crop)
        return tensor, label
