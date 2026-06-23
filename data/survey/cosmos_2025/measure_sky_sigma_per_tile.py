"""Measure per-tile background sky sigma for COSMOS-Web F150W mosaics (seg-masked).

Feeds the per-tile entries of ``SKY_SIGMA["cosmos"]`` in
encoder_image/jwst_dino/data/augmentations.py (scales the relative noise aug).

Method — random 128px patches, each masked twice before the statistic:
  * coverage mask ``finite & nonzero``: tile-edge no-coverage is exact 0 (not NaN),
    so zeros are dropped; a patch with <95% coverage is rejected.
  * source mask ``segmap == 0``: catalogued sources removed so only background enters
    (sigma-clip kept as a light guard). A patch with too few source-free pixels is rejected.
Sky sigma = median over patches of the sigma-clipped background std.

COSMOS: 20 per-tile mosaics (ext 0, MJy/sr, 30 mas) + per-tile segmaps (ext 0, uint32
via BZERO). Segmap is pixel-aligned to its mosaic.

Run:
    .pixi/envs/default/bin/python data/survey/cosmos_2025/measure_sky_sigma_per_tile.py
"""

import glob
import os
import re

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

HERE = os.path.dirname(__file__)
PATCH = 128
N_PATCHES = 300
MAX_TRIES = 8000
MIN_COVERAGE = 0.95
MIN_BACKGROUND = 0.50
RNG_SEED = 0

TILE_RE = re.compile(r"30mas_([A-Z0-9]+)_v1\.0_sci\.fits$")


def field_sky_sigma(data: np.ndarray, seg: np.ndarray) -> tuple[float, int]:
    """Median sigma-clipped background std over random source-free patches (MJy/sr)."""
    rng = np.random.default_rng(RNG_SEED)
    ny, nx = data.shape
    sigmas: list[float] = []
    tries = 0
    while len(sigmas) < N_PATCHES and tries < MAX_TRIES:
        tries += 1
        y = rng.integers(0, ny - PATCH)
        x = rng.integers(0, nx - PATCH)
        sub = data[y:y + PATCH, x:x + PATCH]
        cover = np.isfinite(sub) & (sub != 0)
        if cover.mean() < MIN_COVERAGE:
            continue
        bg = cover & (seg[y:y + PATCH, x:x + PATCH] == 0)
        if bg.mean() < MIN_BACKGROUND:
            continue
        _, _, std = sigma_clipped_stats(sub[bg], sigma=3.0, maxiters=5)
        if np.isfinite(std) and std > 0:
            sigmas.append(std)
    return (float(np.median(sigmas)) if sigmas else float("nan")), len(sigmas)


def main() -> None:
    mosaics = sorted(glob.glob(os.path.join(
        HERE, "f150w", "mosaic_nircam_f150w_COSMOS-Web_30mas_*_v1.0_sci.fits")))
    print(f"# {len(mosaics)} COSMOS F150W tiles\n")
    result: dict[str, float] = {}
    for mp in mosaics:
        m = TILE_RE.search(mp)
        tile = m.group(1) if m else os.path.basename(mp)
        sp = os.path.join(HERE, "segmentation_maps",
                          f"detection_chi2pos_SWLW_{tile}_segmap_v1.3.fits.gz")
        if not (os.path.exists(mp) and os.path.exists(sp)):
            print(f"  {tile:4s} skipped (missing mosaic/segmap)")
            continue
        data = np.asarray(fits.getdata(mp, ext=0), dtype=np.float32)
        seg = np.asarray(fits.getdata(sp, ext=0))  # uint32 via BZERO
        sig, n = field_sky_sigma(data, seg)
        result[tile] = sig
        print(f"  {tile:4s} sky_sigma = {sig:.4e} MJy/sr  (n={n})")
        del data, seg

    vals = np.array([v for v in result.values() if np.isfinite(v)])
    print(f"\n# median = {np.median(vals):.4e}  range = [{vals.min():.3e}, {vals.max():.3e}] MJy/sr")
    print("\n# ---- paste into SKY_SIGMA['cosmos'] in jwst_dino/data/augmentations.py ----")
    print('"cosmos": {')
    for tile, sig in result.items():
        print(f"    {tile!r}: {sig:.4e},")
    print("},")


if __name__ == "__main__":
    main()
