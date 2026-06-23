"""Measure background sky sigma for the CEERS 'fullceers' F150W mosaic (seg-masked).

Feeds ``SKY_SIGMA["ceers"]["EGS"]`` in encoder_image/jwst_dino/data/augmentations.py.

Method — random 128px patches, masked twice before the statistic:
  * coverage mask ``finite & nonzero``: the fullceers mosaic has NO NaN — field-seam
    gaps are encoded as exact 0 (~half the bbox), so zeros are dropped and a patch with
    <95% coverage (straddling a seam) is rejected.
  * source mask ``segmap == 0``: catalogued sources removed (sigma-clip kept as a guard).
Sky sigma = median over patches of the sigma-clipped background std.

CEERS: one mosaic (ext 1, MJy/sr, 30 mas) + one segmap (ext 0, int32, seg==NUMBER),
pixel-aligned. A single pseudo-tile 'EGS' (the EGS field).

Run:
    .pixi/envs/default/bin/python images/CEERS/measure_sky_sigma.py
"""

import os

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

HERE = os.path.dirname(__file__)
MOSAIC = os.path.join(HERE, "hlsp_ceers_jwst_nircam_fullceers_f150w_v1_sci-bkgsub.fits")
SEGMAP = os.path.join(HERE, "ceers_segmap_v1.0.fits")
TILE = "EGS"
PATCH = 128
N_PATCHES = 400
MAX_TRIES = 60000   # sparse diagonal strip -> many random patches land off-coverage
MIN_COVERAGE = 0.95
MIN_BACKGROUND = 0.50
RNG_SEED = 0


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
    data = np.asarray(fits.getdata(MOSAIC, ext=1), dtype=np.float32)
    seg = np.asarray(fits.getdata(SEGMAP, ext=0))
    sig, n = field_sky_sigma(data, seg)
    print(f"  {TILE} sky_sigma = {sig:.4e} MJy/sr  (n={n})")
    print("\n# ---- paste into SKY_SIGMA['ceers'] in jwst_dino/data/augmentations.py ----")
    print('"ceers": {')
    print(f"    {TILE!r}: {sig:.4e},")
    print("},")


if __name__ == "__main__":
    main()
