"""Measure background sky sigma for the JADES GOODS-S + GOODS-N F150W mosaics (seg-masked).

Feeds ``SKY_SIGMA["jades"]`` (tiles 'gds' / 'gdn') in
encoder_image/jwst_dino/data/augmentations.py.

Method — identical two-gate recipe to COSMOS/CEERS/OutThere (random 128px patches):
  * coverage mask ``finite & nonzero``: off-footprint pixels are exact 0 (or NaN), so a
    patch with <95% coverage (off the mosaic / in a gap) is rejected.
  * source mask ``segmap == 0``: catalogued sources removed; a patch needs >=50% clean
    background. Sigma-clip kept as a guard.
Sky sigma = median over patches of the sigma-clipped background std (MJy/sr).

JADES: NIRCam, MJy/sr @ 30 mas (no conversion/resampling). Per field: SCI ext 1, segmap
ext 1 (SEGMENTATION, seg==ID), pixel-aligned. JADES is the DEEPEST survey, so expect the
smallest sky sigma of the four (below CEERS ~6.7e-3).

Run:
    .pixi/envs/default/bin/python images/JADES/measure_sky_sigma.py
"""

import os

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

HERE = os.path.dirname(__file__)
FIELDS = {'gds': 'goods-s', 'gdn': 'goods-n'}  # tile alias -> file slug
PATCH = 128
N_PATCHES = 400
MAX_TRIES = 60000   # irregular footprint -> many random patches land off-coverage
MIN_COVERAGE = 0.95
MIN_BACKGROUND = 0.50
RNG_SEED = 0


def _ext(hdul, name: str) -> int:
    for i, hd in enumerate(hdul):
        if str(hd.header.get('EXTNAME', '')).upper() == name.upper():
            return i
    return next(i for i, hd in enumerate(hdul) if hd.data is not None)


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
    out = {}
    for tile, slug in FIELDS.items():
        mos = os.path.join(HERE, f"hlsp_jades_jwst_nircam_{slug}_f150w_v5.0_drz.fits")
        seg = os.path.join(HERE, f"hlsp_jades_jwst_nircam_{slug}_segmentation_v5.0_drz.fits")
        with fits.open(mos, memmap=True) as h:
            data = np.asarray(h[_ext(h, 'SCI')].data, dtype=np.float32)
        with fits.open(seg, memmap=True) as h:
            segdata = np.asarray(h[_ext(h, 'SEGMENTATION')].data)
        sig, n = field_sky_sigma(data, segdata)
        out[tile] = sig
        print(f"  {tile} ({slug}) sky_sigma = {sig:.4e} MJy/sr  (n={n})")
        del data, segdata

    print("\n# ---- paste into SKY_SIGMA['jades'] in jwst_dino/data/augmentations.py ----")
    print('"jades": {')
    for tile, sig in out.items():
        print(f"    {tile!r}: {sig:.4e},")
    print("},")


if __name__ == "__main__":
    main()
