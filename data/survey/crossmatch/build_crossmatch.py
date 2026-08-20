"""Crossmatch DJA spectra against the JWST image cutout indices (all surveys).

Produces a slim join table — one row per (spectrum, cutout) match — that carries
ONLY the indices needed to go back to each source, plus match metadata. No pixels,
no spectra, no embeddings are duplicated here; rich per-object metadata stays in
the source catalogs (the DJA FITS and the per-survey image_index FITS) and is
recovered by joining on the id columns below.

Both sides are referenced by (source-file index + catalog index):
    spectrum : dja_id                         -> row of DJA_spectra_v4.5.fits
    image    : survey, rel_path, local_idx    -> row local_idx of root/rel_path (.npy shard)
               image_id                       -> row of image_index_<survey>_<filter>.fits

The imaging is the reference side: every cutout is matched to its nearest DJA
spectrum, so a cutout never carries more than one spectrum, while a source imaged
by several surveys emits one row per cutout. Downstream splitting must therefore be
grouped by dja_id to avoid leaking a spectrum across train/val.

The DJA side is restricted to the same quality selection the spectrum encoder uses
(visual grade in {1,2,3} and more than half the pixels valid inside the reference
wavelength window) before matching, so the join table carries only usable spectra.

Output: data/crossmatched/dja_x_<filter>.fits
"""

from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack
import astropy.units as u

# data/survey/crossmatch/<this file>  ->  repository root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE_ROOT   = PROJECT_ROOT / "data/image"
DJA_FITS     = PROJECT_ROOT / "data/spectrum/DJA_spectra_v4.5.fits"
OUT_DIR      = PROJECT_ROOT / "data/crossmatched"

FILTER        = "f150w"
SURVEYS       = ("cosmos", "ceers", "outthere", "jades")
RADIUS_ARCSEC = 0.3   # matches are astrometrically clean (median sep < 0.05"); 0.5" is generous

# Spectrum quality selection applied before matching, mirroring LowResDataset's
# `grades` and `frac_valid_pix` so both sides of the project select the same spectra.
GRADES         = (1, 2, 3)   # DJA visual-inspection grade
WL_MIN, WL_MAX = 1.0, 2.0    # micron; the window the spectrum encoder is trained on
FRAC_VALID_PIX = 0.5         # keep spectra with more than this fraction of valid pixels


def quality_selection(dja_fits=DJA_FITS) -> np.ndarray:
    """Boolean mask over DJA rows: graded spectra with usable window coverage."""
    with fits.open(dja_fits, memmap=False) as hdul:
        wave = np.asarray(hdul["WAVE"].data, dtype=np.float32)
        window = (wave > WL_MIN) & (wave < WL_MAX)
        cat = hdul["CATALOG"].data
        flux = np.asarray(cat["flux"], dtype=np.float32)[:, window]
        valid = np.asarray(cat["valid_spec"], dtype=bool)[:, window] & np.isfinite(flux)
        grade = np.asarray(cat["grade"])
    return np.isin(grade, GRADES) & (valid.mean(axis=1) > FRAC_VALID_PIX)


def build(filter: str = FILTER, surveys=SURVEYS, radius_arcsec: float = RADIUS_ARCSEC) -> Table:
    # ── spectrum side: DJA positions + provenance ids ──
    dja = Table.read(DJA_FITS)
    dja_ra  = np.asarray(dja["ra"], dtype=float)
    dja_dec = np.asarray(dja["dec"], dtype=float)
    dja_id  = np.arange(len(dja), dtype=np.int64)          # row into DJA_spectra FITS
    srcid   = np.asarray(dja["srcid"])
    objid   = np.asarray(dja["objid"])
    usable  = np.isfinite(dja_ra) & np.isfinite(dja_dec) & quality_selection()
    dsc = SkyCoord(dja_ra[usable] * u.deg, dja_dec[usable] * u.deg)
    print(f"DJA: {len(dja)} rows, {usable.sum()} with finite ra/dec and "
          f"grade in {GRADES} and valid-pixel fraction > {FRAC_VALID_PIX}")

    radius = radius_arcsec * u.arcsec
    rows = []
    for s in surveys:
        idx_path = IMAGE_ROOT / f"image_index_{s}_{filter}.fits"
        it = Table.read(idx_path)
        isc = SkyCoord(np.asarray(it["ra"], float) * u.deg,
                       np.asarray(it["dec"], float) * u.deg)
        # nearest DJA spectrum for each image cutout, then threshold
        j, sep, _ = isc.match_to_catalog_sky(dsc)
        keep = sep < radius
        n = int(keep.sum())
        print(f"  {s:9s}: {len(it):7d} cutouts   matched {n:6d}   "
              f"median sep {np.median(sep.arcsec[keep]) if n else float('nan'):.3f}\"")
        if n == 0:
            continue
        di = np.where(usable)[0][np.asarray(j)[keep]]      # DJA rows that matched
        ii = np.where(keep)[0]                             # image_index rows that matched
        rows.append(Table({
            "dja_id":     dja_id[di],
            "srcid":      srcid[di],
            "objid":      objid[di],
            "ra":         dja_ra[di],                      # canonical (DJA) position
            "dec":        dja_dec[di],
            "survey":     np.full(n, s),
            "image_id":   np.asarray(it["id"])[ii],        # row into image_index (its catalog id)
            "rel_path":   np.asarray(it["rel_path"]).astype(str)[ii],
            "local_idx":  np.asarray(it["local_idx"], np.int64)[ii],
            "tile":       np.asarray(it["tile"]).astype(str)[ii],
            "sep_arcsec": sep.arcsec[keep].astype(np.float32),
        }))

    out = vstack(rows)
    n_unique = len(np.unique(out["dja_id"]))
    print(f"\nTotal matched rows: {len(out)}  (unique spectra: {n_unique}; "
          f"{len(out) - n_unique} extra rows from repeat cutouts of one source)")
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = build()
    out_path = OUT_DIR / f"dja_x_{FILTER}.fits"
    out.write(out_path, overwrite=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
