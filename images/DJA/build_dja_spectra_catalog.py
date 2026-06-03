#!/usr/bin/env python
"""Merge the DJA prism spectra with the emission-line catalog into one file.

The prism FITS (`*.prism_spectra.fits`) stores a *shared* wavelength grid
(473 points) and each spectral quantity as a (473, N_obj) array, where column i
corresponds to the i-th PRISM-grating row of the emission-line CSV (verified:
wmin/wmax agree to ~1e-7, npix agrees for all but a handful of edge-pixel cases).

Output `DJA_spectra_v4.5.fits` (multi-extension):
    HDU 'CATALOG' : per-object rows = all CSV metadata columns
                    + vector columns flux/err/full_err/valid_spec, each (473,)
                    (the spectral pixel mask is named `valid_spec` to avoid
                    clobbering the CSV's scalar `valid` review-flag column)
    HDU 'WAVE'    : the shared wavelength grid (473,), microns

No data filtering is applied -- all PRISM rows are kept so filtering can happen
downstream (e.g. in the Dataset class).
"""
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table


def build(csv_path, spec_path, out_path, dtype="float32"):
    print(f"[info] reading metadata : {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    pr = df[df["grating"] == "PRISM"].reset_index(drop=True)
    print(f"[info] PRISM rows       : {len(pr)}")

    print(f"[info] reading spectra  : {spec_path}")
    with fits.open(spec_path) as h:
        d = h[1].data
        wave = np.asarray(d["wave"], dtype=dtype)            # (L,)
        flux = np.asarray(d["flux"], dtype=dtype).T          # (N, L)
        err = np.asarray(d["err"], dtype=dtype).T            # (N, L)
        full_err = np.asarray(d["full_err"], dtype=dtype).T  # (N, L)
        valid = np.asarray(d["valid"], dtype=bool).T         # (N, L)

    n_obj, n_wave = flux.shape
    assert n_obj == len(pr), f"row mismatch: {n_obj} spectra vs {len(pr)} PRISM rows"
    print(f"[info] spectra shape    : {flux.shape} (n_obj, n_wave)")

    # astropy can't write object dtype -> coerce string/object columns to str
    print("[info] building table ...")
    for c in pr.columns:
        if pr[c].dtype == object:
            pr[c] = pr[c].astype(str)
    tab = Table.from_pandas(pr)

    tab["flux"] = flux
    tab["err"] = err
    tab["full_err"] = full_err
    tab["valid_spec"] = valid  # spectral pixel mask; CSV scalar `valid` kept intact

    cat_hdu = fits.BinTableHDU(tab, name="CATALOG")
    wave_hdu = fits.ImageHDU(data=wave, name="WAVE")
    wave_hdu.header["BUNIT"] = "micron"
    wave_hdu.header["COMMENT"] = "Shared wavelength grid; pairs with CATALOG.flux[i]"

    print(f"[info] writing          : {out_path}")
    fits.HDUList([fits.PrimaryHDU(), cat_hdu, wave_hdu]).writeto(
        out_path, overwrite=True
    )
    print("[done]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dja_msaexp_emission_lines_v4.5.csv")
    ap.add_argument("--spec", default="dja_msaexp_emission_lines_v4.5.prism_spectra.fits")
    ap.add_argument("--out", default="DJA_spectra_v4.5.fits")
    a = ap.parse_args()
    build(a.csv, a.spec, a.out)
