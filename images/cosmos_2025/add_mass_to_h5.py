#!/usr/bin/env python3
"""
Add stellar mass, age, and SFR from the LePhare SED catalog into HDF5 files.

Fields added (all float32, NaN where no match):
  mass_minchi2  — stellar mass at minimum chi²
  age_minchi2   — age at minimum chi²
  sfr_minchi2   — SFR at minimum chi²

The LePhare catalog is row-aligned with the primary catalog (same length, same order).
The primary catalog provides the 'id' column used to match HDF5 entries.

Usage:
  ~/conda-envs/astrodino/bin/python add_mass_to_h5.py \
      --primary COSMOSWeb_mastercatalog_v1_photom_primary.fits \
      --lephare COSMOSWeb_mastercatalog_v1_lephare.fits \
      --h5-dir ../jwst/f150w --overwrite
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
from astropy.table import Table, hstack
from tqdm import tqdm


DATASETS = [
    ('mass_minchi2', 'mass_minchi2'),
    ('age_minchi2',  'age_minchi2'),
    ('sfr_minchi2',  'sfr_minchi2'),
]


def process_h5_file(fn, catalog, id_to_row_idx, args):
    """Add mass_minchi2, age_minchi2, sfr_minchi2 to a single HDF5 file."""
    # locking=False: avoids POSIX lock conflicts on NFS/cluster filesystems
    # and when other processes already have the file open in read mode
    with h5py.File(fn, 'r+', locking=False) as h5f:
        if args.id_col not in h5f:
            return fn

        ids = h5f[args.id_col][:]
        n   = len(ids)

        # Skip entirely if all datasets exist and not overwriting
        if not args.overwrite and all(ds_name in h5f for ds_name, _ in DATASETS):
            return fn

        # Remove existing datasets if overwriting
        for ds_name, _ in DATASETS:
            if ds_name in h5f and args.overwrite:
                del h5f[ds_name]

        # Pre-allocate arrays
        arrays = {ds_name: np.full((n,), np.nan, dtype=np.float32) for ds_name, _ in DATASETS}

        # Fill by id lookup
        for i, vid in enumerate(ids):
            key = int(vid)
            if key in id_to_row_idx:
                row_idx = id_to_row_idx[key]
                for ds_name, cat_col in DATASETS:
                    try:
                        # np.ma.filled handles masked astropy columns (masked → NaN)
                        val = np.ma.filled(catalog[cat_col][row_idx], fill_value=np.nan)
                        arrays[ds_name][i] = float(val)
                    except (ValueError, TypeError):
                        arrays[ds_name][i] = np.nan

        # Write datasets
        for ds_name, _ in DATASETS:
            if ds_name not in h5f:
                h5f.create_dataset(ds_name, data=arrays[ds_name], chunks=(min(256, n),))

    return fn


def main():
    p = argparse.ArgumentParser(description='Add mass_minchi2 / age_minchi2 / sfr_minchi2 to HDF5 files')
    p.add_argument('--primary',    required=True, help='Path to primary catalog FITS (with id column)')
    p.add_argument('--lephare',    required=True, help='Path to LePhare FITS (row-aligned with primary, has SED fields)')
    p.add_argument('--h5-dir',     required=True, help='Directory with HDF5 files')
    p.add_argument('--id-col',     default='id',  help='Name of ID column in primary catalog and HDF5 (default: id)')
    p.add_argument('--overwrite',  action='store_true', help='Overwrite existing datasets if present')
    p.add_argument('--numthreads', type=int, default=8, help='Number of threads (default: 8)')
    args = p.parse_args()

    # Load primary catalog (has id column)
    print(f'Loading primary catalog: {args.primary}')
    primary = Table.read(args.primary)
    if args.id_col not in primary.colnames:
        raise SystemExit(f"ID column '{args.id_col}' not found in primary catalog columns: {primary.colnames}")

    # Load LePhare catalog (row-aligned with primary)
    print(f'Loading LePhare catalog: {args.lephare}')
    lephare = Table.read(args.lephare)

    if len(lephare) != len(primary):
        raise SystemExit(
            f"Catalog length mismatch: primary has {len(primary)} rows, "
            f"lephare has {len(lephare)} rows"
        )

    # Combine catalogs horizontally to get id + SED fields in one table
    catalog = hstack([primary, lephare], join_type='exact')

    # Validate SED columns exist in combined catalog
    for _, cat_col in DATASETS:
        if cat_col not in catalog.colnames:
            raise SystemExit(
                f"Column '{cat_col}' not found in LePhare catalog.\n"
                f"Available columns: {lephare.colnames[:30]}..."
            )
    print(f'Catalogs combined: {len(catalog)} rows')

    # Build id → row index mapping
    id_to_row_idx: Dict[int, int] = {
        int(id_val): i for i, id_val in enumerate(catalog[args.id_col])
    }

    # Find HDF5 files
    h5_files = sorted(glob.glob(os.path.join(args.h5_dir, '*.h5')))
    if not h5_files:
        print(f'No HDF5 files found in {args.h5_dir}')
        return
    print(f'Found {len(h5_files)} HDF5 files — processing with {args.numthreads} threads')

    # Process in parallel
    with ThreadPoolExecutor(max_workers=args.numthreads) as executor:
        tasks = [(fn, catalog, id_to_row_idx, args) for fn in h5_files]
        for _ in tqdm(executor.map(lambda t: process_h5_file(*t), tasks),
                      total=len(h5_files), desc='Adding mass/age/SFR'):
            pass

    print('Done.')


if __name__ == '__main__':
    main()
