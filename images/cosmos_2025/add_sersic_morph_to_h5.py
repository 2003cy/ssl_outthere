#!/usr/bin/env python3
"""
Add axis_ratio, axis_ratio_err, radius_sersic, radius_sersic_err into HDF5 files from primary catalog.

Usage:
~/conda-envs/astrodino/bin/python add_sersic_morph_to_h5.py \
      --primary COSMOSWeb_mastercatalog_v1_photom_primary.fits \
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
from astropy.table import Table
from tqdm import tqdm


def process_h5_file(fn, primary_cat, id_to_row_idx, args):
    """Process a single HDF5 file: add axis_ratio, axis_ratio_err, radius_sersic, radius_sersic_err."""
    with h5py.File(fn, 'r+') as h5f:
        if args.id_col not in h5f:
            return fn
        
        ids = h5f[args.id_col][:]
        n = len(ids)
        
        # Define datasets to add: (h5_name, catalog_col)
        datasets = [
            ('axis_ratio', 'axratio_sersic'),
            ('axis_ratio_err', 'axratio_sersic_err'),
            ('radius_sersic', 'radius_sersic'),
            ('radius_sersic_err', 'radius_sersic_err'),
        ]
        
        # Check if all datasets exist (skip if not overwriting)
        if not args.overwrite and all(ds_name in h5f for ds_name, _ in datasets):
            return fn
        
        # Remove existing datasets if overwriting
        for ds_name, _ in datasets:
            if ds_name in h5f and args.overwrite:
                del h5f[ds_name]
        
        # Create and fill arrays
        arrays = {ds_name: np.full((n,), np.nan, dtype=np.float32) for ds_name, _ in datasets}
        
        for i, vid in enumerate(ids):
            key = int(vid)
            if key in id_to_row_idx:
                row_idx = id_to_row_idx[key]
                for ds_name, cat_col in datasets:
                    try:
                        arrays[ds_name][i] = float(primary_cat[cat_col][row_idx])
                    except (ValueError, TypeError):
                        arrays[ds_name][i] = np.nan
        
        # Write datasets
        for ds_name, _ in datasets:
            if ds_name not in h5f:
                h5f.create_dataset(ds_name, data=arrays[ds_name], chunks=(min(256, n),))
    
    return fn


def main():
    p = argparse.ArgumentParser(description='Add axis_ratio (b/a) to HDF5 files')
    p.add_argument('--primary', required=True, help='Path to primary catalog FITS (with a_image, b_image)')
    p.add_argument('--h5-dir', required=True, help='Directory with HDF5 files')
    p.add_argument('--id-col', default='id', help='Name of ID column (default: id)')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing dataset if present')
    p.add_argument('--numthreads', type=int, default=8, help='Number of threads (default: 8)')
    args = p.parse_args()

    # load primary catalog
    print(f"Loading primary catalog: {args.primary}")
    primary_cat = Table.read(args.primary)
    
    # check required columns
    for col in [args.id_col, 'axratio_sersic', 'axratio_sersic_err', 'radius_sersic', 'radius_sersic_err']:
        if col not in primary_cat.colnames:
            raise SystemExit(f"Column '{col}' not found in catalog. Available: {primary_cat.colnames[:20]}...")

    # build mapping id -> row index
    id_to_row_idx: Dict[int, int] = {int(id_val): i for i, id_val in enumerate(primary_cat[args.id_col])}

    # select HDF5 files
    h5_files = sorted(glob.glob(os.path.join(args.h5_dir, '*.h5')))
    if not h5_files:
        print(f'No HDF5 files found in {args.h5_dir}')
        return

    print(f"Processing {len(h5_files)} HDF5 files with {args.numthreads} threads")
    with ThreadPoolExecutor(max_workers=args.numthreads) as executor:
        tasks = [(fn, primary_cat, id_to_row_idx, args) for fn in h5_files]
        for _ in tqdm(executor.map(lambda t: process_h5_file(*t), tasks), 
                      total=len(h5_files), desc='Adding axis_ratio'):
            pass

    print("Done!")


if __name__ == '__main__':
    main()
