#!/usr/bin/env python3
"""
Add morphology catalog fields into f150w HDF5 files and add `ml_morph_tag`.

Usage examples:
  python add_morph_to_h5.py \
      --primary COSMOSWeb_mastercatalog_v1_photom_primary.fits \
      --morph COSMOSWeb_mastercatalog_v1_ml_morph.fits \
      --h5-dir images/jwst/f150w --pattern 'f150w_*.h5' --overwrite

Reads primary catalog (which has 'id' column) and morphology catalog (row-aligned).
For each HDF5 file, looks up rows by the 'id' dataset from primary catalog
and retrieves the corresponding morphology row. Creates datasets with the same
length as the 'image' dataset. `ml_morph_tag` is 1 when an id is found, otherwise 0.
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
from astropy.table import Table
from tqdm import tqdm


def process_h5_file(fn, cat_combined, id_to_row_idx, args):
    """Process a single HDF5 file: add morph_flag_f150w and delta_f150w."""
    fields_to_add = ['morph_flag_f150w', 'delta_f150w']
    
    with h5py.File(fn, 'r+') as h5f:
        # check if id column exists
        if args.id_col not in h5f:
            return fn
        
        ids = h5f[args.id_col][:]
        n = len(ids)
        
        # write each field
        for morph_field in fields_to_add:
            ds_name = morph_field
            
            # skip if field not in catalog
            if morph_field not in cat_combined.colnames:
                continue
            
            # skip if dataset already exists (unless overwrite)
            if ds_name in h5f:
                if not args.overwrite:
                    continue
                else:
                    del h5f[ds_name]
            
            # create and fill array (all float32)
            arr = np.full((n,), np.nan, dtype=np.float32)
            for i, vid in enumerate(ids):
                key = int(vid)
                if key in id_to_row_idx:
                    row_idx = id_to_row_idx[key]
                    try:
                        arr[i] = float(cat_combined[morph_field][row_idx])
                    except (ValueError, TypeError):
                        arr[i] = np.nan
            
            # write dataset
            h5f.create_dataset(ds_name, data=arr, chunks=(min(256, n),))
    
    return fn





def main():
    p = argparse.ArgumentParser(description='Append morphology catalog fields into HDF5 files by matching on id')
    p.add_argument('--primary', required=True, help='Path to primary catalog FITS (with id column)')
    p.add_argument('--morph', required=True, help='Path to morphology FITS (ml_morph, row-aligned with primary)')
    p.add_argument('--h5-dir', required=True, help='Directory with HDF5 files')
    p.add_argument('--id-col', default='id', help='Name of ID column in primary catalog and HDF5 (default: id)')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing datasets if present')
    p.add_argument('--numthreads', type=int, default=8, help='Number of threads for parallel field processing (default: 8)')
    args = p.parse_args()

    # load primary catalog (has id column)
    print(f"Loading primary catalog: {args.primary}")
    primary = Table.read(args.primary)
    if args.id_col not in primary.colnames:
        raise SystemExit(f"ID column '{args.id_col}' not found in primary catalog columns: {primary.colnames}")

    # load morphology catalog (row-aligned with primary)
    print(f"Loading morphology catalog: {args.morph}")
    morph = Table.read(args.morph)
    
    if len(morph) != len(primary):
        raise SystemExit(f"Catalog length mismatch: primary has {len(primary)} rows, morph has {len(morph)} rows")

    # combine catalogs by stacking horizontally
    from astropy.table import hstack
    cat_combined = hstack([primary, morph], join_type='exact')
    
    # build mapping id -> row index
    id_to_row_idx: Dict[int, int] = {int(id_val): i for i, id_val in enumerate(cat_combined[args.id_col])}


    # select HDF5 files in directory
    h5_files = sorted(glob.glob(os.path.join(args.h5_dir, '*.h5')))
    if not h5_files:
        print(f'No HDF5 files found in {args.h5_dir}')
        return

    # process each file in parallel
    print(f"Processing {len(h5_files)} HDF5 files with {args.numthreads} threads")
    with ThreadPoolExecutor(max_workers=args.numthreads) as executor:
        tasks = [(fn, cat_combined, id_to_row_idx, args) for fn in h5_files]
        for result in tqdm(executor.map(lambda t: process_h5_file(*t), tasks), 
                          total=len(h5_files), desc='Processing files'):
            pass
    
    # write ml_morph_tag for each file
    print("Writing ml_morph_tag...")
    for fn in tqdm(h5_files, desc='Writing ml_morph_tag'):
        with h5py.File(fn, 'r+') as h5f:
            if args.id_col not in h5f:
                continue
            
            ids = h5f[args.id_col][:]
            n = len(ids)
            tag_name = 'ml_morph_tag'
            
            if tag_name in h5f and args.overwrite:
                del h5f[tag_name]
            if tag_name not in h5f:
                tag_arr = np.zeros((n,), dtype=np.uint8)
                for i, vid in enumerate(ids):
                    key = int(vid)
                    if key in id_to_row_idx:
                        tag_arr[i] = 1
                h5f.create_dataset(tag_name, data=tag_arr, dtype=np.uint8, chunks=(min(256, n),))


if __name__ == '__main__':
    main()
