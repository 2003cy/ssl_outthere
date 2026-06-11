"""Lightweight helper to write a minimal image-store HDF5 for one filter.

Stores only the contract fields — `image`, `seg`, `id`, `ra`, `dec` — plus
survey/filter/pixscale as file attributes. No catalog metadata is duplicated:
the cutout's `id` is the row index into the COSMOS-Web master catalogs (which are
row-aligned), so metadata is joined on demand by `id`; third-party catalogs match
by `ra`/`dec`. See cutout_export_parallel.py for the parallel per-tile variant.
"""

from __future__ import annotations

import os
import argparse
from typing import Sequence

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table
from tqdm.auto import tqdm

# Catalog columns the exporter needs: id/ra/dec are stored; the rest drive the cutout.
REQUIRED_COLUMNS: tuple[str, ...] = (
    'id', 'ra', 'dec', 'segment-id', 'x_image', 'y_image',
)


def _safe_cutout(image: np.ndarray, x_center: float, y_center: float, size: int) -> np.ndarray | None:
    """Extract a centered square cutout; return None if the box goes out of bounds."""
    half = size // 2
    x = int(round(float(x_center)))
    y = int(round(float(y_center)))
    y0, y1 = y - half, y + half
    x0, x1 = x - half, x + half
    if y0 < 0 or x0 < 0:
        return None
    if y1 > image.shape[-2] or x1 > image.shape[-1]:
        return None
    return image[..., y0:y1, x0:x1]


def export_single_filter_dataset(
    master_cat: Table,
    filter_name: str,
    output_path: str,
    *,
    mask: np.ndarray | None = None,
    survey: str = 'cosmos',
    base_dir: str = '.',
    segmentation_dir: str = 'segmentation_maps',
    filter_dir: str | None = None,
    image_cutout_size: int = 64,
    seg_cutout_size: int | None = None,
    chunk_size: int = 256,
    segmentation_template: str = 'detection_chi2pos_SWLW_{tile}_segmap_v1.3.fits.gz',
    filter_template: str = 'mosaic_nircam_{filter}_COSMOS-Web_30mas_{tile}_v1.0_sci.fits',
    overwrite: bool = True,
    show_progress: bool = True,
    max_sample_per_tile: int | None = None,
) -> int:
    """Stream catalog rows into a single HDF5 file for one filter.

    Output layout: datasets `image`, `seg`, `id`, `ra`, `dec` (plus survey/filter
    attributes), so downstream code loads arrays via `file['image']` and joins any
    metadata by `id`.
    """
    # Use even sizes so center stays aligned with integer pixel indices.
    image_size = image_cutout_size - (image_cutout_size % 2)
    seg_size = (seg_cutout_size or image_size) - ((seg_cutout_size or image_size) % 2)

    table = master_cat if mask is None else master_cat[mask]
    if len(table) == 0:
        print('No rows selected, skip export.')
        return 0

    missing = [name for name in REQUIRED_COLUMNS if name not in table.colnames]
    if missing:
        raise KeyError(f'Catalog is missing required columns: {missing}')

    tiles = np.unique(np.asarray(table['tile']))
    print(f'found {len(tiles)} tiles', '\n')

    mode = 'w' if overwrite else 'x'
    print('opening h5py file in mode:', mode, '\n')

    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    print('creating output directory if needed:', output_dir, '\n')

    total = 0
    filter_subdir = filter_dir or filter_name

    with h5py.File(output_path, mode) as h5f:
        print('creating image dataset with size:', (0, image_size, image_size))
        image_ds = h5f.create_dataset(
            'image', shape=(0, image_size, image_size),
            maxshape=(None, image_size, image_size),
            chunks=(max(1, chunk_size), image_size, image_size), dtype=np.float32,
        )
        print('creating seg dataset with size:', (0, seg_size, seg_size))
        seg_ds = h5f.create_dataset(
            'seg', shape=(0, seg_size, seg_size),
            maxshape=(None, seg_size, seg_size),
            chunks=(max(1, chunk_size), seg_size, seg_size), dtype=np.uint8,
        )
        # Contract identity/position fields.
        id_ds = h5f.create_dataset('id', shape=(0,), maxshape=(None,),
                                   chunks=(max(1, chunk_size),), dtype=np.int64)
        ra_ds = h5f.create_dataset('ra', shape=(0,), maxshape=(None,),
                                   chunks=(max(1, chunk_size),), dtype=np.float64)
        dec_ds = h5f.create_dataset('dec', shape=(0,), maxshape=(None,),
                                    chunks=(max(1, chunk_size),), dtype=np.float64)

        # Self-describing, constant attributes.
        h5f.attrs['survey'] = survey
        h5f.attrs['filter'] = filter_name
        h5f.attrs['pixscale_mas'] = 30.0
        h5f.attrs['bunit'] = 'MJy/sr'
        h5f.attrs['image_size'] = int(image_size)
        h5f.attrs['seg_size'] = int(seg_size)

        print('finished creating datasets, beginning tile loop over tiles:', tiles)

        tile_iter = tqdm(tiles, desc=f"{filter_name} tiles", disable=not show_progress)
        for tile in tile_iter:
            tile_rows = table[np.asarray(table['tile'] == tile)]
            if len(tile_rows) == 0:
                continue
            if max_sample_per_tile is not None:
                tile_rows = tile_rows[:max_sample_per_tile]
                if len(tile_rows) == 0:
                    continue

            tile_str = tile.decode('utf-8') if isinstance(tile, (bytes, np.bytes_)) else str(tile)
            row_progress = tqdm(total=len(tile_rows), desc=f"{filter_name} rows {tile_str}",
                                disable=not show_progress)

            seg_path = os.path.join(base_dir, segmentation_dir, segmentation_template.format(tile=tile_str))
            filter_path = os.path.join(base_dir, filter_subdir, filter_template.format(filter=filter_name, tile=tile_str))

            if not (os.path.exists(seg_path) and os.path.exists(filter_path)):
                print(f'Skip tile {tile}: missing files.')
                row_progress.update(len(tile_rows))
                row_progress.close()
                continue

            tile_count = 0
            with fits.open(seg_path) as seg_hdul, fits.open(filter_path) as filt_hdul:
                seg_data = seg_hdul[0].data
                filt_data = filt_hdul[0].data

                for row in tile_rows:
                    row_progress.update(1)
                    seg_mask = (seg_data == row['segment-id']).astype(np.uint8)
                    seg_cut = _safe_cutout(seg_mask, row['x_image'], row['y_image'], seg_size)
                    img_cut = _safe_cutout(filt_data, row['x_image'], row['y_image'], image_size)
                    if seg_cut is None or img_cut is None:
                        continue

                    idx = image_ds.shape[0]
                    for ds in (image_ds, seg_ds, id_ds, ra_ds, dec_ds):
                        ds.resize(idx + 1, axis=0)
                    image_ds[idx] = img_cut.astype(np.float32)
                    seg_ds[idx] = seg_cut
                    id_ds[idx] = int(row['id'])
                    ra_ds[idx] = float(row['ra'])
                    dec_ds[idx] = float(row['dec'])
                    total += 1
                    tile_count += 1
            row_progress.close()
            print(f'tile {tile_str} done, saved {tile_count} samples in this tile')
        tile_iter.close()
        print('finished tile loop')

    print(f'{filter_name}: wrote {total} samples to {output_path}')
    return total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export per-filter cutouts to a minimal image-store HDF5.')
    parser.add_argument('--catalog', required=True, help='Path to COSMOS master catalog FITS file.')
    parser.add_argument('--filter', dest='filter_name', required=True, help='Filter name, e.g., f115w.')
    parser.add_argument('--output', required=True, help='Output HDF5 path.')
    parser.add_argument('--survey', default='cosmos', help='Survey label stored as an h5 attribute.')
    parser.add_argument('--base-dir', default='.', help='Base directory used to resolve FITS mosaics.')
    parser.add_argument('--segmentation-dir', default='segmentation_maps', help='Folder containing segmentation FITS files.')
    parser.add_argument('--filter-dir', default=None, help='Folder containing per-filter mosaics (defaults to filter name).')
    parser.add_argument('--image-size', type=int, default=64, help='Square size of image cutouts.')
    parser.add_argument('--seg-size', type=int, default=64, help='Square size of segmentation cutouts (defaults to image size).')
    parser.add_argument('--chunk-size', type=int, default=256, help='Chunk length for resizable datasets.')
    parser.add_argument('--max-sample', type=int, default=None, help='Maximum catalog rows to export per tile.')

    # Filename templates for the COSMOS2025 DR1 release; leave unchanged for that release.
    parser.add_argument('--segmentation-template', default='detection_chi2pos_SWLW_{tile}_segmap_v1.3.fits.gz', help='Template for segmentation filenames.')
    parser.add_argument('--filter-template', default='mosaic_nircam_{filter}_COSMOS-Web_30mas_{tile}_v1.0_sci.fits', help='Template for per-filter mosaic filenames.')

    parser.add_argument('--no-progress', action='store_true', help='Disable tqdm progress bars.')
    parser.add_argument('--no-overwrite', action='store_true', help='Fail if output already exists.')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    catalog = Table.read(args.catalog)
    mask = catalog['warn_flag'] <= 2
    catalog = catalog[mask]
    export_single_filter_dataset(
        catalog,
        args.filter_name,
        args.output,
        survey=args.survey,
        base_dir=args.base_dir,
        segmentation_dir=args.segmentation_dir,
        filter_dir=args.filter_dir,
        image_cutout_size=args.image_size,
        seg_cutout_size=args.seg_size,
        chunk_size=args.chunk_size,
        segmentation_template=args.segmentation_template,
        filter_template=args.filter_template,
        overwrite=not args.no_overwrite,
        show_progress=not args.no_progress,
        max_sample_per_tile=args.max_sample,
    )


if __name__ == '__main__':
    main()
