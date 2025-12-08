"""Lightweight helpers to write per-filter cutout HDF5 files."""

from __future__ import annotations

import os
import argparse
from typing import Sequence

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table
from tqdm.auto import tqdm

# Metadata fields shared across both filters.
BASE_COLUMNS: tuple[str, ...] = (
    'id',
    'tile',
    'ra',
    'dec',
    'a_image',
    'b_image',
    'theta_image',
    'sersic',
    'sersic_err',
)

# Metadata fields whose names depend on the chosen filter.
FILTER_COLUMNS: tuple[str, ...] = (
    'snr_{filter}',
    'flux_auto_{filter}',
    'flux_err_auto_{filter}',
    'flux_aper_{filter}',
    'flux_err_aper_{filter}',
)


def build_column_list(filter_name: str, extra_columns: Sequence[str] | None = None) -> list[str]:
    """Return the metadata column names to copy into the HDF5 file."""
    # Create the per-filter names (snr_f115w, flux_auto_f115w, etc.).
    print('building columns for filter:', filter_name, '\n')
    per_filter = [col.format(filter=filter_name) for col in FILTER_COLUMNS]
    columns = list(BASE_COLUMNS) + per_filter
    # Allow callers to append arbitrary extra features without worrying about duplicates.
    if extra_columns:
        for name in extra_columns:
            if name not in columns:
                columns.append(name)
    return columns


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
    column_names: Sequence[str],
    output_path: str,
    *,
    mask: np.ndarray | None = None,
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

    The output layout is extremely simple (datasets named 'image', 'seg', 'id', ...)
    so downstream ML code can load each array via `file['image']` without wrappers.
    """
    # Use even sizes so center stays aligned with integer pixel indices.
    image_size = image_cutout_size - (image_cutout_size % 2)
    seg_size = (seg_cutout_size or image_size) - ((seg_cutout_size or image_size) % 2)

    table = master_cat if mask is None else master_cat[mask]
    if len(table) == 0:
        print('No rows selected, skip export.')
        return 0

    # Keep only the columns that actually exist in the catalog; warn about the rest.
    available_columns = [name for name in column_names if name in table.colnames]
    missing = [name for name in column_names if name not in table.colnames]
    if missing:
        print(f'Missing columns ignored: {missing}')
    if not available_columns:
        print('No valid columns found, skip export.')
        return 0

    tiles = np.unique(np.asarray(table['tile']))
    print(f'found {len(tiles)} tiles:','\n')

    mode = 'w' if overwrite else 'x'
    print('opening h5py file in mode:', mode,'\n')


    output_dir = os.path.dirname(output_path) or '.'    
    os.makedirs(output_dir, exist_ok=True)
    print('creating output directory if needed:', output_dir,'\n')


    total = 0
    filter_subdir = filter_dir or filter_name

    # Open the HDF5 file and create resizable datasets.
    with h5py.File(output_path, mode) as h5f:
        # Store images/segmentations as independent datasets so downstream code can
        # load them lazily (file['image'] etc.).

        print('creating image dataset with size:', (0, image_size, image_size))
        image_ds = h5f.create_dataset(
            'image',
            shape=(0, image_size, image_size),
            maxshape=(None, image_size, image_size),
            chunks=(max(1, chunk_size), image_size, image_size),
            dtype=np.float32,
        )
        print('creating seg dataset with size:', (0, seg_size, seg_size))
        seg_ds = h5f.create_dataset(
            'seg',
            shape=(0, seg_size, seg_size),
            maxshape=(None, seg_size, seg_size),
            chunks=(max(1, chunk_size), seg_size, seg_size),
            dtype=np.uint8,
        )
        # Create datasets for each metadata column.
        print(f'creating metadata datasets for {len(available_columns)} columns:')
        meta = {}
        for name in available_columns:
            column = table[name]
            if column.dtype.kind in {'U', 'S', 'O'}:
                dtype = h5py.string_dtype('utf-8')
            else:
                dtype = column.dtype
            col_shape = column.shape[1:]
            # Each metadata column is stored as a flat dataset so we can pull it out
            # as a NumPy array without reading the images.
            meta[name] = h5f.create_dataset(
                name,
                shape=(0,) + col_shape,
                maxshape=(None,) + col_shape,
                chunks=(max(1, chunk_size),) + col_shape,
                dtype=dtype,
            )
        print('finishing creating datasets, beginning tile loop over tiles:', tiles)

        tile_count = 0
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

            row_progress = tqdm(
                total=len(tile_rows),
                desc=f"{filter_name} rows {tile_str}",
                disable=not show_progress,
            )

            seg_path = os.path.join(base_dir, segmentation_dir, segmentation_template.format(tile=tile_str))
            filter_path = os.path.join(base_dir, filter_subdir, filter_template.format(filter=filter_name, tile=tile_str))

            if not (os.path.exists(seg_path) and os.path.exists(filter_path)):
                print(f'Skip tile {tile}: missing files.')
                row_progress.update(len(tile_rows))
                row_progress.close()
                continue

            with fits.open(seg_path) as seg_hdul, fits.open(filter_path) as filt_hdul:
                seg_data = seg_hdul[0].data
                filt_data = filt_hdul[0].data

                for row in tile_rows:
                    row_progress.update(1)
                    # Segmentation mask isolates the object; filter data carries the flux.
                    seg_mask = (seg_data == row['segment-id']).astype(np.uint8)
                    seg_cut = _safe_cutout(seg_mask, row['x_image'], row['y_image'], seg_size)
                    img_cut = _safe_cutout(filt_data, row['x_image'], row['y_image'], image_size)
                    if seg_cut is None or img_cut is None:
                        # Skip objects too close to the edge to get a clean cutout.
                        continue

                    idx = image_ds.shape[0]
                    image_ds.resize(idx + 1, axis=0)
                    seg_ds.resize(idx + 1, axis=0)
                    image_ds[idx] = img_cut.astype(np.float32)
                    seg_ds[idx] = seg_cut

                    for name in available_columns:
                        # Convert masked/byte values to plain Python numbers/strings for h5py.
                        value = row[name]
                        if isinstance(value, np.ma.MaskedArray):
                            value = value.filled(np.nan)
                        elif getattr(value, 'mask', False):
                            value = np.nan
                        else:
                            arr = np.asarray(value)
                            if arr.shape == ():
                                value = arr.item()
                            else:
                                value = arr
                        if isinstance(value, (bytes, np.bytes_)):
                            value = value.decode('utf-8')
                        meta[name].resize(idx + 1, axis=0)
                        meta[name][idx] = value
                        total += 1; tile_count += 1
                    row_progress.close()
            print(f'tile {tile_str} done, saved {tile_count} samples in this tile')
        tile_iter.close()
        print('finished tile loop')

    print(f'{filter_name}: wrote {total} samples to {output_path}')
    return total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export per-filter cutouts to HDF5.')
    parser.add_argument('--catalog', required=True, help='Path to COSMOS master catalog FITS file.')
    parser.add_argument('--filter', dest='filter_name', required=True, help='Filter name, e.g., f115w.')
    parser.add_argument('--output', required=True, help='Output HDF5 path.')
    parser.add_argument('--extra-column', action='append', default=[], help='Additional catalog column to include (repeatable).')
    parser.add_argument('--base-dir', default='.', help='Base directory used to resolve FITS mosaics.')
    parser.add_argument('--segmentation-dir', default='segmentation_maps', help='Folder containing segmentation FITS files.')
    parser.add_argument('--filter-dir', default=None, help='Folder containing per-filter mosaics (defaults to filter name).')
    parser.add_argument('--image-size', type=int, default=64, help='Square size of image cutouts.')
    parser.add_argument('--seg-size', type=int, default=64, help='Square size of segmentation cutouts (defaults to image size).')
    parser.add_argument('--chunk-size', type=int, default=256, help='Chunk length for resizable datasets.')
    parser.add_argument('--max-sample', type=int, default=None, help='Maximum catalog rows to export per tile.')

    #the templates of image/seg file names
    #this should not change if using same COSMOS2025 dr1 release
    parser.add_argument('--segmentation-template', default='detection_chi2pos_SWLW_{tile}_segmap_v1.3.fits.gz', help='Template for segmentation filenames.')
    parser.add_argument('--filter-template', default='mosaic_nircam_{filter}_COSMOS-Web_30mas_{tile}_v1.0_sci.fits', help='Template for per-filter mosaic filenames.')

    #
    parser.add_argument('--no-progress', action='store_true', help='Disable tqdm progress bars.')
    parser.add_argument('--no-overwrite', action='store_true', help='Fail if output already exists.')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    catalog = Table.read(args.catalog)
    mask = catalog['warn_flag']<=2
    catalog = catalog[mask]
    extra = args.extra_column or None
    columns = build_column_list(args.filter_name, extra)
    export_single_filter_dataset(
        catalog,
        args.filter_name,
        columns,
        args.output,
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
