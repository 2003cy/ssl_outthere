"""Parallel per-tile exporter that writes per-tile cutouts per filter."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Sequence

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table
from tqdm.auto import tqdm

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

FILTER_COLUMNS: tuple[str, ...] = (
    'snr_{filter}',
    'flux_auto_{filter}',
    'flux_err_auto_{filter}',
    'flux_aper_{filter}',
    'flux_err_aper_{filter}',
)


@dataclass
class ExportConfig:
    base_dir: str
    segmentation_dir: str
    filter_dir: str
    image_size: int
    seg_size: int
    chunk_size: int
    segmentation_template: str
    filter_template: str
    tile_output_dir: str
    show_progress: bool


@dataclass
class TileResult:
    tile: str
    path: str | None
    samples: int
    error: str | None = None


def build_column_list(filter_name: str, extra_columns: Sequence[str] | None = None) -> list[str]:
    print(f'building columns for filter: {filter_name}')
    per_filter = [col.format(filter=filter_name) for col in FILTER_COLUMNS]
    columns = list(BASE_COLUMNS) + per_filter
    if extra_columns:
        for name in extra_columns:
            if name not in columns:
                columns.append(name)
    print(f'total metadata columns: {len(columns)}\n')
    return columns


def _safe_cutout(image: np.ndarray, x_center: float, y_center: float, size: int) -> np.ndarray | None:
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


def _sanitize_value(value):
    if isinstance(value, np.ma.MaskedArray):
        value = value.filled(np.nan)
    elif getattr(value, 'mask', False):
        value = np.nan
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode('utf-8')
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, (bytes, np.bytes_)):
            return item.decode('utf-8')
        return item
    return arr


def _meta_array(values: list) -> tuple[np.ndarray, object]:
    first = values[0]
    if isinstance(first, str):
        data = np.asarray(['' if v is None else str(v) for v in values], dtype=object)
        dtype = h5py.string_dtype('utf-8')
        return data, dtype
    if isinstance(first, np.ndarray):
        data = np.stack(values)
        return data, data.dtype
    arr = np.asarray(values)
    if arr.dtype.kind in {'U', 'S'}:
        data = np.asarray(arr.tolist(), dtype=object)
        dtype = h5py.string_dtype('utf-8')
        return data, dtype
    if arr.dtype.kind == 'O' and all(isinstance(v, str) for v in values):
        data = np.asarray(['' if v is None else str(v) for v in values], dtype=object)
        dtype = h5py.string_dtype('utf-8')
        return data, dtype
    return arr, arr.dtype


def _chunk_shape(chunk_len: int, data_shape: tuple[int, ...]) -> tuple[int, ...]:
    chunk0 = max(1, min(chunk_len, data_shape[0]))
    return (chunk0,) + data_shape[1:]


def _process_tile(
    tile_rows: Table,
    tile_label,
    filter_name: str,
    column_names: Sequence[str],
    config: ExportConfig,
) -> TileResult:
    tile_str = tile_label.decode('utf-8') if isinstance(tile_label, (bytes, np.bytes_)) else str(tile_label)
    print(f'start processing tile {tile_str} for {filter_name}')
    seg_path = os.path.join(
        config.base_dir,
        config.segmentation_dir,
        config.segmentation_template.format(tile=tile_str),
    )
    filter_subdir = config.filter_dir or filter_name
    filter_path = os.path.join(
        config.base_dir,
        filter_subdir,
        config.filter_template.format(filter=filter_name, tile=tile_str),
    )

    if not (os.path.exists(seg_path) and os.path.exists(filter_path)):
        print(f'tile {tile_str}: missing FITS files')
        return TileResult(tile=tile_str, path=None, samples=0, error='missing FITS files')

    try:
        with fits.open(seg_path) as seg_hdul, fits.open(filter_path) as filt_hdul:
            seg_data = seg_hdul[0].data
            filt_data = filt_hdul[0].data
    except Exception as exc:  # pragma: no cover - FITS I/O errors
        return TileResult(tile=tile_str, path=None, samples=0, error=str(exc))

    images: list[np.ndarray] = []
    segs: list[np.ndarray] = []
    meta_store: dict[str, list] = {name: [] for name in column_names}
    row_bar = tqdm(
        total=len(tile_rows),
        desc=f"{filter_name} rows {tile_str}",
        disable=not config.show_progress,
    )

    for row in tile_rows:
        row_bar.update(1)
        seg_mask = (seg_data == row['segment-id']).astype(np.uint8)
        seg_cut = _safe_cutout(seg_mask, row['x_image'], row['y_image'], config.seg_size)
        img_cut = _safe_cutout(filt_data, row['x_image'], row['y_image'], config.image_size)
        if seg_cut is None or img_cut is None:
            continue
        images.append(img_cut.astype(np.float32))
        segs.append(seg_cut.astype(np.uint8))
        for name in column_names:
            meta_store[name].append(_sanitize_value(row[name]))

    row_bar.close()

    if not images:
        print(f'tile {tile_str}: no valid cutouts after filtering')
        return TileResult(tile=tile_str, path=None, samples=0, error='no valid cutouts')

    image_arr = np.stack(images)
    seg_arr = np.stack(segs)
    meta_datasets: dict[str, tuple[np.ndarray, object]] = {}
    for name, values in meta_store.items():
        if not values:
            continue
        meta_datasets[name] = _meta_array(values)

    os.makedirs(config.tile_output_dir, exist_ok=True)
    output_filename = f'{filter_name}_{tile_str}.h5'
    tile_path = os.path.join(config.tile_output_dir, output_filename)

    if os.path.exists(tile_path):
        print(f'tile {tile_str}: {tile_path} exists, skipping.')
        return TileResult(tile=tile_str, path=tile_path, samples=0, error='exists')

    with h5py.File(tile_path, 'w') as h5f:
        print(f'tile {tile_str}: writing datasets -> {tile_path}')
        h5f.create_dataset(
            'image',
            data=image_arr,
            chunks=_chunk_shape(config.chunk_size, image_arr.shape),
            dtype=np.float32,
        )
        h5f.create_dataset(
            'seg',
            data=seg_arr,
            chunks=_chunk_shape(config.chunk_size, seg_arr.shape),
            dtype=np.uint8,
        )
        for name, (data, dtype) in meta_datasets.items():
            h5f.create_dataset(
                name,
                data=data,
                chunks=_chunk_shape(config.chunk_size, data.shape),
                dtype=dtype,
            )

    print(f'tile {tile_str}: saved {image_arr.shape[0]} samples to {tile_path}')
    return TileResult(tile=tile_str, path=tile_path, samples=image_arr.shape[0])

def _export_filter_parallel(
    catalog: Table,
    filter_name: str,
    column_names: Sequence[str],
    args: argparse.Namespace,
) -> None:
    image_size = args.image_size - (args.image_size % 2)
    seg_size = args.seg_size - (args.seg_size % 2)
    filter_output_dir = os.path.join(args.output_dir, filter_name)
    os.makedirs(filter_output_dir, exist_ok=True)

    config = ExportConfig(
        base_dir=args.base_dir,
        segmentation_dir=args.segmentation_dir,
        filter_dir=args.filter_dir or filter_name,
        image_size=image_size,
        seg_size=seg_size,
        chunk_size=args.chunk_size,
        segmentation_template=args.segmentation_template,
        filter_template=args.filter_template,
        tile_output_dir=filter_output_dir,
        show_progress=not args.no_progress,
    )

    tiles = np.unique(np.asarray(catalog['tile']))
    print(f'{filter_name}: found {len(tiles)} tiles in catalog')
    tasks = []
    required_columns = list(column_names) + ['x_image', 'y_image', 'segment-id']

    for tile in tiles:
        tile_mask = np.asarray(catalog['tile'] == tile)
        tile_rows = catalog[tile_mask]
        if len(tile_rows) == 0:
            continue
        if args.max_sample is not None:
            tile_rows = tile_rows[:args.max_sample]
        if len(tile_rows) == 0:
            continue
        subset = tile_rows[required_columns].copy()
        tasks.append((subset, tile))
    print(f'{filter_name}: queued {len(tasks)} tiles for workers')

    if not tasks:
        print(f'{filter_name}: no tiles to process.')
        return

    tile_results: list[TileResult] = []
    show_progress = not args.no_progress
    max_workers = max(1, args.max_workers or 1)

    print(f'{filter_name}: launching up to {max_workers} workers')
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_tile, subset, tile, filter_name, column_names, config) for subset, tile in tasks]
        tile_progress = tqdm(total=len(futures), desc=f'{filter_name} tiles', disable=not show_progress)
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - worker crash
                print(f'{filter_name}: worker failed - {exc}')
                tile_progress.update(1)
                continue
            tile_progress.update(1)
            if result.error == 'exists':
                print(f'{filter_name}: tile {result.tile} skipped (exists).')
            elif result.error:
                print(f'{filter_name}: tile {result.tile} skipped ({result.error}).')
            if result.path and result.samples:
                tile_results.append(result)
        tile_progress.close()

    total_tiles = len(tile_results)
    total_samples = sum(result.samples for result in tile_results)
    print(
        f'{filter_name}: wrote {total_tiles} tile files '
        f'({total_samples} samples) to {filter_output_dir}'
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parallel per-tile cutout exporter.')
    parser.add_argument('--catalog', required=True, help='Path to COSMOS master catalog FITS file.')
    parser.add_argument('--filters', nargs='+', required=True, help='One or more filters, e.g., f115w f150w.')
    parser.add_argument('--output-dir', required=True, help='Directory where per-filter HDF5 files will be written.')
    parser.add_argument('--extra-column', action='append', default=[], help='Additional catalog columns to include (repeatable).')
    parser.add_argument('--base-dir', default='.', help='Base directory used to resolve FITS mosaics.')
    parser.add_argument('--segmentation-dir', default='segmentation_maps', help='Folder containing segmentation FITS files.')
    parser.add_argument('--filter-dir', default=None, help='Folder containing per-filter mosaics (defaults to filter name).')
    parser.add_argument('--image-size', type=int, default=64, help='Square size of image cutouts (pixels).')
    parser.add_argument('--seg-size', type=int, default=64, help='Square size of segmentation cutouts (pixels).')
    parser.add_argument('--chunk-size', type=int, default=256, help='Chunk length for HDF5 datasets.')
    parser.add_argument('--max-sample', type=int, default=None, help='Maximum number of catalog rows per tile (optional).')
    parser.add_argument('--max-workers', type=int, default=os.cpu_count() or 4, help='Number of parallel tile workers.')
    parser.add_argument('--segmentation-template', default='detection_chi2pos_SWLW_{tile}_segmap_v1.3.fits.gz', help='Filename template for segmentation maps.')
    parser.add_argument('--filter-template', default='mosaic_nircam_{filter}_COSMOS-Web_30mas_{tile}_v1.0_sci.fits', help='Filename template for per-filter mosaics.')
    parser.add_argument('--no-progress', default=False, action='store_true', help='Disable tqdm progress bars.')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    catalog = Table.read(args.catalog)
    mask = (catalog['warn_flag'] <= 2) & ((catalog['snr_f115w']>5) | (catalog['snr_f150w']>5))
    catalog = catalog[mask]
    extra = args.extra_column or None

    for filter_name in args.filters:
        columns = build_column_list(filter_name, extra)
        _export_filter_parallel(catalog, filter_name, columns, args)


if __name__ == '__main__':
    main()
