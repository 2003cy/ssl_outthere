#!/usr/bin/env python3
"""Download DJA NIRSpec spectra listed in a selected CSV table.

Expected CSV columns: root, file
URL pattern: {base_url}/{root}/{file}
"""

import argparse
import csv
import gzip
import os
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm


DEFAULT_BASE_URL = "https://s3.amazonaws.com/msaexp-nirspec/extractions"


def open_csv(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', newline='')
    return open(path, 'r', newline='')


def load_rows(csv_path):
    rows = []
    with open_csv(csv_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        missing = [c for c in ('root', 'file') if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. Found: {reader.fieldnames}"
            )

        for i, row in enumerate(reader, start=2):
            root = (row.get('root') or '').strip()
            fname = (row.get('file') or '').strip()
            if not root or not fname:
                continue
            rows.append((root, fname, i))
    return rows


def download_one(base_url, download_dir, root, fname, timeout=120):
    url = f"{base_url}/{root}/{fname}"
    out_path = os.path.join(download_dir, fname)

    if os.path.exists(out_path):
        return out_path, "exists", None

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_spec_", dir=download_dir)
    os.close(fd)

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp_path, 'wb') as out:
            shutil.copyfileobj(resp, out)
        os.replace(tmp_path, out_path)
        return out_path, "downloaded", None
    except Exception as e:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return out_path, "failed", f"{url} -> {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Download all spectra files listed in a selected CSV table."
    )
    parser.add_argument('--csv-table', required=True, help='Path to selected CSV/CSV.GZ table')
    parser.add_argument('--num-workers', type=int, default=20, help='Number of download workers')
    parser.add_argument('--download-dir', default='download', help='Output directory')
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL, help='Base URL for spectra files')
    args = parser.parse_args()

    if args.num_workers < 1:
        raise SystemExit('--num-workers must be >= 1')

    os.makedirs(args.download_dir, exist_ok=True)

    rows = load_rows(args.csv_table)

    # Keep notebook behavior: output filename is only row["file"].
    # Dedupe by filename to avoid repeated downloads and write races.
    by_file = {}
    for root, fname, lineno in rows:
        by_file.setdefault(fname, (root, fname, lineno))

    tasks = list(by_file.values())

    total = len(tasks)
    lock = threading.Lock()
    stats = {'downloaded': 0, 'exists': 0, 'failed': 0}
    failed_msgs = []

    print(f"Input rows: {len(rows)}")
    print(f"Unique files: {total}")
    print(f"Workers: {args.num_workers}")
    print(f"Download dir: {args.download_dir}")

    def run_task(task):
        root, fname, _ = task
        return download_one(args.base_url, args.download_dir, root, fname)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(run_task, t) for t in tasks]
        pbar = tqdm(total=total, desc='Downloading spectra', unit='file')
        for fut in as_completed(futures):
            _, status, msg = fut.result()
            with lock:
                stats[status] += 1
                if status == 'failed' and msg:
                    failed_msgs.append(msg)
                pbar.update(1)
                pbar.set_postfix(
                    downloaded=stats['downloaded'],
                    exists=stats['exists'],
                    failed=stats['failed'],
                    refresh=False,
                )
        pbar.close()

    print('--- Summary ---')
    print(f"downloaded: {stats['downloaded']}")
    print(f"exists:     {stats['exists']}")
    print(f"failed:     {stats['failed']}")

    '''
    if failed_msgs:
        print('--- Failed URLs ---')
        for m in failed_msgs:
            print(m)
    '''

if __name__ == '__main__':
    main()
