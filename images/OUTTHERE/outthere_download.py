#! /usr/bin/env python

# Import packages
import os
import argparse
import subprocess
import numpy as np
from astropy.table import Table
from concurrent.futures import ThreadPoolExecutor


# Download product with rsync
def _looks_like_html_error(path: str) -> bool:
    """Return True if file begins with an HTML doctype (failed download)."""
    try:
        with open(path, "rb") as f:
            head = f.read(200).lstrip()
        return head.startswith(b"<!DOCTYPE HTML")
    except OSError:
        return False


def download_spectrum(extract,home_path, remote, password):
    """Download product from remote server."""

    # Product name
    field = extract['field'] 

    # Remote URL
    remote_url = f'{remote}/{field}/spectra'

    # Files
    id = 'id'
    products =  ['1D','full','stack']
    files = [f'{field}_{str(extract[id]).zfill(5)}.{product}.fits' for product in products] 

    # Execute command
    for file in files:
        path = f'data/{field}/{file}'
        command = [
            'curl',
            '-u',
            f'outthere:{password}',
            '-o',
            path,
            f'{remote_url}/{file}',
        ]

        if os.path.exists(path) and not _looks_like_html_error(path):
            print(f'{file} exists and looks ok')
            continue

        # Retry download up to two attempts if the result is an HTML error page
        for attempt in range(2):
            try:
                subprocess.run(command, check=True)
                if _looks_like_html_error(path):
                    print(f'{file} seems to be an HTML error page; retrying ({attempt+1}/2)')
                    continue
                print(f'\n download {file} downloaded \n')
                break
            except subprocess.CalledProcessError as e:
                print(f'Failed to download {file}. Error: {e}')
        else:
            print(f'{file} could not be downloaded successfully after retries')


# Main Function
def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('extracted', type=str, help='Path to extracted table')
    parser.add_argument(
        '--remote',
        type=str,
        help='Remote URL',
        default='https://outthere-mpia.org/s3/data',
    )
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()

    # Prompt User for input
    #print('Enter the password to the remote server')
    password = 'outthere'

    # Load extractePd
    extracted = Table.read(args.extracted)

    # Remote URL
    remote = args.remote

    # Number of CPUs
    ncpu = args.ncpu

    # Create directories
    home = os.getcwd()
    for field in np.unique(extracted['field']):
        os.makedirs(os.path.join(home,'data',f'{field}'), exist_ok=True)
        #os.makedirs(os.path.join(home,'png',f'{field}'), exist_ok=True)

    # Multi-threaded download
    if ncpu > 1:
        with ThreadPoolExecutor(ncpu) as executor:
            executor.map(
                lambda e: download_spectrum(e, home, remote, password),
                extracted,
            )

    # Single-threaded download
    else:
        for extract in extracted:
            download_spectrum(extract, home, remote, password)


if __name__ == '__main__':
    main()