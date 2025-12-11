#! /usr/bin/env python

# Import packages
import os
import argparse
import subprocess
import numpy as np
from astropy.table import Table
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


# Download product with rsync
def download_spectrum(extract):
    """Download product from remote server."""

    # Product field + id
    field = extract['subfield'].lower()
    id    = str(extract['ID']).zfill(5)
    # Files
    files =[]
    save_names = []
    remote_urls = []
    for file_type in ['full','1d']:
        # Remote URL
        remote_urls.append(f"https://archive.stsci.edu/hlsps/clear/data/{file_type}/{field}")
        files.append(f"hlsp_clear_hst_wfc3_{field}-{id}_g102-g141_v4_{file_type}.fits")
        save_names.append(f"{field}_{id}_{file_type}.fits")

    # Execute command
    for save_name,file,remote_url in zip(save_names,files,remote_urls):
        # Download command
        command = [
            'curl',
            '-o',
            f'data/{field}/{save_name}',
            f'{remote_url}/{file}',
        ]

        if os.path.exists(f'data/{field}/{save_name}'):
            print(f'{file}exist')
            continue

        try:
            subprocess.run(command, check=True)
            print(f'\n download {file} downloaded \n')
        except subprocess.CalledProcessError as e:
            print(f'Failed to download {file}. Error: {e}')


# Main Function
def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('extracted', type=str, help='Path to extracted table')

    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()

    # Load extracted
    extracted = Table.read(args.extracted)

    # Number of CPUs
    ncpu = args.ncpu

    # Create directories
    home = os.getcwd()
    for field in np.unique(extracted['subfield']):
        os.makedirs(os.path.join(home,'data',f'{field.lower()}'), exist_ok=True)


# Multi-threaded download
    if ncpu > 1:
        with ThreadPoolExecutor(ncpu) as executor:
            list(tqdm(
                executor.map(download_spectrum, extracted),
                total=len(extracted),
                desc="Downloading spectra"
            ))

    # Single-threaded download
    else:
        for extract in tqdm(extracted, desc="Downloading spectra"):
            download_spectrum(extract)


if __name__ == '__main__':
    main()