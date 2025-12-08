import os, argparse, h5py, numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from tqdm import tqdm

def list_h5_files(data_dir: str):
    # Collect all readable .h5 files under root_dir/north and root_dir/south
    # Skips folders that do not exist and silently ignores unreadable (corrupted) files.
    files = []
    for sub in ["north", "south"]:
        d = os.path.join(data_dir, sub)
        if not os.path.isdir(d):
            continue
        for f in tqdm(sorted(os.listdir(d)), desc=f"Scanning {sub}"):
            if f.endswith(".h5"):
                fp = os.path.join(d, f)
                try:
                    with h5py.File(fp, "r"):
                        files.append(fp)
                except OSError:
                    # Ignore files that cannot be opened
                    continue
    return files

def build_position_catalog(files):
    # Build a unified astropy Table containing:
    # ra, dec          : sky positions (concatenated across all files)
    # file_id          : integer ID mapping back to the original file path (index in 'files')
    # index            : row index inside that file's datasets
    # This avoids storing repeated file path strings for every source.
    ras, decs, file_ids, idxs = [], [], [], []
    for fid, fp in tqdm(enumerate(files), desc="Building position catalog"):
        # Load only RA/DEC arrays; defer image loading to downstream dataset for lazy access.
        with h5py.File(fp, "r") as hf:
            ra = hf["ra"][:]
            dec = hf["dec"][:]
        n = ra.shape[0]
        ras.append(ra)
        decs.append(dec)
        file_ids.append(np.full(n, fid, dtype=np.int32))
        idxs.append(np.arange(n, dtype=np.int32))
    # Concatenate per-file arrays into single large arrays.
    ras = np.concatenate(ras)
    decs = np.concatenate(decs)
    file_ids = np.concatenate(file_ids)
    idxs = np.concatenate(idxs)
    return Table(dict(ra=ras, dec=decs, file_id=file_ids, index=idxs))

def cross_match(morph_table: Table, pos_table: Table, max_sep_arcsec=0.5):
    # Perform nearest-neighbor sky match from morphology (Galaxy Zoo) table to the position catalog.
    # Uses astropy SkyCoord.match_to_catalog_sky (1:1 nearest match).
    # max_sep_arcsec : maximum angular separation (arcsec) to accept a match.
    print(f"Starting cross-match with {len(morph_table)} morphology entries and {len(pos_table)} position entries.")
    c1 = SkyCoord(morph_table["ra"] * u.degree, morph_table["dec"] * u.degree)
    c2 = SkyCoord(pos_table["ra"] * u.degree, pos_table["dec"] * u.degree)
    idx, d2d, _ = c1.match_to_catalog_sky(c2)
    # Boolean mask of matches within allowed separation.
    mask = d2d < (max_sep_arcsec * u.arcsec)
    # Filter both tables to matched rows.
    matched_morph = morph_table[mask]
    matched_pos = pos_table[idx[mask]]
    # Merge: copy morphology row, then append file_id and index from position table.
    out = matched_morph.copy()
    out["file_id"] = matched_pos["file_id"]
    out["index"] = matched_pos["index"]
    print(f"Cross-match completed: {len(out)} matches found within {max_sep_arcsec} arcsec.")
    return out

def main():
    # CLI entry point:
    # --data_dir         : base directory containing 'north' and 'south' subdirectories with .h5 files
    # --morph_path       : Galaxy Zoo (or other morphology) catalog file path
    # --save_index_path  : output HDF5 path for matched index table (no images stored)
    # --max_sep          : maximum angular separation in arcseconds for accepting matches
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default='/ptmp/yacheng/outthere_ssl/images')
    ap.add_argument("--morph_path", default='/ptmp/yacheng/outthere_ssl/images/galaxy_zoo/gz_decals_volunteers_5.csv')
    ap.add_argument("--save_index_path", default='/ptmp/yacheng/outthere_ssl/images/galaxy_zoo/galaxy_zoo_matched_index.hdf5',
                    help="Path to save matched index + labels (no images stored).")
    ap.add_argument("--max_sep", type=float, default=0.5)
    args = ap.parse_args()

    print("Starting cross-match process...")

    # Discover all image HDF5 files.
    files = list_h5_files(args.data_dir)
    print(f"Found {len(files)} valid .h5 files.")

    # Read morphology / label table (expects columns 'ra','dec' plus label columns).
    morph = Table.read(args.morph_path, format="ascii")
    print(f"Loaded morphology table with {len(morph)} entries.")

    # Build positional index across all files.
    pos = build_position_catalog(files)
    print(f"Built position catalog with {len(pos)} entries.")

    # Cross-match morphology table to positional catalog.
    matched = cross_match(morph, pos, max_sep_arcsec=args.max_sep)

    # Save matched table (contains file_id + index for lazy downstream loading).
    matched.write(args.save_index_path, overwrite=True, format="hdf5", serialize_meta=True)
    print(f"Saved {len(matched)} matches → {args.save_index_path}")

    # Also save a simple file map (tab-separated: file_id \t file_path).
    file_map_path = args.save_index_path.replace(".hdf5", "_file_map.txt")
    with open(file_map_path, "w") as fw:
        for i, fp in enumerate(files):
            fw.write(f"{i}\t{fp}\n")
    print(f"Saved file map → {file_map_path}")
    print("Cross-match process completed.")

if __name__ == "__main__":
    main()