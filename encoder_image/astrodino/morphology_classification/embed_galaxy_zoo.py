import os
import sys
from argparse import ArgumentParser
from typing import List, Tuple, Dict

import h5py
import numpy as np
import torch
from astropy.table import Table
from dinov2.eval.setup import build_model_for_eval
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm import tqdm

from astroclip.env import format_with_env
from astroclip.astrodino.data.augmentations import ToRGB


def _load_file_map(file_map_path: str) -> Dict[int, str]:
    mp: Dict[int, str] = {}
    with open(file_map_path) as fr:
        for line in fr:
            fid, fp = line.strip().split("\t")
            if not fid:
                continue
            mp[int(fid)] = fp
    return mp


def _iter_images(index_table: Table, file_map: Dict[int, str], crop_size: int, channel: int = 2):
    print("Preparing image iterator...")
    to_rgb = ToRGB()
    center_crop = transforms.CenterCrop(crop_size)
    # cache opened h5 files
    hf_cache: Dict[int, h5py.File] = {}
    try:
        print(f"Opening HDF5 files for {len(file_map)} file IDs...")
        for row in tqdm(index_table):
            fid = int(row["file_id"])
            idx = int(row["index"])
            if fid not in hf_cache:
                hf_cache[fid] = h5py.File(file_map[fid], "r")
            img = hf_cache[fid]["images"][idx].astype("float32")  # (C,H,W)
            single = img[channel:channel+1, :, :]
            img3 = np.repeat(single, 3, axis=0)  # (3,H,W)
            tensor = torch.from_numpy(img3)
            # apply same preprocessing as compute_embeddings
            tensor = center_crop(tensor)
            tensor = torch.from_numpy(to_rgb(tensor.numpy()))
            yield tensor
    finally:
        for f in hf_cache.values():
            try:
                f.close()
            except Exception:
                pass


def _batched(iterable, batch_size: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == batch_size:
            yield torch.stack(batch, dim=0)
            batch = []
    if batch:
        yield torch.stack(batch, dim=0)


def _build_models(model_cfgs: List[str], model_weights: List[str], use_data_parallel: bool = False) -> List[Tuple[str, torch.nn.Module, int, torch.device]]:

    print("Building models...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Warning: CUDA not available, using CPU")
    models: List[Tuple[str, torch.nn.Module, int, torch.device]] = []
    gpu_id = 0
    used_names: Dict[str, int] = {}
    for cfg_path, w_path in zip(model_cfgs, model_weights):
        cfg = OmegaConf.load(cfg_path)
        model = build_model_for_eval(cfg, pretrained_weights=w_path)
        model.eval()

        #use parallel or single GPU
        if torch.cuda.is_available():
            if use_data_parallel and torch.cuda.device_count() > 1:
                # move to first device then wrap
                print("Using DataParallel on all available GPUs")
                primary_device = torch.device("cuda:0")
                model = model.to(primary_device)
                model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
                device = primary_device
            else:
                print(f"Using single GPU id {gpu_id % torch.cuda.device_count()}")
                device = torch.device(f"cuda:{gpu_id % torch.cuda.device_count()}")
                model = model.to(device)
        else:
            print("Using CPU")
            device = torch.device("cpu")
            model = model.to(device)
        # derive a stable, intended name: use parent directory of cfg file (model root folder)
        raw_name = os.path.basename(os.path.dirname(cfg_path)) or os.path.splitext(os.path.basename(cfg_path))[0]
        # ensure uniqueness if multiple entries would share the same name
        if raw_name in used_names:
            used_names[raw_name] += 1
            run_name = f"{raw_name}__v{used_names[raw_name]}"
        else:
            used_names[raw_name] = 1
            run_name = raw_name

        models.append((run_name, model, cfg.crops.global_crops_size, device))
        gpu_id += 1
    
    print(f"Built {len(models)} models:")
    for name, _, _, device in models:
        print(f" - {name} on {device}")
    return models


def embed_galaxy_zoo(
    index_table_path: str,
    file_map_path: str,
    model_cfgs: List[str],
    model_weights: List[str],
    output_path: str,
    batch_size: int = 256,
    channel: int = 2,
    use_data_parallel: bool = True,
    auto_scale_batch: bool = True,
    max_samples: int = -1,
):
    assert len(model_cfgs) == len(model_weights), "configs and weights must be 1:1"

    tbl = Table.read(index_table_path, format="hdf5")
    if max_samples is not None and max_samples > 0:
        original_len = len(tbl)
        tbl = tbl[: max_samples]
        print(f"Limiting samples: {original_len} -> {len(tbl)}")
    fmap = _load_file_map(file_map_path)
    models = _build_models(model_cfgs, model_weights, use_data_parallel=use_data_parallel)

    # Optionally scale batch size by number of GPUs when using DataParallel
    if use_data_parallel and auto_scale_batch and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        effective_batch_size = batch_size * torch.cuda.device_count()
        print(f"Auto-scaling batch size: {batch_size} -> {effective_batch_size} (num_gpus={torch.cuda.device_count()})")
        batch_size = effective_batch_size

    # prepare per-model collectors
    collectors: Dict[str, List[np.ndarray]] = {name: [] for name, _, _, _ in models}

    # for each model, stream images and compute embeddings
    for name, model, crop_size, device in models:
        print(f"Computing embeddings for model: {name} on {device}")
        image_iter = _iter_images(tbl, fmap, crop_size=crop_size, channel=channel)
        total = len(tbl)
        print(f"Prepared image iterator for {total} images.")
        with torch.no_grad():
            for batch in tqdm(_batched(image_iter, batch_size), total=(total + batch_size - 1)//batch_size):
                batch = batch.to(device, non_blocking=True)
                emb = model(batch)
                if isinstance(emb, tuple):
                    emb = emb[0]
                # Normalize to 2D (B, D)
                if emb.dim() > 2:
                    emb = emb.view(emb.size(0), -1)
                collectors[name].append(emb.detach().cpu().numpy())
        print(f"Done model {name}")

    # assemble final table: add columns per model
    out = tbl.copy()
    for name in collectors.keys():
        # sanity check for consistent dimensionality within a model's collector
        shapes = [arr.shape[1] for arr in collectors[name] if arr.ndim == 2]
        if len(set(shapes)) > 1:
            details = {i: s for i, s in enumerate(shapes)}
            raise ValueError(
                f"Inconsistent embedding dims for model '{name}': {sorted(set(shapes))}. "
                f"This often happens if multiple models share the same name and got merged. "
                f"Please verify --model_cfgs/--model_weights alignment. Shapes by batch idx: {list(details.items())[:5]} ..."
            )
        embs = np.concatenate(collectors[name], axis=0)
        assert embs.shape[0] == len(out), f"Embedding count mismatch for {name}"
        col_name = f"{name}_embeddings"
        out[col_name] = embs

    # write combined embeddings file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.write(output_path, overwrite=True, format="hdf5")
    print(f"Saved combined embeddings → {output_path}")


if __name__ == "__main__":
    ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")
    parser = ArgumentParser()
    parser.add_argument("--index_table_path", type=str, default="/ptmp/yacheng/outthere_ssl/images/galaxy_zoo/galaxy_zoo_matched_index.hdf5",
                        help="Path to matched index HDF5 (from cross_match).")
    parser.add_argument("--file_map_path", type=str, default="/ptmp/yacheng/outthere_ssl/images/galaxy_zoo/galaxy_zoo_matched_index_file_map.txt",
                        help="TSV map of file_id to HDF5 file path.")
    parser.add_argument("--model_cfgs", type=str, nargs='+', required=True,
                        help="List of config.yaml files (one per model).")
    parser.add_argument("--model_weights", type=str, nargs='+', required=True,
                        help="List of pretrained checkpoint files (aligned to cfgs).")
    parser.add_argument("--output_path", type=str, default="/ptmp/yacheng/outthere_ssl/images/galaxy_zoo/galaxy_zoo_embeddings.hdf5",
                        help="Output HDF5 path to write combined embeddings.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--channel", type=int, default=2)
    parser.add_argument("--use_data_parallel", action="store_true", help="Use torch.nn.DataParallel to utilize all GPUs.")
    parser.add_argument("--auto_scale_batch", action="store_true", help="Scale batch size by number of GPUs when using DataParallel.")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Limit number of samples to process/save (take first N). Use -1 for all samples.")
    args = parser.parse_args()

    embed_galaxy_zoo(
        index_table_path=args.index_table_path,
        file_map_path=args.file_map_path,
        model_cfgs=args.model_cfgs,
        model_weights=args.model_weights,
        output_path=args.output_path,
        batch_size=args.batch_size,
        channel=args.channel,
        use_data_parallel=args.use_data_parallel,
        auto_scale_batch=args.auto_scale_batch,
        max_samples=args.max_samples,
    )
