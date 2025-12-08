import argparse
import os
from typing import Any, Dict, List

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Subset
from torchvision import transforms
from tqdm import tqdm

from astroclip.astrodino.data.loaders import make_data_loader, make_dataset
from dinov2.eval.setup import build_model_for_eval
from astroclip.astrodino.data.augmentations import ToRGB

def load_cfgs(cfg_paths: List[str]) -> List[Any]:
    return [OmegaConf.load(path) for path in cfg_paths]


def build_models(cfgs, cfg_paths, weight_paths, use_data_parallel) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    gpu_id = 0
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if cuda_available else 0
    for cfg, cfg_path, weight_path in zip(cfgs, cfg_paths, weight_paths):
        model = build_model_for_eval(cfg, pretrained_weights=weight_path)
        model.eval()

        if cuda_available:
            if use_data_parallel and num_gpus > 1:
                device = torch.device("cuda:0")
                model = model.to(device)
                model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
            else:
                device = torch.device(f"cuda:{gpu_id % max(1, num_gpus)}")
                model = model.to(device)
                gpu_id += 1
        else:
            device = torch.device("cpu")
            model = model.to(device)

        raw_name = os.path.basename(os.path.dirname(cfg_path)) or os.path.splitext(os.path.basename(cfg_path))[0]
        suffix = sum(1 for slot in models if slot["name"] == raw_name)
        run_name = raw_name if suffix == 0 else f"{raw_name}__v{suffix+1}"

        models.append(
            {
                "name": run_name,
                "model": model,
                "device": device,
                "cfg_path": cfg_path,
                "weight_path": weight_path,
            }
        )

    return models



def unwrap_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in (
            "x_norm_clstoken",
            "x_norm_clstoken_mcrop",
            "x_norm_patchtokens",
            "logits",
        ):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
        raise ValueError(f"Unsupported model output keys: {list(output.keys())}")
    if isinstance(output, (list, tuple)) and output:
        return unwrap_output(output[0])
    raise ValueError("Cannot extract embeddings from model output")


def collect_embeddings(models, data_loader):
    collectors: Dict[str, List[np.ndarray]] = {slot["name"]: [] for slot in models}
    flux_store: List[np.ndarray] = []

    iterable = tqdm(data_loader, total=len(data_loader), desc="Embedding batches")
    with torch.no_grad():
        for images, flux in iterable:
            flux_store.append(flux.detach().cpu().numpy())
            for slot in models:
                device_batch = images.to(slot["device"], non_blocking=True)
                embeds = unwrap_output(slot["model"](device_batch))
                collectors[slot["name"]].append(embeds.detach().cpu().numpy())

    flux_array = np.concatenate(flux_store, axis=0).astype("float32", copy=False)
    embeddings = {
        name: np.concatenate(chunks, axis=0).astype("float32", copy=False)
        for name, chunks in collectors.items()
    }
    return embeddings, flux_array


def write_output(path, embeddings, flux, models_meta, args, crop_size):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with h5py.File(path, "w") as hf:
        hf.attrs["dataset_str"] = args.dataset
        hf.attrs["channel"] = args.channel
        hf.attrs["crop_size"] = crop_size
        hf.attrs["batch_size"] = args.batch_size
        hf.create_dataset("flux", data=flux, compression="gzip")
        for slot in models_meta:
            emb = embeddings[slot["name"]]
            hf.create_dataset(f"{slot['name']}_embedding", data=emb, compression="gzip")

def parse_args():
    parser = argparse.ArgumentParser("Embed flux with multi-model evaluation")
    parser.add_argument("--dataset", required=True, help="Dataset string for make_dataset")
    parser.add_argument("--model_cfgs", nargs="+", required=True, help="Config yaml list")
    parser.add_argument("--model_weights", nargs="+", required=True, help="Weight checkpoint list")
    parser.add_argument("--output_path", required=True, help="Final HDF5 path")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--channel", type=int, default=2)
    parser.add_argument("--use_data_parallel", action="store_true")
    parser.add_argument("--auto_scale_batch", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    if len(args.model_cfgs) != len(args.model_weights):
        raise ValueError("model_cfgs and model_weights must have the same length")

    cfgs = load_cfgs(args.model_cfgs)
    crop_sizes = {int(cfg.crops.global_crops_size) for cfg in cfgs}
    if len(crop_sizes) != 1:
        raise ValueError("All configs must share the same crops.global_crops_size")
    crop_size = crop_sizes.pop()

    effective_batch = args.batch_size
    if (
        args.use_data_parallel
        and args.auto_scale_batch
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    ):
        effective_batch *= torch.cuda.device_count()

    transform = transforms.Compose(
        [
            transforms.CenterCrop(crop_size),
            ToRGB(),
        ]
    )

    dataset = make_dataset(
        dataset_str=args.dataset,
        transform=transform,
        channel=args.channel,
        return_flux=True,
    )
    if args.max_samples is not None and args.max_samples > 0:
        max_samples = min(args.max_samples, len(dataset))
        dataset = Subset(dataset, range(max_samples))

    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=effective_batch,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        seed=args.seed,
        sampler_type=None,
    )

    models = build_models(cfgs, args.model_cfgs, args.model_weights, args.use_data_parallel)
    embeddings, flux = collect_embeddings(models, data_loader)
    write_output(args.output_path, embeddings, flux, models, args, crop_size)
    print(f"Saved embeddings + flux → {args.output_path}")


if __name__ == "__main__":
    main()