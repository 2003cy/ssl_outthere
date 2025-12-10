import argparse
from omegaconf import OmegaConf
import torch as nn
from dinov2.eval.setup import build_model_for_eval
from train.data.loaders import make_data_loader, make_dataset
from train.data.augmentations  import ToRGB
from torchvision import transforms
print('imported transforms')
import numpy as np
import os

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument(
        "--config_file",
        "-c",
        "--config",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--pretrained_weights",
        "-p",
        "--pretrained",
        default="",
        type=str,
        help="path to pretrained weights",
    )
    parser.add_argument(
        "--dataset",
        default="",
        type=str,
        help="dataset to compute embeddings",
    )

    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="batch-size to compute embeddings",
    )
    return parser

def compute_embeddings_from_batch(model, batch, device='cuda'):
    model.eval()
    #print model architecture
    print(model)
    embeddings = []
    with nn.no_grad():
            images = batch.to(device)
            embeddings.append(model(images))
    return nn.cat(embeddings, dim=0)

def main():
    argparser = get_args_parser()
    args = argparser.parse_args()

    cfg = OmegaConf.load(args.config_file)
    pretrained_weights = args.pretrained_weights
    print(f"Using pretrained weights from: {pretrained_weights}")
    dataset_str = args.dataset
    print(f"loading from dataset: {dataset_str}")
    device = "cuda" if nn.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    to_rgb = ToRGB()
    transform = transforms.Compose([
        transforms.CenterCrop(cfg.crops.global_crops_size),
        to_rgb,
    ])

    dataset = make_dataset(dataset_str = dataset_str, transform=transform)
   
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=3,
        drop_last=True,
        seed=cfg.train.seed)
    
    batch = next(iter(data_loader))[0]
    #save the corresponding batch
    model = build_model_for_eval(cfg,pretrained_weights=pretrained_weights)
    embeddings = compute_embeddings_from_batch(model, batch, device=device)
    print('finished computing embeddings with shape',embeddings.shape)

    #use the final folder name as run_name, split by / +
    run_name = cfg.train.output_dir.split('/')[-1]
    output_dir = '/ptmp/yacheng/outthere_ssl/AstroCLIP/outputs/image_embeddings_eval'
    os.makedirs(f"{output_dir}/{run_name}", exist_ok=True)
    
    
    print(f"saving embeddings to {output_dir}/{run_name}/embeddings.npy")
    
    #convert embeddings to numpy array and save as numpy
    np.save(f"{output_dir}/{run_name}/embeddings.npy", embeddings.cpu().numpy())
    np.save(f"{output_dir}/{run_name}/batch.npy", batch.cpu().numpy())


if __name__ == "__main__":
    main()


