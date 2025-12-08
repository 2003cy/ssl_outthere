print('starting compute_embeddings.py')
import argparse
print
from omegaconf import OmegaConf
print('imported argparse and OmegaConf')
import torch as nn
print('imported torch')
from dinov2.eval.setup import build_model_for_eval
print('imported build_model_for_eval')
from AstroCLIP.astroclip.astrodino.data.loaders import make_data_loader, make_dataset
print('imported make_data_loader and make_dataset')
from torchvision import transforms
print('imported transforms')
import numpy as np
print('imported numpy')
import os
print('imported os')


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
    print(dataset)
   
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=3,
        drop_last=True,
        seed=cfg.train.seed)
    
    print(next(iter(data_loader))[0].shape)

    batch = next(iter(data_loader))[0]
    #save the corresponding batch
    model = build_model_for_eval(cfg,pretrained_weights=pretrained_weights)
    embeddings = compute_embeddings_from_batch(model, batch, device=device)
    print('finished computing embeddings with shape',embeddings.shape)

    #use the final folder name as run_name, split by / +
    run_name = cfg.train.output_dir.split('/')[-1] + 'to_rgb'
    output_dir = '/ptmp/yacheng/outthere_ssl/AstroCLIP/outputs/image_embeddings_eval'
    os.makedirs(f"{output_dir}/{run_name}", exist_ok=True)
    print(f"saving embeddings to {output_dir}/{run_name}/embeddings.npy")
    #convert embeddings to numpy array and save as numpy
    np.save(f"{output_dir}/{run_name}/embeddings.npy", embeddings.cpu().numpy())
    np.save(f"{output_dir}/{run_name}/batch.npy", batch.cpu().numpy())


class ToRGB:
    """
    Transformation from raw image data (nanomaggies) to the rgb values displayed
    at the legacy viewer https://www.legacysurvey.org/viewer

    Code copied from
    https://github.com/legacysurvey/imagine/blob/master/map/views.py
    """

    def __init__(self, scales=None, m=0.03, Q=20, bands=["z", "z", "z"]):
        rgb_scales = {
            "u": (2, 1.5),
            "g": (2, 6.0),
            "r": (1, 3.4),
            "i": (0, 1.0),
            "z": (0, 2.2),
        }
        if scales is not None:
            rgb_scales.update(scales)

        self.rgb_scales = rgb_scales
        self.m = m
        self.Q = Q
        self.bands = bands
        self.axes, self.scales = zip(*[rgb_scales[bands[i]] for i in range(len(bands))])

        # rearange scales to correspond to image channels after swapping
        self.scales = [self.scales[i] for i in self.axes]

    def __call__(self, imgs):
        # Check image shape and set to C x H x W
        if imgs.shape[0] != len(self.bands):
            imgs = np.transpose(imgs, (2, 0, 1))

        I = 0
        for img, band in zip(imgs, self.bands):
            plane, scale = self.rgb_scales[band]
            img = np.maximum(0, img * scale + self.m)
            I = I + img
        I /= len(self.bands)

        Q = 20
        fI = np.arcsinh(Q * I) / np.sqrt(Q)
        I += (I == 0.0) * 1e-6
        H, W = I.shape
        rgb = np.zeros((H, W, 3), np.float32)
        for img, band in zip(imgs, self.bands):
            plane, scale = self.rgb_scales[band]
            rgb[:, :, plane] = (img * scale + self.m) * fI / I

        rgb = np.clip(rgb, 0, 1)
        return rgb.transpose(2, 0, 1)  # C x H x W



if __name__ == "__main__":
    main()


