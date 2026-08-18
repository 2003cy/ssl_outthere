"""Linear-probe the frozen jwst_dino teacher backbone on COSMOS morphology.

Mirrors astrodino's benchmark/linearprobe (linear_probe_morph.ipynb) end-to-end, but
for the jwst_dino stack: loads the teacher backbone from a Lightning ``*.ckpt``
(hyperparameters travel inside the ckpt), extracts CLS embeddings over
CosmosMorphDataset (labels cross-matched from ml_morph at load time), then fits a
logistic-regression probe and reports accuracy / macro-F1 / per-class report /
confusion matrix.

Run:
    python linear_probe.py \
      --weights ../../outputs/jwst_dino_ps6_st3/version_6/checkpoints/<epoch>.ckpt \
      --morph-catalog ../../../../data/survey/cosmos_2025/COSMOSWeb_mastercatalog_v1_ml_morph.fits \
      --balanced --samples-per-class 4000
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", ".."))  # jwst_dino/ for model + data imports
from model.jwst_dino import load_teacher_backbone  # noqa: E402
from dataset import CosmosMorphDataset, CLASS_NAMES  # noqa: E402


@torch.no_grad()
def extract_embeddings(net, loader, device):
    embs, labels = [], []
    for imgs, ys in tqdm(loader, desc="embeddings"):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            cls = net(imgs.to(device))["cls"]          # (B, embed_dim)
        embs.append(cls.float().cpu().numpy())
        labels.append(np.asarray(ys))
    return np.concatenate(embs), np.concatenate(labels)


def main():
    p = argparse.ArgumentParser(description="jwst_dino COSMOS morphology linear probe.")
    p.add_argument("--weights", required=True,
                   help="Lightning *.ckpt (loaded via JWST_DINO.load_from_checkpoint)")
    p.add_argument("--root", default="~/ssl_outthere/data/image")
    p.add_argument("--morph-catalog", required=True)
    p.add_argument("--filter", default="f150w")
    p.add_argument("--crop-size", type=int, default=None, help="default = config global_crops_size")
    p.add_argument("--delta-threshold", type=float, default=0.5)
    p.add_argument("--effective-radius-min", type=float, default=None, help="min sersic reff [px]")
    p.add_argument("--exclude-irregular", action="store_true", help="drop the irregular class")
    p.add_argument("--balanced", action="store_true")
    p.add_argument("--samples-per-class", type=int, default=-1)
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--test-frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    net = load_teacher_backbone(args.weights, device)
    crop_size = args.crop_size or net.crop_size

    ds = CosmosMorphDataset(
        root=args.root, morph_catalog=args.morph_catalog, filter=args.filter,
        crop_size=crop_size, delta_threshold=args.delta_threshold,
        effective_radius_min=args.effective_radius_min,
        exclude_irregular=args.exclude_irregular,
        balanced=args.balanced, samples_per_class=args.samples_per_class,
        max_samples=args.max_samples, seed=args.seed,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    X, y = extract_embeddings(net, loader, device)
    print(f"embeddings: {X.shape}  labels: {y.shape}")

    # ── linear probe: standardize -> logistic regression ────────────────────────
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_frac,
                                          random_state=args.seed, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xte))

    names = [CLASS_NAMES[c] for c in sorted(np.unique(y))]
    print(f"\n=== linear probe ({len(Xtr)} train / {len(Xte)} test) ===")
    print(f"accuracy   : {accuracy_score(yte, pred):.4f}")
    print(f"macro-F1   : {f1_score(yte, pred, average='macro'):.4f}")
    print(classification_report(yte, pred, target_names=names, digits=3))
    print("confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(yte, pred))


if __name__ == "__main__":
    main()
