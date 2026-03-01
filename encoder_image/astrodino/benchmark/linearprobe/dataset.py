"""
Dataset classes for linear probe evaluation.
"""
import os
import sys
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, "/u/yacheng/projects/ssl_outthere")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import get_torgb


class JWSTMorphDataset(Dataset):
    """
    JWST f150w dataset for linear probe evaluation.
    Filters out samples with NaN in morph_flag_f150w or delta_f150w.
    
    Args:
        root: Path to the data directory containing h5 files
        crop_size: Size to center crop images to
        max_samples: Maximum number of samples to use (-1 for all)
        seed: Random seed for reproducibility
        balanced: If True, undersample majority classes to match minority class count
        samples_per_class: If set (>0), use this many samples per class (overrides balanced)
        delta_threshold: Maximum delta_f150w value to include (default 0.5, set to None to disable)
        effective_radius_min: Minimum allowed effective radius in pixels (from f['radius_sersic'] converted using 30 mas/pixel)
    """
    def __init__(self, root: str, crop_size: int = 90, max_samples: int = -1, seed: int = 42,
                 balanced: bool = False, samples_per_class: int = -1, delta_threshold: float = 0.5,
                 effective_radius_min: float = None, cfg=None):
        self.crop_size = crop_size
        self.to_rgb, self.in_chans = get_torgb(cfg)
        self.center_crop = transforms.CenterCrop(crop_size)
        self.rng = np.random.default_rng(seed=seed)
        
        # Load all h5 files
        self._files = []
        self._file_names = []
        h5_files = sorted(f for f in os.listdir(root) if f.endswith(".h5"))
        
        for fname in h5_files:
            fpath = os.path.join(root, fname)
            try:
                f = h5py.File(fpath, 'r')
                # Check if morph_flag_f150w exists
                if 'morph_flag_f150w' in f.keys():
                    self._files.append(f)
                    self._file_names.append(fname)
                else:
                    f.close()
                    print(f"Skipping {fname}: no morph_flag_f150w")
            except Exception as e:
                print(f"Error loading {fname}: {e}")
        
        print(f"Loaded {len(self._files)} files")
        
        # Build index of valid samples (non-NaN morph_flag and delta, with delta threshold)
        self._valid_indices = []  # (file_idx, local_idx, label)
        
        for file_idx, f in enumerate(tqdm(self._files, desc="Indexing valid samples")):
            morph_flag = f['morph_flag_f150w'][:]
            delta = f['delta_f150w'][:]
            
            # Find valid indices (non-NaN in both + delta threshold)
            valid_mask = np.isfinite(morph_flag) & np.isfinite(delta)
            if delta_threshold is not None:
                valid_mask = valid_mask & (delta < delta_threshold)
            if effective_radius_min is not None:
                if 'radius_sersic' not in f:
                    valid_mask[:] = False
                else:
                    re = f['radius_sersic'][:]
                    re_pix = re * (3600 * 1000 / 30)
                    valid_mask = valid_mask & np.isfinite(re_pix) & (re_pix >= effective_radius_min)
            valid_local_indices = np.where(valid_mask)[0]
            
            for local_idx in valid_local_indices:
                label = int(morph_flag[local_idx])
                self._valid_indices.append((file_idx, local_idx, label))
        
        print(
            f"Total valid samples (delta < {delta_threshold}, radius_sersic_pix >= {effective_radius_min}): "
            f"{len(self._valid_indices)}"
        )
        
        # Apply balanced sampling or samples_per_class
        if samples_per_class > 0 or balanced:
            self._valid_indices = self._balance_samples(samples_per_class, balanced)
        
        # Subsample if max_samples is set (applied after balancing)
        if max_samples > 0 and max_samples < len(self._valid_indices):
            indices = self.rng.choice(len(self._valid_indices), size=max_samples, replace=False)
            self._valid_indices = [self._valid_indices[i] for i in indices]
            print(f"Subsampled to {len(self._valid_indices)} samples (max_samples={max_samples})")
        
        # Get label distribution
        labels = [x[2] for x in self._valid_indices]
        unique, counts = np.unique(labels, return_counts=True)
        print("Label distribution:")
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} samples ({100*c/len(labels):.1f}%)")
    
    def _balance_samples(self, samples_per_class: int, balanced: bool):
        """Balance samples across classes by undersampling majority classes."""
        # Group indices by label
        label_to_indices = {}
        for idx, (file_idx, local_idx, label) in enumerate(self._valid_indices):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        
        # Determine samples per class
        class_counts = {label: len(indices) for label, indices in label_to_indices.items()}
        print(f"Original class counts: {class_counts}")
        
        if samples_per_class > 0:
            n_samples = samples_per_class
        else:  # balanced=True
            n_samples = min(class_counts.values())
        
        print(f"Balancing to {n_samples} samples per class")
        
        # Sample from each class
        balanced_indices = []
        for label, indices in sorted(label_to_indices.items()):
            if len(indices) >= n_samples:
                sampled = self.rng.choice(indices, size=n_samples, replace=False)
            else:
                # If class has fewer samples than n_samples, use all of them
                sampled = indices
                print(f"  Warning: Class {label} only has {len(indices)} samples (< {n_samples})")
            balanced_indices.extend(sampled)
        
        # Shuffle and return
        self.rng.shuffle(balanced_indices)
        return [self._valid_indices[i] for i in balanced_indices]
    
    def __len__(self):
        return len(self._valid_indices)
    
    def __getitem__(self, index):
        file_idx, local_idx, label = self._valid_indices[index]
        
        # Load image  (H, W) single band
        img = self._files[file_idx]['image'][local_idx].astype('float32')

        # For 3-channel models repeat before crop so ToRGB3Band gets (3,H,W);
        # for 1-channel models keep (1,H,W).
        n = self.in_chans if self.in_chans > 1 else 1
        img = np.repeat(img[np.newaxis], n, axis=0)     # (C, H, W)

        tensor = self.center_crop(torch.from_numpy(img))
        tensor = torch.from_numpy(self.to_rgb(tensor.numpy()))  # (C, H, W)

        return tensor, label
    
    def close(self):
        for f in self._files:
            try:
                f.close()
            except:
                pass
