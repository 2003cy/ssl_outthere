"""JADES data preparation from flow_method/flow_sed_jades.ipynb."""

import glob
import re
from pathlib import Path
from typing import Optional, Sequence

import h5py
import lightning.pytorch as L
import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from encoder_fusion.model.fusion import MultimodalFusion
from encoder_spectrum.LowResPT.data.dataset import LowResDataset

ROOT = Path(__file__).resolve().parents[2]


def project_path(path: str) -> Path:
    """Resolve relative data paths against the repository, independent of cwd."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


class JADESDataModule(L.LightningDataModule):
    """Read JADES p50 targets and concatenate frozen image/spectrum CLIP latents.

    Keep the highest-S/N spectrum per JADES ID, then split unique galaxies.
    Scalers are fitted on training galaxies only. Dataloaders return unscaled
    (X, Y); the model stores and applies the scalers in its checkpoint.
    Targets stay in model units: log10(1+z), log mass, log SFR, log Z, log A_V.
    """

    def __init__(
        self,
        embeddings_path: str,
        fits_path: str,
        labels_path: str,
        fusion_checkpoint: str,
        target_columns: Sequence[str] = ("log1pz", "logmass", "logsfr100", "gas_logz", "av"),
        log_columns: Sequence[str] = ("av",),
        grades: Sequence[int] = (1, 2, 3),
        min_sn50: float = 0.0,
        frac_valid_pix: float = 0.5,
        min_redshift: Optional[float] = 1.0,
        max_redshift: Optional[float] = 3.0,
        max_gas_logz: Optional[float] = 0.40,
        z_consistent: bool = True,
        split: Sequence[float] = (0.50, 0.25, 0.25),
        split_seed: int = 0,
        batch_size: int = 64,
        num_workers: int = 0,
        embedding_batch_size: int = 128,
        embedding_device: str = "auto",
    ):
        super().__init__()
        self.save_hyperparameters()
        if len(split) != 3 or min(split) <= 0 or not np.isclose(sum(split), 1):
            raise ValueError("split must contain positive train/validation/test fractions summing to 1")
        if not target_columns or not set(log_columns).issubset(target_columns):
            raise ValueError("log_columns must be a subset of the nonempty target_columns")
        if batch_size < 1 or embedding_batch_size < 1 or num_workers < 0:
            raise ValueError("Batch sizes must be positive and num_workers nonnegative")
        self.train_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return
        hp = self.hparams
        labels = Table.read(project_path(hp.labels_path))
        with fits.open(project_path(hp.fits_path), memmap=True) as hdul:
            all_z = np.asarray(hdul["CATALOG"].data["z_best"], dtype=float).copy()
        # Reuse the spectrum dataset's grades and observed-frame valid-pixel cut.
        spectra = LowResDataset(
            fits_path=str(project_path(hp.fits_path)),
            grades=hp.grades,
            frac_valid_pix=hp.frac_valid_pix,
        )
        keep = spectra.valid_indices.copy()
        del spectra

        quantiles = {}
        for q in ("p16", "p50", "p84"):
            values = np.stack([np.asarray(labels[f"{name}_{q}"], float)
                               for name in hp.target_columns], axis=1)
            for name in hp.log_columns:
                i = list(hp.target_columns).index(name)
                v = values[:, i]
                with np.errstate(divide="ignore", invalid="ignore"):
                    values[:, i] = np.where(v > 0, np.log10(v), np.nan)
            quantiles[q] = values

        with h5py.File(project_path(hp.embeddings_path), "r") as f:
            ids = f["id"][:]
            sn = f["sn50"][:]
            groups = np.asarray(labels["jades_id"])
            if len(labels) != len(ids):
                raise ValueError("JADES labels must be row-aligned with the embeddings")
            if "dja_id" in labels.colnames and not np.array_equal(labels["dja_id"], ids):
                raise ValueError("JADES dja_id order does not match the embedding IDs")
            z = all_z[ids]
            valid_frac = f["spectrum_token_mask"][:].mean(axis=1)
            selected = ((sn > hp.min_sn50) & (valid_frac > hp.frac_valid_pix)
                        & np.isin(ids, keep) & (groups >= 0)
                        & np.isfinite(quantiles["p50"]).all(axis=1))
            if hp.z_consistent:
                selected &= np.asarray(labels["z_consistent"], bool)
            if hp.min_redshift is not None:
                selected &= z > hp.min_redshift
            if hp.max_redshift is not None:
                selected &= z <= hp.max_redshift
            if hp.max_gas_logz is not None:
                selected &= np.asarray(labels["gas_logz_p50"]) <= hp.max_gas_logz

            rows = np.flatnonzero(selected)
            # Scan only candidate spectra, in batches; never load the full HDF5.
            finite = []
            for start in range(0, len(rows), hp.embedding_batch_size):
                rr = rows[start:start + hp.embedding_batch_size]
                values = f["spectrum_patch_embed"][rr]
                finite.append(np.isfinite(values).reshape(len(rr), -1).all(axis=1))
            rows = rows[np.concatenate(finite)] if finite else rows
            n_spectra = len(rows)
            rows = rows[np.argsort(-sn[rows], kind="stable")]
            _, first = np.unique(groups[rows], return_index=True)
            self.rows = np.sort(rows[first])
            if len(self.rows) < 4:
                raise ValueError("Too few unique galaxies remain for train/validation/test splits")
            self.dja_ids = ids[self.rows]
            self.jades_ids = groups[self.rows]
            self.surveys = f["survey"][self.rows].astype(str)
            self.sn50 = sn[self.rows]
            self.redshift = z[self.rows]
            self.Y, self.Y16, self.Y84 = (quantiles[q][self.rows]
                                        for q in ("p50", "p16", "p84"))
            if not (np.isfinite(self.Y16).all() and np.isfinite(self.Y84).all()):
                raise ValueError("Selected catalog p16/p84 labels must be finite (positive for log columns)")
            self.X = self._encode_features(f)

        hold_fraction = hp.split[1] + hp.split[2]
        self.tr, hold = train_test_split(np.arange(len(self.rows)), test_size=hold_fraction,
                                        random_state=hp.split_seed)
        self.va, self.te = train_test_split(hold, test_size=hp.split[2] / hold_fraction,
                                          random_state=hp.split_seed)
        self.xs = StandardScaler().fit(self.X[self.tr])
        self.ys = StandardScaler().fit(self.Y[self.tr])
        self.train_dataset, self.val_dataset, self.test_dataset = (
            TensorDataset(torch.as_tensor(self.X[ii], dtype=torch.float32),
                          torch.as_tensor(self.Y[ii], dtype=torch.float32))
            for ii in (self.tr, self.va, self.te)
        )
        print(f"Flow data: {n_spectra} spectra -> {len(self.rows)} unique JADES galaxies; "
              f"train {len(self.tr)} / val {len(self.va)} / test {len(self.te)}", flush=True)

    def _encode_features(self, f: h5py.File) -> np.ndarray:
        hp = self.hparams
        paths = glob.glob(str(project_path(hp.fusion_checkpoint)))
        if not paths:
            raise FileNotFoundError(f"No fusion checkpoint matches {project_path(hp.fusion_checkpoint)}")

        def score(path):
            match = re.search(r"R10=([0-9]+(?:\.[0-9]+)?)", Path(path).name)
            return float(match.group(1)) if match else float("-inf")

        self.fusion_checkpoint = max(sorted(paths), key=score)
        ck = torch.load(self.fusion_checkpoint, map_location="cpu", weights_only=False)
        fusion_hp = ck["hyper_parameters"]
        fusion = MultimodalFusion(latent_dim=fusion_hp["latent_dim"],
                                  temperature=fusion_hp["temperature"])
        for name, cfg in fusion_hp["modalities"].items():
            fusion.register_modality(
                name, input_dim=f[f"{name}_patch_embed"].shape[-1],
                hidden_dim=cfg.get("hidden_dim"), pool=cfg.get("pool"),
                num_heads=cfg.get("num_heads", 4), stats_dim=cfg.get("stats_dim"),
                stats_scale=cfg.get("stats_scale"),
            )
        fusion.load_state_dict({k[len("fusion."):]: v for k, v in ck["state_dict"].items()
                                if k.startswith("fusion.")})
        device = hp.embedding_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        fusion = fusion.to(device).eval().requires_grad_(False)
        chunks = []
        print(f"Frozen fusion: {self.fusion_checkpoint} ({device})", flush=True)
        with torch.no_grad():
            for start in range(0, len(self.rows), hp.embedding_batch_size):
                rr = self.rows[start:start + hp.embedding_batch_size]
                inputs = {name: torch.as_tensor(f[f"{name}_patch_embed"][rr], device=device)
                          for name in ("image", "spectrum")}
                available = {name: torch.ones(len(rr), dtype=torch.bool, device=device)
                             for name in inputs}
                masks = {"spectrum": torch.as_tensor(f["spectrum_token_mask"][rr],
                                                     dtype=torch.bool, device=device)}
                stats = {name: torch.as_tensor(f[f"{name}_stats"][rr], dtype=torch.float32,
                                               device=device) for name in fusion.stats_encoders}
                encoded = fusion(inputs, available, masks, stats or None)
                chunks.append(torch.cat([encoded["image"], encoded["spectrum"]], dim=1).cpu().numpy())
        features = np.concatenate(chunks)
        if not np.isfinite(features).all():
            raise ValueError("Frozen fusion produced non-finite features")
        return features

    def _loader(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=shuffle,
                          num_workers=self.hparams.num_workers, drop_last=False,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=self.hparams.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset)

    def test_dataloader(self):
        return self._loader(self.test_dataset)

    def predict_dataloader(self):
        return self.test_dataloader()
