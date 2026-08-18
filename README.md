# SSL Outthere — Multi-Modal Self-Supervised Learning  JWST Imaging and low resolution spectroscopy data, 
# Prototyped for OutThere pure parallel survey

Self-supervised representation learning and downstream tasks for JWST **NIRCam/NIRISS images**
and **NIRSpec/prism spectra**. The pipeline pretrains an **image encoder** and a **spectrum
encoder** independently, fuses them with a versitile option including **CLIP-style contrastive head** or a simple **cross-attention** pooling into a final joint latent
space, and probes downstream physical properties (redshift, stellar mass, morphology).

The image encoder is a highly modified version of[AstroCLIP](https://github.com/PolymathicAI/AstroCLIP)'s adaptation
of [DINOv2](https://github.com/facebookresearch/dinov2) (self-distillation + masked image
modeling), customized for versatile hyperparameter choise beyond default viT (base, large ...), single-channel JWST galaxy cutouts, noise augmentations and supports (in progress)variable input resolution via Sinusoial positional encoding.

---

## Repository structure

The repo is **code, not data** — checkpoints and datasets are git-ignored (`*.ckpt`, `*.h5`,
`*.fits`, `model/`, `outputs/`, `images/jwst`). This repo contains the full end to end pipeline from data preparation to downstream tasks. Top-level layout:

| Path | What it is |
|------|------------|
| `encoder_image/` | Image SSL. `astrodino/` — DINOv2/iBOT ViT on F150W cutouts (**primary model**); `mocov2/` — contrastive baseline (experimental). |
| `encoder_spectrum/` | Spectrum SSL. `LowResPT/` — low-res patch-transformer MAE on prism spectra; `ma_specformer/` — point-wise GPT-style spectrum transformer. |
| `encoder_fusion/` | CLIP-style multimodal fusion of precomputed image+spectrum tokens (`model/fusion.py` → `MultimodalFusion`), plus downstream probes. |
| `images/` | Data pipeline + stores: NIRCam/ NIRISS imaging preparation, DJA spectra catalog, image↔spectrum crossmatch. See [Data pipeline](#data-pipeline). |
| `paper/` | Manuscript. |
| `pixi.toml`, `pixi.lock` | Environment definition + locked dependency graph (see [Installation](#installation)). |

`CLEAR/` and `OUTTHERE/` data exporters under `images/` are work-in-progress.

---

## Installation - pixi up!

Environments are managed with [**pixi**](https://pixi.sh). Prerequisites: Linux x86-64, an
NVIDIA GPU + driver, and pixi installed. Python is pinned to `>=3.10,<3.12`.

Three environments are defined in `pixi.toml`:

| Env | Features | Stack |
|-----|----------|-------|
| `default` | `cu117` | torch 2.0.1+cu117, torchvision 0.15.2, xformers 0.0.20 — **the main env** |
| `dev` | `cu117` + `dev` | adds `pytest`, `black`, `isort` |
| `h100` | `h100` | torch 2.0.1+**cu118** build |

```bash
pixi install                       # solve + create the envs from pixi.lock
#`bootstrap` is needed for cutomized installation and code patching for **DINOv2`** 
pixi run bootstrap                 # install dinov2 (--no-deps), patch it
pixi run register-kernel           # (default env) Jupyter kernel "ssl_outthere"
pixi run -e h100 register-kernel   # h100 env kernel "ssl_h100"
pixi run -e dev pytest             # run tests
```

Run any command inside an environment with `pixi run -e <env> <cmd>`, or drop into a shell with
`pixi shell -e <env>`. For modification to the envs, refer to pixi [documentation](https://pixi.prefix.dev/latest/).

> **Which env for which GPU:** Use **`default` (cu117) on A100** and lower (sm_80) — this is the tested
> path. The **`h100` (cu118)** env targets H100 (sm_90) was extensively used for pretraining of the spectrum/CLIP/downstream models, but the image encoder has not been tested, proceed with caution.

---

## Data pipeline

Stores live under `images/` (git-ignored). The image stores are per-tile **HDF5** files written
by `images/cosmos_2025/cutout_export.py` (COSMOS-Web) and the CEERS exporter:

| Dataset | Shape | Type | Notes |
|---------|-------|------|-------|
| `image` | (N, 128, 128) | float32 | 30 mas/pixel cutout, MJy/sr |
| `seg` | (N, 128, 128) | uint8 | binary source mask — **COSMOS tiles only** (CEERS has none) |
| `id` | (N,) | int64 | row index into the source catalog |
| `ra`, `dec` | (N,) | float64 | sky position (deg) |

File attributes: `pixscale_mas=30`, `bunit`, `survey`, `image_size`; CEERS adds `sky_sigma`.

End-to-end flow:

```
JWST mosaics (COSMOS-Web f150w, CEERS)
   └─ cutout_export.py                  → images/jwst/f150w/*.h5   (~700k cutouts, 21 tiles)

DJA NIRSpec prism (DJA_spectra_v4.5.fits, ~42k rows: flux / z_best / sn50 / ...)
   └─ images/dja_crossmatch/crossmatch.ipynb   (1:1 match, ≤ 0.5")
                                        → images/dja_crossmatch/dja_matched.h5
                                          (image cutouts; `id` indexes into the DJA FITS,
                                           so the spectrum is dja[id])

image h5 + spectrum FITS → frozen image/spectrum encoders → token-embedding h5 → encoder_fusion
```

**Paths:** configs reference absolute data paths. On other hosts,
override the data root rather than editing configs — e.g. the AstroDINO dataset string
`jwst:split=train:root=<your/images/jwst/>:filter=f150w`, or the `h5_path` in the fusion config.

---

## Components & quickstart

Each encoder has its own training entrypoint and config; outputs (checkpoints, metrics) are
git-ignored and land under the module's `model/` or `outputs/` directory.

### Image — AstroDINO (`encoder_image/astrodino/`)
DINOv2/iBOT ViT-Base self-supervised on F150W cutouts.
```bash
cd encoder_image/astrodino/train
torchrun --nproc_per_node=<N> trainer.py \
  -c configs/astrodino_f150w_vitb_ps6_st3_bs128.yaml --run-name <name>
```
FSDP-sharded (one shard per rank); a single GPU is auto-detected (`world_size=1`). Outputs:
`model/<name>/` (`*.rank_*.pth` shards, `eval/training_*/teacher_checkpoint.pth` consolidated
teacher). Eval: linear-probe notebooks in `benchmark/`. Training **requires working xformers**
(see the env↔GPU note above).

### Spectrum — LowResPT (`encoder_spectrum/LowResPT/`)
Masked-autoencoder patch transformer on low-res prism spectra.
```bash
cd encoder_spectrum/LowResPT
python trainer.py fit --config low_res_pt.yaml
```
Data: `images/DJA/DJA_spectra_v4.5.fits`. Outputs: `outputs/<run>/version_*/checkpoints/*.ckpt`.
Eval: `linear_probe_redshift.ipynb`, `recon.ipynb`.

### Fusion (`encoder_fusion/`)
CLIP-style contrastive fusion of **frozen** image + spectrum tokens into a shared latent.
First extract token embeddings (`train_dja.ipynb` → an embeddings HDF5), then:
```bash
cd encoder_fusion
python trainer.py fit --config config_dja.yaml
```
Outputs: `outputs/fusion_*/version_*/`. Eval: `probe_mass.ipynb`, `downstream_mass.ipynb`.

---

## License & citation

MIT (see `LICENSE`). Author: Yang Cheng. Repo: <https://github.com/2003cy/ssl_outthere>.

Image encoder is built based on [AstroCLIP](https://github.com/PolymathicAI/AstroCLIP) and
[DINOv2](https://github.com/facebookresearch/dinov2) — please credit those works when using the
image encoder.
