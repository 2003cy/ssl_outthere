# LowResPT — Optuna HPO

## Layout
```
optuna/
├── optuna_train.py     # objective + Lightning trainer wrapper
├── launch_study.py     # multi-process launcher (4-8 workers on one GPU)
├── configs/base.yaml   # fixed/default hyperparameters
└── studies/
    └── {study_name}/
        ├── study.db         # SQLite — concurrent multi-process safe
        ├── launch_meta.yaml # snapshot of CLI args
        └── trials/
            └── trial_XXXX/
                ├── hparams.yaml
                ├── checkpoints/best-*.ckpt
                ├── metrics.csv     # CSVLogger
                ├── result.json
                └── trial.log
```

## Quickstart

```bash
cd /home/yacheng/ssl_outthere/encoder_spectrum/LowResPT
# H100 NVL 96GB → 4 parallel trials comfortably; bump to 6-8 if util < 70%.
pixi run -e h100 python optuna/launch_study.py \
    --study-name sweep_v1 \
    --n-workers  4 \
    --n-trials   100 \
    --max-epochs 200
```

Resume / extend the same study (just re-run with more trials):
```bash
pixi run -e h100 python optuna/launch_study.py --study-name sweep_v1 --n-workers 4 --n-trials 50
```

Live dashboard (other shell):
```bash
pixi run -e h100 optuna-dashboard sqlite:///optuna/studies/sweep_v1/study.db
```

## Search space (edit in `optuna_train.py :: suggest_hparams`)

| Param              | Type     | Range                       |
|--------------------|----------|-----------------------------|
| `embed_dim`        | cat      | 128, 192, 256, 384          |
| `num_layers`       | cat      | 4, 6, 8, 12                 |
| `head_dim`         | cat      | 16, 32, 64  (num_heads = embed_dim / head_dim) |
| `dropout`          | uniform  | 0.0 – 0.3                   |
| `mask_ratio`       | uniform  | 0.3 – 0.7                   |
| `line_loss_weight` | log      | 1.0 – 10.0                  |
| `lr`               | log      | 1e-4 – 1e-3                 |
| `weight_decay`     | log      | 1e-5 – 1e-1                 |

Fixed (override in `base.yaml` if you want to widen): `patch_size`, `stride`,
`mlp_ratio`, `warmup_steps`, `batch_size`, `min_std`, `decoder_hidden_dims`.

## Notes
- **Pruner**: MedianPruner (kills underperformers after 20 epochs of warmup).
- **Sampler**: TPE, multivariate + group sampling for categorical+continuous mix.
- **Per-worker GPU memory**: ~2-4 GB → 4 workers ≈ 12-16 GB on H100.
- **CUDA MPS** (optional, for >4 workers): start `nvidia-cuda-mps-control -d`
  before launching; gives real hardware concurrency instead of time-slicing.
- Failed trials (OOM/NaN) return `inf`; don't crash the worker.
- `OptunaPruningCallback` reports `val_loss` every epoch — no `optuna.integration`
  dependency, so it survives Lightning/Optuna version drift.
