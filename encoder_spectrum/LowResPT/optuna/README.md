# LowResPT — Optuna hyperparameter search

One trial = one **300-epoch** run at the production configuration. Every
**20 epochs** a frozen-encoder **redshift probe** is fitted on the live model,
averaged over **40 group-aware splits** (as the stellar-mass and Sersic probes
are), and its **σ_NMAD** is reported to Optuna. The trial's objective is the
**median of the last three reports** (minimize).

Not the best-over-run: the best-scoring epoch is a live model that no checkpoint
ever saved, and a minimum over 15 reports rewards a trial for being noisy. The
median of the final reports describes the state the run actually ends in, which
is the checkpoint that gets used. Every report is still needed — pruning
compares trials at equal epochs.

`val_hid_loss` is deliberately *not* the objective. It still selects the
checkpoint and drives early stopping *inside* a run — it just doesn't rank
trials. Reconstruction loss is dominated by the noise floor and says little
about whether redshift is linearly decodable from the representation.

## Run

```bash
cd /home/yacheng/ssl_outthere/encoder_spectrum/LowResPT
nohup pixi run -e default python optuna/launch_study.py \
    --study-name r1 --n-trials 200 --timeout 43200 > r1.log 2>&1 &
```

`--timeout` (seconds) stops the worker on the clock even if `--n-trials` is not
reached, which is the reliable way to hit a wall-clock budget: a trial costs
2.2 min at d=32/L=4 and 6.6 min at d=192/L=12, so a fixed trial count can
overshoot by 3x depending on where TPE settles. 12 h fits roughly 180 trials at
the 4.0 min grid average.

`max_line_blocks > 0` must stay out of the search space: it falls back to the
Python-loop masking, which is ~10x slower per trial and would blow the budget.

**Run one worker.** Two processes sharing this GPU measured 6.5x *slower* each
(4.86 -> 31.5 s/epoch) and three were 12.5x slower: the model issues many tiny
kernels, so time-slicing between CUDA contexts costs more than it buys. GPU
utilisation reading 28% at one worker is not headroom for a second.

Use the `default` pixi environment. Re-running the same command resumes the
study; running it again in another shell adds a worker against the same sqlite
DB. **`--n-trials` counts trials for that worker**, so three workers of 20 make a
60-trial study, not 180.

A trial leaves the GPU mostly idle (28 tokens at batch 1024 — the bottleneck is
launch and dataloader overhead, not compute), so workers scale well. On a
24-core node three workers is the sweet spot:

```bash
export OMP_NUM_THREADS=1          # and set probe.n_jobs: 6 in base.yaml
pixi run -e default python optuna/launch_study.py --study-name r1 --n-trials 20
```

3 workers x 6 probe threads = 18 of 24 cores, leaving room for the dataloaders.

```python
import optuna
s = optuna.load_study(study_name="r1", storage="sqlite:///optuna/studies/r1/study.db")
print(s.best_value, s.best_params)
s.trials_dataframe().to_csv("r1.csv")
```

Per-trial artifacts (`hparams.yaml`, `metrics.csv`, `result.json`,
`checkpoints/`) land in `studies/<name>/trials/trial_XXXX/`. `result.json`
carries the full per-epoch probe `history`.

## Searched — 10 dimensions

| param | space | why |
|---|---|---|
| `embed_dim` | {32, 64, 128, 192} | Shifted down from the obvious {128…384}, for two independent reasons. **Capacity:** the training set is 22,522 × 56 px = 1.26M scalars, so this grid spans N_params/N_data = 0.04–4.2× and straddles 1:1, where {128…384} spans 0.63–16.9× and sits almost entirely on the over-parameterised side. Both independently tuned optima land at 1–2× (LowResPT d=128/L=8 → 1.26×; SpecML d=512/L=12 on a 16× larger set → 1.91×), so a grid whose minimum is already at the optimum will return its own boundary — which is exactly what happened. **Readout:** the probe reads `concat(27 × d)` against 4582 training rows; d=192 is already 5184 features and anything wider lets a trial win on feature count rather than representation quality. |
| `head_dim` | {8, 16, 32}, heads = $d$/head_dim | Every pair divides exactly, so no trial is wasted. 4 is excluded: dot-product attention over a 4-dimensional head is close to noise, and d=192 would give 48 of them. Strongly interacts with `embed_dim` — hence `multivariate=True, group=True`. |
| `num_layers` | {4, 6, 8, 12} | With 27 tokens one layer is already global, so depth buys non-linear composition, not receptive field. Expect the optimum below SpecML's 12 (which had 233 tokens). |
| `dropout` | [0.0, 0.3] | ~22.5k spectra of 56 numbers against 0.1–4M parameters; the model is over-parameterised by orders of magnitude. |
| `continuous_patch_length` | int {1..4} | Highest leverage. At `patch_size=4, stride=2` adjacent tokens share 2 of their 4 pixels, so a C-token block hides only `2C−2` pixels exclusively: C=1 hides **none** — every pixel is copyable from a visible neighbour. C=2/3/4 hide 2/4/6 px = 0.036/0.072/0.108 µm. The Hβ–[OIII] separation (146 Å rest) is 1.6/2.4/3.3 px at z=1/2/3, so **C=3 is the smallest block that can hide the pair together**. |
| `mask_ratio` | [0.30, 0.75] | With 2–4 strong lines in the window each spanning 1–2 tokens, the chance that a random mask hides *all* of them is ≈ r³ — 13% at r=0.5, 34% at r=0.7, 42% at r=0.75. The upper end is where the pretext task starts destroying the redshift anchor. |
| `err_weight_sigma_min` | log [0.01, 2.0] | The inverse-variance floor `w = 1/max(σ², ε)`. Because flux is standardised to unit variance, **σ²=1 is exactly S/N=1**, so this range is S/N ∈ [1, 10] with a tail past the noise-dominated boundary. See the σ² table below: 0.01 clamps 5% of tokens, 2.0 clamps ~95% — the top of the range therefore contains the *no weighting* ablation for free. |
| `lr` | log [1e-4, 1e-3] | At batch 1024 a run is only ~6300 updates, so under-training is the more likely failure than over-fitting. |
| `weight_decay` | log [1e-5, 1e-1] | In AdamW the effective regularisation is lr × wd, and lr is searched over a full decade — pinning wd would leave the high-lr trials under-regularised. Partly redundant with `dropout`. |
| `warmup_steps` | log [200, 2000] | A run is ~6300 steps at 21 steps/epoch, so this spans 3–32% of it (the shipped 1000 is 16%, ≈48 epochs). The dry run showed the objective *degrading* through warmup — 0.0543 at epoch 19 → 0.0600 at epoch 39 — and improving only once cosine decay takes over. Strongly coupled to the searched `lr`, so it is searched jointly rather than pinned. The bound is 2000 (epoch 95), not 3000, so warmup always ends before the pruner's first decision — see below. |

`decoder_hidden_dims` is **not** searched: a deeper head would let the
reconstruction be solved inside the decoder instead of being forced into the
encoder representation that every downstream probe uses.

## Fixed — and why

Values follow **the checkpoint the paper benchmarks**
(`outputs/low_res_pt_1_2_micron_noz_cut_tokenweight/version_0`), not
`low_res_pt.yaml`, which has drifted from it (the yaml says
`continuous_patch_length: 1` and enables line-aware masking; the benchmarked
model used C=2 and pure random masking).

- **`patch_size=4`, `stride=2`** — they set the 27 tokens and hence the readout
  width, they set the positional-encoding resolution, and they are quoted in the
  paper's architecture description. The 50% overlap they create is handled
  through `continuous_patch_length`.
- **`max_line_blocks=0`** — pure random masking, as in the benchmarked
  checkpoint. Line-aware masking is a separate A/B, not a sweep dimension.
- **`selected_mask_prob=1.0`** — the effective hidden fraction is
  `mask_ratio × selected_mask_prob`, so it is multiplicatively confounded with
  the searched `mask_ratio`: two knobs fighting over one quantity.
- **`batch_size=1024`, `max_epochs=300`, `warmup_steps=1000`** — production
  values, so the trial run *is* the production run. This is the direct lesson
  from SpecML, where `warmup_steps` was tuned at 100 epochs and then had to be
  hand-rescaled for the 250-epoch production run.
- **All data cuts** — changing the training set between trials would make them
  incomparable.
- `mlp_ratio=4.0`, `bias`, `betas`, `min_lr=1e-6`, `min_unmasked=4`,
  `patch_invalid_threshold=0.25` — conventional or data-validity guards.
- **`loss_weighting=invvar`, `use_patch_stats=true`** — these define *which
  model* the study is about.

## Budget

180 trials. The reasoning, none of which depends on wall-clock:

**Coverage of the discrete skeleton.** `embed_dim`(4) x `head_dim`(3) x
`num_layers`(4) x `continuous_patch_length`(4) = 192 discrete combinations, on
top of 6 continuous dimensions. Below ~200 trials a sizeable fraction of those
combinations is never touched at all.

**Quantile guarantee.** Random search reaches the top-q fraction of the space
with probability `1 - (1-q)^n`, independent of dimensionality
\citep{bergstra2012}. At 95% confidence that gives `q = 1 - 0.05^(1/n)`:

| trials | 95% confidence of reaching top |
|---|---|
| 60 | 4.9% |
| 80 | 3.7% |
| **180** | **1.65%** |
| 300 | 1.0% |

TPE should beat random search, so these are conservative upper bounds. Returns
fall off fast: 80 -> 180 halves the quantile, 180 -> 300 barely moves it.

**What TPE itself can use.** Optuna splits finished trials into a "below" set of
size `min(ceil(0.1 n), 25)` (`default_gamma`, `_tpe/sampler.py`) and fits the
Parzen estimator on it. So the good-sample set stops growing at n=250 and 180
gives it 18 points. Note `n` counts pruned trials too, and the below set is
filled from COMPLETE trials first.

**Signal to noise.** Configurations differ by ~0.002 in sigma_NMAD, against a
0.00014 standard error on the 40-split mean — comfortable. This covers split
noise only; run-to-run training noise has not been measured.

## Speed

A trial is ~5.5 min: 2.5 min of training plus ~3 min of probing. Two changes got
it there from 28 min, both verified against the implementations they replace:

| | s/epoch | 300 epochs |
|---|---|---|
| `LowResDataModule` + `masking.mask_patches` | 4.77 | 23.8 min |
| `GPULowResDataModule` + `masking_gpu.mask_patches_fast` | **0.50** | **2.5 min** |

`model/masking_gpu.py` was the 9.5x. The reference `mask_patches` loops over the
batch in Python with several `.tolist()` calls per spectrum, forcing ~3000 GPU
synchronisations per step; the replacement loops over *blocks* instead, so the
trip count is `N // C + 1` (7-28) regardless of batch size. Measured 232-485x on
the masking call alone. Constraints are identical (block length exactly C, >=1
token gap for C>1, never on an invalid token, `min_unmasked` respected) and the
masked-count distributions match the original to within 0.4 tokens across
C=1..4 x ratio=0.3/0.6/0.75. `max_line_blocks > 0` falls straight through to the
original implementation -- verified bit-identical -- so line-aware masking still
works, just without the speedup. Switched by `model.use_fast_masking`.

`data/gpu_datamodule.py` keeps the ~13 MB sample on the device and slices batches
by index, removing ~1000 `__getitem__` calls per step. Worth only 4% before the
masking fix but ~28% after it, since its 0.19 s/epoch is a fixed cost that no
longer hides behind masking. The 90/10 split is reused verbatim from
`random_split`, and all five batch fields match the DataLoader path elementwise.

## Calibration

Both numbers below were measured on the benchmarked checkpoint; re-measure if
the tokenisation or the flux normalisation changes.

**Token σ² (164,971 tokens, production cuts).** Sets the
`err_weight_sigma_min` range.

```
p1 0.0037  p5 0.0103  p25 0.0748  p50 0.293  p75 0.654  p90 1.17  p99 2.97
eps=0.01 clamps  4.8%   eps=0.1 clamps 29.2%   eps=1.0 clamps 86.8%
eps=0.03 clamps 13.9%   eps=0.3 clamps 50.7%   eps=3.0 clamps 99.0%
```

The shipped default 0.1 clamps 29% of tokens, so it is a live knob, not inert.

**Lasso probe.** `LassoCV` is unusable at 3456 features — it spends minutes on
the smallest alphas of its default grid. A warm-started `lasso_path` over 12
alphas takes 7.6 s, and a single fit with `selection="random"` (deterministic
under a fixed `random_state`) is ~19× faster than cyclic descent at small α
(3.3 s vs 63.9 s). Fixed-α scan on the benchmarked checkpoint, split seed 7:

| alpha | time | n_active | R² | σ_NMAD | outliers |
|---|---|---|---|---|---|
| 3e-4 | 3.5 s | 1145 | 0.861 | 0.0466 | 4.8% |
| **5e-4** | **2.2 s** | **888** | **0.861** | **0.0473** | **5.1%** |
| 1e-3 | 1.4 s | 622 | 0.851 | 0.0488 | 5.7% |
| 2e-3 | 1.0 s | 370 | 0.833 | 0.0516 | 6.8% |
| 5e-3 | 0.7 s | 218 | 0.800 | 0.0591 | 7.8% |

`alpha=5e-4` is the default: genuinely sparse (26% of features) and within 4% of
ridge on σ_NMAD, at 2.2 s. All of these converge.

Worth knowing: Lasso does **not** beat ridge on this readout — σ_NMAD improves
monotonically as α → 0 and approaches ridge's 0.0456 from above. L1 buys
sparsity and interpretability, not accuracy.

Selecting α per-eval on a holdout of the train half was tried and rejected: the
holdout is too small, so it lands on heavy regularisation (σ_NMAD 0.0703 against
the 0.0466 the same data supports). The code path remains available via
`lasso_alpha: null`.

## Notes

- **Probe split seed is 7**, not the paper's 42, so re-scoring the winner at seed
  42 in `neurips_spectrum_bench.ipynb` is an out-of-selection number.
- The probe sample (9172 spectra, z ∈ (1,3], `frac_valid_pix>0.9`) is a subset of
  the ~23.9k the encoder pre-trains on. No label ever reaches pre-training, and
  redshift is never used to transform model input.
- k-NN (k=5) σ_NMAD is logged at every eval but never optimised — it is the guard
  against a config that wins on one readout by degrading the space.
- Pruning: `MedianPruner`, no trial cut before **epoch 180** (first decision at
  epoch 199), decisions at every probe report. The threshold is late for two
  reasons. Searching `warmup_steps` up to 2000 means warmup can run to epoch 95,
  and a run then needs roughly 200 more epochs of cosine decay to approach its
  best (the dry run was at 0.0531 at epoch 139 and 0.0495 at epoch 259) — judging
  earlier would systematically cut every long-warmup configuration regardless of
  where it would have ended up. The threshold is deliberately late: a full-length dry run
  (30 min 20 s, 15 reports) showed the objective is *not* monotonic — 0.0543 at
  epoch 19, degrading to 0.0600 by epoch 39 and staying worse until ~epoch 99,
  then improving steadily to 0.0495 at epoch 259 and flat thereafter. Pruning at
  epoch 79 would cut configurations on that transient.
- **Reference to beat**: the benchmarked checkpoint scores σ_NMAD
  **0.0471 ± 0.0009** under this exact objective (Lasso α=5e-4, 40 splits). The
  ±0.0009 is split-to-split scatter, so the 40-split mean has a standard error of
  0.00014 — differences of ~0.002 between configurations are resolvable.
- k-NN σ_NMAD *worsens* slightly over a full run (0.0054 → 0.0059) while Lasso
  improves by 9%. It saturates almost immediately, is not a usable objective, and
  is therefore evaluated on a **single** split — 40 neighbour searches in a
  5184-dimensional space would cost more than the objective itself.
- The in-loop probe uses `tol=0.01`, which is 4.8x faster than the `tol=0.001`
  default (58 s -> 12 s per 40-split evaluation) and shifts σ_NMAD by 0.00025,
  far below the 0.0009 split scatter. The benchmark path keeps `tol=0.001`.
- Splits are fitted in a thread pool (`probe.n_jobs`); scikit-learn's coordinate
  descent releases the GIL, so threads share the feature matrix instead of
  pickling it per worker. Drop `n_jobs` when running several sweep workers.
- `eval/` is shared with `neurips_spectrum_bench.ipynb`, so the number the sweep
  optimises and the number the paper reports cannot drift apart.
- The datasets are built **once** in `make_objective` and shared by every trial
  (`dm.dataset = ds_pt`), so the FITS is not re-read 40 times.
