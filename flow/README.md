# JADES conditional flow

The MLP + conditional Zuko NSF follows `flow_method/flow_sed_jades.ipynb`.
The Lightning model and CLI live in `model/flow.py`, with data preparation in
`data/datamodule_jades.py` (`JADESDataModule`), following the LowResPT
model/data/config layout. This DataModule is specific to the JADES catalog.

- `z_1_3.yaml`: keep `1 < z_spec <= 3`.
- `full.yaml`: no redshift-range cut; the other quality cuts are identical.
- [posterior/flow_jades.ipynb](posterior/flow_jades.ipynb): notebook training,
  posterior comparisons, PIT and coverage.
- [posterior/flow_jades_individual_posterior.ipynb](posterior/flow_jades_individual_posterior.ipynb): independent notebook training/loading
  and individual cutout, spectrum and corner plots.
- [posterior/data_label_check.ipynb](posterior/data_label_check.ipynb): label
  histograms before the metallicity cut, high-end zoom, threshold counts and
  catalog percentile checks; no flow training or fusion extraction required.

All notebooks live in `posterior/`. In either notebook, select `CONFIG` and
`SEEDS`, then run its cells. The first cell sets the working directory to
`flow/`, keeping the config, training command and output paths consistent.
The training cell uses the current kernel's interpreter:

```python
!{sys.executable} model/flow.py fit --config z_1_3.yaml
!{sys.executable} model/flow.py fit --config full.yaml
```

There is no separate trainer script. Model, data and Trainer parameters come
from YAML, and Lightning CLI overrides work, e.g. `--model.passes=2` or
`--trainer.max_epochs=100`. Relative **data paths** resolve from the repository
root; logger output paths resolve from the working directory (`flow/` in the
provided notebooks).

Data preparation keeps one highest-S/N valid spectrum per JADES galaxy and
uses the original notebook's actual 50/25/25 split, fixed by `data.split_seed`.
The configurable metallicity cut is currently `gas_logz_p50 <= 0.40`, relaxed
from 0.30; it is a sample-selection choice, not the fit prior boundary (0.50).
Set `data.max_gas_logz: null` to retain the full metallicity range. Use the label
check notebook to inspect the discarded objects before interpreting them as
unreliable fits. Existing trained checkpoints retain their original data cuts.
Changing a model seed does not change the split. Image and spectrum CLIP features
are extracted in batches using the frozen fusion checkpoint with the highest
R10 matching `data.fusion_checkpoint`; an explicit checkpoint path is also
accepted. Feature extraction runs once per process, before flow training.

The scalers are fitted on training galaxies only and stored as model buffers.
`Flow.sample(X)` returns CPU samples of shape `(n, n_objects, features)` with
standardization already undone. `Flow.log_prob(X, Y)` includes the corresponding
Jacobian. `X` is the unstandardized concatenated CLIP latent. `Y` and samples
remain in **model-label units**: log10(1+z), log mass, log SFR100, log Zgas, and
log10(A_V / mag). Convert the first coordinate to z only for plotting; do not
undo the scaler a second time.

`N_POST` can be any positive integer. With multiple flows, the notebooks draw
member counts from an equal-weight multinomial mixture; their sum is exactly
`N_POST`, even when it is smaller than or not divisible by the number of flows.

The model checkpoint also records target names and the data configuration,
including the exact selected fusion checkpoint. The notebooks load the best
validation-NLL checkpoint, rather than the final training weights. To reuse
existing runs, skip the training cell; choose the same config and seeds.
