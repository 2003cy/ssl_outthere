"""Frozen-encoder downstream evaluation for LowResPT.

`run_eval.eval_redshift` is the single implementation shared by the benchmark
notebooks and the Optuna sweep, so the number the sweep optimises and the number
the paper reports can never drift apart.
"""

from .probes import knn_zeroshot, lasso_probe, metrics, ridge_probe, sigma_nmad
from .run_eval import (
    PROBE_CUTS,
    best_ckpt,
    build_probe_dataset,
    compute_embeddings,
    eval_redshift,
    group_split,
    label_from_catalog,
)

__all__ = [
    "PROBE_CUTS",
    "best_ckpt",
    "build_probe_dataset",
    "compute_embeddings",
    "eval_redshift",
    "group_split",
    "knn_zeroshot",
    "label_from_catalog",
    "lasso_probe",
    "metrics",
    "ridge_probe",
    "sigma_nmad",
]
