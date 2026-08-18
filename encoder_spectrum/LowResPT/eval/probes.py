"""Probes and metrics applied to frozen LowResPT embeddings.

Every probe standardises features with a `StandardScaler` fitted on the training
half only, and none of them touches the encoder.
"""

import numpy as np
from sklearn.linear_model import Lasso, RidgeCV, lasso_path
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHAS = np.logspace(-2, 5, 15)
# Descending, as lasso_path requires. Bounded below at ~3e-4: past that the
# active set explodes, coordinate descent slows by orders of magnitude, and the
# fit has already converged onto ridge (see the calibration table in README.md).
LASSO_ALPHAS = np.logspace(-1, -3.5, 12)
KNN_K = 5


def sigma_nmad(dz):
    """Normalised median absolute deviation, centred on the median of `dz`.

    Centring on the median rather than zero makes this a pure scatter measure,
    insensitive to a constant bias.
    """
    return float(1.4826 * np.median(np.abs(dz - np.median(dz))))


def metrics(y, yp):
    """Redshift metrics; `dz` is the conventional (z_pred - z_true) / (1 + z_true)."""
    dz = (yp - y) / (1.0 + y)
    return dict(r2=float(r2_score(y, yp)),
                mae=float(mean_absolute_error(y, yp)),
                rmse=float(np.sqrt(np.mean((y - yp) ** 2))),
                snmad=sigma_nmad(dz),
                out=float(np.mean(np.abs(dz) > 0.15)))


def knn_zeroshot(X_tr, y_tr, X_va, k=KNN_K):
    """Zero-shot retrieval probe: distance-weighted k-NN average over the train half."""
    sc = StandardScaler().fit(X_tr)
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance").fit(sc.transform(X_tr), y_tr)
    return knn.predict(sc.transform(X_va))


def ridge_probe(X_tr, y_tr, X_va, alphas=RIDGE_ALPHAS):
    """Closed-form ridge; the penalty comes from RidgeCV's internal CV on the train half."""
    sc = StandardScaler().fit(X_tr)
    r = RidgeCV(alphas=alphas).fit(sc.transform(X_tr), y_tr)
    return r.predict(sc.transform(X_va)), {"alpha": float(r.alpha_),
                                           "n_active": int(np.sum(r.coef_ != 0))}


def lasso_probe(X_tr, y_tr, X_va, alphas=LASSO_ALPHAS, alpha=None, val_frac=0.2,
                max_iter=50_000, tol=1e-3, selection="random", random_state=0):
    """L1 probe. `alpha=None` selects the penalty on a holdout of the train half.

    `LassoCV` is unusable at this width (3456 features): it spends minutes on the
    smallest alphas of its default grid. Instead one warm-started `lasso_path`
    over `alphas` is fitted on part of the training half and scored by sigma_NMAD
    on the rest, so the reported validation metrics never inform the penalty.
    Pass an explicit `alpha` to skip selection entirely.

    The selection holdout is a plain random split, not grouped by object; it only
    ever picks a scalar penalty, so a straddling repeat observation cannot leak
    anything of substance.

    `selection="random"` is not a source of nondeterminism here (`random_state`
    is fixed) and is ~19x faster than cyclic descent at the small alphas where
    the active set is large: 3.3 s against 63.9 s at alpha=3e-4.
    """
    sc = StandardScaler().fit(X_tr)
    Ztr, Zva = sc.transform(X_tr), sc.transform(X_va)

    if alpha is None:
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(len(Ztr))
        cut = int(len(idx) * (1.0 - val_frac))
        i_fit, i_sel = idx[:cut], idx[cut:]
        # lasso_path assumes centred X (StandardScaler guarantees it) and no
        # intercept, so the train mean of y is added back by hand.
        _, coefs, _ = lasso_path(Ztr[i_fit], y_tr[i_fit], alphas=alphas,
                                 max_iter=max_iter, tol=tol)
        ybar = y_tr[i_fit].mean()
        ysel = y_tr[i_sel]
        scores = [sigma_nmad(((Ztr[i_sel] @ c + ybar) - ysel) / (1.0 + ysel))
                  for c in coefs.T]
        alpha = float(alphas[int(np.argmin(scores))])

    m = Lasso(alpha=alpha, max_iter=max_iter, tol=tol,
              selection=selection, random_state=random_state).fit(Ztr, y_tr)
    lo, hi = float(np.min(alphas)), float(np.max(alphas))
    edge = "FLOOR" if alpha <= lo * 1.001 else "CEIL" if alpha >= hi * 0.999 else "ok"
    return m.predict(Zva), {"alpha": float(alpha), "edge": edge,
                            "n_active": int(np.sum(m.coef_ != 0))}
