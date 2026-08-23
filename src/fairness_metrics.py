"""
Statistical significance tools shared across arms.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error


def bootstrap_ci_classification(y_true, y_pred, y_probs, n_bootstrap: int = 1000,
                                 ci: float = 0.95, seed: int = 42):
    """
    Bootstraps AUC on the test set to give a confidence interval, rather
    than reporting a single point estimate -- directly addresses the
    "single small-sample AUC" problem flagged in the literature review.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    if len(y_true) == 0:
        return {"note": "No test samples available for bootstrap CI."}

    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue  # skip resamples with only one class present
        try:
            aucs.append(roc_auc_score(y_true[idx], y_probs[idx]))
        except ValueError:
            continue

    if not aucs:
        return {"note": "Could not compute bootstrap CI (insufficient class diversity in resamples)."}

    aucs = np.array(aucs)
    lower = np.percentile(aucs, (1 - ci) / 2 * 100)
    upper = np.percentile(aucs, (1 + ci) / 2 * 100)

    return {
        "mean_auc": float(aucs.mean()),
        f"ci_{int(ci*100)}_lower": float(lower),
        f"ci_{int(ci*100)}_upper": float(upper),
        "n_bootstrap_valid": len(aucs),
    }


def bootstrap_ci_forecast_gap(africa_true, africa_pred, nonafrica_true, nonafrica_pred,
                               n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42):
    """
    Bootstraps the MAE gap between African and non-African forecast test sets.
    """
    rng = np.random.default_rng(seed)

    if len(africa_true) == 0 or len(nonafrica_true) == 0:
        return {"note": "Insufficient data in one or both groups to compute bootstrap CI."}

    gaps = []
    for _ in range(n_bootstrap):
        a_idx = rng.integers(0, len(africa_true), len(africa_true))
        n_idx = rng.integers(0, len(nonafrica_true), len(nonafrica_true))

        africa_mae = mean_absolute_error(africa_true[a_idx], africa_pred[a_idx])
        nonafrica_mae = mean_absolute_error(nonafrica_true[n_idx], nonafrica_pred[n_idx])
        gaps.append(africa_mae - nonafrica_mae)

    gaps = np.array(gaps)
    lower = np.percentile(gaps, (1 - ci) / 2 * 100)
    upper = np.percentile(gaps, (1 + ci) / 2 * 100)

    return {
        "mean_mae_gap": float(gaps.mean()),
        f"ci_{int(ci*100)}_lower": float(lower),
        f"ci_{int(ci*100)}_upper": float(upper),
        "n_bootstrap": n_bootstrap,
    }
