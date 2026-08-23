"""
Evaluation.

Arm A / Arm B: held-out test set metrics + honest CV fold spread (mean,
std, min, max across folds) -- reporting the spread is the point, not
just a single headline number.

Arm C: forecast accuracy on the held-out test set, split and compared
between African and non-African countries.
"""

import json
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, mean_squared_error,
)

from fairness_metrics import bootstrap_ci_classification, bootstrap_ci_forecast_gap


# ---------------------------------------------------------------------------
# Arm A / Arm B
# ---------------------------------------------------------------------------

def evaluate_classifiers(fitted_models: dict, cv_fold_scores: dict,
                          X_test, y_test, config: dict):
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    summary = {}

    for name, model in fitted_models.items():
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "test_accuracy": accuracy_score(y_test, preds),
            "test_precision": precision_score(y_test, preds, zero_division=0),
            "test_recall": recall_score(y_test, preds, zero_division=0),
            "test_f1": f1_score(y_test, preds, zero_division=0),
        }
        if probs is not None:
            try:
                metrics["test_auc"] = roc_auc_score(y_test, probs)
            except ValueError:
                metrics["test_auc"] = None

        metrics["confusion_matrix"] = confusion_matrix(y_test, preds).tolist()

        fold_scores = [s for s in cv_fold_scores.get(name, []) if not np.isnan(s)]
        if fold_scores:
            metrics["cv_auc_mean"] = float(np.mean(fold_scores))
            metrics["cv_auc_std"] = float(np.std(fold_scores))
            metrics["cv_auc_min"] = float(np.min(fold_scores))
            metrics["cv_auc_max"] = float(np.max(fold_scores))
            metrics["cv_fold_scores"] = fold_scores

        if hasattr(model, "feature_importances_"):
            importances = dict(zip(X_test.columns, model.feature_importances_.tolist()))
            metrics["feature_importances"] = dict(
                sorted(importances.items(), key=lambda x: -x[1])[:15]
            )

        if config["evaluation"]["bootstrap_ci"] and probs is not None:
            metrics["bootstrap_ci"] = bootstrap_ci_classification(
                y_test.values, preds, probs, n_bootstrap=config["evaluation"]["n_bootstrap"]
            )

        summary[name] = metrics
        print(f"[RESULT] {name}: test_AUC={metrics.get('test_auc')}, "
              f"CV mean AUC={metrics.get('cv_auc_mean'):.4f} "
              f"(std={metrics.get('cv_auc_std'):.4f})" if fold_scores else
              f"[RESULT] {name}: test_AUC={metrics.get('test_auc')}")

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[INFO] Full results saved to {results_dir}/summary.json")
    return summary


# ---------------------------------------------------------------------------
# Arm C
# ---------------------------------------------------------------------------

def evaluate_forecast(model, africa_test_df, nonafrica_test_df, config: dict):
    lag_cols = [f"lag_{l}" for l in range(1, config["data"]["lag_features"] + 1)]
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    def _score(df, label):
        if len(df) == 0:
            print(f"[WARN] No test rows for {label} -- check country name matching.")
            return {"n_samples": 0}
        X = df[lag_cols]
        y_true = df["target"].values
        y_pred = model.predict(X)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        nonzero_mask = y_true != 0
        mape = (np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]).mean() * 100
                if nonzero_mask.sum() > 0 else None)

        return {
            "n_samples": len(df),
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape) if mape is not None else None,
            "_y_true": y_true.tolist(),
            "_y_pred": y_pred.tolist(),
        }

    africa_metrics = _score(africa_test_df, "Africa")
    nonafrica_metrics = _score(nonafrica_test_df, "non-Africa")

    summary = {
        "africa": {k: v for k, v in africa_metrics.items() if not k.startswith("_")},
        "non_africa": {k: v for k, v in nonafrica_metrics.items() if not k.startswith("_")},
    }

    if africa_metrics.get("n_samples", 0) > 0 and nonafrica_metrics.get("n_samples", 0) > 0:
        summary["forecast_gap"] = {
            "mae_gap": africa_metrics["mae"] - nonafrica_metrics["mae"],
            "rmse_gap": africa_metrics["rmse"] - nonafrica_metrics["rmse"],
        }

        if config["evaluation"]["bootstrap_ci"]:
            summary["bootstrap_ci"] = bootstrap_ci_forecast_gap(
                np.array(africa_metrics["_y_true"]), np.array(africa_metrics["_y_pred"]),
                np.array(nonafrica_metrics["_y_true"]), np.array(nonafrica_metrics["_y_pred"]),
                n_bootstrap=config["evaluation"]["n_bootstrap"],
            )

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[RESULT] Africa MAE: {summary['africa'].get('mae')}")
    print(f"[RESULT] Non-Africa MAE: {summary['non_africa'].get('mae')}")
    if "forecast_gap" in summary:
        print(f"[RESULT] MAE gap (Africa - non-Africa): {summary['forecast_gap']['mae_gap']:.4f}")

    return summary
