"""
Training loops.

Arm A / Arm B: trains with stratified k-fold cross-validation and reports
EVERY fold's score, not just the best split -- this is the direct fix for
the small-sample overfitting pattern documented in the literature review
(README.md methodological note).

Arm C: trains a single forecasting model on the pooled multi-country
training set (all countries contribute to training; the held-out test
sets are recent years per country, split by Africa/non-Africa).
"""

import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from model import build_classifiers, build_forecast_model


# ---------------------------------------------------------------------------
# Arm A / Arm B
# ---------------------------------------------------------------------------

def train_classifiers_with_cv(X_train, y_train, config: dict):
    """
    Returns:
        fitted_models: {name: model fitted on FULL X_train/y_train}
        cv_fold_scores: {name: [auc_fold_1, auc_fold_2, ...]}
    """
    models_dict = build_classifiers(config)
    cv_folds = config["data"]["cv_folds"]
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    cv_fold_scores = {name: [] for name in models_dict}

    for name, model in models_dict.items():
        print(f"[INFO] Cross-validating {name} ({cv_folds} folds)...")
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            fold_model = build_classifiers(config)[name]
            fold_model.fit(X_fold_train, y_fold_train)

            if hasattr(fold_model, "predict_proba"):
                probs = fold_model.predict_proba(X_fold_val)[:, 1]
                try:
                    auc = roc_auc_score(y_fold_val, probs)
                except ValueError:
                    auc = float("nan")
            else:
                auc = float("nan")

            cv_fold_scores[name].append(auc)
            print(f"  [Fold {fold_idx+1}/{cv_folds}] {name} AUC: {auc:.4f}")

        mean_auc = np.nanmean(cv_fold_scores[name])
        std_auc = np.nanstd(cv_fold_scores[name])
        print(f"[INFO] {name} CV summary: mean AUC = {mean_auc:.4f} (+/- {std_auc:.4f})")

    print("[INFO] Fitting final models on full training set...")
    fitted_models = {}
    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

    return fitted_models, cv_fold_scores


# ---------------------------------------------------------------------------
# Arm C
# ---------------------------------------------------------------------------

def train_forecast_model(train_df, config: dict):
    lag_cols = [f"lag_{l}" for l in range(1, config["data"]["lag_features"] + 1)]

    X_train = train_df[lag_cols]
    y_train = train_df["target"]

    model = build_forecast_model(config)

    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    print(f"[INFO] Training forecast model on {len(X_train)} rows...")
    model.fit(X_train, y_train)

    return model
