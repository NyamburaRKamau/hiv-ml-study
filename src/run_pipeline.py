"""
Entry point. Run with:
    python src/run_pipeline.py --arm A --config configs/arm_a_actg175.yaml
    python src/run_pipeline.py --arm B --config configs/arm_b_nigeria.yaml
    python src/run_pipeline.py --arm C --config configs/arm_c_who_forecast.yaml
"""

import argparse
import yaml

from data_loader import (
    load_tabular_data, prepare_tabular_features,
    load_who_timeseries, build_country_lag_features, split_africa_nonafrica,
)
from train import train_classifiers_with_cv, train_forecast_model
from evaluate import evaluate_classifiers, evaluate_forecast


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_tabular_arm(config: dict, arm_label: str):
    print(f"[INFO] Running {arm_label}: {config['experiment_name']}")

    print("[STAGE 1/3] Loading data...")
    df = load_tabular_data(config, arm_label)
    if df is None:
        print(f"\n[STOP] Could not load data for {arm_label}. See warnings above "
              f"and README.md for download instructions.")
        return

    if not config["data"]["categorical_columns"] and not config["data"]["numeric_columns"]:
        print("\n[STOP] No feature columns configured. Fill in categorical_columns "
              "and/or numeric_columns in the config using the column names printed above.")
        return

    X_train, X_test, y_train, y_test = prepare_tabular_features(df, config)

    print("[STAGE 2/3] Training with stratified cross-validation...")
    fitted_models, cv_fold_scores = train_classifiers_with_cv(X_train, y_train, config)

    print("[STAGE 3/3] Evaluating on held-out test set...")
    summary = evaluate_classifiers(fitted_models, cv_fold_scores, X_test, y_test, config)

    print(f"\n[DONE] {arm_label} complete. See {config['output']['results_dir']}/ for output.")
    return summary


def run_arm_c(config: dict):
    print(f"[INFO] Running Arm C: {config['experiment_name']}")

    print("[STAGE 1/4] Loading WHO/UNESCO time series...")
    df = load_who_timeseries(config)
    if df is None:
        print("\n[STOP] Could not load Arm C data. See warnings above and README.md.")
        return

    print("[STAGE 2/4] Building lagged features and Africa/non-Africa split...")
    feature_df = build_country_lag_features(df, config)
    if len(feature_df) == 0:
        print("\n[STOP] No usable rows after building lag features. Check "
              "min_years_per_country and lag_features settings in the config "
              "against how much history your data actually has.")
        return

    train_df, africa_test_df, nonafrica_test_df = split_africa_nonafrica(feature_df, config)
    if len(train_df) == 0:
        print("\n[STOP] No training rows produced. Check test_years isn't larger "
              "than the available history for most countries.")
        return

    print("[STAGE 3/4] Training forecast model...")
    model = train_forecast_model(train_df, config)

    print("[STAGE 4/4] Evaluating Africa vs non-Africa forecast accuracy...")
    summary = evaluate_forecast(model, africa_test_df, nonafrica_test_df, config)

    print(f"\n[DONE] Arm C complete. See {config['output']['results_dir']}/ for output.")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, required=True, choices=["A", "B", "C"])
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.arm == "A":
        run_tabular_arm(config, "Arm A (ACTG175)")
    elif args.arm == "B":
        run_tabular_arm(config, "Arm B (Nigeria)")
    else:
        run_arm_c(config)


if __name__ == "__main__":
    main()
