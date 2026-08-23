"""
Models for the HIV study.

Arm A / Arm B: classification models (shared builder, different configs)
Arm C: regression model for lagged time-series forecasting
"""


def build_classifiers(config: dict) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    models_dict = {}
    algos = config["model"]["algorithms"]

    if "random_forest" in algos:
        rf_cfg = config["model"]["random_forest"]
        models_dict["random_forest"] = RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            random_state=42,
        )

    if "xgboost" in algos:
        xgb_cfg = config["model"]["xgboost"]
        models_dict["xgboost"] = XGBClassifier(
            n_estimators=xgb_cfg["n_estimators"],
            max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"],
            random_state=42,
            eval_metric="logloss",
        )

    if "logistic_regression" in algos:
        lr_cfg = config["model"]["logistic_regression"]
        models_dict["logistic_regression"] = LogisticRegression(
            max_iter=lr_cfg["max_iter"],
            class_weight=lr_cfg["class_weight"],
        )

    return models_dict


def build_forecast_model(config: dict):
    from xgboost import XGBRegressor

    xgb_cfg = config["model"]["xgboost"]
    return XGBRegressor(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        random_state=42,
    )
