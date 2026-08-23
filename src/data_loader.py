"""
Data loading for the HIV ML study.

Arm A: ACTG175 clinical trial data
Arm B: Nigeria symptom-based clinical data
Arm C: WHO/UNESCO country-year time series
"""

from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def _find_csv(data_dir: str) -> Path:
    data_dir = Path(data_dir)
    if not data_dir.exists() or not any(data_dir.iterdir()):
        return None
    csvs = list(data_dir.glob("*.csv"))
    if not csvs:
        return None
    return csvs[0]


# ---------------------------------------------------------------------------
# Arm A / Arm B share the same tabular-classification loading logic
# ---------------------------------------------------------------------------

def load_tabular_data(config: dict, arm_label: str):
    csv_path = _find_csv(config["data"]["csv_dir"])
    if csv_path is None:
        print(f"[WARN] No CSV found in {config['data']['csv_dir']}. "
              f"Download the {arm_label} dataset per README.md.")
        return None

    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {arm_label} data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[INFO] Columns found: {list(df.columns)}")

    target_col = config["data"]["target_column"]
    if target_col not in df.columns:
        print(f"[ACTION NEEDED] target_column '{target_col}' not found in data. "
              f"Available columns: {list(df.columns)}. "
              f"Update the config with the correct column name before proceeding.")
        return None

    return df


def prepare_tabular_features(df: pd.DataFrame, config: dict):
    target_col = config["data"]["target_column"]
    cat_cols = [c for c in config["data"]["categorical_columns"] if c in df.columns]
    num_cols = [c for c in config["data"]["numeric_columns"] if c in df.columns]

    missing_cat = set(config["data"]["categorical_columns"]) - set(cat_cols)
    missing_num = set(config["data"]["numeric_columns"]) - set(num_cols)
    if missing_cat or missing_num:
        print(f"[WARN] Some configured columns were not found in the data and will "
              f"be skipped: categorical={missing_cat}, numeric={missing_num}. "
              f"Check the config against the actual CSV headers.")

    X = df[cat_cols + num_cols].copy()
    y = df[target_col].copy()

    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    n_samples = len(df)
    small_sample_threshold = config["evaluation"].get("small_sample_warning_threshold")
    if small_sample_threshold and n_samples < small_sample_threshold:
        print(f"[METHODOLOGICAL WARNING] Sample size (n={n_samples}) is below the "
              f"configured threshold ({small_sample_threshold}). Small samples in this "
              f"literature have repeatedly produced inflated, likely overfit AUCs "
              f"(e.g. near-perfect scores from <150-sample cohorts in published "
              f"African HIV studies). Report results with wide confidence intervals "
              f"and avoid over-claiming from this arm alone.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["data"]["test_split"], random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Arm C: country-year time series
# ---------------------------------------------------------------------------

def load_who_timeseries(config: dict):
    csv_path = _find_csv(config["data"]["csv_dir"])
    if csv_path is None:
        print(f"[WARN] No CSV found in {config['data']['csv_dir']}. "
              f"Download the WHO/UNESCO dataset per README.md.")
        return None

    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded WHO time series: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[INFO] Columns found: {list(df.columns)}")

    required = [config["data"]["country_column"], config["data"]["year_column"],
                config["data"]["value_column"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ACTION NEEDED] Required columns not found: {missing}. "
              f"Available columns: {list(df.columns)}. "
              f"This dataset ships multiple CSVs -- make sure you picked the one "
              f"with country/year/value structure, and update the config.")
        return None

    return df


def load_african_countries_list(config: dict) -> set:
    list_path = Path(config["data"]["african_countries_list"])
    with open(list_path) as f:
        data = yaml.safe_load(f)
    return set(data["african_countries"])


def build_country_lag_features(df: pd.DataFrame, config: dict):
    """
    Builds lagged-feature rows per country for time-series forecasting:
    predicting value at year t from values at t-1, t-2, ..., t-lag.
    """
    country_col = config["data"]["country_column"]
    year_col = config["data"]["year_column"]
    value_col = config["data"]["value_column"]
    lag = config["data"]["lag_features"]
    min_years = config["data"]["min_years_per_country"]

    african_countries = load_african_countries_list(config)

    rows = []
    unmatched_africa_names = set(african_countries)

    for country, group in df.groupby(country_col):
        group = group.sort_values(year_col)
        if len(group) < min_years:
            continue

        is_africa = country in african_countries
        if is_africa:
            unmatched_africa_names.discard(country)

        values = group[value_col].values
        years = group[year_col].values

        for i in range(lag, len(values)):
            row = {
                "country": country,
                "year": years[i],
                "target": values[i],
                "is_africa": is_africa,
            }
            for l in range(1, lag + 1):
                row[f"lag_{l}"] = values[i - l]
            rows.append(row)

    if unmatched_africa_names:
        print(f"[WARN] {len(unmatched_africa_names)} countries in the Africa list "
              f"were not found in the dataset's country names (naming mismatch likely): "
              f"{sorted(unmatched_africa_names)[:10]}{'...' if len(unmatched_africa_names) > 10 else ''}")

    feature_df = pd.DataFrame(rows)
    print(f"[INFO] Built {len(feature_df)} lagged-feature rows across "
          f"{feature_df['country'].nunique() if len(feature_df) else 0} countries")
    return feature_df


def split_africa_nonafrica(feature_df: pd.DataFrame, config: dict):
    """
    Splits into train (mix of all countries, most years) and two held-out
    test sets: recent years for African countries vs. recent years for
    non-African countries, so forecast accuracy can be compared directly.
    """
    test_years_n = config["data"]["test_years"]

    train_rows, africa_test_rows, nonafrica_test_rows = [], [], []

    for country, group in feature_df.groupby("country"):
        group = group.sort_values("year")
        train_part = group.iloc[:-test_years_n] if len(group) > test_years_n else group.iloc[0:0]
        test_part = group.iloc[-test_years_n:] if len(group) > test_years_n else group

        train_rows.append(train_part)
        if group["is_africa"].iloc[0]:
            africa_test_rows.append(test_part)
        else:
            nonafrica_test_rows.append(test_part)

    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
    africa_test_df = pd.concat(africa_test_rows, ignore_index=True) if africa_test_rows else pd.DataFrame()
    nonafrica_test_df = pd.concat(nonafrica_test_rows, ignore_index=True) if nonafrica_test_rows else pd.DataFrame()

    return train_df, africa_test_df, nonafrica_test_df
