# ML in HIV Care: Three-Arm Study Across Clinical, Symptom-Based, and Epidemiological Data

## Research question
Where and how does machine learning add value across the HIV care and
surveillance pipeline — clinical trial outcome prediction, symptom-based
risk stratification in an African clinical setting, and country-level
epidemiological forecasting — and where does existing published work
overstate performance due to small samples or weak validation?

All datasets below are **free, direct-download, no registration/approval
required** — confirmed before building this pipeline. (DHS and PHIA survey
data were investigated and explicitly ruled out: both require a signed
Terms of Use and account approval, which breaks the "free and open" goal
of this project.)

## Three study arms

### Arm A: ACTG175 — clinical trial outcome prediction (standard benchmark)

| Dataset | Role | Source | Access |
|---|---|---|---|
| ACTG175 (AIDS Clinical Trial Group study 175) | Standard/benchmark clinical dataset | Kaggle | Free Kaggle account, instant download |

**Download link:** https://www.kaggle.com/datasets/aadarshvelu/aids-virus-infection-prediction
(also mirrored under other Kaggle listings — search "ACTG175" or
"AIDS Clinical Trials" if this specific link changes)

Place the CSV in:
```
data/actg175/
```

This is a well-known, heavily benchmarked dataset (US-based clinical
trial, ~2,100 patients) with lab values (CD4, CD8 counts), treatment arm,
and demographic variables. Target: predicting whether a patient's disease
status worsens by a defined endpoint (censoring indicator in the original
trial). Used here as a **rigor benchmark** — the goal is a properly
cross-validated, honestly reported model, in contrast to several published
African-cohort studies that report suspiciously high AUCs from very small
positive-class samples (see README section "Methodological note" below).

### Arm B: Nigeria symptom-based clinical dataset (African context, standalone)

| Dataset | Role | Source | Access |
|---|---|---|---|
| HIV Infection Rate Based on Endogenous Variables (Osun State, Nigeria) | Standalone African clinical study | Mendeley Data | Direct download, no account needed |

**Download link:** https://data.mendeley.com/datasets/37syp7hj8n/1

Place the file in:
```
data/nigeria_clinical/
```

Important: this dataset contains **138 already-HIV-positive patients**
with symptom presence/absence (fever, diarrhea, cough, weight loss, etc.)
— there is no HIV-negative control group, so this cannot be used to build
an infection *diagnosis* classifier. Instead, this arm predicts a clinical
severity/status outcome (**Medical Fitness**, coded 0/1 in the dataset;
~20% positive) from the symptom profile — a legitimate standalone
predictive task, and NOT directly comparable to Arm A's features (no
shared columns — this was checked before building the pipeline, so the
two arms are run and reported separately, not as a train/test transfer
pair).

### Arm C: WHO/UNESCO country-level HIV time series (epidemiological forecasting)

| Dataset | Role | Source | Access |
|---|---|---|---|
| HIV/AIDS Dataset (country-year prevalence/incidence, WHO & UNESCO) | Cross-country forecasting comparison | Kaggle | Free Kaggle account, instant download |

**Download link:** https://www.kaggle.com/datasets/imdevskp/hiv-aids-dataset

Place the CSV(s) in:
```
data/who_timeseries/
```

This arm trains a time-series forecasting model (e.g., gradient-boosted
trees on lagged features, or a simple LSTM) on country-year HIV prevalence
data and compares forecast accuracy **between African and non-African
countries** — this is the arm that most directly parallels the
generalizability-gap framing from the earlier cervical cancer study, but
applied at the epidemiological/surveillance level rather than clinical
imaging.

## Methodological note: why Arm A is framed around rigor, not just accuracy

Several published African HIV viral-suppression prediction studies report
wildly inconsistent AUCs — from ~0.65 up to a near-perfect 0.999 — and the
near-perfect results tend to come from studies with very small positive-class
sample sizes (a recurring, likely overfitting-driven pattern in this
literature). Arm A is deliberately designed to demonstrate honest,
properly cross-validated benchmarking (stratified k-fold, reported
confidence intervals, no single-split "best case" reporting) as a
methodological contribution in its own right, not just another prediction
model.

## Pipeline stages (src/)
1. `data_loader.py` — loads all three datasets, handles the very different
   schemas per arm
2. `model.py` — classifiers for Arm A/B (Random Forest, XGBoost, Logistic
   Regression), forecasting model for Arm C
3. `train.py` — training/cross-validation loops for all three arms
4. `evaluate.py` — arm-specific evaluation, including Arm C's
   African-vs-non-African forecast accuracy comparison
5. `fairness_metrics.py` — bootstrap CIs, used in Arm C's cross-country gap
6. `run_pipeline.py` — orchestrates all three arms; run with `--arm A`,
   `--arm B`, or `--arm C`

## How to run once data is downloaded
```bash
pip install -r requirements.txt

python src/run_pipeline.py --arm A --config configs/arm_a_actg175.yaml
python src/run_pipeline.py --arm B --config configs/arm_b_nigeria.yaml
python src/run_pipeline.py --arm C --config configs/arm_c_who_forecast.yaml
```

## Before running Arm A or B: confirm actual column names
Both configs have placeholder column names. Kaggle dataset column naming
can change between uploads/versions. Open each CSV once after downloading,
confirm the real column names against the config, and update
`configs/arm_a_actg175.yaml` / `configs/arm_b_nigeria.yaml` accordingly —
the pipeline will print the actual columns found and stop with a clear
error if there's a mismatch, rather than silently running on wrong data.

## What "success" looks like
- **Arm A**: a properly validated, honestly reported AUC (expect somewhere
  in the 0.65-0.85 range based on the wider literature) — not a
  suspiciously perfect score.
- **Arm B**: identifying which symptom combinations most strongly predict
  the medical fitness outcome — a directly interpretable clinical finding.
- **Arm C**: a quantified forecast-accuracy gap (or lack thereof) between
  African and non-African countries, with bootstrap confidence intervals.
