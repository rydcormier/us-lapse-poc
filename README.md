# US Life Lapse Prediction PoC (uslapseagent)

A **local-only, end-to-end** proof of concept for **policy lapse / surrender early warning** using a public US dataset.  

---

## Table of contents

- [US Life Lapse Prediction PoC (uslapseagent)](#us-life-lapse-prediction-poc-uslapseagent)
  - [Table of contents](#table-of-contents)
  - [Project purpose](#project-purpose)
  - [Data used](#data-used)
    - [Primary dataset: `uslapseagent` (US Whole Life policies)](#primary-dataset-uslapseagent-us-whole-life-policies)
  - [What this repo builds](#what-this-repo-builds)
    - [1) Raw export (from R)](#1-raw-export-from-r)
    - [2) Policy-quarter modeling table (DuckDB + pandas-friendly output)](#2-policy-quarter-modeling-table-duckdb--pandas-friendly-output)
      - [Labels produced](#labels-produced)
      - [Feature contract (default)](#feature-contract-default)
  - [Models included](#models-included)
    - [1) Logistic Regression baseline (scikit-learn)](#1-logistic-regression-baseline-scikit-learn)
    - [2) PyTorch TabularNet (categorical embeddings + MLP)](#2-pytorch-tabularnet-categorical-embeddings--mlp)
  - [Metrics reported](#metrics-reported)
  - [Model Drivers](#model-drivers)
    - [Logistic Regression — Top Positive Coefficients](#logistic-regression--top-positive-coefficients)
    - [PyTorch TabularNet — Permutation Importance (PR-AUC drop)](#pytorch-tabularnet--permutation-importance-pr-auc-drop)
  - [Build \& run dependencies](#build--run-dependencies)
    - [Prerequisites](#prerequisites)
    - [1) Create environment and install](#1-create-environment-and-install)
    - [2) Export the dataset using R](#2-export-the-dataset-using-r)
    - [3) Build the policy-quarter panel](#3-build-the-policy-quarter-panel)
    - [4) Train models](#4-train-models)
    - [5) Run the API](#5-run-the-api)
  - [How to use](#how-to-use)
    - [A) Call the FastAPI endpoint](#a-call-the-fastapi-endpoint)
    - [B) Offline scoring in Python (using saved artifacts)](#b-offline-scoring-in-python-using-saved-artifacts)
    - [C) Makefile shortcuts](#c-makefile-shortcuts)
  - [Directory contents](#directory-contents)
  - [Notes on leakage and time splits](#notes-on-leakage-and-time-splits)
    - [DJIA default behavior](#djia-default-behavior)
    - [Time splits](#time-splits)
  - [Troubleshooting](#troubleshooting)
    - [CASdatasets installation issues](#casdatasets-installation-issues)
    - [Export to parquet not working](#export-to-parquet-not-working)
    - [Duration looks like decimals (e.g., 0.01)](#duration-looks-like-decimals-eg-001)
  - [Roadmap](#roadmap)
  - [Further Documentation](#further-documentation)
  - [Citation / licensing](#citation--licensing)

---

## Project purpose

Insurance lapse/surrender risk is a classic **retention / persistency** problem:

- Business teams have limited outreach capacity (calls/emails/letters).
- A model ranks in-force policies by risk of lapse/surrender.
- Scores are converted into a **budgeted contact list** and (optionally) expected-value decisioning.

This repo demonstrates an end-to-end workflow:

1. **Obtain public data** (no proprietary inputs)
2. **Engineer a leakage-safe, time-aware modeling table** (policy-quarter panel)
3. Train:
   - a strong **baseline** (logistic regression)
   - a **PyTorch tabular deep learning model** (categorical embeddings + MLP)
4. **Evaluate** using imbalanced-friendly metrics and lift
5. **Deploy** locally via **FastAPI** (`/predict` endpoint)

---

## Data used

### Primary dataset: `uslapseagent` (US Whole Life policies)

This project uses the public `uslapseagent` dataset from the **CASdatasets** R package.

High-level characteristics (from CASdatasets documentation):

- US Whole Life policies sold through a tied-agent channel
- Observation window roughly **1995–2008**
- Includes termination cause flags: surrender vs death vs other, plus policy attributes

**Important**: This repository does **not** include the dataset itself. You export it locally using R.

---

## What this repo builds

### 1) Raw export (from R)

`scripts/export_uslapseagent.R` writes:

- `data/raw/uslapseagent.csv` (default) or `.parquet` if you choose and have `arrow`

It also adds a stable `policy_id` if one isn’t present.

### 2) Policy-quarter modeling table (DuckDB + pandas-friendly output)

`python -m lapse_poc.data.build_features` builds a discrete-time hazard / early-warning panel:

- Each row is a **(policy_id, quarter-of-tenure)** snapshot.
- Label indicates whether an event occurs **in the next `horizon` quarters**.

Output:

- `data/processed/policy_quarter.parquet`
- `data/processed/manifest.json` (build metadata: split rates, duration scaling heuristic, etc.)

#### Labels produced

- `y_surrender_next` — surrender within next `horizon` quarters  
- `y_death_next` — death within next `horizon` quarters  
- `y_other_next` — other termination within next `horizon` quarters  
- `y_any_next` — any termination within next `horizon` quarters  

Default training target is: **`y_surrender_next`**.

#### Feature contract (default)

Categorical:

- `gender`
- `premium_frequency`
- `risk_state`
- `underwriting_age`
- `living_place`
- `acc_death_rider`

Numeric:

- `tenure_qtr`
- `annual_premium`

Optional numeric (disabled by default):

- `djia` (see leakage notes below)

---

## Models included

### 1) Logistic Regression baseline (scikit-learn)

- One-hot encoding for categoricals
- Standard scaling for numeric features
- Class imbalance handled via `class_weight="balanced"`

Command:

```bash
python -m lapse_poc.models.train_logreg \
  --data data/processed/policy_quarter.parquet \
  --out artifacts/logreg \
  --target y_surrender_next
```

Artifacts:

- `artifacts/logreg/model.joblib`
- `artifacts/logreg/metrics.json`

### 2) PyTorch TabularNet (categorical embeddings + MLP)

- Categorical embeddings (learned)
- Stadardized numeric features
- `BCEWithLogitsLoss(pos_weights=...)` for class imbalance
- Early stopping on validation PR-AUC

Command:

```bash
python -m lapse_poc.models.train_torch \
    --data data/processed/policy_quarter.parquet \
    --out artifacts/torch \
    --target y_surrender_next
```

Artifacts:

- `artifacts/torch/model.pt`
- `artifacts/torch/preproccessor.joblib`
- `artifacts/torch/metrics.json`
- `artifacts/torch/test_preds.parquet`

## Metrics reported

For each split (train/val/test):

- **PR-AUC** (primary for imbalanced classification)
- **ROC-AUC** (secondary)
- **Brier score** (probability calibration quality)
- **Precision@K** (defaults to K = 10% fo the split)
- **Lift@Decile** (event rate in top 10% scored) / (overall event rate)

See:

- `src/lapse_poc/eval/metrics`

## Model Drivers

### Logistic Regression — Top Positive Coefficients
<!-- BEGIN:LOGREG_TOP10 -->

| feature | coef |
| --- | --- |
| premium_frequency_InfraAnnual | 0.260731 |
| annual_premium | 0.193399 |
| underwriting_age_Middle | 0.132068 |
| acc_death_rider_NoRider | 0.098208 |
| living_place_EastCoast | 0.0524 |
| underwriting_age_Young | 0.020385 |
| risk_state_NonSmoker | -2.6e-05 |
| living_place_Other | -0.008679 |
| gender_Male | -0.011316 |
| premium_frequency_Annual | -0.065264 |

<!-- END:LOGREG_TOP10 -->

### PyTorch TabularNet — Permutation Importance (PR-AUC drop)
<!-- BEGIN:TABNET_PERMIMP_TOP10 -->

| feature | pr_auc_drop_mean | pr_auc_drop_std |
| --- | --- | --- |
| annual_premium | 0.003656 | 0.000387 |
| premium_frequency | 0.002579 | 0.000261 |
| underwriting_age | 0.00114 | 0.00022 |
| acc_death_rider | 0.000402 | 7.7e-05 |
| risk_state | 0.00017 | 0.000103 |
| gender | 6.8e-05 | 0.000118 |
| tenure_qtr | -3.9e-05 | 0.000218 |
| living_place | -0.000178 | 8.7e-05 |

<!-- END:TABNET_PERMIMP_TOP10 -->

## Build & run dependencies

### Prerequisites

- Python **3.10+**
- R **4.x** (recommended)
- (Optional) `arrow` in R for parquet export
- Works CPU-only; CUDA is optional if available

### 1) Create environment and install

```bash
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows powershell
# .venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -e ".[dev]"
```

### 2) Export the dataset using R

Install CASdatasets (choose one approach):

**Option A**: Install from CAS/UQAM repo (source)

```r
install.packages(
  "CASdatasets",
  repos="https://cas.uqam.ca/pub/",
  type="source"
)
```

**Option B**: Install from GitHub

```r
install.packages(c("ps", "processx", "remotes"))
remotes::install_github("dutangc/CASdatasets")

```

Export to CSV:

```bash
Rscript scripts/export_uslapseagent.R --out data/raw/uslapseagent.csv
```

If `Rscript` is not on PATH (common on Windows), use a full path, e.g.:

```powershell
& "C:\\Program Files\\R\\R-4.3.1\\bin\\Rscript.exe" .\\scripts\\export_uslapseagent.R --out data\\raw\\uslapseagent.csv
```

Or via the Makefile:

```bash
make data RSCRIPT="C:\\Program Files\\R\\R-4.3.1\\bin\\Rscript.exe"
```

### 3) Build the policy-quarter panel

Default horizon = 1 quarter; DJIA excluded by default:

```bash
python -m lapse_poc.data.build_features \
    --raw data/raw/uslapseagent.csv \
    --out data/processed/policy_quarter.parquet \
    --horizon 1
```

If you *explicitly* want DJIA included:

```bash
python -m lapse_poc.data.build_features \
    --raw data/raw/uslapseagent.csv \
    --out data/processed/policy_quarter.parquet \
    --horizon 1 \
    --include-djiq
```

### 4) Train models

Logistic regression:

```bash
python -m lapse_poc.models.train_logreg \
    --data data/processed/policy_quarter.parquet \
    --out artifacts/logreg
```

PyTorch

```bash
python -m lapse_poc.models.train_torch \
    --data data/processed/policy_quarter.parquet \
    --out artifacts/torch
```

### 5) Run the API

By default, the API loads PyTorch artifacts from `artifacts/torch`.

```bash
uvicorn lapse_poc.api.app:app --reload --port 8000
```

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

## How to use

### A) Call the FastAPI endpoint

Example request:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_qtr": 12,
    "gender": "Male",
    "premium_frequency": "Annual",
    "risk_state": "NonSmoker",
    "underwriting_age": "Middle",
    "living_place": "Other",
    "acc_death_rider": "NoRider",
    "annual_premium": 0.5
  }'
```

Response:

```json
{"lapse_prob": 0.1234}
```

> Note: the categorical string values must match what exists in the training data (unknown categories map to an `__UNK__` token internally for the PyTorch model).

### B) Offline scoring in Python (using saved artifacts)

```python
import joblib
import pandas as pd
import torch

from lapse_poc.data.preprocessing import TabularPreprocessor
from lapse_poc.models.tabular import TabularNet

artifact_dir = "artifacts/torch"

pre = TabularPreprocessor.from_dict(joblib.load(f"{artifact_dir}/preprocessor.joblib"))
model = TabularNet(cat_cardinalities=pre.cat_cardinalities(), n_num=len(pre.num_cols))
model.load_state_dict(torch.load(f"{artifact_dir}/model.pt", map_location="cpu"))
model.eval()

df = pd.DataFrame([{
    "tenure_qtr": 12,
    "annual_premium": 0.5,
    "gender": "Male",
    "premium_frequency": "Annual",
    "risk_state": "NonSmoker",
    "underwriting_age": "Middle",
    "living_place": "Other",
    "acc_death_rider": "NoRider",
}])

x_cat, x_num = pre.transform(df)
with torch.no_grad():
    prob = torch.sigmoid(model(torch.from_numpy(x_cat), torch.from_numpy(x_num))).item()

print(prob)
```

### C) Makefile shortcuts

```bash
make install
make data
make features
make train_logreg
make train_torch
make api
make test
make lint
make fmt
```

## Directory contents

```graphql
us-lapse-poc/
  README.md                           # You are here
  pyproject.toml                      # Python deps + tooling config (ruff/black/pytest)
  Makefile                            # Convenience commands
  .vscode/                            # VS Code recommended settings/launch configs

  scripts/
    export_uslapseagent.R             # Exports uslapseagent from CASdatasets -> data/raw/

  src/lapse_poc/
    settings.py                       # Feature contract + defaults
    utils.py                          # Seed setting utility

    analysis/
      embed_drivers_into_readme.py    # Embed top-10 drivers for each model
      logreg_drivers.py               # Extract and display top positive coefficients
      torch_permutation_importance.py # permutation importance (model-agnostic)

    data/
      build_features.py               # DuckDB builder for policy-quarter panel + manifest
      preprocessing.py                # Simple tabular preprocessor (cats->ids, nums->zscore)
      torch_dataset.py                # PyTorch Dataset wrapper
    
    docs/
      application.md                  # How-to apply this to production data
      CASdatasets-manual.pdf          # Information on where dataset comes from
      dataset.md                      # Dataset description
      metrics.md                      # Metrics used for model performance
      results.md                      # results of initial training

    models/
      train_logreg.py                 # sklearn baseline training + metrics
      tabular.py                      # PyTorch TabularNet definition (embeddings + MLP)
      train_torch.py                  # PyTorch training loop + artifact saving

    eval/
      metrics.py                      # PR-AUC, ROC-AUC, Brier, precision@k, lift@decile

    api/
      app.py                          # FastAPI app that loads artifacts + serves /predict

  data/
    raw/                              # (gitignored) raw exported dataset
    processed/                        # (gitignored) policy_quarter.parquet + manifest.json
    external/                         # placeholder for future public macro data joins

  artifacts/                          # (gitignored) trained models + metrics

  tests/
    test_build_features_smoke.py
    test_tabular_forward.py
```

## Notes on leakage and time splits

### DJIA default behavior

`uslapseagent` contains a `DJIA` field described as "last observed quarterly variation." "Last observed" style fields can accidentally leak termination timing depending on how they were constructed, so:

- This scaffold drops `DJIA` by default.
- You can include it with `--include-djia` for experimentation.

Recommended portfolio approach:

- Keep DJIA excluded initially
- Later: replace with your own quarter-level US macro table from public sources and join by date

### Time splits

`build_features.py` assigns splits by `as_of_date`:

- train: `as_of_date <= 2006-12-31`
- val: `2006-12-31 < as_of_date <= 2007-12-31`
- test: `as_of_date > 2007-12-31`

These are parameters in `build_policy_quarter_panel()` if you want to adjust them.

## Troubleshooting

### CASdatasets installation issues

- If installing from source fails, ensure you have compilers set up (Rtools on Windows, Xcode CLT on macOS, build-essential on Linux).
- If you see: `there is no package called 'ps'` while installing from GitHub, run:

  ```r
  install.packages(c("ps", "processx", "remotes"))
  remotes::install_github("dutangc/CASdatasets")
  ```

### Export to parquet not working

Parquet export requires the R package `arrow`. If `arrow` is not installed, export as CSV (default).

### Duration looks like decimals (e.g., 0.01)

The feature builder detects whether duration appears "scaled" and applies a heuristic:

- if p99(duration) < 5, it multiplies duration by 100 and rounds
- otherwise it rounds duration directly

You can inspect the applied scaling factor in:

- `data/processed/manifest.json` → `"duration_scale"`

## Roadmap

High-impact upgrades after the first successful run:

1. **Calibration**: temperature scaling / isotonic; add reliability plots
2. **Decisioning**: budgeted outreach optimizer + expected value curves
3. **Monitoring**: drift checks (PSI) for key features and score distributions
4. **Competing risks**: multi-head PyTorch model for surrender vs death vs other
5. **Public macro enrichment**: join quarter-level FRED/macro series by `as_of_date`

## Further Documentation

Further documentation found in [docs folder](./docs).

## Citation / licensing

- See CASdatasets documentation for recommended citation and licensing.
- You can retrieve the package citation directly in R:

  ```r
  citation("CASdatasets")
  ```

This project is intended as a proof of concept and does not include any proprietary insurer data.
