# Applying the Lapse/Surrender Models to Production Data (and Extending to Term Policies)

This document explains how to take the PoC models (Logistic Regression + PyTorch TabularNet) built on `uslapseagent` and apply the same architecture to **production Life & Health data**, including how to extend the solution to support **term life policies**.

> Scope: local-first “production-like” approach—repeatable batch scoring, feature contracts, validations, monitoring, and model governance artifacts.

---

## 1) What the PoC models actually do

### Prediction target
The PoC is framed as a **discrete-time hazard** / early-warning classification problem:

> For each in-force policy at an “as-of” date (typically quarter-end), predict whether it will **surrender/lapse within the next horizon** (default: next quarter).

### Modeling table
The core modeling unit is a **policy-period panel**:

- Key identifiers: `policy_id`, `as_of_date`
- Time feature: `tenure_qtr`
- Features: policyholder/policy fields + optionally macro/behavioral signals
- Labels: `y_surrender_next`, `y_any_next`, etc. (not available for production scoring at inference time)

### Outputs
- **Risk score**: `p(lapse/surrender within horizon)` per policy and as-of date
- Used for: ranked outreach lists, portfolio monitoring, QA controls, financial controls.

---

## 2) Production data requirements (what you need upstream)

### Minimal production inputs (policy snapshot)
For each policy on each run date (`as_of_date`), you need a “snapshot” dataset that contains:

- **Policy identifiers**: `policy_id`, `product_code`, `issue_date`, `status`, `admin_system`
- **Policyholder attributes**: age or DOB, gender (if permitted), state, smoker class (life), underwriting class
- **Billing/premium**: premium amount, modal premium, premium mode (annual/monthly/etc.), next due date, paid-to-date, delinquency indicators
- **Riders / coverages**: rider flags, face amount/benefit, plan options
- **Term-specific** fields (for extension): term length, renewal/level period, conversion options, reprice dates

### Optional but high-signal production inputs (recommended)
These are usually the strongest lapse predictors:
- **Payment behavior**: missed payments, days past due buckets, reversals, reinstatements
- **Customer interactions**: calls, complaints, portal activity, change requests
- **Agent/channel**: distribution channel, agent id, commission changes (if allowed)
- **Macro**: unemployment, rates, inflation, market returns aligned by time

---

## 3) Production pipeline design (SQL-first + feature contract)

### Step A — Build a policy “feature view”
Create a reproducible feature view using SQL (DuckDB/Snowflake/SQL Server) with:

- one row per `(policy_id, as_of_date)`
- deterministic joins and aggregations
- strict schema contract (types + allowed nulls + categorical domains)

**Recommended pattern**
- `dim_policy` (static-ish)
- `fact_policy_period` (time-varying snapshot)
- `fact_payments` (events)
- `fact_service_events` (events)
- `dim_product` (product metadata)
- `dim_calendar` (quarter-end dates)

### Step B — Validate the inputs (quality gates)
Before scoring, validate:
- required columns exist
- value ranges (e.g., premium >= 0, tenure >= 0)
- categorical domains (unknown values mapped to `__UNK__`)
- duplicates and join explosion checks
- time consistency (issue_date <= as_of_date)

### Step C — Feature engineering consistency
Production must match training transformations exactly:
- same categorical mappings (or unknown bucket)
- same numeric standardization (mean/std)
- same missing handling defaults

For PoC:
- Logistic regression pipeline stores its preprocessing within `model.joblib`
- TabularNet stores preprocessing in `preprocessor.joblib`

---

## 4) Scoring in “production mode” (batch)

### Recommended batch scoring workflow
1. Extract policy snapshot for run date (e.g., month/quarter-end)
2. Validate schema/quality
3. Apply model artifact:
   - LogReg: `model.joblib` pipeline
   - TabularNet: load `preprocessor.joblib` + `model.pt`
4. Write scored outputs:
   - `policy_id`, `as_of_date`, `score`, `model_version`, `feature_version`
5. Produce operational outputs:
   - Top K list for outreach
   - Score distribution summaries
   - Drift checks vs prior run

### Output schema (example)
| field | description |
|---|---|
| policy_id | stable identifier |
| as_of_date | scoring date |
| lapse_risk | predicted probability (0–1) |
| model_name | e.g., `tabularnet_v1` |
| model_version | git hash or artifact version |
| feature_version | feature contract version |
| scored_at | timestamp |

---

## 5) Decisioning: turning scores into actions

A production lapse score is most valuable when paired with an outreach policy:

### Simple rule-based decisioning
- Contact top `K` policies by risk score
- OR contact those above a threshold `t`

### Cost-sensitive decisioning (better)
Define:
- contact_cost (call/mail/SMS)
- expected margin saved if retained
- save_rate (probability outreach prevents lapse)

Compute expected value (EV) per policy:
- `EV = (margin_saved * save_rate * score) - contact_cost`

Then select the top policies by EV under a budget constraint.

---

## 6) Monitoring & controls (must-have for Ops Insights)

### Data drift
Track feature distribution changes over time (monthly/quarterly):
- PSI for top features (tenure, premium, delinquency)
- category mix shifts (product mix, channel mix)

### Score drift
Monitor score distributions:
- mean/median score
- top-decile average score
- percent above threshold

### Performance monitoring (delayed labels)
Once you observe actual lapses after the horizon:
- PR-AUC / lift recalculated on completed cohorts
- calibration checks (Brier, reliability)

### Operational controls
- run completeness (expected count vs delivered count)
- exception counts and unresolved issues
- audit trail: model/version, input snapshot version, logs

---

## 7) Extending the solution to Term Policies

### Why term needs special handling
Term lapse behavior differs due to:
- level term period and renewal cliffs
- repricing events
- conversion windows
- shorter product durations and different premium dynamics

### Two recommended approaches

#### Approach A (fastest): single unified model + product features
Train one model for all products and include:
- `product_type` (UL/WL/Term)
- `term_length`, `level_period_remaining`, `renewal_date`, `reprice_date`
- conversion eligibility flags
- product-specific premium features

Pros:
- One scoring pipeline
- Shares learnings across products
Cons:
- Can underfit product-specific patterns unless data is rich

#### Approach B (stronger): separate models per product family
- Train a Term-specific model with term-specific features
- Keep a UL/WL model for permanent products
- Optional: a meta-model to choose model routing by product

Pros:
- Best fit to product-specific drivers
Cons:
- More governance + monitoring effort

### Term-specific feature ideas (high-signal)
- `months_to_renewal` / `quarters_to_renewal`
- `months_to_level_period_end`
- `conversion_window_open` (bool)
- `reprice_in_next_horizon` (bool)
- `rate_increase_percent` (if known)
- premium affordability proxies (premium / income if available, or changes in premium)

### Labels for term
Use the same definition:
- `y_lapse_next_horizon = 1` if policy terminates due to lapse/surrender within next horizon
Ensure you exclude “death claim” and non-lapse terminations as needed depending on business definition.

---

## 8) Training refresh for production + term addition

### Step 1 — Build a training dataset from production history
To train a production-grade model you need:
- historical as-of snapshots
- termination outcome tables (lapse vs other)
- aligned label horizon

Key point: **avoid leakage**
- features must be known at `as_of_date`
- do not include “last observed” fields that encode termination timing

### Step 2 — Splits
Use time-based splits:
- Train: older years
- Val: recent year
- Test: most recent holdout period

### Step 3 — Refit preprocessing
For TabularNet:
- rebuild categorical maps and numeric scalers from train split
- version and store them with artifacts

### Step 4 — Compare against baseline
Always keep:
- Logistic regression baseline for interpretability and calibration
- TabularNet (or tree model) for lift/ranking

---

## 9) Suggested “productionization” deliverables (portfolio-ready)

- **Feature contract** (`schema.json`)
- **Batch scoring script** (`score_batch.py`)
- **Model artifact registry** folder structure (`artifacts/model_name/version/`)
- **Monitoring report** (`reports/monitoring.md`)
- **Runbook** for Ops/Insights:
  - what to do when validations fail
  - what thresholds trigger escalation
- **Model card**:
  - intended use
  - limitations and drift risks
  - fairness considerations (where applicable)

---

## 10) Minimal checklist (what “done” looks like)

✅ Input snapshot extraction + deterministic joins  
✅ Validation gates + exception reporting  
✅ Batch scoring outputs with versions and audit trail  
✅ Lift/top-K reporting for outreach use-case  
✅ Drift monitoring + score drift monitoring  
✅ Term extension implemented via unified model or separate model  
✅ Documentation: runbook + model card + feature contract

---

## Appendix: Practical sequencing

1) Production scoring for permanent products using current feature contract  
2) Add payment behavior features (highest ROI)  
3) Add term policy features and route term vs permanent  
4) Retrain models on production historical snapshots  
5) Add calibration + EV decisioning for outreach  
6) Add monitoring + alerting thresholds