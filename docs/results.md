# Model Results — Lapse/Surrender Prediction (y_surrender_next)

This document summarizes model performance for the **US Whole Life surrender/lapse early warning** PoC.  
Target: **`y_surrender_next`** (predict surrender within the next horizon; default horizon = 1 quarter).

Both models were evaluated on **time-based** train/validation/test splits to reflect realistic forward-looking performance.

---

## Models evaluated

1. **Logistic Regression (baseline)**  
   - One-hot encoding for categorical features  
   - Standardization for numeric features  
   - Class imbalance handled via `class_weight="balanced"`

2. **PyTorch TabularNet**  
   - Learned embeddings for categorical features  
   - MLP backbone  
   - Class imbalance handled via `BCEWithLogitsLoss(pos_weight=neg/pos)`  
   - Early stopping on validation PR-AUC  
   - Trained on CPU in this run

Feature set used in both models:

- Categorical: `gender`, `premium_frequency`, `risk_state`, `underwriting_age`, `living_place`, `acc_death_rider`
- Numeric: `tenure_qtr`, `annual_premium`

---

## Why these metrics

Because surrender events are **rare**, we focus on metrics that reflect:

- **Ranking quality under imbalance**: PR-AUC
- **General discrimination**: ROC-AUC
- **Operational targeting value**: Precision@10% and Lift@Decile
- **Probability quality (for EV decisioning)**: Brier score

---

## Base rates (event prevalence)

From your run:

- **Train base rate:** 0.01208 (≈ 1.21%)
- **Val base rate:** 0.00813 (≈ 0.81%)
- **Test base rate:** 0.01160 (≈ 1.16%)

These low base rates are typical for lapse/surrender prediction tasks and make PR-AUC and lift particularly important.

---

## Results summary tables

### PyTorch TabularNet

| Split | PR-AUC  | ROC-AUC |  Brier  | Precision@10% | Base Rate | Top Decile Rate | Lift@Decile |
|-------|--------:|--------:|--------:|--------------:|----------:|----------------:|------------:|
| Train | 0.01916 | 0.63838 | 0.21985 |       0.02262 |   0.01208 |         0.02262 |       1.872 |
| Val   | 0.01186 | 0.62580 | 0.17663 |       0.01441 |   0.00813 |         0.01441 |       1.772 |
| Test  | 0.01816 | 0.64618 | 0.16598 |       0.02090 |   0.01160 |         0.02090 |       1.802 |

### Logistic Regression baseline

| Split | PR-AUC  | ROC-AUC |  Brier  | Precision@10% | Base Rate | Top Decile Rate | Lift@Decile |
|-------|--------:|--------:|--------:|--------------:|----------:|----------------:|------------:|
| Train | 0.01769 | 0.61329 | 0.24064 |       0.02105 |   0.01208 |         0.02105 |       1.742 |
| Val   | 0.01132 | 0.59351 | 0.18053 |       0.01279 |   0.00813 |         0.01279 |       1.573 |
| Test  | 0.01656 | 0.60548 | 0.16177 |       0.01893 |   0.01160 |         0.01893 |       1.633 |

---

## Head-to-head comparison (TabularNet vs LogReg)

### Ranking and discrimination

TabularNet is higher on **PR-AUC** and **ROC-AUC** across all splits:

- **Train:** PR-AUC +0.00147, ROC-AUC +0.02509  
- **Val:** PR-AUC +0.00054, ROC-AUC +0.03230  
- **Test:** PR-AUC +0.00160, ROC-AUC +0.04069  

Interpretation: TabularNet is better at ranking surrender outcomes above non-surrenders given the same features.

### Operational targeting (outreach use-case)

TabularNet improves **Lift@Decile** meaningfully:

- **Train:** 1.872× vs 1.742×  
- **Val:** 1.772× vs 1.573×  
- **Test:** 1.802× vs 1.633×

Interpretation: If a retention team contacts the top 10% highest-risk policies, TabularNet’s top decile contains ~**1.80×** the base surrender rate on the test period, compared to ~**1.63×** for logistic regression.

### Calibration (Brier score)

- TabularNet has lower (better) Brier on **train** and **val**
- Logistic regression has slightly lower (better) Brier on **test** (0.1618 vs 0.1660)

Interpretation: Logistic regression remains a strong, stable baseline for probability quality; TabularNet likely benefits from an explicit calibration step if probabilities will be used for EV-based decisioning.

---

## Practical conclusion

- **Best model for ranking/targeting:** **PyTorch TabularNet**
  - Higher PR-AUC and ROC-AUC
  - Higher lift in top-decile outreach lists
- **Best model for interpretability and strong baseline calibration:** **Logistic Regression**
  - Slightly better test Brier
  - Easy-to-explain coefficients

Recommended next step:

- Calibrate TabularNet (temperature scaling) and re-evaluate Brier + reliability plots.
- Add EV-based decisioning (expected value vs threshold / budget) to translate lift into business outcomes.
