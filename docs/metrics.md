# Metrics used in lapse / surrender prediction project

This project predicts the probability that an in-force policy will **surrender/lapse in the next N quarters**
(default: next quarter). The event rate is typically **imbalanced** (few lapses compared to many non-lapses),
and business users act on **ranked risk lists** under a limited outreach budget. For that reason, we report
metrics that measure:

1) **Ranking quality under imbalance**: can we surface likely lapses at the top?
2) **Probability quality**: do predicted probabilities mean what they say?
3) **Operational usefulness**: how good are the top-K policies we would contact?

---

## 1) PR-AUC (Average Precision)

**What it is**: Area under the Precision-Recall curve (often reported as Average Precision).

**Why it matters here**:

- Lapse events are usually rare, so ROC-AUC can look “good” even when top-of-list performance is weak.
- PR-AUC focuses on the positive class and better reflects real performance when positives are scare.

**How to interpret**:

- Higher is better.
- Compare models primarily on **validation/test PR-AUC** (time-based splits).

---

## 2) ROC-AUC

**What it is**: Area under the ROC (Receiver Operating Characteristic) curve (TPR vs FPR across thresholds).

**Why we still include it**:

- Useful for broad comparability and sanity checks.
- Helpful to understand performance across all thresholds.

**Limitations in this domain**:

- With heavy class imbalance, ROC-AUC can overstate usefulness because it weights true negatives heavily.

---

## 3) Precision@K (Top-K precision)

**What it is**: Precision among top K% (or top K) highest-risk policies.

Example: `Precision@0.10` = event rate among the top 10% scored.

**Why it matters here**:

- Retention teams often have fixed capacity (e.g., "call 5,000 policies/week").
- The real question is: **Are the policies we contact actually high risk?**

**How to interpret**:

- Higher means better targeting efficiency.
- Should be compared at a business-relevant K (e.g., 1%, 5%, 10%).

---

## 4) Lift@Decile

**What it is**: The event rate in the top 10% divided by the overall event rate.

Lift@Decile = (Top decile event rate) / (Base event rate)

**Why it matters here**:

- Lift is a classic insurance/marketing metric.
- It translates directly to "how much better than random" the model is at selecting outreach targets.

**How to interpret**:

- Lift > 1 is better than random; Lift of 3 means the top decile is 3x the base lapse rate.

---

## 5) Brier Score (probability calibration)

**What it is**: Mean squared error of predicted probabilities:

Brier = mean$\left(\left(p-y\right)^2\right)$

**Why it matters here**:

- Retention decisioning often uses probability to estimate expected value and chooses thresholds.
- A well-calibrated model enables more reliable cost/benefit decisions.

**How to interpret**:

- Lower is better.
- Two models can have a similar PR-AUC, but the one with better calibration is often easier to operationalize.

---

## Recommended "model drivers" reporting

In addition to performance metrics, this project supports driver reporting:

- Logistic Regression: top positive coefficients (interpret directionality)
- TabularNet: permutation importance (model-agnostic importance based on performance drop)

These help communicate **why** the model flags certain policies and support stakeholder trust.

---

## Why we emphasize time-based splits

Lapse behavior can change over time (economic cycles, distribution behavior, underwriting shifts).
Random splits often inflate metrics. We therefore evaluate using **time-based splits**
(train → val → test in chronological order) to reflect deployment conditions.
