# Model comparison

Two different models are used to solve the prediction problem but they differ in how they represent features, what kinds of relationships they can learn, and how you train/serve them.

---

## 1) What each model "is" mathematically

### Logistic Regression baseline (sklearn)

A linear model on top of engineered features that becomes linear over:

- **One-hot-encoded** categoricals (gender, smoker status, premium frequency, etc.)
- **standardized numeric** features (tenure, annual premium)

You can think of this as:
$$P\left(y=1 \vert x\right) = \sigma\left(\omega^\top\phi\left(x\right)+b\right)$$

Where $\phi\left(x\right)$ is the one-hot + scaled vector.

**Implication**: It learns additive effects. Interactions (e.g. "smoker *and* early tenure") aren't captured unless interaction features are explicitly added.

### PyTorch TabularNet (embeddings + MLP)

A **neural network** with:

- **Embeddings** for categoricals (dense vectors learned during training)
- A **multi-layer perception (MLP)** over concatenated embeddings + numeric features

Conceptually:
$$z = \left[\mathrm{emb}(\mathit{cat}_1),\dots,\mathrm{emb}(\mathit{cat}_k),x_{num}\right]$$
$$\hat{y} = \phi\left(\mathrm{MLP}(z)\right)$$

**Implication**: It can learn non-linearities and feature interactions automatically.

---

## 2) How they represent features

### Categoricals

### Logistic regression

- Uses **OneHotEncoder** → high-dimensional sparse vector.
- Every category gets its own coefficient.
- Works great when:
  - categories are low/medium cardinality
  - effects are mostly additive

### TabularNet

- Uses **embedding tables**:
  - each category maps to a learned vector (e.g. 4–16 dims)
- Often better when:
  - there are many categories or multiple categorical fields
  - there are nuanced interactions between categories and numerics

In the `uslapseagent` feature set, categoricals are not huge, so TabularNet's advantage won't automatically appear because it is "deep." Its advantage usually shows up with:

- more features
- richer signals (payments, service events, macro joins)
- more complex interactions

---

## 3) What patterns can they learn (concrete examples)

Assume lapse risk increase for:

- infra-annual payers in early tenure, *especially for smokers*
- high annual premium, but only after some tenure threshold
- accidental death rider depends on age bucket

### Logistic regression

- Captures:
  - "smoker adds +X risk"
  - "infra-annual adds +Y risk"
  - "tenure adds +Z risk" (linearly, unless bucketed)
- Struggles with:
  - "infra-annual AND smoker AND tenure < 4" unless interaction features are explicitly added.

### TabularNet

Captures those interactions naturally because the MLP can model:

- "if tenure is small, weight smoker/infra-annual differently"
- non-linear tenure effects (e.g. early spike, then plateau)

---

## 4) Training differences

### Logistic regression training

- One pass (fast)
- Convex optimization (generally stable)
- Imbalance handled via:
  - `class-weight="balanced"`
- very quick iteration and easy debugging

### TabularNet training

- Mini-batch SGD/AdamW for multiple epochs
- Needs choices for:
  - architecture (hidden sizes, dropout)
  - learning rate
  - batch size
  - early stopping
- Imbalance handled via:
  - `BCEWithLogitsLoss(pos_weight=neg/pos)`
- More moving parts, but closer to "AI engineering" work

---

## 5) Calibration & probability behavior

This matters a lot for retention decisioning.

### Logistic regression

- Often **reasonably calibrated out of the box** (not always, but often better than many flexible models)

### TabularNet

- Often has **worse raw calibration** unless you calibrate (temperature scaling, isotonic, etc.)
- Can still rank well (high PR-AUC) but probabilities may be overconfident

In a portfolio setting, it's strong to:

- show TabularNet improves ranking/lift
- then add a calibration step to improve probability quality

## 6) Interpretability & stakeholder friendliness

### Logistic regression

- Easy to explain:
  - coefficients by feature/category
  - directionality is obvious
- Great for:
  - “Here are the top drivers of surrender risk”
  - fast stakeholder trust

### TabularNet

- Harder to explain directly
- You can still do:
  - permutation importance
  - SHAP (more work)
  - partial dependence style plots

---

## 7) Serving / deployment differences

### Logistic regression (sklearn joblib pipeline)

- One artifact: `model.joblib`
- Encodes preprocessing inside the pipeline
- very simple to serve

### TabularNet

- Two key artifacts:
  - `model.pt` (weights)
  - `preprocessor.joblib` (category mappings + numeric stats)
- Serving requires:
  - applying exact preprocessor
  - feeding tensors to PyTorch model
- More realistic for AI engineering roles, but slightly more complex

## When TabularNet should be expected to beat Logistic Regression

TabularNet tends to win when you introduce:

- richer behavioral signals (missed payments, call/service events)
- more category complexity (agents, channels, product variants)
- non-linear effects (tenure curves, premium thresholds)
- interactions (macro regime x policy type x payment mode)

When feature set is small and mostly additive, **logistic regression can be surprisingly strong**.

---

## How to decide which is "better" for this PoC

Use a decision matrix:

- If **PR-AUC and Lift@Decile** are similar:
  - logistic regression may be "best" due to simplicity + interpretability
- If TabularNet provides a meaningful life in top-K capture:
  - keep TabularNet as your "AI engineering" highlight
  - add calibration to make it decision-ready
