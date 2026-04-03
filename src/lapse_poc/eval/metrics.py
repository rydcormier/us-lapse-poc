from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def classification_report(
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        k_frac: float=0.1
    ) -> dict[str, float]:
    """
    Compute classification metrics for binary classification.

    Args:
        y_true (np.ndarray): True binary labels (0 or 1).
        y_prob (np.ndarray): Predicted probabilities for the positive class.
        k_frac (float): Fraction of top predictions to consider for Precision@K.
    Returns:
        dict[str, float]: Dictionary containing various classification metrics.
    """
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    out: dict[str, float] = {}
    out['pr_auc'] = float(average_precision_score(y_true, y_prob))
    # roc_auc requires both classes to be present in y_true
    if len(np.unique(y_true)) == 2:
        out['roc_auc'] = float(roc_auc_score(y_true, y_prob))
    else:
        out['roc_auc'] = float('nan')

    out['brier'] = float(brier_score_loss(y_true, y_prob))

    # Precision@K (K fraction of highest scores)
    n = len(y_true)
    k = max(1, int(round(n * k_frac)))
    idx = np.argsort(-y_prob)[:k]
    out[f'precision@{k_frac:.2f}'] = float(y_true[idx].mean())

    # Lift@decile
    base_rate = float(y_true.mean()) if n > 0 else 0.0
    top_decile = max(1, int(round(0.1 * n)))
    top_idx = np.argsort(-y_prob)[:top_decile]
    top_rate = float(y_true[top_idx].mean())
    out['base_rate'] = base_rate
    out['top_decile_rate'] = top_rate
    out['lift@decile'] =float(top_rate / base_rate) if base_rate > 0 else float('inf')

    return out