from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from lapse_poc.data.preprocessing import TabularPreprocessor
from lapse_poc.models.tabular import TabularNet


@torch.no_grad()
def predict_proba(pre: TabularPreprocessor, model: TabularNet, df: pd.DataFrame) -> np.ndarray:
    x_cat, x_num = pre.transform(df)
    logits = model(torch.from_numpy(x_cat), torch.from_numpy(x_num))
    return torch.sigmoid(logits).cpu().numpy()


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(int)
    return float(average_precision_score(y_true, y_prob))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", default="data/processed/policy_quarter.parquet", help="Modeling table parquet"
    )
    ap.add_argument("--artifact-dir", default="artifacts/torch", help="Torch artifact directory")
    ap.add_argument("--target", default="y_surrender_next", help="Label column")
    ap.add_argument("--repeats", type=int, default=3, help="Permutation repeats per feature")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out", default="artifacts/torch/permutation_importance.csv", help="Output CSV"
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    artifact_dir = Path(args.artifact_dir)

    # Load artifacts
    pre = TabularPreprocessor.from_dict(joblib.load(artifact_dir / "preprocessor.joblib"))

    model = TabularNet(cat_cardinalities=pre.cat_cardinalities(), n_num=len(pre.num_cols))
    state = torch.load(artifact_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Load test data
    df = pd.read_parquet(args.data)
    df_test = df[df["split"] == "test"].copy()
    y = df_test[args.target].to_numpy(dtype=int)

    # Baseline PR-AUC
    base_prob = predict_proba(pre, model, df_test)
    base = pr_auc(y, base_prob)
    print(f"Baseline test PR-AUC: {base:.6f}")

    # Determine feature list from the preprocessor feature contract
    # pre.cat_cols / pre.num_cols are the truth for what Torch model expects.
    features = list(pre.cat_cols) + list(pre.num_cols)

    results = []
    for feat in features:
        drops = []
        for _ in range(args.repeats):
            df_perm = df_test.copy()

            # Permute within test split
            perm_idx = rng.permutation(len(df_perm))
            df_perm[feat] = df_perm[feat].to_numpy()[perm_idx]

            prob = predict_proba(pre, model, df_perm)
            score = pr_auc(y, prob)
            drops.append(base - score)

        results.append(
            {
                "feature": feat,
                "pr_auc_drop_mean": float(np.mean(drops)),
                "pr_auc_drop_std": float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0,
            }
        )

    imp = (
        pd.DataFrame(results)
        .sort_values("pr_auc_drop_mean", ascending=False)
        .reset_index(drop=True)
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out_path, index=False)

    print("\nTop permutation importances (higher PR-AUC drop => more important):")
    print(imp.head(25).to_string(index=False))

    # Also write a small JSON snippet you can paste into README/slides if you want
    (out_path.parent / "permutation_importance_top10.json").write_text(
        json.dumps(imp.head(10).to_dict(orient="records"), indent=2)
    )


if __name__ == "__main__":
    main()
