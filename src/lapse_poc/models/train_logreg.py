from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lapse_poc.eval.metrics import classification_report
from lapse_poc.settings import CAT_COLS, NUM_COLS


def main() -> None:
    """
    Logistic Regression training script.

    Steps:
        1. Load data from parquet file.
        2. Split into train/val/test based on 'split' column.
        3. Preprocess categorical (OneHot) and numerical (StandardScaler) features.
        4. Train Logistic Regression with class balancing.
        5. Evaluate on all splits and save metrics and model artifacts.
    """
    ap = argparse.ArgumentParser(description="Train Logistic Regression Model")
    ap.add_argument("--data", required=True, help="policy_quarter.parquet from build_features")
    ap.add_argument("--out", required=True, help="Output dir for artifacts")
    ap.add_argument("--target", default="y_surrender_next", help="label column")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    df = df.dropna(subset=["split"])

    y_col = args.target

    # Split data
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    # Preprocessing
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS),
        ]
    )

    # classifier
    clf = LogisticRegression(max_iter=200, class_weight="balanced", solver="lbfgs")

    # Pipeline
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(train[CAT_COLS + NUM_COLS], train[y_col].astype(int))

    def eval_split(d: pd.DataFrame) -> dict:
        y_true = d[y_col].to_numpy(dtype=int)
        y_prob = pipe.predict_proba(d[CAT_COLS + NUM_COLS])[:, 1]
        return classification_report(y_true, y_prob)

    metrics = {
        "train": eval_split(train),
        "val": eval_split(val),
        "test": eval_split(test),
        "target": y_col,
        "cat_cols": CAT_COLS,
        "num_cols": NUM_COLS,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    joblib.dump(pipe, out_dir / "model.joblib")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
