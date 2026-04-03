from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def _get_feature_names(preprocessor) -> list[str]:
    """
    ColumnTransformer feature names:
    - cat: OneHotEncoder -> get_feature_names_out
    - num: StandardScaler -> passthrough names
    """
    feature_names: list[str] = []

    # Assumes transformers named ("cat", OneHotEncoder, CAT_COLS), ("num", StandardScaler, NUM_COLS)
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            # OneHotEncoder supports this; include original column prefixes
            fn = transformer.get_feature_names_out(cols)
            feature_names.extend(fn.tolist())
        else:
            # StandardScaler doesn't expand features
            feature_names.extend(list(cols))
    return feature_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/logreg/model.joblib", help="Path to saved sklearn pipeline")
    ap.add_argument("--top", type=int, default=25, help="Top N positive coefficients to display")
    ap.add_argument("--out", default="artifacts/logreg/top_positive_coeffs.csv", help="Output CSV path")
    args = ap.parse_args()

    pipe: Pipeline = joblib.load(args.model)

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    # Binary logistic regression: coef_ shape is (1, n_features)
    coefs = clf.coef_.reshape(-1)
    names = _get_feature_names(pre)

    if len(names) != len(coefs):
        raise RuntimeError(f"Feature name length {len(names)} != coef length {len(coefs)}")

    df = pd.DataFrame({"feature": names, "coef": coefs})
    df["abs_coef"] = df["coef"].abs()

    top_pos = df.sort_values("coef", ascending=False).head(args.top).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top_pos.to_csv(out_path, index=False)

    # Pretty print
    pd.set_option("display.max_colwidth", 120)
    print("\nTop positive coefficients (increase lapse risk):")
    print(top_pos[["feature", "coef"]].to_string(index=False))


if __name__ == "__main__":
    main()
