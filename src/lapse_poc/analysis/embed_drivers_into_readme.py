from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Render a GitHub-flavored Markdown table (no extra deps)."""
    headers = df.columns.tolist()
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        vals = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def replace_block(text: str, begin: str, end: str, replacement: str) -> str:
    """
    Replace content between markers (inclusive markers remain).
    Markers are HTML comments, e.g. <!-- BEGIN:XYZ --> ... <!-- END:XYZ -->
    """
    pattern = re.compile(
        rf"({re.escape(begin)})(.*?)(\s*{re.escape(end)})",
        flags=re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Could not find marker block: {begin} ... {end}")

    # Keep markers; normalize to: BEGIN + newline + replacement + newline + END
    new_block = f"{begin}\n\n{replacement}\n\n{end}"
    return text[: m.start()] + new_block + text[m.end() :]


def load_top10_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing expected columns: {missing}. Found: {list(df.columns)}")
        df = df[columns]

    return df.head(10).copy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed top-10 driver tables into README.md")
    ap.add_argument("--readme", default="README.md", help="Path to README.md to update")
    ap.add_argument("--logreg-csv", default="artifacts/logreg/top_positive_coeffs.csv")
    ap.add_argument("--tabnet-csv", default="artifacts/torch/permutation_importance.csv")
    ap.add_argument("--dry-run", action="store_true", help="Print updated README to stdout, don't write")
    args = ap.parse_args()

    readme_path = Path(args.readme)
    if not readme_path.exists():
        raise FileNotFoundError(f"README not found: {readme_path}")

    text = readme_path.read_text(encoding="utf-8")

    # ----- Logistic Regression top10 -----
    logreg_df = load_top10_csv(
        Path(args.logreg_csv),
        columns=["feature", "coef"],  # keep clean; add odds_ratio if you included it in your CSV
    )
    # (Optional) round coef a bit for readability
    logreg_df["coef"] = logreg_df["coef"].astype(float).round(6)

    logreg_md = df_to_markdown_table(logreg_df)

    text = replace_block(
        text,
        begin="<!-- BEGIN:LOGREG_TOP10 -->",
        end="<!-- END:LOGREG_TOP10 -->",
        replacement=logreg_md,
    )

    # ----- TabularNet permutation importance top10 -----
    tab_df = load_top10_csv(
        Path(args.tabnet_csv),
        columns=["feature", "pr_auc_drop_mean", "pr_auc_drop_std"],
    )
    tab_df["pr_auc_drop_mean"] = tab_df["pr_auc_drop_mean"].astype(float).round(6)
    tab_df["pr_auc_drop_std"] = tab_df["pr_auc_drop_std"].astype(float).round(6)

    tab_md = df_to_markdown_table(tab_df)

    text = replace_block(
        text,
        begin="<!-- BEGIN:TABNET_PERMIMP_TOP10 -->",
        end="<!-- END:TABNET_PERMIMP_TOP10 -->",
        replacement=tab_md,
    )

    if args.dry_run:
        print(text)
    else:
        readme_path.write_text(text, encoding="utf-8")
        print(f"Updated {readme_path} with embedded top-10 driver tables.")


if __name__ == "__main__":
    main()
