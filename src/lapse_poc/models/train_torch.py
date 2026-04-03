from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lapse_poc.data.preprocessing import TabularPreprocessor
from lapse_poc.data.torch_dataset import TabularDataset
from lapse_poc.eval.metrics import classification_report
from lapse_poc.models.tabular import TabularNet
from lapse_poc.settings import CAT_COLS, DJIA_COL, NUM_COLS
from lapse_poc.utils import set_seed

# run cleaner when piped
USE_TQDM = sys.stdout.isatty() and sys.stderr.isatty()


@torch.no_grad()
def predict_proba(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """
    Predict probabilities using the given model and data loader.

    Args:
        model (torch.nn.Module): The trained model.
        loader (DataLoader): DataLoader for the dataset to predict on.
        device (torch.device): Device to run the model on.
    Returns:
        np.ndarray: Predicted probabilities.
    """
    model.eval()
    probs: list[np.ndarray] = []
    for batch in loader:
        x_cat = batch[0].to(device)
        x_num = batch[1].to(device)
        logits = model(x_cat, x_num)
        batch_probs = torch.sigmoid(logits).cpu().numpy()
        probs.append(batch_probs)
    return np.concatenate(probs, axis=0)


def main() -> None:
    """
    Main training script for Tabular Neural Network.

    Steps:
        1. Load data from parquet file.
        2. Split into train/val/test based on 'split' column.
        3. Preprocess categorical and numerical features.
        4. Train Tabular Neural Network with class balancing.
        5. Evaluate on all splits and save metrics and model artifacts.
    """
    ap = argparse.ArgumentParser(description="Train Tabular Neural Network")
    ap.add_argument("--data", required=True, help="policy_quarter.parquet from build_features")
    ap.add_argument("--out", required=True, help="Output dir for artifacts")
    ap.add_argument("--target", default="y_surrender_next", help="label column")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    ap.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    ap.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    ap.add_argument(
        "--emb-dim", type=int, default=8, help="Embedding dimension for categorical features"
    )
    ap.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    ap.add_argument(
        "--hidden", type=str, default="128,64", help="Comma-separated hidden layer sizes"
    )
    ap.add_argument("--include-dija", action="store_true", help="Include DJIA column if present")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load data
    df = pd.read_parquet(args.data)

    y_col = args.target

    cat_cols = list(CAT_COLS)
    num_cols = list(NUM_COLS)
    if args.include_dija and DJIA_COL in df.columns:
        num_cols.append(DJIA_COL)

    # Split data
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    # Preprocessing
    pre = TabularPreprocessor.fit(train, cat_cols=cat_cols, num_cols=num_cols)
    x_cat_tr, x_num_tr = pre.transform(train)
    x_cat_va, x_num_va = pre.transform(val)
    x_cat_te, x_num_te = pre.transform(test)

    # Prepare datasets and dataloaders
    y_tr = train[y_col].to_numpy(dtype=np.float32)
    y_va = val[y_col].to_numpy(dtype=np.float32)
    y_te = test[y_col].to_numpy(dtype=np.float32)

    ds_tr = TabularDataset(x_cat=x_cat_tr, x_num=x_num_tr, y=y_tr)
    ds_va = TabularDataset(x_cat=x_cat_va, x_num=x_num_va, y=y_va)
    ds_te = TabularDataset(x_cat=x_cat_te, x_num=x_num_te, y=y_te)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip().isdigit())

    # Model setup
    model = TabularNet(
        cat_cardinalities=pre.cat_cardinalities(),
        n_num=len(num_cols),
        emb_dim=args.emb_dim,
        hidden=hidden,
        dropout=args.dropout,
    ).to(device)

    # class imbalance handling
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32).to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_pr = -1.0
    best_state = None
    patience = 3
    bad = 0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"Epoch {epoch}/{args.epochs}", leave=False, disable=not USE_TQDM)
        for x_cat, x_num, y in pbar:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x_cat, x_num)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        # Validate
        val_prob = predict_proba(
            model, DataLoader(ds_va, batch_size=args.batch_size, shuffle=False), device
        )
        val_rep = classification_report(y_va.astype(int), val_prob)

        if val_rep["pr_auc"] > best_val_pr:
            best_val_pr = val_rep["pr_auc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping triggered")
                break

        print(
            f"[epoch {epoch}] val PR-AUC: {val_rep['pr_auc']:.4f} lift@decile: {val_rep['lift@decile']:.2f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    tr_prob = predict_proba(
        model, DataLoader(ds_tr, batch_size=args.batch_size, shuffle=False), device
    )
    va_prob = predict_proba(model, dl_va, device)
    te_prob = predict_proba(model, dl_te, device)

    metrics = {
        "train": classification_report(y_tr.astype(int), tr_prob),
        "val": classification_report(y_va.astype(int), va_prob),
        "test": classification_report(y_te.astype(int), te_prob),
        "target": y_col,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "seed": args.seed,
        "device": str(device),
    }

    # Save artifacts
    torch.save(model.state_dict(), out_dir / "model.pt")
    joblib.dump(pre.to_dict(), out_dir / "preprocessor.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save test predictions for convenience
    pd.DataFrame(
        {
            "y_true": y_te.astype(int),
            "y_prob": te_prob,
        }
    ).to_parquet(out_dir / "test_preds.parquet")

    # Print metrics
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
