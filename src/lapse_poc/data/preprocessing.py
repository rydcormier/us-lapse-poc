from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TabularPreprocessor:
    """
    Preprocessor for tabular data with categorical and numerical features.

    Attributes:
        cat_cols: List of categorical column names.
        num_cols: List of numerical column names.
        cat_maps: Mapping from categorical values to integer indices for each categorical column.
        num_mean: Mean values for each numerical column.
        num_std: Standard deviation values for each numerical column.
    """
    cat_cols: list[str]
    num_cols: list[str]
    cat_maps: dict[str, dict[str, int]]
    num_mean: dict[str, float]
    num_std: dict[str, float]

    @staticmethod
    def fit(df: pd.DataFrame, cat_cols: list[str], num_cols: list[str]) -> TabularPreprocessor:
        """
        Fit a TabularPreprocessor to the given DataFrame.

        Args:
            df:         Input DataFrame.
            cat_cols:   List of categorical column names.
            num_cols:   List of numerical column names.
        Returns:
            Fitted TabularPreprocessor instance.
        """
        cat_maps: dict[str, dict[str, int]] = {}
        for c in cat_cols:
            s = df[c].astype("string").fillna("__NA__")
            uniq = sorted(s.unique().tolist())
            mapping = {"__UNK__": 0}
            mapping.update({v: i + 1 for i, v in enumerate(uniq)})
            cat_maps[c] = mapping

        num_mean: dict[str, float] = {}
        num_std: dict[str, float] = {}
        for c in num_cols:
            x = pd.to_numeric(df[c], errors="coerce")
            mu = float(x.mean())
            sd = float(x.std(ddof=0))
            if sd == 0.0 or not np.isfinite(sd):
                sd = 1.0
            num_mean[c] = mu
            num_std[c] = sd

        return TabularPreprocessor(
            cat_cols=cat_cols, num_cols=num_cols, cat_maps=cat_maps, num_mean=num_mean, num_std=num_std
        )

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform the given DataFrame into categorical and numerical numpy arrays.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (categorical array, numerical array).
        """
        # Categories -> int64, with unknown=0
        x_cat = np.zeros((len(df), len(self.cat_cols)), dtype=np.int64)
        for j, c in enumerate(self.cat_cols):
            mapping = self.cat_maps[c]
            s = df[c].astype("string").fillna("__NA__")
            x_cat[:, j] = [mapping.get(v, 0) for v in s.tolist()]

        # Numerics -> float32 standardized, NaNs -> mean
        x_num = np.zeros((len(df), len(self.num_cols)), dtype=np.float32)
        for j, c in enumerate(self.num_cols):
            x = pd.to_numeric(df[c], errors="coerce").astype("float32")
            mu = self.num_mean[c]
            sd = self.num_std[c]
            x = x.fillna(mu)
            x_num[:, j] = ((x - mu) / sd).to_numpy(dtype=np.float32)

        return x_cat, x_num

    def cat_cardinalities(self) -> list[int]:
        """Return cardinalities of categorical features."""
        # +1 because mapping includes unknown index 0
        return [len(self.cat_maps[c]) for c in self.cat_cols]

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary representation."""
        return {
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "cat_maps": self.cat_maps,
            "num_mean": self.num_mean,
            "num_std": self.num_std,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TabularPreprocessor:
        """Create a TabularPreprocessor from a dictionary representation."""
        return TabularPreprocessor(
            cat_cols=list(d["cat_cols"]),
            num_cols=list(d["num_cols"]),
            cat_maps=dict(d["cat_maps"]),
            num_mean={k: float(v) for k, v in d["num_mean"].items()},
            num_std={k: float(v) for k, v in d["num_std"].items()},
        )