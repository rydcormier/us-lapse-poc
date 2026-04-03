from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data with categorical and numerical features."""

    def __init__(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray | None = None) -> None:
        """Initialize the TabularDataset with categorical, numerical features and optional targets.

        Args:
            x_cat: Categorical features as a numpy array.
            x_num: Numerical features as a numpy array.
            y: Optional target values as a numpy array.
        """
        self.x_cat = torch.from_numpy(x_cat.astype("int64"))
        self.x_num = torch.from_numpy(x_num.astype("float32"))
        self.y = None if y is None else torch.from_numpy(y.astype("float32"))

    def __len__(self) -> int:
        return self.x_cat.shape[0]

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.x_cat[idx], self.x_num[idx]
        return self.x_cat[idx], self.x_num[idx], self.y[idx]
