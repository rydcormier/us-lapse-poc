from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn


class TabularNet(nn.Module):
    """Neural network for tabular data with categorical and numerical features."""

    def __init__(
        self,
        cat_cardinalities: list[int],
        n_num: int,
        emb_dim: int = 8,
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        """Initialize the TabularNet model.

        Args:
            cat_cardinalities:  List of cardinalities for each categorical feature.
            n_num:              Number of numerical features.
            emb_dim:            Embedding dimension for categorical features.
            hidden:             Tuple of hidden layer sizes.
            dropout:            Dropout rate.
        """
        super().__init__()
        self.embs = nn.ModuleList(
            [
                nn.Embedding(card, min(emb_dim, max(2, (card + 1) // 2)))
                for card in cat_cardinalities
            ]
        )
        emb_out = sum(cast(nn.Embedding, e).embedding_dim for e in self.embs)
        in_dim = emb_out + n_num
        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TabularNet model.

        Args:
            x_cat: Categorical features tensor.
            x_num: Numerical features tensor.
        Returns:
            Output tensor with lapse probability logits.
        """
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(embs + [x_num], dim=1)
        out = self.mlp(x)
        return out.squeeze(1)  # logit output for binary classification
