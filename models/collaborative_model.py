"""
Matrix Factorization Collaborative Filtering (explicit ratings) using PyTorch.

High-level idea
---------------
We learn low-dimensional embeddings for users and movies such that the dot product
(user_vec · movie_vec) approximates the observed rating. Optional bias terms improve
accuracy by capturing that some users rate higher on average and some movies are
generally liked.

Assumptions / Notes
-------------------
- This is *explicit* collaborative filtering (uses observed ratings as regression targets).
- Missing ratings are treated as unknown (not zeros).
- Recommendations only consider movies seen in training.
- `ratings_df` must contain columns: userId, movieId, rating (timestamp ignored).
- Designed to be a clear baseline for a larger recommender pipeline.
- Embeddings require a sequential index, so we build user2idx and item2idx mappings from training data.
Usage (demo)
------------
python -m models.mf_cf --ratings data/ratings.csv --user 1 --topn 10
"""
from dataclasses import dataclass
from typing import Optional, Union, Dict, List, Tuple

import numpy as np
import pandas as pd
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset

from data.load_data import load_movielens
@dataclass
class TrainStats:
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None


class RatingsDataset(Dataset):
    """PyTorch Dataset for (user_idx, item_idx, rating) triples."""
    def __init__(self, user_indices: np.ndarray, item_indices: np.ndarray, ratings: np.ndarray):
        self.user_indices = torch.as_tensor(user_indices, dtype=torch.long)
        self.item_indices = torch.as_tensor(item_indices, dtype=torch.long)
        self.ratings = torch.as_tensor(ratings, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.ratings.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_indices[idx], self.item_indices[idx], self.ratings[idx]

class MFModel(nn.Module):
    """
    Basic Matrix Factorization model with optional bias terms.

    pred(u,i) = dot(U[u], V[i]) + bu[u] + bi[i] + global_bias
    -->the predicted rating for user u and item i is the dot product of their latent vectors
    """

    def __init__(self, num_users: int, num_items: int, latent_dim: int, use_bias: bool = True):
        super().__init__()
        self.use_bias = bool(use_bias)

        self.user_emb = nn.Embedding(num_users, latent_dim)
        self.item_emb = nn.Embedding(num_items, latent_dim)

        if self.use_bias:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))
        else:
            self.user_bias = None
            self.item_bias = None
            self.global_bias = None

        self._init_weights()

    def _init_weights(self) -> None:
        # Standard small init for embeddings (helps stabilize early training)
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
        if self.use_bias:
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
            with torch.no_grad():
                self.global_bias.fill_(0.0)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_idx: (batch,)
            item_idx: (batch,)
        Returns:
            preds: (batch,) float tensor
        """
        u = self.user_emb(user_idx)  # (batch, k)
        v = self.item_emb(item_idx)  # (batch, k)
        dot = (u * v).sum(dim=1)     # (batch,)

        if self.use_bias:
            bu = self.user_bias(user_idx).squeeze(1)  # (batch,)
            bi = self.item_bias(item_idx).squeeze(1)  # (batch,)
            return dot + bu + bi + self.global_bias
        else:
            return dot

