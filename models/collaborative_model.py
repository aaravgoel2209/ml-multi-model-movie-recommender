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

