from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


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

