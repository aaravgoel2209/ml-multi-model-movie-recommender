"""
Matrix Factorization Collaborative Filtering (explicit ratings) using PyTorch.

High-level idea
---------------
learn low-dimensional embeddings for users and movies such that the dot product
(user_vec · movie_vec) approximates the observed rating. Optional bias terms per user and movie improve accuracy.


Assumptions / Notes
-------------------
- Recommendations only consider movies seen in training.
- expects files train.csv/val.csf with columns user_idx, item_idx, rating (created by data/preprocess.py)
    -> b.c we need sequential idx for users/items to use as embedding indices
- recommendation for new users is done via a simple weighted average of item embeddings based on their provided ratings (cold-start strategy)
------------
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, List

import numpy as np
import pandas as pd
from torch import nn
import torch
from torch.utils.data import DataLoader

from models.mf.data import RatingsDataset
from models.mf.model import MFModel


@dataclass
class TrainStats:
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None

class CollaborativeFilterTrainer:
    def __init__(
        self,
        latent_dim: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        use_bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 42,
        movie_data: pd.DataFrame= None,
    ):
        self.latent_dim = latent_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_bias = use_bias
        self.seed = seed
        self.movie_id2idx = movie_data.set_index("movieId")["item_idx"].to_dict()
        self.movie_idx2id = movie_data.set_index("item_idx")["movieId"].to_dict()
        self.movie_id2title = movie_data.set_index("movieId")["title"].to_dict()
        self.num_users = 0
        self.num_items = 0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Learned / populated after fit()
        self.model: Optional[MFModel] = None
        # For recommend(): track items each user has already seen (in training data)
        self._seen_items_by_user_seen_items_by_user =  {}
        self._is_fit: bool = False

    def fit(
        self,
        ratings_df: pd.DataFrame,
        epochs: int = 10,
        batch_size: int = 2048,
        val_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> List[TrainStats]:
        """
        Train the MF model on explicit ratings.

        Args:
            ratings_df: DataFrame with columns userId, movieId, rating
            epochs: number of passes over training data
            batch_size: mini-batch size
            val_df: optional validation DataFrame with same columns; used only for reporting
            verbose: print epoch losses

        Returns:
            list of TrainStats (epoch-level)
        """
        self._set_seed(self.seed)

        # Build mappings based on training data only

        # Prepare training dataset/loader
        train_ds = self._df_to_dataset(ratings_df)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        self.num_users = ratings_df["user_idx"].nunique()
        self.num_items = ratings_df["item_idx"].nunique()
        # Optional val dataset/loader
        val_loader = None
        if val_df is not None and len(val_df) > 0:
            val_ds = self._df_to_dataset(val_df)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        # Create model
        self.model = MFModel(
            num_users=self.num_users,
            num_items=self.num_items,
            latent_dim=self.latent_dim,
            use_bias=self.use_bias
        ).to(self.device)

        # Slightly better init for global bias: mean rating (optional but helpful)
        if self.use_bias:
            with torch.no_grad():
                mean_rating = float(ratings_df["rating"].mean())
                self.model.global_bias.fill_(mean_rating)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        stats: List[TrainStats] = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0
            n_train = 0

            for user_idx, item_idx, rating in train_loader:
                user_idx = user_idx.to(self.device)
                item_idx = item_idx.to(self.device)
                rating = rating.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                pred = self.model(user_idx, item_idx)
                loss = loss_fn(pred, rating)
                loss.backward()
                optimizer.step()

                current_batch_size = int(rating.shape[0])
                train_loss += float(loss.item()) * current_batch_size
                n_train += current_batch_size

            train_loss /= max(n_train, 1)

            val_loss = None
            if val_loader is not None:
                val_loss = self._eval_loss(val_loader, loss_fn)

            stats.append(TrainStats(epoch=epoch, train_loss=train_loss, val_loss=val_loss))

            if verbose:
                if val_loss is None:
                    print(f"Epoch {epoch:03d} | train MSE: {train_loss:.4f}")
                else:
                    print(f"Epoch {epoch:03d} | train MSE: {train_loss:.4f} | val MSE: {val_loss:.4f}")

        # Precompute "seen items" per user from the *training* ratings
        self._build_seen_items_index(ratings_df)

        self._is_fit = True
        return stats

    def save(self, path: str) -> None:
        """
        Save model weights + mappings. Can be loaded via `MatrixFactorizationCF.load(path)`.

        Note: saves a torch checkpoint dict (use .pt extension typically).
        """
        assert self.model is not None

        ckpt = {
            "latent_dim": self.latent_dim,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "use_bias": self.use_bias,
            "seed": self.seed,
            "seen_items_by_user": self._seen_items_by_user,
            "model_state_dict": self.model.state_dict(),
            "num_users": self.num_users,
            "num_items": self.num_items,
            "is_fit": self._is_fit,

        }
        torch.save(ckpt, path)


    @staticmethod
    def _set_seed(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def _df_to_dataset(self, df: pd.DataFrame) -> RatingsDataset:
        user_idx = df["user_idx"].astype(np.int64).to_numpy()
        item_idx = df["item_idx"].astype(np.int64).to_numpy()
        ratings = df["rating"].astype(np.float32).to_numpy()
        return RatingsDataset(user_idx, item_idx, ratings)

    @torch.no_grad()
    def _eval_loss(self, loader: DataLoader, loss_fn: nn.Module) -> float:
        assert self.model is not None
        self.model.eval()
        total = 0.0
        n = 0

        for user_idx, item_idx, rating in loader:
            user_idx = user_idx.to(self.device)
            item_idx = item_idx.to(self.device)
            rating = rating.to(self.device)

            pred = self.model(user_idx, item_idx)
            loss = loss_fn(pred, rating)

            bs = int(rating.shape[0])
            total += float(loss.item()) * bs
            n += bs

        return total / max(n, 1)

    def _build_seen_items_index(self, train_df: pd.DataFrame) -> None:
        """
        Build {user_idx: np.array([item_idx, ...])} for fast filtering in recommend().
        Uses training interactions only.
        """
        user_idx = pd.Series(train_df["user_idx"].astype(np.int64).to_numpy())
        item_idx = pd.Series(train_df["item_idx"].astype(np.int64).to_numpy())

        # Group by internal user_idx
        seen: Dict[int, List[int]] = {}
        for u, i in zip(user_idx, item_idx):
            seen.setdefault(int(u), []).append(int(i))

        # Deduplicate to keep it smaller
        self._seen_items_by_user = {u: np.unique(np.array(items, dtype=np.int64)) for u, items in seen.items()}


def _main():

    # Paths (we need to make this smarter)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MODEL_DIR = PROJECT_ROOT / "mf" / "mf_model_small.pt"
    TRAIN_DF_DIR = PROJECT_ROOT / ".." / "data" / "processed" / "train.csv"
    VAL_DF_DIR = PROJECT_ROOT / ".." / "data" / "processed" / "val.csv"
    MOVIE_DATA_DIR = PROJECT_ROOT / ".." /"data" / "processed" / "items.csv"

    # Load Data
    train_df = pd.read_csv(TRAIN_DF_DIR)
    val_df = pd.read_csv(VAL_DF_DIR)
    movie_data = pd.read_csv(MOVIE_DATA_DIR)

    # Training Config
    latent_dim = 20
    lr = 1e-3
    weight_decay = 1e-5
    epochs = 10
    use_bias = True
    batch_size = 2048

    # Create Model Service Wrapper and Train
    mf = CollaborativeFilterTrainer(
        latent_dim=latent_dim,
        lr=lr,
        weight_decay=weight_decay,
        use_bias=use_bias,
        movie_data=movie_data)
    mf.fit(train_df, epochs=epochs, batch_size=batch_size, val_df=val_df, verbose=True)

    # Save Model
    mf.save(str(MODEL_DIR))

    # Ask for a recommendation

if __name__ == "__main__":
    _main()