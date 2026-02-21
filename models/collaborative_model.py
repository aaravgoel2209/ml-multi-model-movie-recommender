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
from typing import Optional, Union, Dict, List, Tuple

import numpy as np
import pandas as pd
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset

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

class CollaborativeModel:
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
        self._seen_items_by_user =  {}
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
        self._require_fit()
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

    @classmethod
    def load(cls, path: str, movie_data: pd.DataFrame, device: Optional[Union[str, torch.device]] = None) -> "CollaborativeModel":
        """
        Load a saved MF model.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(
            latent_dim=int(ckpt["latent_dim"]),
            lr=float(ckpt["lr"]),
            weight_decay=float(ckpt["weight_decay"]),
            use_bias=bool(ckpt["use_bias"]),
            device=device,
            seed=int(ckpt.get("seed", 42)),
            movie_data=movie_data
        )
        # seen_items_by_user keys are internal user_idx
        obj._seen_items_by_user = {
            int(k): np.array(v, dtype=np.int64) for k, v in ckpt.get("seen_items_by_user", {}).items()
        }
        obj.num_users = ckpt["num_users"]
        obj.num_items = ckpt["num_items"]
        obj.is_fit = ckpt["is_fit"]

        obj.model = MFModel(
            num_users=int(ckpt["num_users"]),
            num_items=int(ckpt["num_items"]),
            latent_dim=obj.latent_dim,
            use_bias=obj.use_bias,
        )
        obj.model.load_state_dict(ckpt["model_state_dict"])
        obj.model.to(obj.device)
        obj.model.eval()

        obj._is_fit = True
        return obj

    def _require_fit(self) -> None:
        """Helper to check if model is fit before allowing predict/recommend."""
        if not self._is_fit or self.model is None:
            raise RuntimeError(f"Model is not fit yet. model = {self.model}, _is_fit = {self._is_fit}")


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

    @torch.no_grad()
    def recommend_from_ratings(
            self,
            ratings: list[tuple[int, float]],
            top_n: int = 10,
            exclude_rated: bool = True,
            return_scores: bool = False,
    ):
        """
        Cold-start recommendation based on a small set of (movieId, rating) pairs.

        Args:
            ratings: List of (movieId, rating) provided by new user
            top_n: number of recommendations
            exclude_rated: remove movies already rated by the user
            return_scores: if True, returns (movieId, score)

        Returns:
            List of movieIds or (movieId, score)
        """

        self._require_fit()

        device = self.device
        item_emb = self.model.item_emb.weight  # (num_items, latent_dim)

        # Collect weighted item embeddings
        weighted_sum = torch.zeros(self.latent_dim, device=device)
        total_weight = 0.0

        rated_item_indices = []

        for movie_id, rating in ratings:

            idx = self.get_movie_idx(movie_id)
            rated_item_indices.append(idx)

            # Simple centered weight: rating - 3.0
            weight = float(rating) - 3.0

            if weight == 0:
                continue

            weighted_sum += weight * item_emb[idx]
            total_weight += abs(weight)

        if total_weight == 0:
            raise ValueError("No valid ratings provided for cold-start recommendation.")

        # Create pseudo-user vector
        user_vec = weighted_sum / total_weight  # (latent_dim,)

        # Score all items
        scores = item_emb @ user_vec  # (num_items,)

        if self.use_bias:
            scores = scores + self.model.item_bias.weight.squeeze(1)
            scores = scores + self.model.global_bias.squeeze(0)

        scores_np = scores.detach().cpu().numpy()

        # Optionally remove already rated movies
        if exclude_rated and rated_item_indices:
            scores_np[rated_item_indices] = -np.inf

        # Get Top-N
        n_items = scores_np.shape[0]
        k = min(top_n, n_items)

        if not np.isfinite(scores_np).any():
            return []

        candidate_idx = np.argpartition(-scores_np, kth=k - 1)[:k]
        candidate_idx = candidate_idx[np.argsort(-scores_np[candidate_idx])]

        movie_ids = [self.get_movie_id(i)for i in candidate_idx.tolist()]

        if not return_scores:
            return movie_ids

        return [(mid, float(scores_np[self.get_movie_idx(mid)])) for mid in movie_ids]

    def get_movie_idx(self, id: int) -> int:
        """Helper to convert internal item_idx back to original movieId."""
        return self.movie_id2idx.get(id, -1)

    def get_movie_id(self, idx: int) -> int:
        """Helper to convert internal item_idx back to original movieId."""
        return self.movie_idx2id.get(idx, -1)
    def get_movie_title(self, id: int) -> str:
        """Helper to get movie title from original movieId."""
        return self.movie_id2title.get(id, "Unknown Title")


def _main():
    latent_dim = 20
    lr = 1e-3
    weight_decay = 1e-5
    epochs = 1
    use_bias = True
    batch_size = 2048
    user = 1
    top_n = 10
    train_df = pd.read_csv("../data/processed/train.csv")
    val_df = pd.read_csv("../data/processed/val.csv")
    movie_data = pd.read_csv("../data/processed/items.csv")
   # mf = CollaborativeModel(
   #     latent_dim=latent_dim,
   #     lr=lr,
   #     weight_decay=weight_decay,
   #     use_bias=use_bias,
   #     movie_data=movie_data
    #)
    #mf.fit(train_df, epochs=epochs, batch_size=batch_size, val_df=val_df, verbose=True)
    #mf.save("mf_model.pt")
    #print(mf.get_movie_title(0))
    mf = CollaborativeModel.load("mf_model.pt", movie_data=movie_data)
    recs = mf.recommend_from_ratings(
        ratings=[
            (1221, 5.0),  # LOTR
            (4306, 5.0),  # Harry Potter
            (597, 1.0),  # Titanic
        ],
        top_n=10,
        return_scores = True

    )
    print(f"\nTop-{top_n} recommendations for userId={user}:")
    for movie_id, score in recs:
        print(f" movieId={movie_id}  score={score:.3f} title={mf.get_movie_title(movie_id)}")

    print(mf.get_movie_id(30))
    print(mf.get_movie_title(357))
if __name__ == "__main__":
    _main()