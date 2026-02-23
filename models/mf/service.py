from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

from models.mf.model import MFModel
from models.mf.trainer import CollaborativeFilterTrainer


class CollaborativeFilteringService:
    def __init__(self,
                 device: Optional[Union[str, torch.device]] = None,
        movie_data: pd.DataFrame = None,

    ):
        self.device = device

        self.movie_data = movie_data
        self.movie_id2idx = movie_data.set_index("movieId")["item_idx"].to_dict()
        self.movie_idx2id = movie_data.set_index("item_idx")["movieId"].to_dict()
        self.movie_id2title = movie_data.set_index("movieId")["title"].to_dict()


        # Will be populated after loading
        self.is_fit = False
        self.latent_dim = None
        self.model = None
        self.use_bias = None

    def predict(self, user_id, item_id):
        return self.model.predict(user_id, item_id)

    def recommend(self, user_id, top_n=10):
        return self.model.recommend(user_id, top_n)

    def load(self, path: str, device: Optional[Union[str, torch.device]] = None):
        """
        Load a saved MF model.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.latent_dim = ckpt["latent_dim"]
        self.is_fit = ckpt["is_fit"]

        model = MFModel(
            num_users=int(ckpt["num_users"]),
            num_items=int(ckpt["num_items"]),
            latent_dim=ckpt["latent_dim"],
            use_bias=bool(ckpt["use_bias"]),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()
        self.model = model

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
        Requires movieId to exist in items mapping!
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

    def _require_fit(self) -> None:
        """Helper to check if model is fit before allowing predict/recommend."""
        if not self.is_fit or self.model is None:
            raise RuntimeError(f"Model is not fit yet. model = {self.model}, _is_fit = {self.is_fit}")

    def get_movie_idx(self, id: int) -> int:
        """Helper to convert internal item_idx back to original movieId."""
        return self.movie_id2idx.get(id, -1)

    def get_movie_id(self, idx: int) -> int:
        """Helper to convert internal item_idx back to original movieId."""
        return self.movie_idx2id.get(idx, -1)
    def get_movie_title(self, id: int) -> str:
        """Helper to get movie title from original movieId."""
        return self.movie_id2title.get(id, "Unknown Title")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MOVIE_DATA_DIR = PROJECT_ROOT / ".." /"data" / "processed" / "mf_items.csv"
    MODEL_DIR = PROJECT_ROOT / "mf" / "mf_model_small.pt"


    movie_data = pd.read_csv(MOVIE_DATA_DIR)
    service = CollaborativeFilteringService(movie_data=movie_data)
    service.load(str(MODEL_DIR))

    recs = service.recommend_from_ratings(
        ratings=[
            (4993, 5.0),  # LOTR (The Fellowship of the Ring)
            (5816, 5.0),  # Harry Potter 2
            (1721, 1.0),  # Titanic
        ],
        top_n=10,
        return_scores = True

    )

    for movie_id, score in recs:
        print(f" movieId={movie_id}  score={score:.3f} title={service.get_movie_title(movie_id)}")

