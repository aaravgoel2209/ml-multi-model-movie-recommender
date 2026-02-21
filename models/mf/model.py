import torch
from torch import nn


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
