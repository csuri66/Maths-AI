from __future__ import annotations

import torch
import torch.nn as nn


class GreedyMatchingDecoder(nn.Module):
    """
    Minimális, tanulható greedy matching decoder.

    Input:
        edge_embedding: [E, D]

    Output:
        edge_logits: [E]

    A decoder csak azt tanulja meg, hogy egy edge embedding
    alapján mennyire érdemes az élt választani.

    A degree <= 1 kényszert nem az MLP, hanem a greedy decode
    közbeni hard mask garantálja.
    """

    def __init__(
        self,
        edge_embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.edge_scorer = nn.Sequential(
            nn.Linear(edge_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        edge_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        edge_embedding: [E, D]

        return:
            edge_logits: [E]

        A kimenet LOGIT, nem sigmoid valószínűség.
        """

        return self.edge_scorer(
            edge_embedding
        ).squeeze(-1)