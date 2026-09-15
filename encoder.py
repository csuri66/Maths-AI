from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float = 0.0,
    final_norm: bool = False,
) -> nn.Sequential:
    layers = [
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    ]
    if final_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


def scatter_max(
    values: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """
    values: [E, D]
    index:  [E], a célcsoport indexe minden élhez
    return: [dim_size, D]
    """
    out = values.new_full(
        (dim_size, values.size(-1)),
        float("-inf"),
    )

    expanded_index = index[:, None].expand(-1, values.size(-1))
    out.scatter_reduce_(
        dim=0,
        index=expanded_index,
        src=values,
        reduce="amax",
        include_self=True,
    )

    return torch.where(
        torch.isfinite(out),
        out,
        torch.zeros_like(out),
    )


@dataclass
class EncoderOutput:
    node_embedding: torch.Tensor
    edge_embedding: torch.Tensor
    auxiliary: Dict[str, torch.Tensor]


class EdgeAwarePreferenceBlock(nn.Module):
    """
    Egy réteg:
      1. Minden irányított élre készít message-et.
      2. Node-onként max-poololja az outgoing élüzeneteket.
      3. Frissíti a node embeddingeket.
      4. Frissíti a persistent edge embeddingeket.

    edge_index[0] = source agent
    edge_index[1] = target candidate partner
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        d = hidden_dim

        self.edge_message = make_mlp(
            in_dim=3 * d,
            hidden_dim=2 * d,
            out_dim=d,
            dropout=dropout,
        )

        self.node_update = make_mlp(
            in_dim=2 * d,
            hidden_dim=2 * d,
            out_dim=d,
            dropout=dropout,
        )

        self.edge_update = make_mlp(
            in_dim=5 * d,
            hidden_dim=2 * d,
            out_dim=d,
            dropout=dropout,
        )

        self.node_norm = nn.LayerNorm(d)
        self.edge_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        z0: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h:          [N, D] node embedding
        z:          [E, D] aktuális edge embedding
        z0:         [E, D] eredeti edge embedding
        edge_index: [2, E]
        """
        src, dst = edge_index
        num_nodes = h.size(0)

        # Az u -> v él üzenete:
        # forrás agent állapota + cél agent állapota + élállapot.
        message_input = torch.cat(
            [
                h[src],
                h[dst],
                z,
            ],
            dim=-1,
        )
        edge_message = self.edge_message(message_input)  # [E, D]

        # Egy agent a saját outgoing preferenciaéleinek legerősebb
        # strukturális jelét kapja meg.
        aggregated = scatter_max(
            values=edge_message,
            index=src,
            dim_size=num_nodes,
        )  # [N, D]

        node_delta = self.node_update(
            torch.cat([h, aggregated], dim=-1)
        )
        h_new = self.node_norm(
            h + self.dropout(node_delta)
        )

        # Persistent edge-state update.
        # z0 skip connection: az eredeti rank / compatibility feature
        # explicit módon elérhető marad minden rétegben.
        edge_input = torch.cat(
            [
                z,
                z0,
                h[src],
                h[dst],
                edge_message,
            ],
            dim=-1,
        )
        edge_delta = self.edge_update(edge_input)
        z_new = self.edge_norm(
            z + self.dropout(edge_delta)
        )

        return h_new, z_new


class StableMatchingEdgeEncoder(nn.Module):
    """
    Általános, edge-centric GNN encoder stable matchinghez.

    Kimenet:
      - node_embedding: [N, D]
      - edge_embedding: [E, D]
      - auxiliary headek: interpretálható segédpredikciók

    A decodernek elsődlegesen az edge_embeddinget add át.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.10,
        use_auxiliary_heads: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_auxiliary_heads = use_auxiliary_heads

        self.node_input = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.edge_input = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [
                EdgeAwarePreferenceBlock(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Az utolsó edge embedding egyszerre használja:
        # aktuális edge-state, source node, target node, első edge-state.
        self.edge_output = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        if use_auxiliary_heads:
            self.auxiliary_heads = nn.ModuleDict(
                {
                    "rank_score": nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                    ),
                    # Az él potenciálisan jó kölcsönös pár-e.
                    "mutual_quality": nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                    ),

                    # Mennyire versengő / konfliktusos az él.
                    "contention": nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                    ),

                    # Például egy downstream optimalizálási cél
                    # edge-utility komponense.
                    "utility": nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                    ),

                    # Az u, illetve v oldal alternatíváinak erőssége.
                    "replaceability": nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 2),
                    ),
                    "stable_membership_logit": nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                    ),
                }
            )
        else:
            self.auxiliary_heads = nn.ModuleDict()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> EncoderOutput:
        """
        x:          [N, node_feature_dim]
        edge_index: [2, E]
        edge_attr:  [E, edge_feature_dim]
        """
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                "edge_index shape must be [2, E]."
            )

        if edge_attr.size(0) != edge_index.size(1):
            raise ValueError(
                "edge_attr first dimension must equal E."
            )

        h = self.node_input(x)       # [N, D]
        z0 = self.edge_input(edge_attr)  # [E, D]
        z = z0
        for block in self.blocks:
            h, z = block(h, z, z0, edge_index)

        src, dst = edge_index

        edge_embedding = self.edge_output(
            torch.cat(
                [
                    z,
                    z0,
                    h[src],
                    h[dst],
                ],
                dim=-1,
            )
        )  # [E, D]

        auxiliary = {
            name: head(edge_embedding).squeeze(-1)
            if name != "replaceability"
            else head(edge_embedding)
            for name, head in self.auxiliary_heads.items()
        }

        return EncoderOutput(
            node_embedding=h,
            edge_embedding=edge_embedding,
            auxiliary=auxiliary,
        )