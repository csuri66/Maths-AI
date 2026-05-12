import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.1):
        super().__init__()

        self.heads = 8
        self.out_dim = hidden_channels * self.heads

        self.conv1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=self.heads,
            edge_dim=1,
            concat=True,
            dropout=dropout
        )

        self.skip1 = torch.nn.Linear(in_channels, self.out_dim)

        self.conv2 = GATv2Conv(
            self.out_dim,
            hidden_channels,
            heads=self.heads,
            edge_dim=1,
            concat=True,
            dropout=dropout
        )

        self.skip2 = torch.nn.Linear(self.out_dim, self.out_dim)

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.out_dim * 4 + 1, self.out_dim),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(self.out_dim, hidden_channels),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        edge_attr = torch.as_tensor(
            data.edge_attr,
            dtype=torch.float,
            device=x.device
        ).view(-1, 1)

        x0 = x
        x = self.conv1(x, edge_index, edge_attr)
        x = x + self.skip1(x0)
        x = F.relu(x)

        x1 = x
        x = self.conv2(x, edge_index, edge_attr)
        x = x + self.skip2(x1)
        x = F.relu(x)

        src, dst = edge_index

        x_src = x[src]
        x_dst = x[dst]
        x_diff = torch.abs(x_src - x_dst)
        x_mul = x_src * x_dst

        edge_emb = torch.cat(
            [x_src, x_dst, x_diff, x_mul, edge_attr],
            dim=-1
        )

        logits = self.edge_mlp(edge_emb).squeeze(-1)
        return logits

