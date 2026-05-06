import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.7):
        super().__init__()

        self.conv1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=8,
            edge_dim=1,
            concat=False,
            dropout=dropout
        )
        self.skip1 = torch.nn.Linear(in_channels, hidden_channels)

        self.conv2 = GATv2Conv(
            hidden_channels,
            hidden_channels,
            heads=8,
            edge_dim=1,
            concat=False,
            dropout=dropout
        )
        self.skip2 = torch.nn.Linear(hidden_channels, hidden_channels)

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 4 + 1, hidden_channels),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, 1)
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
        x = F.elu(x)

        x1 = x
        x = self.conv2(x, edge_index, edge_attr)
        x = x + self.skip2(x1)
        x = F.elu(x)

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

