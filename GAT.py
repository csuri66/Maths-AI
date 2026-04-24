import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=1,edge_dim=1,concat=False)
        self.skip1 = torch.nn.Linear(in_channels, hidden_channels)
        self.conv2 = GATv2Conv(
            hidden_channels,
            hidden_channels,
            heads=1,
            edge_dim=1,
            concat=False
        )
        self.skip2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
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
        x = F.relu(x + self.skip1(x0))

        x1 = x
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x + self.skip2(x1))

        src, dst = edge_index
        edge_emb = torch.cat([x[src], x[dst]], dim=-1)
        logits = self.edge_mlp(edge_emb).squeeze(-1)
        return logits

