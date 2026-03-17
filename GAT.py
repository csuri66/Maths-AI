import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=2, concat=False)
        self.skip1 = torch.nn.Linear(in_channels, hidden_channels)

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x0 = x
        x = self.gat1(x, edge_index)
        x = F.relu(x + self.skip1(x0))

        src, dst = edge_index
        edge_emb = torch.cat([x[src], x[dst]], dim=-1)
        logits = self.edge_mlp(edge_emb).squeeze(-1)
        return logits

