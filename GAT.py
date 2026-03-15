import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels)
        self.gat2 = GATConv(hidden_channels, out_channels)
        self.lin = torch.nn.Linear(out_channels * 2, 1)  # src + dst + edge_attr

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.gat1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gat2(x, edge_index, edge_attr)

        src, dst = edge_index
        edge_emb = torch.cat([x[src], x[dst]], dim=-1)
        edge_pred = torch.sigmoid(self.lin(edge_emb)).squeeze()
        return edge_pred

