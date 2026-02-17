import torch
from torch_geometric.nn import GATConv

class GATAttention(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x, _ = self.conv1(x, edge_index, return_attention_weights=True)
        x, (edge_index_out, alpha) = self.conv2(x, edge_index, return_attention_weights=True)
        return x, alpha
