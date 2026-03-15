from torch import device
import torch
from torch_geometric.loader import DataLoader
import GAT
import data_generator
import torch.nn.functional as F

import gale_shapley

group_size=10

raw_train_graphs =[]

# Tanító
for i in range(100):
    raw_train_graphs.append(data_generator.generate_graph(group_size))

train_data = []
for graph in raw_train_graphs:
    train_data.append(data_generator.graph_to_pyg_data(graph,group_size))

# Validációs
val_graphs = []
for i in range(1000):
    val_graphs.append(data_generator.generate_graph(group_size))

val_data = []
for graph in val_graphs:
    val_data.append(data_generator.graph_to_pyg_data(graph,group_size))



model = GAT.GATEdgeClassifier(-1, 16, 16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


loader = DataLoader(train_data, batch_size=16, shuffle=True)

for data in loader:
    optimizer.zero_grad()
    logits = model(data)
    print("Logits mean/std:", logits.mean().item(), logits.std().item())
    print("Labels mean:", data.edge_attr.float().mean().item())  # ~0.5 legyen!
    print("Pred probs:", torch.sigmoid(logits[:10]))
    loss = F.binary_cross_entropy_with_logits(logits, data.edge_attr.float())
    loss.backward()
    optimizer.step()



val_loader = DataLoader(val_data, batch_size=16, shuffle=True)
model.eval()
total_loss = 0
total_edges = 0
for data in val_loader:
    logits = model(data)
    loss = F.binary_cross_entropy_with_logits(logits, data.edge_attr.float())

    total_loss += loss.item() * data.edge_attr.numel()  # vagy data.num_edges ha teljes
    total_edges += data.edge_attr.numel()

avg_loss = total_loss / total_edges if total_edges > 0 else 0  # normalizáld megfelelően
print(avg_loss)

proba = data_generator.generate_graph(group_size)
proba = data_generator.graph_to_pyg_data(proba,group_size)
proba_logits = model(proba)
probs = torch.sigmoid(proba_logits)
preds = (probs > 0.5).int()
topk_edges = torch.topk(probs, k=10).indices
edge_scores = torch.topk(probs, k=10).values
print(topk_edges)

src_nodes = proba.edge_index[0, topk_edges]  # forrás csúcsok
dst_nodes = proba.edge_index[1, topk_edges]  # cél csúcsok

for i in range(10):
    print(f"#{i+1}: {src_nodes[i].item()} → {dst_nodes[i].item()} "
          f"(p={edge_scores[i].item():.3f})")