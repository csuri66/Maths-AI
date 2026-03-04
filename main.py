from torch import device
import torch
from torch_geometric.loader import DataLoader
import GAT
import data_generator
import torch.nn.functional as F

group_size=10

raw_train_graphs =[]

# Tanító
for i in range(1000):
    raw_train_graphs.append(data_generator.generate_graph(group_size))

train_data = []
for graph in raw_train_graphs:
    train_data.append(data_generator.graph_to_pyg_data(graph,group_size))

# Validációs
val_graphs = []
for i in range(100):
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
    loss = F.binary_cross_entropy_with_logits(logits, data.edge_label.float())
    loss.backward()
    optimizer.step()


val_loader = DataLoader(val_data, batch_size=16, shuffle=True)
model.eval()
total_loss = 0
total_edges = 0
for data in val_loader:  # batch_size >1, több gráf
    logits = model(data)
    loss = F.binary_cross_entropy_with_logits(logits, data.edge_label.float())

    total_loss += loss.item() * data.edge_label.numel()  # vagy data.num_edges ha teljes
    total_edges += data.edge_label.numel()

avg_loss = total_loss / total_edges if total_edges > 0 else 0  # normalizáld megfelelően
print(avg_loss)

# TODO : duplázva vannak az élek de az edge atributomok nem