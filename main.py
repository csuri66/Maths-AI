from torch import device
import torch
from torch.nn import BCEWithLogitsLoss
from torch_geometric.loader import DataLoader
import GAT
import data_generator
import torch.nn.functional as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx

def best_pairing_for_selected_nodes(selected_nodes, edge_index, edge_probs):
    """
    selected_nodes: pl. [0, 2, 4, 5, 7, 8, 10, 12, 15, 18]
    edge_index: torch.Tensor [2, E]
    edge_probs: torch.Tensor [E] vagy [E, 1]

    Visszatér:
      matching_edges: lista [(u, v, prob, best_dir, edge_idx), ...]
      unmatched: lista [node, ...]
    """
    selected_nodes = list(selected_nodes)
    selected_set = set(selected_nodes)

    ei = edge_index.detach().cpu()
    probs = edge_probs.detach().cpu().view(-1)

    # Irányított -> irányítatlan összevonás
    # Minden irányítatlan élhez a jobbik irány valószínűségét tartjuk meg
    pair_best = {}  # (u,v) -> (prob, src, dst, edge_idx)

    E = ei.size(1)
    for e in range(E):
        src = int(ei[0, e].item())
        dst = int(ei[1, e].item())

        # Csak a kijelölt csúcsok közötti élek érdekelnek
        if src not in selected_set or dst not in selected_set:
            continue
        if src == dst:
            continue

        u, v = sorted((src, dst))
        p = float(probs[e].item())

        if (u, v) not in pair_best or p > pair_best[(u, v)][0]:
            pair_best[(u, v)] = (p, src, dst, e)

    # Súlyozott irányítatlan gráf építése
    G = nx.Graph()
    G.add_nodes_from(selected_nodes)

    for (u, v), (p, src, dst, e) in pair_best.items():
        G.add_edge(u, v, weight=p, best_dir=(src, dst), edge_idx=e)

    # Maximum súlyú matching
    matching = nx.max_weight_matching(G, maxcardinality=True, weight="weight")

    matching_edges = []
    used_nodes = set()

    for u, v in matching:
        a, b = sorted((u, v))
        data = G[a][b]
        matching_edges.append((a, b, data["weight"], data["best_dir"], data["edge_idx"]))
        used_nodes.add(a)
        used_nodes.add(b)

    unmatched = [n for n in selected_nodes if n not in used_nodes]

    matching_edges.sort(key=lambda x: (x[0], x[1]))
    return matching_edges, unmatched


def is_stable_matching(match_a, prefs_a, prefs_b):
    """
    match_a: dict, pl. {'a1': 'b2', 'a2': 'b1', ...}
    prefs_a: dict, pl. {'a1': ['b1','b2','b3'], ...}
    prefs_b: dict, pl. {'b1': ['a2','a1','a3'], ...}

    Visszaad:
      (stable, blocking_pairs)
    """

    match_b = {b: a for a, b in match_a.items() if b is not None}

    rank_a = {
        a: {b: i for i, b in enumerate(pref_list)}
        for a, pref_list in prefs_a.items()
    }
    rank_b = {
        b: {a: i for i, a in enumerate(pref_list)}
        for b, pref_list in prefs_b.items()
    }

    blocking_pairs = []

    for a in prefs_a:
        current_b = match_a.get(a, None)

        for b in prefs_a[a]:
            if current_b == b:
                continue

            current_a_for_b = match_b.get(b, None)

            a_prefers_b = (
                current_b is None or rank_a[a][b] < rank_a[a][current_b]
            )
            b_prefers_a = (
                current_a_for_b is None or rank_b[b][a] < rank_b[b][current_a_for_b]
            )

            if a_prefers_b and b_prefers_a:
                blocking_pairs.append((a, b))

    return len(blocking_pairs) == 0, blocking_pairs

group_size=6

raw_train_graphs =[]

# Tanító
for i in range(1000):
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


model = GAT.GATEdgeClassifier(train_data[0].x.size(-1), 16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
pos = sum(data.edge_attr.sum().item() for data in train_data)
total = sum(data.edge_attr.numel() for data in train_data)
neg = total - pos
pos_weight = torch.tensor([neg / pos], dtype=torch.float)
criterion = torch.nn.BCEWithLogitsLoss()


for data in train_data:
    optimizer.zero_grad()
    logits = model(data)
    #print("Logits mean/std:", logits.mean().item(), logits.std().item())
    #print("Labels mean:", data.edge_attr.float().mean().item())
    loss =criterion(logits, data.edge_attr.float())
    loss.backward()
    optimizer.step()



model.eval()
total_loss = 0
total_edges = 0
for data in val_data:
    logits = model(data)
    loss = criterion(logits, data.edge_attr.float())

    total_loss += loss.item() * data.edge_attr.numel()  # vagy data.num_edges ha teljes
    total_edges += data.edge_attr.numel()

avg_loss = total_loss / total_edges if total_edges > 0 else 0  # normalizáld megfelelően
print(avg_loss)

proba = data_generator.generate_graph(group_size)
sel_nodes = proba.nodes
proba = data_generator.graph_to_pyg_data(proba,group_size,verbose=True)
proba_logits = model(proba)
probs = torch.sigmoid(proba_logits)
print(probs)
preds = (probs > 0.5).int()
topk_edges = torch.topk(probs, k=10).indices
edge_scores = torch.topk(probs, k=10).values
#print(topk_edges)

src_nodes = proba.edge_index[0, topk_edges]  # forrás csúcsok
dst_nodes = proba.edge_index[1, topk_edges]  # cél csúcsok

#for i in range(10):
    #print(f"#{i+1}: {src_nodes[i].item()} → {dst_nodes[i].item()} "
          #f"(p={edge_scores[i].item():.3f})")



pairs, unmatched = best_pairing_for_selected_nodes(
    selected_nodes=sel_nodes,
    edge_index=proba.edge_index,
    edge_probs=probs
)

print("Kiválasztott párok:")
pair_dict= {}
for u, v, p, best_dir, edge_idx in pairs:
    pair_dict[u]=v
    print(f"{u} -- {v} | p={p:.4f} | jobb irány: {best_dir[0]}->{best_dir[1]} | edge_idx={edge_idx}")

print("Pár nélkül maradt csúcsok:", unmatched)
print(is_stable_matching(pair_dict,proba.proposer_pref,proba.proposee_pref))