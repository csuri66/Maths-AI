
import torch
import GAT
import data_generator
import networkx as nx

def best_pairing_for_selected_nodes(selected_nodes, edge_index, edge_probs):

    selected_nodes = list(selected_nodes)
    selected_set = set(selected_nodes)

    ei = edge_index.detach().cpu()
    probs = edge_probs.detach().cpu().view(-1)

    pair_best = {}  # (u,v) -> (prob, src, dst, edge_idx)

    E = ei.size(1)
    for e in range(E):
        src = int(ei[0, e].item())
        dst = int(ei[1, e].item())

        if src not in selected_set or dst not in selected_set:
            continue
        if src == dst:
            continue

        u, v = sorted((src, dst))
        p = float(probs[e].item())

        if (u, v) not in pair_best or p > pair_best[(u, v)][0]:
            pair_best[(u, v)] = (p, src, dst, e)

    G = nx.Graph()
    G.add_nodes_from(selected_nodes)

    for (u, v), (p, src, dst, e) in pair_best.items():
        G.add_edge(u, v, weight=p, best_dir=(src, dst), edge_idx=e)

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

group_size=3

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
pos = sum(data.edge_attr.sum().item() for data in train_data)
total = sum(data.edge_attr.numel() for data in train_data)
neg = total - pos
pos_weight = torch.tensor(3.0, dtype=torch.float)
#criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion = torch.nn.BCEWithLogitsLoss()

patience = 10
min_delta = 1e-4
best_val_loss = float("inf")
patience_counter = 0
best_path = "best_model.pt"

patience = 5
min_delta = 1e-4
eval_every = 10

best_val_loss = float("inf")
bad_checks = 0
global_step = 0
stop_training = False

for epoch in range(2):
    model.train()

    for data in train_data:
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, data.edge_attr.float())
        loss.backward()
        optimizer.step()

        global_step += 1

        if global_step % eval_every == 0:
            model.eval()
            val_loss_sum = 0.0
            val_count = 0

            with torch.no_grad():
                for val_batch in val_data:
                    val_logits = model(val_batch)
                    val_loss = criterion(val_logits, val_batch.edge_attr.float())
                    val_loss_sum += val_loss.item()
                    val_count += 1

            mean_val_loss = val_loss_sum / max(val_count, 1)
            print(f"step={global_step}, val_loss={mean_val_loss:.4f}")

            if mean_val_loss < best_val_loss - min_delta:
                best_val_loss = mean_val_loss
                bad_checks = 0
                torch.save(model.state_dict(), "best_model.pt")
            else:
                bad_checks += 1

            model.train()

            if bad_checks >= patience:
                print("Early stopping")
                stop_training = True
                break

    if stop_training:
        break

model.load_state_dict(torch.load("best_model.pt", weights_only=True))



model.eval()
total_loss = 0
total_edges = 0
for data in val_data:
    logits = model(data)
    loss = criterion(logits, data.edge_attr.float())

    total_loss += loss.item() * data.edge_attr.numel()
    total_edges += data.edge_attr.numel()

avg_loss = total_loss / total_edges if total_edges > 0 else 0
print(avg_loss)

proba = data_generator.generate_graph(group_size)
sel_nodes = proba.nodes
proba = data_generator.graph_to_pyg_data(proba,group_size,verbose=True)
proba_logits = model(proba)
probs = torch.sigmoid(proba_logits)
print(probs)
preds = (probs > 0.5).int()



pairs, unmatched = best_pairing_for_selected_nodes(
    selected_nodes=sel_nodes,
    edge_index=proba.edge_index,
    edge_probs=probs
)

print("Kiválasztott párok:")
pair_dict= {}
for u, v, p, best_dir, edge_idx in pairs:
    pair_dict[u]=v
    print(f"{u} -- {v} | p={p:.4f}  | edge_idx={edge_idx}")

print("Pár nélkül maradt csúcsok:", unmatched)
print(is_stable_matching(pair_dict,proba.proposer_pref,proba.proposee_pref))