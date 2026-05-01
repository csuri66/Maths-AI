import random

import torch
import GAT
import data_generator
import networkx as nx
import torch.nn.functional as F
import copy

def best_pairing_for_selected_nodes(selected_nodes, edge_index, edge_probs):
    selected_nodes = list(selected_nodes)
    selected_set = set(selected_nodes)

    ei = edge_index.detach().cpu()
    probs = edge_probs.detach().cpu().view(-1)

    G = nx.Graph()
    G.add_nodes_from(selected_nodes)

    E = ei.size(1)
    for e in range(E):
        src = int(ei[0, e].item())
        dst = int(ei[1, e].item())

        if src not in selected_set or dst not in selected_set:
            continue
        if src == dst:
            continue

        p = float(probs[e].item())

        G.add_edge(src, dst, weight=p, edge_idx=e)

    matching = nx.max_weight_matching(G, maxcardinality=True, weight="weight")

    matching_edges = []
    used_nodes = set()

    for u, v in matching:
        a, b = sorted((u, v))
        data = G[a][b]
        matching_edges.append((a, b, data["weight"], data["edge_idx"]))
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

    return len(blocking_pairs) == 0


def scatter_sum_1d(values, index, n):
    out = values.new_zeros(n)
    out.index_add_(0, index, values)
    return out


def soft_current_rank(probs, node_index, edge_rank, num_nodes):

    mass = scatter_sum_1d(probs, node_index, num_nodes)

    deg = scatter_sum_1d(torch.ones_like(probs), node_index, num_nodes)
    unmatched_rank = deg

    weighted_rank = scatter_sum_1d(probs * edge_rank, node_index, num_nodes)
    unmatched_mass = (1.0 - mass).clamp(min=0.0)

    denom = (mass + unmatched_mass).clamp(min=1e-8)
    current_rank = (weighted_rank + unmatched_mass * unmatched_rank) / denom
    return current_rank, mass



def stable_matching_loss(
    logits,
    edge_label,
    src,
    dst,
    rank_src,
    rank_dst,
    n_left,
    n_right,
    lambda_match,
    lambda_stab,
    tau
):
    probs = torch.sigmoid(logits)

    loss_edge = F.binary_cross_entropy_with_logits(
            logits, edge_label
    )



    left_mass = scatter_sum_1d(probs, src, n_left)
    right_mass = scatter_sum_1d(probs, dst, n_right)

    loss_match = (
        F.relu(left_mass - 1.0).pow(2).mean() +
        F.relu(right_mass - 1.0).pow(2).mean()
    )

    curr_rank_left, _ = soft_current_rank(probs, src, rank_src, n_left)
    curr_rank_right, _ = soft_current_rank(probs, dst, rank_dst, n_right)

    better_for_left = torch.sigmoid((curr_rank_left[src] - rank_src.float()) / tau)
    better_for_right = torch.sigmoid((curr_rank_right[dst] - rank_dst.float()) / tau)

    not_selected = 1.0 - probs

    loss_stab = (not_selected * better_for_left * better_for_right).mean()

    loss = loss_edge + lambda_match * loss_match + lambda_stab * loss_stab

    return loss



def train(global_step,best_val_loss,stop_training,lambda_match,lambda_stab,tau):
    for epoch in range(500):
        model.train()

        for data in train_data:
            optimizer.zero_grad()
            logits = model(data)
            src_global = data.edge_index[0]
            dst_global = data.edge_index[1]
            dst_local = data.edge_index[1] - group_size
            proposer_rank = {
                u: {partner: rank for rank, partner in enumerate(pref_list)}
                for u, pref_list in data.proposer_pref.items()
            }
            proposee_rank = {
                u: {partner: rank for rank, partner in enumerate(pref_list)}
                for u, pref_list in data.proposee_pref.items()
            }
            rank_src = torch.tensor(
                [proposer_rank[int(u)][int(vg)] for u, vg in zip(src_global.tolist(), dst_global.tolist())],
                dtype=torch.float,
                device=src_global.device
            )
            rank_dst = torch.tensor(
                [proposee_rank[int(vg)][int(u)] for u, vg in zip(src_global.tolist(), dst_global.tolist())],
                dtype=torch.float,
                device=src_global.device
            )
            loss_dict = stable_matching_loss(
                logits=logits,
                edge_label=data.edge_y.float(),
                src=src_global,
                dst=dst_local,
                rank_src=rank_src,
                rank_dst=rank_dst,
                n_left=group_size,
                n_right=group_size,
                lambda_match=lambda_match,
                lambda_stab=lambda_stab,
                tau=tau
            )
            loss = loss_dict
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % eval_every == 0:
                model.eval()
                val_loss_sum = 0.0
                val_count = 0

                with torch.no_grad():
                    for val in val_data:
                        val_logits = model(val)
                        src_global = val.edge_index[0]
                        dst_global = val.edge_index[1]
                        dst_local = val.edge_index[1] - group_size
                        proposer_rank = {
                            u: {partner: rank for rank, partner in enumerate(pref_list)}
                            for u, pref_list in val.proposer_pref.items()
                        }
                        proposee_rank = {
                            u: {partner: rank for rank, partner in enumerate(pref_list)}
                            for u, pref_list in val.proposee_pref.items()
                        }
                        rank_src = torch.tensor(
                            [proposer_rank[int(u)][int(vg)] for u, vg in zip(src_global.tolist(), dst_global.tolist())],
                            dtype=torch.float,
                            device=src_global.device
                        )
                        rank_dst = torch.tensor(
                            [proposee_rank[int(vg)][int(u)] for u, vg in zip(src_global.tolist(), dst_global.tolist())],
                            dtype=torch.float,
                            device=src_global.device
                        )
                        val_loss  = stable_matching_loss(
                            logits=val_logits,
                            edge_label=val.edge_y.float(),
                            src=src_global,
                            dst=dst_local,
                            rank_src=rank_src,
                            rank_dst=rank_dst,
                            n_left=group_size,
                            n_right=group_size,
                            lambda_match=lambda_match,
                            lambda_stab=lambda_stab,
                            tau=tau
                        )
                        val_loss_sum += val_loss
                        val_count += 1

                mean_val_loss = val_loss_sum / max(val_count, 1)

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
            print(global_step)
            print(best_val_loss)
            break



class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, val_loss, model):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop



def train_with_early_stopping(train_data,val_data,model):
    early_stopper = EarlyStopping(patience=10, min_delta=1e-5)
    for epoch in range(20000):
        model.train()
        optimizer.zero_grad()
        src_global = train_data.edge_index[0]
        dst_global = train_data.edge_index[1]
        dst_local = train_data.edge_index[1] - group_size
        proposer_rank = {
            u: {partner: rank for rank, partner in enumerate(pref_list)}
            for u, pref_list in train_data.proposer_pref.items()
        }
        proposee_rank = {
            u: {partner: rank for rank, partner in enumerate(pref_list)}
            for u, pref_list in train_data.proposee_pref.items()
        }
        rank_src = torch.tensor(
            [proposer_rank[int(u)][int(vg)] for u, vg in zip(src_global.tolist(), dst_global.tolist())],
            dtype=torch.float,
            device=src_global.device
        )
        rank_dst = torch.tensor(
            [proposee_rank[int(vg)][int(u)] for u, vg in zip(src_global.tolist(), dst_global.tolist())],
            dtype=torch.float,
            device=src_global.device
        )
        logits = model(train_data)
        train_loss = stable_matching_loss(
            logits=logits,
            edge_label=train_data.edge_y.float(),
            src=src_global,
            dst=dst_local,
            rank_src=rank_src,
            rank_dst=rank_dst,
            n_left=group_size,
            n_right=group_size,
            lambda_match=0.5,
            lambda_stab=0.5,
            tau=0.3
        )

        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_data)
            src_global = val_data.edge_index[0]
            dst_global = val_data.edge_index[1]
            dst_local = val_data.edge_index[1] - group_size
            proposer_rank = {
                u: {partner: rank for rank, partner in enumerate(pref_list)}
                for u, pref_list in train_data.proposer_pref.items()
            }
            proposee_rank = {
                u: {partner: rank for rank, partner in enumerate(pref_list)}
                for u, pref_list in train_data.proposee_pref.items()
            }
            rank_src = torch.tensor(
                [proposer_rank[int(u)][int(vg)] for u, vg in zip(src_global.tolist(), dst_global.tolist())],
                dtype=torch.float,
                device=src_global.device
            )
            rank_dst = torch.tensor(
                [proposee_rank[int(vg)][int(u)] for u, vg in zip(src_global.tolist(), dst_global.tolist())],
                dtype=torch.float,
                device=src_global.device
            )
            val_loss = stable_matching_loss(
                logits=val_logits,
                edge_label=val_data.edge_y.float(),
                src=src_global,
                dst=dst_local,
                rank_src=rank_src,
                rank_dst=rank_dst,
                n_left=group_size,
                n_right=group_size,
                lambda_match=0.5,
                lambda_stab=0.5,
                tau=0.3
            )

        stop = early_stopper.step(val_loss.item(), model)

        if stop:
            print("Early stopping triggered.")
            break

    model.load_state_dict(early_stopper.best_state)


group_size=3


train_data = []

# Tanító

for i in range(12):
    train_data.append(data_generator.graph_to_pyg_data_low_diff(data_generator.generate_graph(group_size), group_size))

for i in range(6):
    train_data.append(data_generator.graph_to_pyg_data_dominant_proposee(data_generator.generate_graph(group_size), group_size))

for i in range(6):
    train_data.append(data_generator.graph_to_pyg_data_dominant_proposer(data_generator.generate_graph(group_size), group_size))

for i in range(12):
    train_data.append(data_generator.graph_to_pyg_data_high_diff(data_generator.generate_graph(group_size), group_size))

# Validációs
val_graphs = []
for i in range(100):
    val_graphs.append(data_generator.generate_graph(group_size))

val_data = []
for graph in val_graphs:
    val_data.append(data_generator.graph_to_pyg_data_random(graph,group_size))


all_train_x = torch.cat([g.x for g in train_data], dim=0)
all_train_edge = torch.cat([g.edge_attr for g in train_data], dim=0)

mean = all_train_x.mean(dim=0, keepdim=True)
mean_edge = all_train_edge.mean(dim=0, keepdim=True)

std = all_train_x.std(dim=0, keepdim=True)
std_edge =  all_train_edge.std(dim=0, keepdim=True)

for g in train_data:
    g.x = (g.x - mean) / std
    g.edge_attr=(g.edge_attr - mean_edge) / std_edge

for g in val_data:
    g.x = (g.x - mean) / std
    g.edge_attr = (g.edge_attr - mean_edge) / std_edge


model = GAT.GATEdgeClassifier(train_data[0].x.size(-1), 16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
pos = sum(data.edge_y.sum().item() for data in train_data)
total = sum(data.edge_y.numel() for data in train_data)
neg = total - pos
pos_weight = torch.tensor(neg/pos, dtype=torch.float)




patience_counter = 0
best_path = "best_model.pt"
patience = 30
min_delta = 1e-4
eval_every = 10

best_val_loss = float("inf")
bad_checks = 0
global_step = 0
stop_training = False

#train(global_step,best_val_loss,stop_training,0.8,0.4,0.05)

model.load_state_dict(torch.load("best_model.pt", weights_only=True))



model.eval()
"""
total_loss = 0
total_edges = 0
with torch.no_grad():
    for data in val_data:
        logits = model(data)
        loss = F.binary_cross_entropy_with_logits(
            logits, data.edge_y
        )

        total_loss += loss.item() * data.edge_y.numel()
        total_edges += data.edge_y.numel()

avg_loss = total_loss / total_edges if total_edges > 0 else 0
print(avg_loss)
"""
group_size = 3

good = 0
bad = 0
for i in range(5000):
    acc_graph = data_generator.generate_graph(group_size)
    acc_data = data_generator.graph_to_pyg_data_low_diff(acc_graph, group_size)
    acc_data.x = (acc_data.x - mean) / std
    acc_data.edge_attr = (acc_data.edge_attr - mean_edge) / std_edge
    logits = model(acc_data)
    probs = torch.sigmoid(logits)
    pairs, unmatched = best_pairing_for_selected_nodes(
        selected_nodes=acc_graph.nodes,
        edge_index=acc_data.edge_index,
        edge_probs=probs
    )
    pair_dict = {}
    for u, v, p, edge_idx in pairs:
        pair_dict[u] = v
    if is_stable_matching(pair_dict, acc_data.proposer_pref, acc_data.proposee_pref):
        good += 1
    else:
        bad += 1

print(good/(good+bad))