
import torch
from torch_geometric.data import Data
import GAT
import data_generator
import torch.nn.functional as F
import gale_shapley
import random

def build_current_data_from_original(original_data, remaining_orig_ids):
    device = original_data.x.device
    num_nodes = original_data.x.size(0)

    if not isinstance(remaining_orig_ids, torch.Tensor):
        remaining_orig_ids = torch.tensor(
            list(remaining_orig_ids), dtype=torch.long, device=device
        )
    else:
        remaining_orig_ids = remaining_orig_ids.to(device)

    keep_node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    keep_node_mask[remaining_orig_ids] = True

    kept_orig_id = torch.arange(num_nodes, device=device)[keep_node_mask]

    src, dst = original_data.edge_index
    keep_edge_mask = keep_node_mask[src] & keep_node_mask[dst]

    old_to_new = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    old_to_new[kept_orig_id] = torch.arange(kept_orig_id.numel(), device=device)

    new_edge_index = original_data.edge_index[:, keep_edge_mask]
    new_edge_index = old_to_new[new_edge_index]

    current_data = Data(
        x=original_data.x[keep_node_mask],
        edge_index=new_edge_index,
        edge_attr=original_data.edge_attr[keep_edge_mask]
        if getattr(original_data, "edge_attr", None) is not None else None,
        edge_y=original_data.edge_y[keep_edge_mask]
        if getattr(original_data, "edge_y", None) is not None else None,
        proposee_pref=filter_node_field(
            getattr(original_data, "proposee_pref", None),
            keep_node_mask,
            old_to_new
        ),
        proposer_pref=filter_node_field(
            getattr(original_data, "proposer_pref", None),
            keep_node_mask,
            old_to_new
        ),
        orig_id=kept_orig_id
    )

    return current_data

def best_pairing_for_selected_node(selected_nodes, edge_index, edge_probs):
    selected_set = set(selected_nodes)

    ei = edge_index.detach().cpu()
    probs = edge_probs.detach().cpu().view(-1)

    best_e = None
    best_p = float("-inf")
    best_src = None
    best_dst = None

    E = ei.size(1)
    for e in range(E):
        src = int(ei[0, e].item())
        dst = int(ei[1, e].item())

        if src not in selected_set or dst not in selected_set:
            continue
        if src == dst:
            continue

        p = float(probs[e].item())
        if p > best_p:
            best_p = p
            best_e = e
            best_src = src
            best_dst = dst

    if best_e is None:
        return None, None, None

    return best_e, (best_src, best_dst)

def filter_node_field(field, keep_node_mask, old_to_new=None):
    if field is None:
        return None

    if isinstance(field, torch.Tensor):
        return field[keep_node_mask]

    if isinstance(field, list):
        return [v for i, v in enumerate(field) if keep_node_mask[i].item()]

    if isinstance(field, tuple):
        return tuple(v for i, v in enumerate(field) if keep_node_mask[i].item())

    if isinstance(field, dict):
        new_field = {}
        for old_idx, value in field.items():
            if keep_node_mask[old_idx].item():
                new_idx = int(old_to_new[old_idx].item()) if old_to_new is not None else old_idx
                new_field[new_idx] = value
        return new_field

    return field

def remove_best_pair_from_data(data, best_src, best_dst):
    if not hasattr(data, "orig_id") or data.orig_id is None:
        data.orig_id = torch.arange(data.x.size(0), device=data.x.device)

    removed_original = data.orig_id[[best_src, best_dst]].tolist()

    device = data.edge_index.device
    num_nodes = data.x.size(0)

    keep_node_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    keep_node_mask[best_src] = False
    keep_node_mask[best_dst] = False

    src, dst = data.edge_index
    keep_edge_mask = keep_node_mask[src] & keep_node_mask[dst]

    old_to_new = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    old_to_new[keep_node_mask] = torch.arange(keep_node_mask.sum(), device=device)

    new_edge_index = data.edge_index[:, keep_edge_mask]
    new_edge_index = old_to_new[new_edge_index]

    new_data = Data(
        x=data.x[keep_node_mask],
        edge_index=new_edge_index,
        edge_attr=data.edge_attr[keep_edge_mask] if getattr(data, "edge_attr", None) is not None else None,
        edge_y=data.edge_y[keep_edge_mask] if getattr(data, "edge_y", None) is not None else None,
        proposee_pref=filter_node_field(getattr(data, "proposee_pref", None), keep_node_mask, old_to_new),
        proposer_pref=filter_node_field(getattr(data, "proposer_pref", None), keep_node_mask, old_to_new),
        orig_id=data.orig_id[keep_node_mask]
    )

    return new_data, removed_original

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
    tau,
    pos_weight
):
    probs = torch.sigmoid(logits)

    loss_edge = F.binary_cross_entropy_with_logits(
            logits, edge_label,pos_weight=pos_weight
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



def train(global_step,best_val_loss,stop_training,lambda_match,lambda_stab,tau,model,optimizer,train_data,group_size,eval_every,val_data,patience,min_delta,pos_weight):
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
            loss = stable_matching_loss(
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
                tau=tau,
                pos_weight=pos_weight
            )
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
                            tau=tau,
                            pos_weight=pos_weight
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
            print("BEST TOTAL LOSS")
            print(best_val_loss)
            break
    return best_val_loss



def main(training=False):

    group_size=3

    train_data = []
    for j in range(200):
        train_data.append(
            data_generator.graph_to_pyg_data_low_diff(data_generator.generate_graph_m(group_size), group_size))
        train_data.append(
            data_generator.graph_to_pyg_data_high_diff(data_generator.generate_graph_m(group_size), group_size))


    val_data = []
    for i in range(100):
        val_data.append(
            data_generator.graph_to_pyg_data_low_diff(data_generator.generate_graph_m(group_size), group_size))
        val_data.append(
            data_generator.graph_to_pyg_data_high_diff(data_generator.generate_graph_m(group_size), group_size))




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
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.015)
    pos = sum(data.edge_y.sum().item() for data in train_data)
    total = sum(data.edge_y.numel() for data in train_data)
    neg = total - pos
    pos_weight = torch.tensor(neg/pos, dtype=torch.float)


    patience = 10
    min_delta = 1e-4
    eval_every = 10

    best_val_loss = float("inf")
    global_step = 0
    stop_training = False


    if training:
        train(global_step, best_val_loss, stop_training, 0.25, 0.45, 0.1,model,optimizer,train_data, group_size, eval_every, val_data, patience, min_delta,pos_weight)
    model.load_state_dict(torch.load("best_model.pt", weights_only=True))



    model.eval()

    good=0
    bad=0
    with torch.no_grad():
        for i in range(10):

            pair_dict = {}

            group_size=random.randint(2,5)
            acc_graph = data_generator.generate_graph_m(group_size)

            structure = 0
            num = structure %4
            match num:
                case 0:
                    acc_data = data_generator.graph_to_pyg_data_low_diff(acc_graph, group_size)
                case 1:
                    acc_data = data_generator.graph_to_pyg_data_high_diff(acc_graph, group_size)
                case 2:
                    acc_data = data_generator.graph_to_pyg_data_dominant_proposer(acc_graph, group_size)
                case 3:
                    acc_data = data_generator.graph_to_pyg_data_dominant_proposee(acc_graph, group_size)

            acc_data.x = (acc_data.x - mean) / std
            acc_data.edge_attr = (acc_data.edge_attr - mean_edge) / std_edge

            logits = model(acc_data)
            probs = torch.sigmoid(logits)
            propr_pref=acc_data.proposer_pref
            prope_pref = acc_data.proposee_pref
            original_data = acc_data
            if not hasattr(original_data, "orig_id") or original_data.orig_id is None:
                original_data.orig_id = torch.arange(
                    original_data.x.size(0),
                    device=original_data.x.device
                )

            remaining_orig_ids = original_data.orig_id.clone()

            for i in range(group_size - 1):

                current_data = build_current_data_from_original(
                    original_data,
                    remaining_orig_ids
                )

                if i == 0:
                    current_probs = probs
                else:
                    logits = model(current_data)
                    current_probs = torch.sigmoid(logits)
                best_e, (best_src, best_dst) = best_pairing_for_selected_node(
                    selected_nodes=range(current_data.x.size(0)),
                    edge_index=current_data.edge_index,
                    edge_probs=current_probs
                )

                if best_e is None:
                    break

                orig_u, orig_v = current_data.orig_id[[best_src, best_dst]].tolist()
                pair_dict[orig_u] = orig_v

                remaining_orig_ids = remaining_orig_ids[
                    (remaining_orig_ids != orig_u) & (remaining_orig_ids != orig_v)
                    ]

                if acc_graph.has_node(orig_u):
                    acc_graph.remove_node(orig_u)
                if acc_graph.has_node(orig_v):
                    acc_graph.remove_node(orig_v)

            if remaining_orig_ids.numel() == 2:
                u, v = remaining_orig_ids.tolist()
                pair_dict[u] = v


            if gale_shapley.is_stable_matching(pair_dict, propr_pref,  prope_pref):
                good += 1
            else:
                bad += 1

    print("GRAPH LEVEL ACCURACY")
    print(good/(good+bad))

    graph_r = data_generator.generate_graph_r(2)
    data_r = data_generator.graph_to_pyg_data_r_random(graph_r,2)
    data_r.x = (data_r.x - mean) / std
    data_r.edge_attr = (data_r.edge_attr - mean_edge) / std_edge
    logits = model(data_r)
    probs = torch.sigmoid(logits)
    print(logits)

if __name__ == "__main__":
    training = False
    main(training)