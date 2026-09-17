import ast
import torch
from torch_geometric.data import Data
import GAT
import data_generator
import torch.nn.functional as F
import gale_shapley
import random
import encoder
import decoder

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

def best_pairing_for_selected_node(
    selected_nodes,
    edge_index,
    edge_probs,
    preference_lists,
    tolerance=0.01
):
    selected_set = set(selected_nodes)

    ei = edge_index.detach().cpu()
    probs = edge_probs.detach().cpu().view(-1)

    # preference_lists[node] = [partner1, partner2, ...]
    pref_rank = {
        node: {partner: i for i, partner in enumerate(prefs)}
        for node, prefs in preference_lists.items()
    }

    def get_rank(node, partner):
        return pref_rank.get(node, {}).get(partner, float("inf"))

    valid_edges = []
    E = ei.size(1)

    for e in range(E):
        a = int(ei[0, e].item())
        b = int(ei[1, e].item())

        if a not in selected_set or b not in selected_set:
            continue
        if a == b:
            continue

        p = float(probs[e].item())
        valid_edges.append((e, a, b, p))

    if not valid_edges:
        return None, None, None

    # 1) Legnagyobb valószínűségű él
    base_e, u, v, base_p = max(valid_edges, key=lambda x: x[3])

    # Az eredeti él rangja a két végpont preferencialistájában
    base_rank_u = get_rank(u, v)
    base_rank_v = get_rank(v, u)

    # Alapból marad a base edge
    chosen_e, chosen_a, chosen_b, chosen_p = base_e, u, v, base_p
    chosen_pref_rank = float("inf")

    # 2) Nézzük a közeli, egyik végpontot megosztó éleket
    for e, a, b, p in valid_edges:
        if e == base_e:
            continue

        # csak a base edge-hez közeli valószínűségek érdekelnek
        if abs(p - base_p) > tolerance:
            continue

        better = False
        cand_pref_rank = float("inf")

        # Megosztja u-t a base edge-dzsel?
        if a == u or b == u:
            other = b if a == u else a
            if other != v:
                r = get_rank(u, other)
                if r < base_rank_u:
                    better = True
                    cand_pref_rank = min(cand_pref_rank, r)

        # Megosztja v-t a base edge-dzsel?
        if a == v or b == v:
            other = b if a == v else a
            if other != u:
                r = get_rank(v, other)
                if r < base_rank_v:
                    better = True
                    cand_pref_rank = min(cand_pref_rank, r)

        # Ha preferencia szerint jobb, akkor jelölt lehet
        if better:
            if (
                cand_pref_rank < chosen_pref_rank or
                (cand_pref_rank == chosen_pref_rank and p > chosen_p)
            ):
                chosen_e, chosen_a, chosen_b, chosen_p = e, a, b, p
                chosen_pref_rank = cand_pref_rank

    return chosen_e, (chosen_a, chosen_b)


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

    loss = loss_edge+ lambda_match * loss_match + lambda_stab * loss_stab

    return loss

def preference_lists_to_edge_ranks(
    preferences: dict[int, list[int]],
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Preferencialistákból [E] rank tensor.

    preferences[u] = [legjobb, ..., legrosszabb]

    output[e] = rankja annak a partnernek, amelyhez
    edge_index[:, e] = [u, v] tartozik.

    0 = legjobb partner.
    """
    rank_dict = {
        u: {
            v: rank
            for rank, v in enumerate(pref_list)
        }
        for u, pref_list in preferences.items()
    }

    values = []

    for edge_id in range(edge_index.size(1)):
        u = edge_index[0, edge_id].item()
        v = edge_index[1, edge_id].item()

        if u not in rank_dict:
            raise KeyError(
                f"A {u} node-hoz nincs preferencialista."
            )

        if v not in rank_dict[u]:
            raise KeyError(
                f"A {u} -> {v} él szerepel az edge_indexben, "
                "de v nincs u preferencialistájában."
            )

        values.append(float(rank_dict[u][v]))

    return torch.tensor(
        values,
        dtype=torch.float32,
        device=edge_index.device,
    )

def train(global_step,best_val_loss,stop_training,lambda_match,lambda_stab,tau,model,optimizer,train_data,group_size,eval_every,val_data,patience,min_delta):
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
                tau=tau
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
            print("BEST TOTAL LOSS")
            print(best_val_loss)
            break
    return best_val_loss


def encoder_loss(
    out,
    batch,
    pos_weight_membership: float = 10.0,
):
    """
    out:
      out.node_embedding           [N, D]
      out.edge_embedding           [E, D]
      out.auxiliary["mutual_quality"] [E]
      out.auxiliary["utility"]        [E]

    batch:
      batch.rank_better_idx       [P]
      batch.rank_worse_idx        [P]
      batch.mutual_target         [E]
      batch.member_a_target       [E]
      batch.member_b_target       [E]
    """

    z = out.edge_embedding

    # Egy irányított edge-rank score headet célszerű
    # magában az encoder modellben regisztrálni.
    rank_score = out.auxiliary["rank_score"]
    better = rank_score[batch.rank_better_idx]
    worse = rank_score[batch.rank_worse_idx]

    rank_loss = -F.logsigmoid(
        better - worse
    ).mean()

    mutual_pred = out.auxiliary["mutual_quality"]
    mutual_loss = F.smooth_l1_loss(
        mutual_pred,
        batch.edge_attr,
    )

    pos_weight = torch.tensor(
        pos_weight_membership,
        device=z.device,
    )


    total = (
        rank_loss
        +   mutual_loss
    )

    metrics = {
        "loss": total.detach(),
        "rank_loss": rank_loss.detach(),
        "mutual_loss": mutual_loss.detach(),
    }
    return total, metrics

from typing import Dict

@torch.no_grad()
def greedy_decode_pairs(
    edge_index: torch.Tensor,
    directed_logits: torch.Tensor,
    num_nodes: int,
    debug: bool = False,
) -> torch.Tensor:
    """
    Greedy one-to-one decoder irányított preferenciaélekhez.

    Input
    -----
    edge_index:
        [2, E], ahol edge_index[:, e] = [u, v] az u -> v él.

    directed_logits:
        [E], a decoder által adott nyers logitok.

    num_nodes:
        Node-ok száma.

    Output
    ------
    partner:
        [N], ahol partner[u] = v vagy -1.

    Megjegyzés:
    - Csak matching-feasibilityt garantál.
    - Stabilitást nem garantál általánosan.
    """
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(
            "edge_index shape-ja [2, E] legyen, "
            f"de ezt kaptam: {tuple(edge_index.shape)}"
        )

    if directed_logits.ndim != 1:
        raise ValueError(
            "directed_logits shape-ja [E] legyen, "
            f"de ezt kaptam: {tuple(directed_logits.shape)}"
        )

    if edge_index.size(1) != directed_logits.numel():
        raise ValueError(
            "Eltér az edge_index és directed_logits élszáma: "
            f"{edge_index.size(1)} vs {directed_logits.numel()}"
        )

    device = edge_index.device

    # GPU -> CPU: a Python dictionary és rendezés céljára.
    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()
    logits = directed_logits.detach().cpu().tolist()

    # Minden u -> v él logitja.
    directed_score: Dict[Tuple[int, int], float] = {}

    for u, v, score in zip(src, dst, logits):
        if u == v:
            continue

        # Ha duplikált edge van, a nagyobb score-t tartjuk meg.
        if (u, v) not in directed_score:
            directed_score[(u, v)] = float(score)
        else:
            directed_score[(u, v)] = max(
                directed_score[(u, v)],
                float(score),
            )

    # Irányítatlan, kölcsönösen elfogadható pairök.
    candidate_pairs: List[Tuple[float, int, int]] = []

    for (u, v), score_uv in directed_score.items():
        # Csak canonical irányból készítjük el a párt:
        # így {u,v} egyszer kerül a listába.
        if u >= v:
            continue

        score_vu = directed_score.get((v, u))

        # Csak kölcsönös preferenciaélből lehet matching-pár.
        if score_vu is None:
            continue

        pair_score = 0.5 * (score_uv + score_vu)
        candidate_pairs.append((pair_score, u, v))

    # Nagyobb score előre.
    candidate_pairs.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    partner = torch.full(
        (num_nodes,),
        -1,
        dtype=torch.long,
        device=device,
    )

    if debug:
        print("\n--- Candidate pairök score szerint ---")
        for score, u, v in candidate_pairs:
            print(
                f"({u}, {v}) | "
                f"score={score:+.6f}"
            )

    for score, u, v in candidate_pairs:
        if (
            partner[u].item() == -1
            and partner[v].item() == -1
        ):
            partner[u] = v
            partner[v] = u

            if debug:
                print(
                    f"SELECT ({u}, {v}) "
                    f"score={score:+.6f}"
                )
        elif debug:
            print(
                f"SKIP   ({u}, {v}) "
                f"score={score:+.6f}, "
                f"partner[{u}]={partner[u].item()}, "
                f"partner[{v}]={partner[v].item()}"
            )

    return partner

def soft_blocking_pair_loss(
    edge_index: torch.Tensor,
    ranks: torch.Tensor,
    directed_logits: torch.Tensor,
    num_nodes: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Differenciálható soft blocking-pair loss.

    Parameters
    ----------
    edge_index:
        [2, E] directed preferenciaélek.

    ranks:
        [E] vagy [E, F].
        ranks[e] = a source agent preferenciarangja a targetre.
        Kisebb rank = jobb partner.

    directed_logits:
        [E], raw decoder logits.

    num_nodes:
        Node-ok száma.

    temperature:
        A sigmoid hőmérséklete.
        Kisebb érték -> közelebb bináris kiválasztáshoz.

    Returns
    -------
    Scalar tensor, amelyhez van gradient directed_logits felé.
    """
    if ranks.ndim == 2:
        ranks = ranks[:, 0]

    if ranks.ndim != 1:
        raise ValueError(
            "ranks shape-ja [E] vagy [E,F] legyen."
        )

    if directed_logits.ndim != 1:
        raise ValueError(
            "directed_logits shape-ja [E] legyen."
        )

    if edge_index.size(1) != directed_logits.numel():
        raise ValueError(
            "edge_index és directed_logits élszáma eltér."
        )

    device = directed_logits.device
    dtype = directed_logits.dtype

    src = edge_index[0]
    dst = edge_index[1]
    num_edges = edge_index.size(1)

    # p(u -> v), differentiálható a decoder logitokra.
    directed_prob = torch.sigmoid(
        directed_logits / temperature
    )  # [E]

    # Python dict csak indexeket tárol, nem tensorértékeket:
    # ettől a directed_prob felé a gradient nem szakad meg.
    edge_id_of = {
        (int(src[e].item()), int(dst[e].item())): e
        for e in range(num_edges)
    }

    # Minden kölcsönös párra tároljuk:
    # u, v, uv_eid, vu_eid, rank_u(v), rank_v(u)
    pair_data = []

    for (u, v), uv_eid in edge_id_of.items():
        if u >= v:
            continue

        vu_eid = edge_id_of.get((v, u))

        if vu_eid is None:
            continue

        pair_data.append(
            (
                u,
                v,
                uv_eid,
                vu_eid,
            )
        )

    if len(pair_data) == 0:
        return directed_logits.sum() * 0.0

    # P kölcsönös, irányítatlan candidate pair.
    pair_u = torch.tensor(
        [item[0] for item in pair_data],
        dtype=torch.long,
        device=device,
    )
    pair_v = torch.tensor(
        [item[1] for item in pair_data],
        dtype=torch.long,
        device=device,
    )

    uv_eids = torch.tensor(
        [item[2] for item in pair_data],
        dtype=torch.long,
        device=device,
    )
    vu_eids = torch.tensor(
        [item[3] for item in pair_data],
        dtype=torch.long,
        device=device,
    )

    # p_{u,v}: mindkét irány magas legyen.
    p_pair = (
        directed_prob[uv_eids]
        * directed_prob[vu_eids]
    )  # [P]

    rank_u_to_v = ranks[uv_eids]  # [P]
    rank_v_to_u = ranks[vu_eids]  # [P]

    total_loss = directed_logits.new_zeros(())

    # Egyszerűség kedvéért P^2 ciklus.
    # Kis graphokra / első debugra teljesen jó.
    for p in range(len(pair_data)):
        u = pair_u[p]
        v = pair_v[p]

        # Minden olyan pair q-t keresünk, amely:
        # - u-t tartalmazza,
        # - u a q másik endpointját v-nél rosszabbnak preferálja.
        desire_u_for_v = directed_logits.new_zeros(())

        # Ugyanez v oldalról.
        desire_v_for_u = directed_logits.new_zeros(())

        for q in range(len(pair_data)):
            if p == q:
                continue

            a = pair_u[q]
            b = pair_v[q]

            # q = {u, w}; keressük rank_u(w)-t.
            if a.item() == u.item():
                w = b
                q_rank_u_to_w = rank_u_to_v[q]

                if rank_u_to_v[p] < q_rank_u_to_w:
                    desire_u_for_v = (
                        desire_u_for_v + p_pair[q]
                    )

            elif b.item() == u.item():
                w = a
                q_rank_u_to_w = rank_v_to_u[q]

                if rank_u_to_v[p] < q_rank_u_to_w:
                    desire_u_for_v = (
                        desire_u_for_v + p_pair[q]
                    )

            # q = {v, z}; keressük rank_v(z)-t.
            if a.item() == v.item():
                z = b
                q_rank_v_to_z = rank_u_to_v[q]

                if rank_v_to_u[p] < q_rank_v_to_z:
                    desire_v_for_u = (
                        desire_v_for_u + p_pair[q]
                    )

            elif b.item() == v.item():
                z = a
                q_rank_v_to_z = rank_v_to_u[q]

                if rank_v_to_u[p] < q_rank_v_to_z:
                    desire_v_for_u = (
                        desire_v_for_u + p_pair[q]
                    )

        # Ha p pair nem kiválasztott, és mindkét oldal
        # soft módon rosszabb partnerhez van rendelve,
        # akkor büntetjük.
        blocking_mass = (
            (1.0 - p_pair[p])
            * desire_u_for_v
            * desire_v_for_u
        )

        total_loss = total_loss + blocking_mass

    # Az instance méretétől kevésbé függjön a loss.
    return total_loss / len(pair_data)

from typing import List, Tuple
import torch


def find_blocking_pairs(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    partner: torch.Tensor,
) -> List[Tuple[int, int]]:
    """
    Csak blocking paireket keres strict, one-to-one matchinghez.

    Parameters
    ----------
    edge_index:
        [2, E] irányított preferenciaélek.
        edge_index[:, e] = [u, v] azt jelenti:
        u elfogadja v-t.

    edge_attr:
        [E] vagy [E, F].
        Az első feature az u -> v preferenciarangja.
        Kisebb érték = jobb partner.

    partner:
        [N] tensor.
        partner[u] = v, ha u és v párosítva vannak.
        partner[u] = -1, ha u unmatched.

    Returns
    -------
    blocking_pairs:
        Python lista canonical párokkal:
        [(min(u,v), max(u,v)), ...]
    """
    if edge_attr.ndim == 2:
        ranks = edge_attr[:, 0]
    elif edge_attr.ndim == 1:
        ranks = edge_attr
    else:
        raise ValueError(
            f"edge_attr shape-ja [E] vagy [E,F] legyen, "
            f"de ez: {tuple(edge_attr.shape)}"
        )

    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()
    ranks = ranks.detach().cpu().tolist()
    partner = partner.detach().cpu().tolist()

    # rank_of[u][v] = u milyen rankre teszi v-t.
    rank_of = {}

    for u, v, rank in zip(src, dst, ranks):
        if u not in rank_of:
            rank_of[u] = {}

        rank_of[u][v] = float(rank)

    def prefers(agent: int, candidate: int, current: int) -> bool:
        """
        True, ha agent a candidate-et szigorúan jobban preferálja,
        mint a current partnert.

        Ha unmatched, akkor minden elfogadható candidate jobb.
        """
        if candidate not in rank_of.get(agent, {}):
            return False

        if current == -1:
            return True

        # Ha a jelenlegi partner nincs a preferencialistán,
        # tekintsd rossz / invalid partnernek.
        if current not in rank_of.get(agent, {}):
            return True

        return rank_of[agent][candidate] < rank_of[agent][current]

    blocking_pairs = []
    seen = set()

    for u, neighbors in rank_of.items():
        current_u_partner = partner[u]

        for v in neighbors:
            # Self-loopokat hagyd ki.
            if u == v:
                continue

            # Csak kölcsönös elfogadhatóság:
            # u -> v ÉS v -> u is létezik.
            if u not in rank_of.get(v, {}):
                continue

            # Ha már egymással vannak párosítva, nem blocking pair.
            if current_u_partner == v:
                continue

            # Mivel u->v és v->u is bejárásra kerülne,
            # egy canonical tuple-t használunk duplikáció ellen.
            pair = (min(u, v), max(u, v))

            if pair in seen:
                continue

            current_v_partner = partner[v]

            u_prefers_v = prefers(
                agent=u,
                candidate=v,
                current=current_u_partner,
            )

            v_prefers_u = prefers(
                agent=v,
                candidate=u,
                current=current_v_partner,
            )

            if u_prefers_v and v_prefers_u:
                blocking_pairs.append(pair)
                seen.add(pair)

    return blocking_pairs

def main(training=False,roommate =  False,LLM_FILE_GEN = False,LLM_TEST = False,GAT_TEST=False,LOCAL_LLM = False):

    group_size=3

    train_data = []
    for j in range(300):
        train_data.append(
            data_generator.graph_to_pyg_data_random(data_generator.generate_graph_m(group_size), group_size))
    val_data = []
    for i in range(100):
        val_data.append(
            data_generator.graph_to_pyg_data_random(data_generator.generate_graph_m(group_size), group_size))




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

    enc = encoder.StableMatchingEdgeEncoder(
        node_feature_dim=train_data[0].x.shape[1],
        edge_feature_dim=1,
        hidden_dim=64,
        num_layers=4,
    )
    train_enc = False
    if train_enc:
        optimizer = torch.optim.AdamW(
            enc.parameters(),
            lr=3e-4,
            weight_decay=1e-4,
        )
        for epoch in range(100):
            enc.train()
            print("Epoch: " + str(epoch))
            for batch in train_data:
                out = enc(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr.float().unsqueeze(-1),
                )

                loss, metrics = encoder_loss(
                    out=out,
                    batch=batch,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    enc.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()
        torch.save(enc.state_dict(), "enc.pt")
    #enc.load_state_dict(torch.load("enc.pt", weights_only=True))
    dec = decoder.GreedyMatchingDecoder(
        edge_embedding_dim=66,
        hidden_dim=66,
    )
    optimizer = torch.optim.AdamW(
        list(dec.parameters())+list(enc.parameters()),
        lr=3e-3,
        weight_decay=1e-4,
    )
    """
    dec_train=True
    if dec_train:
        for graph in train_data:
            enc.eval()
            dec.train()
            out = enc(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr.float().unsqueeze(-1),
            )
            pair_logits=dec(out.edge_embedding)
            loss = F.binary_cross_entropy_with_logits(
                pair_logits,
                graph.edge_y
            )
            optimizer_dec.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                dec.parameters(),
                max_norm=1.0,
            )
            print(loss)
            optimizer_dec.step()
        print("finished")
        torch.save(dec.state_dict(), "dec.pt")
    dec.load_state_dict(torch.load("dec.pt", weights_only=True))
    """
    enc.train()
    dec.train()
    for i in range(0,50):
        print("Epoch: " + str(i))
        for graph in train_data:
            optimizer.zero_grad(set_to_none=True)

            edge_attr = graph.edge_attr.float()
            if edge_attr.ndim == 1:
                edge_attr = edge_attr.unsqueeze(-1)

            out = enc(
                x=graph.x.float(),
                edge_index=graph.edge_index,
                edge_attr=edge_attr,
            )
            loss_enc, metrics = encoder_loss(
                out=out,
                batch=graph,
            )
            out.edge_embedding = torch.cat(
                [
                    out.edge_embedding,  # [E, 128]
                    out.auxiliary["mutual_quality"].unsqueeze(-1),  # [E, 1]
                    out.auxiliary["rank_score"].unsqueeze(-1),  # [E, 1]
                ],
                dim=-1,
            )
            edge_logits = dec(
                out.edge_embedding
            )
            loss_solver = F.binary_cross_entropy_with_logits(edge_logits, graph.edge_y)
            ranks = preference_lists_to_edge_ranks(graph.proposee_pref | graph.proposer_pref,graph.edge_index)
            total_loss = loss_enc*0.5+ soft_blocking_pair_loss(
                edge_index=graph.edge_index,ranks =ranks,directed_logits=edge_logits,num_nodes=6) +loss_solver*0.3

            total_loss.backward()
            optimizer.step()

    good = 0
    bad = 0
    for graph in train_data:
        enc.eval()
        dec.eval()
        out = enc(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr.float().unsqueeze(-1),
        )
        out.edge_embedding = torch.cat(
            [
                out.edge_embedding,  # [E, 128]
                out.auxiliary["mutual_quality"].unsqueeze(-1),  # [E, 1]
                out.auxiliary["rank_score"].unsqueeze(-1),  # [E, 1]
            ],
            dim=-1,
        )
        pair_logits = dec(out.edge_embedding)
        partner = greedy_decode_pairs(
            edge_index=graph.edge_index,
            directed_logits=pair_logits,
            num_nodes=6,
            debug=False,
        )
        blocking_pairs = find_blocking_pairs(
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            partner=partner,
        )
        if len(blocking_pairs)==0:
            good+=1
        else:
            bad += 1

    print(good/(good+bad))
    print("finished")

    """
    model = GAT.GATEdgeClassifier(train_data[0].x.size(-1), 16)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.015)


    patience = 10
    min_delta = 1e-4
    eval_every = 10

    best_val_loss = float("inf")
    global_step = 0
    stop_training = False


    if training:
        train(global_step, best_val_loss, stop_training, 0.2, 0.3, 0.5,model,optimizer,train_data, group_size, eval_every, val_data, patience, min_delta)
    model.load_state_dict(torch.load("best_model.pt", weights_only=True))



    model.eval()
    

    good=0
    bad=0
    if GAT_TEST:
        with torch.no_grad():
            for i in range(2500):
                pair_dict = {}

                group_size=3
                acc_graph = data_generator.generate_graph_m(group_size)

                structure = random.randint(0,100)
                num = structure %2
                match num:
                    case 0:
                        acc_data = data_generator.graph_to_pyg_data_low_diff(acc_graph, group_size)
                    case 1:
                        acc_data = data_generator.graph_to_pyg_data_high_diff(acc_graph, group_size)

                acc_data.x = (acc_data.x - mean) / std
                acc_data.edge_attr = (acc_data.edge_attr - mean_edge) / std_edge

                logits = model(acc_data)
                probs = torch.sigmoid(logits)
                propr_pref=acc_data.proposer_pref
                prope_pref = acc_data.proposee_pref
                ag = propr_pref |prope_pref
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
                        edge_probs=current_probs,
                        preference_lists=ag,
                        tolerance=0.01
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

        print("GRAPH LEVEL ACCURACY GATv2 TEST")
        print(f"Accuracy: {(good/(good+bad))*100} %")


    if roommate:
        print("Stable roommate case")
        g_size = 4
        graph_r = data_generator.generate_graph_r(g_size)

        data_r = data_generator.graph_to_pyg_data_r_random(graph_r,g_size)
        print(data_r.prefs)
        print(data_r.x)
        print(data_r.edge_attr)
        data_r.x = (data_r.x - mean) / std
        data_r.edge_attr = (data_r.edge_attr - mean_edge) / std_edge
        logits = model(data_r)
        probs = torch.sigmoid(logits)
        print(probs)

    if LLM_FILE_GEN:
        with open("LLM_PREF_DATA_M.txt", "w", encoding="utf-8") as f:
            for i in range(100):
                low_diff = data_generator.graph_to_pyg_data_low_diff(data_generator.generate_graph_m(3), 3)
                f.write(str(low_diff.proposer_pref)+";")
                f.write(str(low_diff.proposee_pref)+ "\n")
                high_diff=  data_generator.graph_to_pyg_data_high_diff(data_generator.generate_graph_m(3), 3)
                f.write(str(high_diff.proposer_pref)+";")
                f.write(str(high_diff.proposee_pref)+ "\n")

    good = 0
    bad = 0
    if LLM_TEST:
        proposer_prefs = []
        proposee_prefs = []
        with open("LLM_PREF_DATA_M.txt", "r", encoding="utf-8") as f:
            for line in f:
                prefs = line.strip().split(";")
                proposer_prefs.append(ast.literal_eval(prefs[0]))
                proposee_prefs.append(ast.literal_eval(prefs[1]))
        index = 0
        with open("stable_matchings_only.txt", "r", encoding="utf-8") as f:
            for line in f:
                result = ast.literal_eval(line)
                if gale_shapley.is_stable_matching(result, proposer_prefs[index], proposee_prefs[index]):
                    good +=1
                else:
                    bad +=1
                index += 1
        print("GRAPH LEVEL ACCURACY LLM TEST")
        print(good/(good+bad))


    if LOCAL_LLM:
        import requests

        prompts = []
        with open("prompts_for_local_llm.txt", "r", encoding="utf-8") as f:
            for line in f:
                prompts.append(line)

        for i, prompt in enumerate(prompts, 1):
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False
                }
            )
            data = r.json()
            print(f"\n--- Prompt {i} ---")
            print(prompt)
            print("\n--- Válasz ---")
            print(data["response"])
    """

if __name__ == "__main__":
    training = False
    roommate = False
    LLM_FILE_GEN = False
    LLM_TEST = False
    GAT_TEST = False
    LOCAL_LLM = False
    main(training, roommate,LLM_FILE_GEN,LLM_TEST,GAT_TEST,LOCAL_LLM)