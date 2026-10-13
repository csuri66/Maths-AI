import ast
import os
import data_generator
import random
import encoder
import decoder
from collections import Counter
from dataclasses import dataclass
from itertools import permutations
import torch.nn.functional as F
import torch
import numpy as np
from typing import Dict
from typing import List, Tuple
import random

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


@dataclass
class MatchingRankingData:
    uv_edge_ids: torch.Tensor       # [P]
    vu_edge_ids: torch.Tensor       # [P]
    matching_pair_ids: torch.Tensor # [K, n]
    blocking_counts: torch.Tensor   # [K]
    better_matching_ids: torch.Tensor  # [Q]
    worse_matching_ids: torch.Tensor   # [Q]
    num_edges: int

    def to(self, device):
        return MatchingRankingData(
            uv_edge_ids=self.uv_edge_ids.to(device),
            vu_edge_ids=self.vu_edge_ids.to(device),
            matching_pair_ids=self.matching_pair_ids.to(device),
            blocking_counts=self.blocking_counts.to(device),
            better_matching_ids=self.better_matching_ids.to(device),
            worse_matching_ids=self.worse_matching_ids.to(device),
            num_edges=self.num_edges,
        )


def matching_ranking_loss(
    directed_logits: torch.Tensor,
    ranking_data: MatchingRankingData,
    temperature: float = 1.0,
    score_reg_weight: float = 1e-4,
):
    """
    directed_logits:
        [E], a decoder raw outputja.

    ranking_data:
        Az adott gráfhoz előkészített matchingadat.

    score_reg_weight:
        Kis regularizáció a pair-score-ok túlzott
        növekedésének korlátozására.
    """
    if temperature <= 0:
        raise ValueError("temperature legyen pozitív.")

    if directed_logits.shape != (ranking_data.num_edges,):
        raise ValueError(
            f"Logit shape: [{ranking_data.num_edges}] szükséges, "
            f"kapott: {tuple(directed_logits.shape)}"
        )

    # [P]: irányítatlan párok score-jai.
    pair_logits = 0.5 * (
        directed_logits[ranking_data.uv_edge_ids]
        + directed_logits[ranking_data.vu_edge_ids]
    )

    # [K, n] -> [K]: teljes matchingek score-jai.
    matching_scores = pair_logits[
        ranking_data.matching_pair_ids
    ].sum(dim=-1)

    better_ids = ranking_data.better_matching_ids
    worse_ids = ranking_data.worse_matching_ids

    if better_ids.numel() == 0:
        # Minden matching azonos minőségű:
        # nincs ranking tanítójel.
        rank_loss = directed_logits.sum() * 0.0
    else:
        better_scores = matching_scores[better_ids]
        worse_scores = matching_scores[worse_ids]

        rank_loss = F.softplus(
            (worse_scores - better_scores) / temperature
        ).mean()

    score_reg = pair_logits.square().mean()

    total = rank_loss + score_reg_weight * score_reg

    with torch.no_grad():
        best_id = matching_scores.argmax()
        top1_blocking_count = ranking_data.blocking_counts[best_id]

        if better_ids.numel() > 0:
            ranking_accuracy = (
                matching_scores[better_ids]
                > matching_scores[worse_ids]
            ).float().mean()
        else:
            ranking_accuracy = matching_scores.new_tensor(
                float("nan")
            )

    metrics = {
        "rank_loss": rank_loss.detach(),
        "score_reg": score_reg.detach(),
        "ranking_accuracy": ranking_accuracy.detach(),
        "top1_blocking_count": top1_blocking_count.detach(),
        "top1_stable": (top1_blocking_count == 0).detach(),
    }

    return total, metrics, matching_scores


def prepare_matching_ranking_data(
    edge_index: torch.Tensor,
    edge_ranks: torch.Tensor,
    num_nodes: int,
    left_nodes: list[int],
    right_nodes: list[int],
) -> MatchingRankingData:
    """
    Előkészítés kis bipartit gráfokra.

    edge_ranks:
        [E], eredeti preferenciarangok.
        Kisebb érték = jobb partner.

    K:
        Megengedett teljes matchingek száma.

    Q:
        Olyan rendezett matchingpárok száma, ahol
        az első matching blocking-countja kisebb.
    """
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index shape-ja [2, E] legyen.")

    ei = edge_index.detach().cpu()
    ranks = edge_ranks.detach().cpu()

    num_edges = ei.size(1)

    if ranks.shape != (num_edges,):
        raise ValueError("edge_ranks shape-ja [E] legyen.")

    if len(left_nodes) != len(right_nodes):
        raise ValueError("A két oldal mérete legyen azonos.")

    if len(left_nodes) == 0:
        raise ValueError("Legalább egy node kell oldalanként.")

    all_nodes = left_nodes + right_nodes

    if (
        len(all_nodes) != num_nodes
        or len(set(all_nodes)) != num_nodes
        or set(all_nodes) != set(range(num_nodes))
    ):
        raise ValueError(
            "A két oldal legyen diszjunkt, és fedje le "
            "a 0..num_nodes-1 node-okat."
        )

    edge_id = {}

    for eid, (u, v) in enumerate(ei.t().tolist()):
        if (u, v) in edge_id:
            raise ValueError(f"Duplikált irányított él: {(u, v)}")
        edge_id[(u, v)] = eid

    # A -> B orientációjú kölcsönös párjelöltek.
    candidate_pairs = [
        (u, v)
        for u in left_nodes
        for v in right_nodes
        if (u, v) in edge_id and (v, u) in edge_id
    ]

    pair_id = {
        pair: pid
        for pid, pair in enumerate(candidate_pairs)
    }

    uv_edge_ids = torch.tensor(
        [edge_id[(u, v)] for u, v in candidate_pairs],
        dtype=torch.long,
    )

    vu_edge_ids = torch.tensor(
        [edge_id[(v, u)] for u, v in candidate_pairs],
        dtype=torch.long,
    )

    matching_pair_ids = []
    blocking_counts = []

    for right_order in permutations(right_nodes):
        pairs = list(zip(left_nodes, right_order))

        if any(pair not in pair_id for pair in pairs):
            continue

        partner = torch.full(
            (num_nodes,),
            -1,
            dtype=torch.long,
        )

        for u, v in pairs:
            partner[u] = v
            partner[v] = u

        blocking = find_blocking_pairs(
            edge_index=ei,
            edge_attr=ranks,
            partner=partner,
        )

        matching_pair_ids.append(
            [pair_id[pair] for pair in pairs]
        )
        blocking_counts.append(len(blocking))

    if not matching_pair_ids:
        raise ValueError(
            "A gráfban nincs megengedett teljes matching."
        )

    matching_pair_ids = torch.tensor(
        matching_pair_ids,
        dtype=torch.long,
    )

    blocking_counts = torch.tensor(
        blocking_counts,
        dtype=torch.long,
    )

    # cost[i] < cost[j] esetén i-t magasabb score-ra tanítjuk.
    comparison_mask = (
        blocking_counts[:, None]
        < blocking_counts[None, :]
    )

    better_ids, worse_ids = comparison_mask.nonzero(
        as_tuple=True
    )

    return MatchingRankingData(
        uv_edge_ids=uv_edge_ids,
        vu_edge_ids=vu_edge_ids,
        matching_pair_ids=matching_pair_ids,
        blocking_counts=blocking_counts,
        better_matching_ids=better_ids,
        worse_matching_ids=worse_ids,
        num_edges=num_edges,
    )


def _enumerate_perfect_matchings(nodes):
    """
    Egy csomóponthalmaz ÖSSZES tökéletes párosítását felsorolja.

    A legkisebb indexű szabad node-ot párosítja minden más szabad
    node-dal, rekurzívan. Minden párosítás kanonikus (min, max) párok
    listája. A node-ok száma páros kell legyen; páratlan esetben üres.
    """
    if len(nodes) % 2 != 0:
        return

    if not nodes:
        yield []
        return

    first = nodes[0]
    rest = nodes[1:]

    for i, partner in enumerate(rest):
        pair = (min(first, partner), max(first, partner))
        remaining = rest[:i] + rest[i + 1:]
        for sub in _enumerate_perfect_matchings(remaining):
            yield [pair] + sub


def prepare_matching_ranking_data_roommate(
    edge_index: torch.Tensor,
    edge_ranks: torch.Tensor,
    num_nodes: int,
) -> MatchingRankingData:
    """
    A `prepare_matching_ranking_data` stabil-szobatárs (roommate)
    változata: nincs bal/jobb oldal.

    Minden kölcsönös (u,v) & (v,u) él párt lehetséges párosítás-jelöltnek
    tekint, és felsorolja az ÖSSZES tökéletes párosítást a csomópontok
    egyetlen halmazán (num_nodes páros kell legyen). A blocking-count-ot a
    `find_blocking_pairs` adja, a visszatérési struktúra megegyezik a
    páros esetben használttal, így a `matching_ranking_loss` változatlanul
    működik.
    """
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index shape-ja [2, E] legyen.")

    ei = edge_index.detach().cpu()
    ranks = edge_ranks.detach().cpu()

    num_edges = ei.size(1)

    if ranks.shape != (num_edges,):
        raise ValueError("edge_ranks shape-ja [E] legyen.")

    if num_nodes % 2 != 0:
        raise ValueError(
            "Szobatárs tökéletes párosításhoz páros számú node kell."
        )

    edge_id = {}
    for eid, (u, v) in enumerate(ei.t().tolist()):
        if (u, v) in edge_id:
            raise ValueError(f"Duplikált irányított él: {(u, v)}")
        edge_id[(u, v)] = eid

    # Kanonikus (min, max) párjelöltek, ahol mindkét irány létezik.
    candidate_pairs = []
    seen = set()
    for (u, v) in edge_id:
        pair = (min(u, v), max(u, v))
        if pair in seen:
            continue
        a, b = pair
        if (a, b) in edge_id and (b, a) in edge_id:
            candidate_pairs.append(pair)
            seen.add(pair)

    pair_id = {pair: pid for pid, pair in enumerate(candidate_pairs)}

    uv_edge_ids = torch.tensor(
        [edge_id[(a, b)] for a, b in candidate_pairs],
        dtype=torch.long,
    )
    vu_edge_ids = torch.tensor(
        [edge_id[(b, a)] for a, b in candidate_pairs],
        dtype=torch.long,
    )

    matching_pair_ids = []
    blocking_counts = []

    for matching in _enumerate_perfect_matchings(list(range(num_nodes))):
        if any(pair not in pair_id for pair in matching):
            continue

        partner = torch.full((num_nodes,), -1, dtype=torch.long)
        for a, b in matching:
            partner[a] = b
            partner[b] = a

        blocking = find_blocking_pairs(
            edge_index=ei,
            edge_attr=ranks,
            partner=partner,
        )

        matching_pair_ids.append(
            [pair_id[pair] for pair in matching]
        )
        blocking_counts.append(len(blocking))

    if not matching_pair_ids:
        raise ValueError(
            "A gráfban nincs megengedett tökéletes szobatárs-párosítás."
        )

    matching_pair_ids = torch.tensor(
        matching_pair_ids,
        dtype=torch.long,
    )
    blocking_counts = torch.tensor(
        blocking_counts,
        dtype=torch.long,
    )

    comparison_mask = (
        blocking_counts[:, None]
        < blocking_counts[None, :]
    )
    better_ids, worse_ids = comparison_mask.nonzero(as_tuple=True)

    return MatchingRankingData(
        uv_edge_ids=uv_edge_ids,
        vu_edge_ids=vu_edge_ids,
        matching_pair_ids=matching_pair_ids,
        blocking_counts=blocking_counts,
        better_matching_ids=better_ids,
        worse_matching_ids=worse_ids,
        num_edges=num_edges,
    )


@dataclass
class RankedMatching:
    rank: int
    pairs: list[tuple[int, int]]
    score: float
    partner: torch.Tensor
    blocking_pairs: list[tuple[int, int]]

    @property
    def is_stable(self) -> bool:
        return len(self.blocking_pairs) == 0


@torch.no_grad()
def rank_all_complete_matchings(
    edge_index: torch.Tensor,
    directed_logits: torch.Tensor,
    edge_ranks: torch.Tensor,
    num_nodes: int,
    left_nodes: list[int],
    right_nodes: list[int],
) -> list[RankedMatching]:
    """
    Kis bipartit gráf összes megengedett teljes matchingjét
    felsorolja és decoder-score szerint rangsorolja.

    edge_index:
        [2, E], irányított preferenciaélek.

    directed_logits:
        [E], a decoder nyers logitjai.

    edge_ranks:
        [E], az EREDETI preferenciarangok az edge_index sorrendjében.
        Kisebb érték = jobb partner.

    left_nodes, right_nodes:
        A két oldal node ID-i.
        Együtt pontosan a 0..num_nodes-1 node-okat fedjék le.

    Return:
        RankedMatching lista csökkenő össz-score szerint.

    Fontos:
        Faktorális számú jelöltet vizsgál.
        Kis példányok diagnosztikájára való.
    """
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index shape-ja [2, E] legyen.")

    num_edges = edge_index.size(1)

    if directed_logits.shape != (num_edges,):
        raise ValueError("directed_logits shape-ja [E] legyen.")

    if edge_ranks.shape != (num_edges,):
        raise ValueError("edge_ranks shape-ja [E] legyen.")

    if len(left_nodes) != len(right_nodes):
        raise ValueError(
            "Teljes matchinghez a két oldal mérete legyen azonos."
        )

    all_nodes = left_nodes + right_nodes
    if (
        len(all_nodes) != num_nodes
        or len(set(all_nodes)) != num_nodes
        or set(all_nodes) != set(range(num_nodes))
    ):
        raise ValueError(
            "A két oldal legyen diszjunkt, és fedje le "
            "a 0..num_nodes-1 node-okat."
        )

    if not torch.isfinite(directed_logits).all().item():
        raise ValueError("A logitok között NaN vagy inf szerepel.")

    # A diagnosztika CPU-n fut.
    ei = edge_index.detach().cpu()
    logits = directed_logits.detach().cpu()
    ranks = edge_ranks.detach().cpu()

    directed_scores = {}

    for eid, (u, v) in enumerate(ei.t().tolist()):
        if (u, v) in directed_scores:
            raise ValueError(f"Duplikált irányított él: {(u, v)}")

        directed_scores[(u, v)] = float(logits[eid].item())

    # Csak kölcsönösen elfogadható bipartit párjelöltek.
    pair_scores = {}

    for u in left_nodes:
        for v in right_nodes:
            if (
                (u, v) in directed_scores
                and (v, u) in directed_scores
            ):
                pair_scores[(u, v)] = 0.5 * (
                    directed_scores[(u, v)]
                    + directed_scores[(v, u)]
                )

    candidates = []

    for right_order in permutations(right_nodes):
        pairs = list(zip(left_nodes, right_order))

        # Hiányzó kölcsönös él esetén ez nem megengedett matching.
        if any(pair not in pair_scores for pair in pairs):
            continue

        score = sum(pair_scores[pair] for pair in pairs)

        partner = torch.full(
            (num_nodes,),
            -1,
            dtype=torch.long,
        )

        for u, v in pairs:
            partner[u] = v
            partner[v] = u

        blocking_pairs = find_blocking_pairs(
            edge_index=ei,
            edge_attr=ranks,
            partner=partner,
        )

        candidates.append(
            RankedMatching(
                rank=0,
                pairs=pairs,
                score=score,
                partner=partner,
                blocking_pairs=blocking_pairs,
            )
        )

    # Egyező score-nál determinisztikus sorrend.
    candidates.sort(
        key=lambda result: (
            -result.score,
            tuple(result.pairs),
        )
    )

    for rank, result in enumerate(candidates, start=1):
        result.rank = rank

    return candidates


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

    # Stable-membership supervisio: a stable_membership_logit head
    # azt tanulja, hogy az él benne van-e a stabil párosításban.
    # Cél: edge_y (1, ha az él a stabil matching része).
    membership_logit = out.auxiliary["stable_membership_logit"]
    membership_loss = F.binary_cross_entropy_with_logits(
        membership_logit,
        batch.edge_y.float(),
        pos_weight=pos_weight,
    )

    total = (
        rank_loss
        +   mutual_loss
    )

    metrics = {
        "loss": total.detach(),
        "rank_loss": rank_loss.detach(),
        "mutual_loss": mutual_loss.detach(),
        "membership_loss": membership_loss.detach(),
    }
    return total, metrics


class BlockingPairLoss(torch.nn.Module):
    """
    Egy fix gráfhoz előkészített blocking-pair surrogate loss.

    Az edge_index sorrendjének training közben változatlannak
    kell maradnia.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_ranks: torch.Tensor,
        num_nodes: int,
        degree_weight: float = 1.0,
        temperature: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature legyen pozitív.")

        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction: 'mean' vagy 'sum'.")

        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index shape: [2, E].")

        edge_ranks = edge_ranks.detach().cpu().reshape(-1)
        edges = edge_index.detach().cpu().t().tolist()

        if len(edges) != edge_ranks.numel():
            raise ValueError("Az élek és rankek száma eltér.")

        edge_id = {}

        for eid, (u, v) in enumerate(edges):
            if not (0 <= u < num_nodes and 0 <= v < num_nodes):
                raise ValueError(f"Érvénytelen node ID: {(u, v)}")

            if (u, v) in edge_id:
                raise ValueError(f"Duplikált irányított él: {(u, v)}")

            edge_id[(u, v)] = eid

        pairs = [
            (u, v)
            for u, v in edges
            if u < v and (v, u) in edge_id
        ]

        uv_ids = [edge_id[(u, v)] for u, v in pairs]
        vu_ids = [edge_id[(v, u)] for u, v in pairs]

        num_pairs = len(pairs)

        # incidence[node, pair]: a node végpontja-e a párnak?
        incidence = torch.zeros(num_nodes, num_pairs)

        # better_u[p, q]:
        # q pár u-nak jobb partnert ad-e, mint a p pár?
        better_u = torch.zeros(num_pairs, num_pairs)
        better_v = torch.zeros(num_pairs, num_pairs)

        incident_pairs = [[] for _ in range(num_nodes)]

        for q, (a, b) in enumerate(pairs):
            incidence[a, q] = 1.0
            incidence[b, q] = 1.0

            incident_pairs[a].append((q, b))
            incident_pairs[b].append((q, a))

        for p, (u, v) in enumerate(pairs):
            rank_uv = edge_ranks[edge_id[(u, v)]].item()
            rank_vu = edge_ranks[edge_id[(v, u)]].item()

            for q, w in incident_pairs[u]:
                rank_uw = edge_ranks[edge_id[(u, w)]].item()
                if rank_uw < rank_uv:
                    better_u[p, q] = 1.0

            for q, w in incident_pairs[v]:
                rank_vw = edge_ranks[edge_id[(v, w)]].item()
                if rank_vw < rank_vu:
                    better_v[p, q] = 1.0

        self.register_buffer(
            "uv_ids", torch.tensor(uv_ids, dtype=torch.long)
        )
        self.register_buffer(
            "vu_ids", torch.tensor(vu_ids, dtype=torch.long)
        )
        self.register_buffer("incidence", incidence)
        self.register_buffer("better_u", better_u)
        self.register_buffer("better_v", better_v)

        self.num_edges = len(edges)
        self.degree_weight = degree_weight
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, directed_logits: torch.Tensor):
        if directed_logits.shape != (self.num_edges,):
            raise ValueError(
                f"directed_logits shape legyen ({self.num_edges},), "
                f"kapott: {tuple(directed_logits.shape)}"
            )

        if self.uv_ids.numel() == 0:
            zero = directed_logits.sum() * 0.0
            return zero, {
                "blocking": zero.detach(),
                "degree": zero.detach(),
            }

        # Egyetlen score minden kölcsönös, irányítatlan párra.
        pair_logits = 0.5 * (
            directed_logits[self.uv_ids]
            + directed_logits[self.vu_ids]
        )

        x = torch.sigmoid(pair_logits / self.temperature)

        better_u = self.better_u.to(dtype=x.dtype)
        better_v = self.better_v.to(dtype=x.dtype)
        incidence = self.incidence.to(dtype=x.dtype)

        better_mass_u = better_u @ x
        better_mass_v = better_v @ x

        desire_u = F.relu(1.0 - x - better_mass_u)
        desire_v = F.relu(1.0 - x - better_mass_v)

        blocking_per_pair = desire_u * desire_v

        soft_degree = incidence @ x
        degree_per_node = F.relu(soft_degree - 1.0).square()

        if self.reduction == "mean":
            blocking_loss = blocking_per_pair.mean()
            degree_loss = degree_per_node.mean()
        else:
            blocking_loss = blocking_per_pair.sum()
            degree_loss = degree_per_node.sum()

        total = blocking_loss + self.degree_weight * degree_loss

        return total, {
            "blocking": blocking_loss.detach(),
            "degree": degree_loss.detach(),
        }


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
    temperature: float = 0.0001,
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

def get_or_generate_dataset(
    group_size,
    gen_fn,
    n_train=1000,
    n_val=500,
    cache_path="dataset.pt",
    seed=None,
    graph_fn=None,
):
    """
    Betölti a train/val adathalmazt lemezről, ha van cache; különben
    egyszer legenerálja és elmenti.

    Az adat NYERS (normalizálatlan) formában kerül mentésre, hogy a
    normalizációs statisztikák (mean/std) minden futásban konzisztensen
    újraszámolhatók legyenek.

    gen_fn: a használt PyG-konverziós függvény, pl.
            data_generator.graph_to_pyg_data_normal (páros/bipartite) vagy
            data_generator.graph_to_pyg_data_r_normal (szobatárs/roommate).

    graph_fn: a bázisgráf-generátor függvény. Ha None, akkor
            data_generator.generate_graph_m (páros gráf). Szobatárs
            esethez data_generator.generate_graph_r adható át.
    """
    if graph_fn is None:
        graph_fn = data_generator.generate_graph_m

    gen_name = getattr(gen_fn, "__name__", str(gen_fn))
    graph_name = getattr(graph_fn, "__name__", str(graph_fn))

    if os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False)
        if (
            blob.get("group_size") == group_size
            and blob.get("gen_name") == gen_name
            and blob.get("graph_name") == graph_name
            and len(blob["train_data"]) == n_train
            and len(blob["val_data"]) == n_val
        ):
            print("Dataset betöltve innen: " + cache_path)
            return blob["train_data"], blob["val_data"]
        print("A cache paraméterei eltérnek a kérttől; újragenerálás.")

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    train_data = [
        gen_fn(graph_fn(group_size), group_size)
        for _ in range(n_train)
    ]
    val_data = [
        gen_fn(graph_fn(group_size), group_size)
        for _ in range(n_val)
    ]

    torch.save(
        {
            "train_data": train_data,
            "val_data": val_data,
            "group_size": group_size,
            "gen_name": gen_name,
            "graph_name": graph_name,
        },
        cache_path,
    )
    print("Dataset legenerálva és elmentve ide: " + cache_path)
    return train_data, val_data


def main(training=False,roommate =  True,LLM_FILE_GEN = False,LLM_TEST = False,GAT_TEST=False,LOCAL_LLM = False):

    group_size=3

    if roommate:
        # Stabil szobatárs (roommate) eset: nem-páros bázisgráf
        # (generate_graph_r) és normális eloszlású preferenciák.
        # A tökéletes szobatárs-párosításhoz PÁROS számú node kell, ezért
        # 2*group_size node-ot használunk (a páros eset összes node-jával
        # megegyező méret).
        roommate_nodes = 2 * group_size
        train_data, val_data = get_or_generate_dataset(
            roommate_nodes,
            gen_fn=data_generator.graph_to_pyg_data_r_normal,
            graph_fn=data_generator.generate_graph_r,
            n_train=500,
            n_val=250,
            cache_path="dataset_roommate.pt",
            seed=42,
        )
    else:
        # Páros (bipartite) eset: generate_graph_m bázisgráf.
        train_data, val_data = get_or_generate_dataset(
            group_size,
            gen_fn=data_generator.graph_to_pyg_data_normal,
            graph_fn=data_generator.generate_graph_m,
            n_train=1000,
            n_val=500,
            cache_path="dataset.pt",
            seed=42,
        )


    """
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
    """
    enc = encoder.StableMatchingEdgeEncoder(
        node_feature_dim=train_data[0].x.shape[1],
        edge_feature_dim=1,
        hidden_dim=32,
        num_layers=6,
    )
    dec = decoder.GreedyMatchingDecoder(
        edge_embedding_dim=34,
        hidden_dim=34,
    )

    optimizer_joint = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()),
        lr=3e-4,
        weight_decay=1e-4,
    )

    def build_decoder_input(out):
        return torch.cat(
            [
                out.edge_embedding,  # [E, 128]
                out.auxiliary["mutual_quality"].unsqueeze(-1),  # [E, 1]
                out.auxiliary["rank_score"].unsqueeze(-1),  # [E, 1]
            ],
            dim=-1,
        )

    def eval_joint_loss(dataset):
        enc.eval()
        dec.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for graph in dataset:
                edge_attr = graph.edge_attr.float()
                if edge_attr.ndim == 1:
                    edge_attr = edge_attr.unsqueeze(-1)
                out = enc(
                    x=graph.x.float(),
                    edge_index=graph.edge_index,
                    edge_attr=edge_attr,
                )
                loss_enc, _ = encoder_loss(out=out, batch=graph)
                edge_logits = dec(build_decoder_input(out))

                ranks = preference_lists_to_edge_ranks(
                    graph.proposee_pref | graph.proposer_pref, graph.edge_index
                )
                criterion_eval = BlockingPairLoss(
                    edge_index=graph.edge_index,
                    edge_ranks=ranks,
                    num_nodes=graph.num_nodes,
                    degree_weight=1.0,
                    temperature=0.01,
                    reduction="mean",
                )
                loss_dec_eval, parts_ = criterion_eval(edge_logits)
                total += (
                    loss_enc*0.5 + loss_dec_eval*0.5
                ).item()
                count += 1
        return total / max(count, 1)

    joint_patience = 5
    min_delta = 1e-4
    def prepare_dataset_ranking(
            dataset,
            left_nodes,
            right_nodes,
            roommate=False,
    ):
        ranking_dataset = []

        for graph in dataset:
            preferences = (
                    graph.proposee_pref | graph.proposer_pref
            )

            edge_ranks = preference_lists_to_edge_ranks(
                preferences,
                graph.edge_index,
            )

            if roommate:
                ranking_data = prepare_matching_ranking_data_roommate(
                    edge_index=graph.edge_index,
                    edge_ranks=edge_ranks,
                    num_nodes=graph.num_nodes,
                )
            else:
                ranking_data = prepare_matching_ranking_data(
                    edge_index=graph.edge_index,
                    edge_ranks=edge_ranks,
                    num_nodes=graph.num_nodes,
                    left_nodes=left_nodes,
                    right_nodes=right_nodes,
                )

            ranking_dataset.append(ranking_data)

        return ranking_dataset


    train_joint = False
    if train_joint:
        best_joint_val = float("inf")
        joint_bad_checks = 0
        for i in range(0, 200):
            enc.train()
            dec.train()
            for graph in train_data:
                optimizer_joint.zero_grad(set_to_none=True)

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

                edge_logits = dec(build_decoder_input(out))


                ranks = preference_lists_to_edge_ranks(
                    graph.proposee_pref | graph.proposer_pref, graph.edge_index
                )
                criterion_eval = BlockingPairLoss(
                    edge_index=graph.edge_index,
                    edge_ranks=ranks,
                    num_nodes=graph.num_nodes,
                    degree_weight=1.0,
                    temperature=0.01,
                    reduction="mean",
                )
                loss_dec_eval, parts_ = criterion_eval(edge_logits)
                loss_joint = loss_enc*0.5  + loss_dec_eval*0.5

                loss_joint.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(enc.parameters()) + list(dec.parameters()),
                    max_norm=1.0,
                )
                optimizer_joint.step()

            val_joint = eval_joint_loss(val_data)
            print("Joint epoch: " + str(i) + " | val loss: " + str(val_joint))

            if val_joint < best_joint_val - min_delta:
                best_joint_val = val_joint
                joint_bad_checks = 0
                torch.save(enc.state_dict(), "enc.pt")
                torch.save(dec.state_dict(), "dec.pt")
            else:
                joint_bad_checks += 1
                if joint_bad_checks >= joint_patience:
                    print("Joint early stopping at epoch " + str(i))
                    break


    optimizer_joint = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()),
        lr=3e-4,
        weight_decay=1e-4,
    )
    # Páros esetben a bal/jobb oldali node-ok a group_size-ból adódnak;
    # szobatárs esetben nincs oldal-felosztás (a ranking-prep figyelmen
    # kívül hagyja a left/right argumentumokat).
    left_nodes = list(range(group_size))
    right_nodes = list(range(group_size, 2 * group_size))

    train_ranking = prepare_dataset_ranking(
        train_data,
        left_nodes,
        right_nodes,
        roommate=roommate,
    )

    val_ranking = prepare_dataset_ranking(
        val_data,
        left_nodes,
        right_nodes,
        roommate=roommate,
    )

    encoder_weight = 0.2
    decoder_weight = 1.0

    def eval_ranking_loss(dataset, ranking_dataset):
        """
        Validációs loss ugyanazzal a kombinált célfüggvénnyel, mint a
        tanítás (encoder_loss + matching_ranking_loss), gradient nélkül.
        """
        enc.eval()
        dec.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for graph, ranking_data in zip(dataset, ranking_dataset):
                edge_attr = graph.edge_attr.float()
                if edge_attr.ndim == 1:
                    edge_attr = edge_attr.unsqueeze(-1)

                out = enc(
                    x=graph.x.float(),
                    edge_index=graph.edge_index,
                    edge_attr=edge_attr,
                )
                directed_logits = dec(build_decoder_input(out))

                loss_enc, _ = encoder_loss(out=out, batch=graph)
                loss_dec, _, _ = matching_ranking_loss(
                    directed_logits=directed_logits,
                    ranking_data=ranking_data,
                    temperature=1.0,
                    score_reg_weight=1e-4,
                )
                loss = (
                    encoder_weight * loss_enc
                    + decoder_weight * loss_dec
                )
                total += loss.item()
                count += 1
        return total / max(count, 1)

    # Early stopping a validációs ranking-loss alapján.
    ranking_patience = 15
    ranking_min_delta = 1e-4
    best_ranking_val = float("inf")
    ranking_bad_checks = 0
    best_ranking_epoch = -1
    trainjoint=False
    if trainjoint:
        for epoch in range(5):
            enc.train()
            dec.train()

            order = list(range(len(train_data)))
            random.shuffle(order)

            total_loss = 0.0
            total_rank_loss = 0.0
            total_encoder_loss = 0.0

            for graph_id in order:
                graph = train_data[graph_id]
                ranking_data = train_ranking[graph_id]

                edge_attr = graph.edge_attr.float()
                if edge_attr.ndim == 1:
                    edge_attr = edge_attr.unsqueeze(-1)

                optimizer_joint.zero_grad(set_to_none=True)

                out = enc(
                    x=graph.x.float(),
                    edge_index=graph.edge_index,
                    edge_attr=edge_attr,
                )

                directed_logits = dec(build_decoder_input(out))

                loss_enc, _ = encoder_loss(
                    out=out,
                    batch=graph,
                )

                loss_dec, metrics, _ = matching_ranking_loss(
                    directed_logits=directed_logits,
                    ranking_data=ranking_data,
                    temperature=1.0,
                    score_reg_weight=1e-4,
                )

                loss = (
                        encoder_weight * loss_enc
                        + decoder_weight * loss_dec
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(enc.parameters()) + list(dec.parameters()),
                    max_norm=1.0,
                )

                optimizer_joint.step()

                total_loss += loss.item()
                total_rank_loss += metrics["rank_loss"].item()
                total_encoder_loss += loss_enc.item()

            count = len(train_data)

            print(
                f"epoch={epoch} | "
                f"train_total={total_loss / count:.6f} | "
                f"train_enc={total_encoder_loss / count:.6f} | "
                f"train_rank={total_rank_loss / count:.6f}"
            )



    # A legjobb (early-stopped) közös súlyok visszatöltése az értékeléshez.
    enc.load_state_dict(torch.load("enc.pt", weights_only=True))
    dec.load_state_dict(torch.load("dec.pt", weights_only=True))
    counts = []
    good = 0
    bad = 0
    for graph in val_data:
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
            num_nodes=graph.x.size(0),
            debug=False,
        )
        ranks = preference_lists_to_edge_ranks(
            graph.proposee_pref | graph.proposer_pref,
            graph.edge_index,
        )
        blocking_pairs = find_blocking_pairs(
            edge_index=graph.edge_index,
            edge_attr=ranks,
            partner=partner,
        )
        counts.append(len(blocking_pairs))

        # Olvasható párosítás: kanonikus (u, v) párok u < v szerint,
        # plusz a párosítatlan node-ok külön.
        partner_list = partner.detach().cpu().tolist()
        matched_pairs = sorted(
            {
                (min(u, v), max(u, v))
                for u, v in enumerate(partner_list)
                if v != -1
            }
        )
        unmatched = [
            u for u, v in enumerate(partner_list) if v == -1
        ]

        print("Párosítás (matching pairs):", matched_pairs)
        if unmatched:
            print("Párosítatlan node-ok:", unmatched)

        if blocking_pairs:
            print(
                f"Blokkoló párok ({len(blocking_pairs)}):",
                sorted(blocking_pairs),
            )
        else:
            print("Blokkoló párok: nincs (stabil párosítás)")

        if len(blocking_pairs)==0:
            good+=1
        else:
            bad += 1
    print("Blocking-count eloszlás:", dict(sorted(Counter(counts).items())))
    print("Átlag:", sum(counts) / len(counts))
    print("Stabilitási arány:",
          sum(c == 0 for c in counts) / len(counts))
    print(good/(good+bad))
    print("finished")

    # A teljes-matching rangsoroló kiértékelés (rank_all_complete_matchings)
    # kizárólag páros gráfokra készült (left/right oldal). Szobatárs esetben
    # a fenti blocking-count alapú kiértékelés adja a stabilitási metrikát,
    # így itt kilépünk.
    if roommate:
        return

    enc.eval()
    dec.eval()

    evaluated = 0
    no_complete_matching = 0
    no_stable_complete_matching = 0

    top1_stable = 0
    best_stable_ranks = []

    for graph in val_data:
        with torch.no_grad():
            edge_attr = graph.edge_attr.float()

            if edge_attr.ndim == 1:
                edge_attr = edge_attr.unsqueeze(-1)

            out = enc(
                x=graph.x.float(),
                edge_index=graph.edge_index,
                edge_attr=edge_attr,
            )

            directed_logits = dec(build_decoder_input(out))

        edge_ranks = preference_lists_to_edge_ranks(
            graph.proposee_pref | graph.proposer_pref,
            graph.edge_index,
        )

        results = rank_all_complete_matchings(
            edge_index=graph.edge_index,
            directed_logits=directed_logits,
            edge_ranks=edge_ranks,
            num_nodes=graph.num_nodes,
            left_nodes=[0, 1, 2],
            right_nodes=[3, 4, 5],
        )

        if not results:
            no_complete_matching += 1
            continue

        evaluated += 1
        top1_stable += int(results[0].is_stable)

        stable_ranks = [
            result.rank
            for result in results
            if result.is_stable
        ]

        if stable_ranks:
            best_stable_ranks.append(stable_ranks[0])
        else:
            no_stable_complete_matching += 1

    def print_matching_ranking(results: list[RankedMatching]):
        if not results:
            print("Nincs megengedett teljes matching.")
            return

        print("\nMatchingek össz-score szerint:")

        for result in results:
            print(
                f"rank={result.rank} | "
                f"score={result.score:+.6f} | "
                f"stable={result.is_stable} | "
                f"blocking_count={len(result.blocking_pairs)} | "
                f"pairs={result.pairs}"
            )

        stable_ranks = [
            result.rank
            for result in results
            if result.is_stable
        ]

        print("\nLegmagasabb score-ú matching:")
        print(results[0].pairs)
        print("Stabil:", results[0].is_stable)

        print("Stabil matchingek ranghelyei:", stable_ranks)
        print(
            "Legjobb stabil matching ranghelye:",
            stable_ranks[0] if stable_ranks else None,
        )

    enc.eval()
    dec.eval()

    graph = val_data[0]

    with torch.no_grad():
        edge_attr = graph.edge_attr.float()

        if edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        out = enc(
            x=graph.x.float(),
            edge_index=graph.edge_index,
            edge_attr=edge_attr,
        )

        directed_logits = dec(build_decoder_input(out))

    # Az eredeti preferenciákból képzett rankek.
    # Ne mutual-quality feature-t adj a checkernek.
    edge_ranks = preference_lists_to_edge_ranks(
        graph.proposee_pref | graph.proposer_pref,
        graph.edge_index,
    )

    results = rank_all_complete_matchings(
        edge_index=graph.edge_index,
        directed_logits=directed_logits,
        edge_ranks=edge_ranks,
        num_nodes=graph.num_nodes,
        left_nodes=[0, 1, 2],
        right_nodes=[3, 4, 5],
    )

    print_matching_ranking(results)
    print("Kiértékelt gráfok:", evaluated)
    print("Teljes matching nélküli gráfok:", no_complete_matching)
    print(
        "Stabil teljes matching nélküli gráfok:",
        no_stable_complete_matching,
    )

    if evaluated > 0:
        print(
            "Legmagasabb score-ú matching stabilitási aránya:",
            top1_stable / evaluated,
        )

        print(
            "Legjobb stabil matching rangeloszlása:",
            dict(sorted(Counter(best_stable_ranks).items())),
        )

        # Jelöltlista-lefedettség, nem automatikus solver-pontosság.
        for k in [1, 2, 3, 6]:
            contains_stable = sum(
                rank <= k
                for rank in best_stable_ranks
            )

            print(
                f"Top-{k} között van stabil matching: "
                f"{contains_stable / evaluated:.3f}"
            )


if __name__ == "__main__":
    training = False
    roommate = False
    LLM_FILE_GEN = False
    LLM_TEST = False
    GAT_TEST = False
    LOCAL_LLM = False
    main(training, roommate,LLM_FILE_GEN,LLM_TEST,GAT_TEST,LOCAL_LLM)