import networkx as nx
import torch
from torch_geometric.data import Data
import random
import gale_shapley
from itertools import chain
import statistics
import numpy as np

def generate_graph_m(group_nodes=10):
    G = nx.Graph()

    G.add_nodes_from(list(range(group_nodes)), bipartite=0)
    G.add_nodes_from(list(range(group_nodes)), bipartite=1)

    for i in range(group_nodes):
        for j in range(group_nodes):
            G.add_edge(i,group_nodes+j)
            G.add_edge(group_nodes + j,i)
    return G

def generate_graph_r(group_nodes=10):
    G = nx.Graph()

    G.add_nodes_from(list(range(group_nodes)))

    for i in range(group_nodes):
        for j in range(group_nodes):
            if i!=j:
                G.add_edge(i,j)
    return G

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

def calculate_embeddings(set1,set2,group_size,x_final,x):
    who_am_i = 0
    where_am_i = 0
    for node in set1:
        look = group_size
        x_final.append([])
        for i in range(4):
            x_final[who_am_i].append(0)

        ranks = []
        for node2 in set2:
            if x[look][0] == who_am_i:
                x_final[who_am_i][0]  +=1
            if x[look][-1] == who_am_i:
                x_final[who_am_i][2] -=1
            x_final[who_am_i][1]-=x[look].index(who_am_i)+1
            ranks.append(x[look].index(who_am_i)+1)
            look+=1
        ranks.sort()
        x_final[who_am_i][3]=statistics.median(ranks)
        where_am_i+=1
        who_am_i+=1


    who_am_i = 0
    where_am_i = group_size
    for node in set2:
        look = 0
        x_final.append([])
        for i in range(4):
            x_final[where_am_i].append(0)
        ranks = []
        for node2 in set1:
            if x[look][0] == who_am_i:
                x_final[where_am_i][0] += 1
            if x[look][-1] == who_am_i:
                x_final[where_am_i][2] -=1
            x_final[where_am_i][1] -= x[look].index(who_am_i)+1
            ranks.append(x[look].index(who_am_i)+1)
            look += 1
        ranks.sort()
        x_final[where_am_i][3]=statistics.median(ranks)
        where_am_i+=1
        who_am_i += 1


def graph_to_pyg_data_random(G, group_size):

    nodes = list(G.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes)}



    set1, set2 = nx.bipartite.sets(G)

    x = []
    proposer_pref = {}
    proposee_pref = {}

    edge_weight = []

    for node in set1:
        prefs_for_gale = list(range(group_size,group_size*2))
        random.shuffle(prefs_for_gale)
        prefs_for_model = [x - group_size for x in prefs_for_gale]
        edge_weight += prefs_for_model
        x.append(prefs_for_model)
        proposer_pref[node] = prefs_for_gale

    temp = []
    for node in set2:
        prefs_for_gale = list(range(group_size))
        random.shuffle(prefs_for_gale)
        prefs_for_model = prefs_for_gale
        temp += prefs_for_model

        edge_weight= list(chain(*zip(temp, edge_weight)))
        x.append(prefs_for_model)
        proposee_pref[node] = prefs_for_gale
    e_w = []
    for i in range(0,len(edge_weight),2):
        e_w.append(1.0-(edge_weight[i] + edge_weight[i+1]/2))
    edge_weight=list(chain(*zip(e_w, e_w)))
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)


    x_final = []

    calculate_embeddings(set1,set2,group_size,x_final,x)


    edges = []
    for u, v in G.edges():
        ui, vi = node_id_map[u], node_id_map[v]
        edges.append([ui, vi])
        edges.append([vi, ui])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    matching = gale_shapley.stable_matching_with_preferences(G,set1,set2,proposer_pref,proposee_pref)

    e_attr= []

    for u,v in G.edges():
        if u in matching and matching[u] == v:
            e_attr.append(1)
            e_attr.append(1)
        else:
            e_attr.append(0)
            e_attr.append(0)

    edge_attr = torch.tensor(e_attr, dtype=torch.float)
    data_x = torch.tensor(x_final, dtype=torch.float)

    prefs_ = proposee_pref | proposer_pref

    edge_id_of = {}
    src_list = []
    dst_list = []
    for u, pref_list in prefs_.items():

        for rank, v in enumerate(pref_list):
            edge_id = len(src_list)

            src_list.append(u)
            dst_list.append(v)

            edge_id_of[(u, v)] = edge_id

    better_indices = []
    worse_indices = []

    for u, pref_list in prefs_.items():
        edge_ids = [
            edge_id_of[(u, v)]
            for v in pref_list
        ]

        if len(edge_ids) < 2:
            continue
        pairs_per_agent = None
        if pairs_per_agent is None:
            # Csak szomszédos rangok:
            # 1. preferencia > 2. preferencia,
            # 2. preferencia > 3. preferencia, stb.
            for i in range(len(edge_ids) - 1):
                better_indices.append(edge_ids[i])
                worse_indices.append(edge_ids[i + 1])
        """
        else:
            # Véletlen rank-párok mintázása:
            for _ in range(pairs_per_agent):
                i = torch.randint(
                    low=0,
                    high=len(edge_ids) - 1,
                    size=(1,),
                ).item()

                j = torch.randint(
                    low=i + 1,
                    high=len(edge_ids),
                    size=(1,),
                ).item()

                better_indices.append(edge_ids[i])
                worse_indices.append(edge_ids[j])
        """
    data = Data(
        x=data_x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        edge_y=edge_attr,
        proposee_pref = proposee_pref,
        proposer_pref = proposer_pref,
        rank_better_idx = better_indices,
        rank_worse_idx = worse_indices
    )
    return data


def graph_to_pyg_data_r_random(G, group_size):
    nodes = list(G.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes)}

    prefs = {}
    for node in nodes:
        temp = nodes.copy()
        temp.remove(node)
        random.shuffle(temp)
        pref = temp
        prefs[node] = pref

    edge_attr = []
    edges = []
    for u, v in G.edges():
        ui, vi = node_id_map[u], node_id_map[v]
        edge_attr.append(prefs[u].index(v)+1 + prefs[v].index(u)+1)
        edges.append([ui, vi])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)


    who_am_i = 0
    data_x = []
    for node in nodes:
        data_x.append([])
        for i in range(4):
            data_x[who_am_i].append(0)

        ranks = []
        for node_ in nodes:
            if node_ == node:
                continue
            if prefs[node_][0] == node:
                data_x[who_am_i][0]+=1
            if prefs[node_][-1] == node:
                data_x[who_am_i][2]-=1
            data_x[who_am_i][1]+=prefs[node_].index(node)+1
            ranks.append(prefs[node_].index(node))
        ranks.sort()
        data_x[who_am_i][3] = statistics.median(ranks)
        who_am_i += 1
    data_x = torch.tensor(data_x, dtype=torch.float)

    data = Data(
        x=data_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        prefs = prefs
    )
    return data


def _ranking_from_scores(candidates, scores):
    """Rend egy jelöltlistát a hozzájuk tartozó pontszámok szerint (növekvő sorrend).

    A `candidates` és `scores` azonos hosszúságú listák; a visszatérési érték a
    `candidates` egy permutációja, így ugyanolyan formátumú preferenciasorrendet
    kapunk, mint a random.shuffle-nél, csak az adott eloszlás szerint rangsorolva.
    """
    order = sorted(range(len(candidates)), key=lambda i: scores[i])
    return [candidates[i] for i in order]


def graph_to_pyg_data_r_normal(G, group_size):
    """
    A `graph_to_pyg_data_normal` stabil-szobatárs (roommate) változata.

    Egyetlen, nem-partícionált csomóponthalmazon dolgozik (lásd
    `generate_graph_r`), a preferenciákat normális eloszlású pontszámokból
    rangsorolva állítja elő (mint a `graph_to_pyg_data_normal` a páros
    esetben), de olyan `Data` objektumot ad vissza, amely KOMPATIBILIS a
    tanítási loop-pal:

      - minden irányítatlan párhoz MINDKÉT irányított él bekerül
        (u->v és v->u), ahogy a páros változatban is;
      - edge_attr [E]: irányított él-"minőség" (jobb partner -> nagyobb),
        az encoder_loss mutual_quality regresszióhoz;
      - edge_y [E]: nullák (nincs stabil-szobatárs megoldó; az
        encoder_loss membership_loss-a nincs a total-ban, csak jelen kell
        lennie a mezőnek);
      - proposer_pref / proposee_pref: a szobatárs esetben nincs oldal-
        felosztás, ezért a teljes preferencia a proposer_pref-be kerül,
        a proposee_pref üres. A `proposee_pref | proposer_pref` únió így
        a teljes preferenciát adja vissza, amit a ranking-prep és a
        preference_lists_to_edge_ranks használ;
      - rank_better_idx / rank_worse_idx: szomszédos preferencia-párok
        irányított él-indexei (mint a páros változatban).

    group_size itt a csomópontok száma (a `generate_graph_r(group_nodes)`
    paraméterével megegyezően).
    """
    nodes = list(G.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes)}

    # Preferenciák normális eloszlású pontszámok alapján rangsorolva.
    # Minden csomópont minden másik csomópontot rangsorol (önmagát nem).
    prefs = {}
    for node in nodes:
        candidates = [other for other in nodes if other != node]
        scores = np.random.normal(0.0, 1.0, size=len(candidates))
        prefs[node] = _ranking_from_scores(candidates, scores)

    # rank_of[u][v] = u mennyire rangsorolja v-t (0 = legjobb).
    rank_of = {
        u: {v: r for r, v in enumerate(pref_list)}
        for u, pref_list in prefs.items()
    }

    # Mindkét irányított él bekerül minden irányítatlan párhoz, hogy a
    # (u,v) és (v,u) él is létezzen (a ranking/blocking logika ezt várja).
    edges = []
    edge_weight = []
    num_others = max(group_size - 1, 1)
    for u, v in G.edges():
        ui, vi = node_id_map[u], node_id_map[v]

        # u -> v irányított él "minősége": jobb partner -> nagyobb érték.
        q_uv = 1.0 - rank_of[u][v] / num_others
        # v -> u irányított él "minősége".
        q_vu = 1.0 - rank_of[v][u] / num_others

        edges.append([ui, vi])
        edge_weight.append(q_uv)

        edges.append([vi, ui])
        edge_weight.append(q_vu)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weight, dtype=torch.float)

    # Nincs stabil-szobatárs megoldó: a membership célt nullára állítjuk.
    edge_y = torch.zeros(edge_index.size(1), dtype=torch.float)

    # 4 dimenziós, kézzel számolt node-embedding (mint az r_random-ben).
    who_am_i = 0
    data_x = []
    for node in nodes:
        data_x.append([])
        for i in range(4):
            data_x[who_am_i].append(0)

        ranks = []
        for node_ in nodes:
            if node_ == node:
                continue
            if prefs[node_][0] == node:
                data_x[who_am_i][0] += 1
            if prefs[node_][-1] == node:
                data_x[who_am_i][2] -= 1
            data_x[who_am_i][1] += prefs[node_].index(node) + 1
            ranks.append(prefs[node_].index(node))
        ranks.sort()
        data_x[who_am_i][3] = statistics.median(ranks)
        who_am_i += 1
    data_x = torch.tensor(data_x, dtype=torch.float)

    # Szobatárs esetben nincs proposer/proposee felosztás: a teljes
    # preferencia a proposer_pref-be kerül, a proposee_pref üres.
    # A kulcsokat és értékeket a leképezett (mapped) node-id-kkel adjuk meg,
    # hogy egyezzenek az edge_index indexeivel (preference_lists_to_edge_ranks
    # ezekkel az id-kkal keres).
    proposer_pref = {
        node_id_map[u]: [node_id_map[v] for v in pref_list]
        for u, pref_list in prefs.items()
    }
    proposee_pref = {}

    # Szomszédos preferencia-párok irányított él-indexei (ranking-tanítójel).
    edge_id_of = {}
    for eid, (a, b) in enumerate(edges):
        edge_id_of[(a, b)] = eid

    better_indices = []
    worse_indices = []
    for u, pref_list in prefs.items():
        ui = node_id_map[u]
        edge_ids = []
        for v in pref_list:
            vi = node_id_map[v]
            if (ui, vi) in edge_id_of:
                edge_ids.append(edge_id_of[(ui, vi)])

        if len(edge_ids) < 2:
            continue
        for i in range(len(edge_ids) - 1):
            better_indices.append(edge_ids[i])
            worse_indices.append(edge_ids[i + 1])

    better_indices = torch.tensor(better_indices, dtype=torch.long)
    worse_indices = torch.tensor(worse_indices, dtype=torch.long)

    data = Data(
        x=data_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_y=edge_y,
        prefs=prefs,
        proposer_pref=proposer_pref,
        proposee_pref=proposee_pref,
        rank_better_idx=better_indices,
        rank_worse_idx=worse_indices,
    )
    return data


def graph_to_pyg_data_normal(G, group_size):

    nodes = list(G.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes)}



    set1, set2 = nx.bipartite.sets(G)

    x = []
    proposer_pref = {}
    proposee_pref = {}

    edge_weight = []

    for node in set1:
        candidates = list(range(group_size, group_size * 2))
        scores = np.random.normal(0.0, 1.0, size=len(candidates))
        prefs_for_gale = _ranking_from_scores(candidates, scores)
        prefs_for_model = [x - group_size for x in prefs_for_gale]
        edge_weight += prefs_for_model
        x.append(prefs_for_model)
        proposer_pref[node] = prefs_for_gale

    temp = []
    for node in set2:
        candidates = list(range(group_size))
        scores = np.random.normal(0.0, 1.0, size=len(candidates))
        prefs_for_gale = _ranking_from_scores(candidates, scores)
        prefs_for_model = prefs_for_gale
        temp += prefs_for_model

        edge_weight = list(chain(*zip(temp, edge_weight)))
        x.append(prefs_for_model)
        proposee_pref[node] = prefs_for_gale
    e_w = []
    for i in range(0, len(edge_weight), 2):
        e_w.append(1.0 - (edge_weight[i] + edge_weight[i + 1] / 2))
    edge_weight = list(chain(*zip(e_w, e_w)))
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)


    x_final = []

    calculate_embeddings(set1, set2, group_size, x_final, x)


    edges = []
    for u, v in G.edges():
        ui, vi = node_id_map[u], node_id_map[v]
        edges.append([ui, vi])
        edges.append([vi, ui])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    matching = gale_shapley.stable_matching_with_preferences(G, set1, set2, proposer_pref, proposee_pref)

    e_attr = []

    for u, v in G.edges():
        if u in matching and matching[u] == v:
            e_attr.append(1)
            e_attr.append(1)
        else:
            e_attr.append(0)
            e_attr.append(0)

    edge_attr = torch.tensor(e_attr, dtype=torch.float)
    data_x = torch.tensor(x_final, dtype=torch.float)

    prefs_ = proposee_pref | proposer_pref

    edge_id_of = {}
    src_list = []
    dst_list = []
    for u, pref_list in prefs_.items():

        for rank, v in enumerate(pref_list):
            edge_id = len(src_list)

            src_list.append(u)
            dst_list.append(v)

            edge_id_of[(u, v)] = edge_id

    better_indices = []
    worse_indices = []

    for u, pref_list in prefs_.items():
        edge_ids = [
            edge_id_of[(u, v)]
            for v in pref_list
        ]

        if len(edge_ids) < 2:
            continue
        pairs_per_agent = None
        if pairs_per_agent is None:
            for i in range(len(edge_ids) - 1):
                better_indices.append(edge_ids[i])
                worse_indices.append(edge_ids[i + 1])
    data = Data(
        x=data_x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        edge_y=edge_attr,
        proposee_pref=proposee_pref,
        proposer_pref=proposer_pref,
        rank_better_idx=better_indices,
        rank_worse_idx=worse_indices
    )
    return data


def graph_to_pyg_data_uniform(G, group_size):

    nodes = list(G.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes)}



    set1, set2 = nx.bipartite.sets(G)

    x = []
    proposer_pref = {}
    proposee_pref = {}

    edge_weight = []

    for node in set1:
        candidates = list(range(group_size, group_size * 2))
        scores = np.random.uniform(0.0, 1.0, size=len(candidates))
        prefs_for_gale = _ranking_from_scores(candidates, scores)
        prefs_for_model = [x - group_size for x in prefs_for_gale]
        edge_weight += prefs_for_model
        x.append(prefs_for_model)
        proposer_pref[node] = prefs_for_gale

    temp = []
    for node in set2:
        candidates = list(range(group_size))
        scores = np.random.uniform(0.0, 1.0, size=len(candidates))
        prefs_for_gale = _ranking_from_scores(candidates, scores)
        prefs_for_model = prefs_for_gale
        temp += prefs_for_model

        edge_weight = list(chain(*zip(temp, edge_weight)))
        x.append(prefs_for_model)
        proposee_pref[node] = prefs_for_gale
    e_w = []
    for i in range(0, len(edge_weight), 2):
        e_w.append(1.0 - (edge_weight[i] + edge_weight[i + 1] / 2))
    edge_weight = list(chain(*zip(e_w, e_w)))
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)


    x_final = []

    calculate_embeddings(set1, set2, group_size, x_final, x)


    edges = []
    for u, v in G.edges():
        ui, vi = node_id_map[u], node_id_map[v]
        edges.append([ui, vi])
        edges.append([vi, ui])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    matching = gale_shapley.stable_matching_with_preferences(G, set1, set2, proposer_pref, proposee_pref)

    e_attr = []

    for u, v in G.edges():
        if u in matching and matching[u] == v:
            e_attr.append(1)
            e_attr.append(1)
        else:
            e_attr.append(0)
            e_attr.append(0)

    edge_attr = torch.tensor(e_attr, dtype=torch.float)
    data_x = torch.tensor(x_final, dtype=torch.float)

    prefs_ = proposee_pref | proposer_pref

    edge_id_of = {}
    src_list = []
    dst_list = []
    for u, pref_list in prefs_.items():

        for rank, v in enumerate(pref_list):
            edge_id = len(src_list)

            src_list.append(u)
            dst_list.append(v)

            edge_id_of[(u, v)] = edge_id

    better_indices = []
    worse_indices = []

    for u, pref_list in prefs_.items():
        edge_ids = [
            edge_id_of[(u, v)]
            for v in pref_list
        ]

        if len(edge_ids) < 2:
            continue
        pairs_per_agent = None
        if pairs_per_agent is None:
            for i in range(len(edge_ids) - 1):
                better_indices.append(edge_ids[i])
                worse_indices.append(edge_ids[i + 1])
    data = Data(
        x=data_x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        edge_y=edge_attr,
        proposee_pref=proposee_pref,
        proposer_pref=proposer_pref,
        rank_better_idx=better_indices,
        rank_worse_idx=worse_indices
    )
    return data


def graph_to_pyg_data_binomial(G, group_size):

    nodes = list(G.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes)}



    set1, set2 = nx.bipartite.sets(G)

    x = []
    proposer_pref = {}
    proposee_pref = {}

    edge_weight = []

    for node in set1:
        candidates = list(range(group_size, group_size * 2))
        # Diszkrét pontszámok + apró zaj a holtversenyek feloldására
        scores = np.random.binomial(len(candidates), 0.5, size=len(candidates)).astype(float)
        scores += np.random.uniform(0.0, 1e-6, size=len(candidates))
        prefs_for_gale = _ranking_from_scores(candidates, scores)
        prefs_for_model = [x - group_size for x in prefs_for_gale]
        edge_weight += prefs_for_model
        x.append(prefs_for_model)
        proposer_pref[node] = prefs_for_gale

    temp = []
    for node in set2:
        candidates = list(range(group_size))
        scores = np.random.binomial(len(candidates), 0.5, size=len(candidates)).astype(float)
        scores += np.random.uniform(0.0, 1e-6, size=len(candidates))
        prefs_for_gale = _ranking_from_scores(candidates, scores)
        prefs_for_model = prefs_for_gale
        temp += prefs_for_model

        edge_weight = list(chain(*zip(temp, edge_weight)))
        x.append(prefs_for_model)
        proposee_pref[node] = prefs_for_gale
    e_w = []
    for i in range(0, len(edge_weight), 2):
        e_w.append(1.0 - (edge_weight[i] + edge_weight[i + 1] / 2))
    edge_weight = list(chain(*zip(e_w, e_w)))
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)


    x_final = []

    calculate_embeddings(set1, set2, group_size, x_final, x)


    edges = []
    for u, v in G.edges():
        ui, vi = node_id_map[u], node_id_map[v]
        edges.append([ui, vi])
        edges.append([vi, ui])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    matching = gale_shapley.stable_matching_with_preferences(G, set1, set2, proposer_pref, proposee_pref)

    e_attr = []

    for u, v in G.edges():
        if u in matching and matching[u] == v:
            e_attr.append(1)
            e_attr.append(1)
        else:
            e_attr.append(0)
            e_attr.append(0)

    edge_attr = torch.tensor(e_attr, dtype=torch.float)
    data_x = torch.tensor(x_final, dtype=torch.float)

    prefs_ = proposee_pref | proposer_pref

    edge_id_of = {}
    src_list = []
    dst_list = []
    for u, pref_list in prefs_.items():

        for rank, v in enumerate(pref_list):
            edge_id = len(src_list)

            src_list.append(u)
            dst_list.append(v)

            edge_id_of[(u, v)] = edge_id

    better_indices = []
    worse_indices = []

    for u, pref_list in prefs_.items():
        edge_ids = [
            edge_id_of[(u, v)]
            for v in pref_list
        ]

        if len(edge_ids) < 2:
            continue
        pairs_per_agent = None
        if pairs_per_agent is None:
            for i in range(len(edge_ids) - 1):
                better_indices.append(edge_ids[i])
                worse_indices.append(edge_ids[i + 1])
    data = Data(
        x=data_x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        edge_y=edge_attr,
        proposee_pref=proposee_pref,
        proposer_pref=proposer_pref,
        rank_better_idx=better_indices,
        rank_worse_idx=worse_indices
    )
    return data


def graph_to_pyg_data_poisson(G, group_size):

    nodes = list(G.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes)}



    set1, set2 = nx.bipartite.sets(G)

    x = []
    proposer_pref = {}
    proposee_pref = {}

    edge_weight = []

    for node in set1:
        candidates = list(range(group_size, group_size * 2))
        scores = np.random.poisson(group_size, size=len(candidates)).astype(float)
        scores += np.random.uniform(0.0, 1e-6, size=len(candidates))
        prefs_for_gale = _ranking_from_scores(candidates, scores)
        prefs_for_model = [x - group_size for x in prefs_for_gale]
        edge_weight += prefs_for_model
        x.append(prefs_for_model)
        proposer_pref[node] = prefs_for_gale

    temp = []
    for node in set2:
        candidates = list(range(group_size))
        scores = np.random.poisson(group_size, size=len(candidates)).astype(float)
        scores += np.random.uniform(0.0, 1e-6, size=len(candidates))
        prefs_for_gale = _ranking_from_scores(candidates, scores)
        prefs_for_model = prefs_for_gale
        temp += prefs_for_model

        edge_weight = list(chain(*zip(temp, edge_weight)))
        x.append(prefs_for_model)
        proposee_pref[node] = prefs_for_gale
    e_w = []
    for i in range(0, len(edge_weight), 2):
        e_w.append(1.0 - (edge_weight[i] + edge_weight[i + 1] / 2))
    edge_weight = list(chain(*zip(e_w, e_w)))
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)


    x_final = []

    calculate_embeddings(set1, set2, group_size, x_final, x)


    edges = []
    for u, v in G.edges():
        ui, vi = node_id_map[u], node_id_map[v]
        edges.append([ui, vi])
        edges.append([vi, ui])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    matching = gale_shapley.stable_matching_with_preferences(G, set1, set2, proposer_pref, proposee_pref)

    e_attr = []

    for u, v in G.edges():
        if u in matching and matching[u] == v:
            e_attr.append(1)
            e_attr.append(1)
        else:
            e_attr.append(0)
            e_attr.append(0)

    edge_attr = torch.tensor(e_attr, dtype=torch.float)
    data_x = torch.tensor(x_final, dtype=torch.float)

    prefs_ = proposee_pref | proposer_pref

    edge_id_of = {}
    src_list = []
    dst_list = []
    for u, pref_list in prefs_.items():

        for rank, v in enumerate(pref_list):
            edge_id = len(src_list)

            src_list.append(u)
            dst_list.append(v)

            edge_id_of[(u, v)] = edge_id

    better_indices = []
    worse_indices = []

    for u, pref_list in prefs_.items():
        edge_ids = [
            edge_id_of[(u, v)]
            for v in pref_list
        ]

        if len(edge_ids) < 2:
            continue
        pairs_per_agent = None
        if pairs_per_agent is None:
            for i in range(len(edge_ids) - 1):
                better_indices.append(edge_ids[i])
                worse_indices.append(edge_ids[i + 1])
    data = Data(
        x=data_x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        edge_y=edge_attr,
        proposee_pref=proposee_pref,
        proposer_pref=proposer_pref,
        rank_better_idx=better_indices,
        rank_worse_idx=worse_indices
    )
    return data


def graph_to_pyg_data_exponential(G, group_size):

    nodes = list(G.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes)}



    set1, set2 = nx.bipartite.sets(G)

    x = []
    proposer_pref = {}
    proposee_pref = {}

    edge_weight = []

    for node in set1:
        candidates = list(range(group_size, group_size * 2))
        scores = np.random.exponential(1.0, size=len(candidates))
        prefs_for_gale = _ranking_from_scores(candidates, scores)
        prefs_for_model = [x - group_size for x in prefs_for_gale]
        edge_weight += prefs_for_model
        x.append(prefs_for_model)
        proposer_pref[node] = prefs_for_gale

    temp = []
    for node in set2:
        candidates = list(range(group_size))
        scores = np.random.exponential(1.0, size=len(candidates))
        prefs_for_gale = _ranking_from_scores(candidates, scores)
        prefs_for_model = prefs_for_gale
        temp += prefs_for_model

        edge_weight = list(chain(*zip(temp, edge_weight)))
        x.append(prefs_for_model)
        proposee_pref[node] = prefs_for_gale
    e_w = []
    for i in range(0, len(edge_weight), 2):
        e_w.append(1.0 - (edge_weight[i] + edge_weight[i + 1] / 2))
    edge_weight = list(chain(*zip(e_w, e_w)))
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)


    x_final = []

    calculate_embeddings(set1, set2, group_size, x_final, x)


    edges = []
    for u, v in G.edges():
        ui, vi = node_id_map[u], node_id_map[v]
        edges.append([ui, vi])
        edges.append([vi, ui])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    matching = gale_shapley.stable_matching_with_preferences(G, set1, set2, proposer_pref, proposee_pref)

    e_attr = []

    for u, v in G.edges():
        if u in matching and matching[u] == v:
            e_attr.append(1)
            e_attr.append(1)
        else:
            e_attr.append(0)
            e_attr.append(0)

    edge_attr = torch.tensor(e_attr, dtype=torch.float)
    data_x = torch.tensor(x_final, dtype=torch.float)

    prefs_ = proposee_pref | proposer_pref

    edge_id_of = {}
    src_list = []
    dst_list = []
    for u, pref_list in prefs_.items():

        for rank, v in enumerate(pref_list):
            edge_id = len(src_list)

            src_list.append(u)
            dst_list.append(v)

            edge_id_of[(u, v)] = edge_id

    better_indices = []
    worse_indices = []

    for u, pref_list in prefs_.items():
        edge_ids = [
            edge_id_of[(u, v)]
            for v in pref_list
        ]

        if len(edge_ids) < 2:
            continue
        pairs_per_agent = None
        if pairs_per_agent is None:
            for i in range(len(edge_ids) - 1):
                better_indices.append(edge_ids[i])
                worse_indices.append(edge_ids[i + 1])
    data = Data(
        x=data_x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        edge_y=edge_attr,
        proposee_pref=proposee_pref,
        proposer_pref=proposer_pref,
        rank_better_idx=better_indices,
        rank_worse_idx=worse_indices
    )
    return data
