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


def _tiered_ranking_from_scores(candidates, scores, bucket=0.6):
    """Preferencia-RÉTEGEKET (tiers) állít elő a pontszámokból, HOLTVERSENYEKKEL.

    A szobatárs (roommate) esetben megengedjük, hogy több partner AZONOS
    rangot kapjon (holtverseny). A megvalósítás: a pontszámokat `bucket`
    felbontással diszkretizáljuk (kvantáljuk), majd az azonos kvantált
    értékű jelöltek egy közös rétegbe (tier) kerülnek. A rétegeket a
    kvantált érték szerint rendezzük (növekvő = jobb elöl), így a kimenet
    rétegek listája:

        [[legjobb réteg tagjai], ..., [legrosszabb réteg tagjai]]

    Egy réteg (tier) minden tagja azonos rangú (holtverseny).

    `candidates` és `scores` azonos hosszúságú listák. `bucket` szabályozza
    a holtversenyek gyakoriságát: nagyobb érték -> több holtverseny.
    """
    # Kvantált kulcs: azonos kulcsú jelöltek holtversenyben vannak.
    keyed = [
        (round(scores[i] / bucket), candidates[i])
        for i in range(len(candidates))
    ]
    # Rendezés a kvantált kulcs szerint (növekvő = jobb elöl).
    keyed.sort(key=lambda kc: kc[0])

    tiers = []
    current_key = None
    for key, cand in keyed:
        if key != current_key:
            tiers.append([cand])
            current_key = key
        else:
            tiers[-1].append(cand)
    return tiers


def _strict_tiers_from_scores(candidates, scores):
    """Réteges formátumú, de HOLTVERSENY NÉLKÜLI preferencia a pontszámokból.

    Minden jelölt a SAJÁT egyelemű rétegébe (tier) kerül, a pontszám
    szerinti (növekvő = jobb) sorrendben. Így az eredmény ugyanabban a
    réteges reprezentációban van, mint `_tiered_ranking_from_scores`, de
    numerikusan megegyezik a régi, szigorú (holtverseny nélküli) adattal:

        [[legjobb], [második], ..., [legrosszabb]]
    """
    flat = _ranking_from_scores(candidates, scores)
    return [[c] for c in flat]


def _is_tiered(pref_list):
    """True, ha a preferencialista RÉTEGELT (tiers = listák listája).

    A réteges (szobatárs, holtversenyes) formátumban az első elem maga is
    lista. A sima (házassági) formátumban az elemek node-id-k (int).
    """
    return len(pref_list) > 0 and isinstance(pref_list[0], (list, tuple))


def _tier_rank_of(pref_list):
    """{partner_id: tier_index} leképezés egy (réteges VAGY sima) listából.

    Réteges formátumnál egy réteg minden tagja AZONOS rangot (a réteg
    indexét) kapja -> holtverseny. Sima listánál a rang az elem indexe
    (minden partner külön rangot kap).
    """
    rank_of = {}
    if _is_tiered(pref_list):
        for tier_idx, tier in enumerate(pref_list):
            for member in tier:
                rank_of[member] = tier_idx
    else:
        for idx, member in enumerate(pref_list):
            rank_of[member] = idx
    return rank_of


def _flatten_tiers(pref_list):
    """Réteges listát sima (best->worst) listává lapít; sima listát változatlanul ad."""
    if _is_tiered(pref_list):
        flat = []
        for tier in pref_list:
            flat.extend(tier)
        return flat
    return list(pref_list)


def graph_to_pyg_data_r_normal(G, group_size, allow_ties=False, score_fn=None):
    """
    A `graph_to_pyg_data_normal` stabil-szobatárs (roommate) változata.

    Egyetlen, nem-partícionált csomóponthalmazon dolgozik (lásd
    `generate_graph_r`), a preferenciákat normális eloszlású pontszámokból
    rangsorolva állítja elő (mint a `graph_to_pyg_data_normal` a páros
    esetben), de olyan `Data` objektumot ad vissza, amely KOMPATIBILIS a
    tanítási loop-pal:

    allow_ties:
        Ha False (alapértelmezett), a preferenciák HOLTVERSENY NÉLKÜLIEK:
        minden partner külön rangot kap (numerikusan a régi adattal
        egyezik, csak a reprezentáció réteges). Ha True, a közeli
        pontszámú partnerek azonos rangot (holtversenyt) kaphatnak.

    score_fn:
        Opcionális `size -> 1D pontszám-tömb` függvény, amely a
        preferencia-pontszámokat adja (kisebb pontszám = jobb partner, a
        rangsor ezek szerint növekvő). Ha None, a pontszámok STANDARD
        NORMÁLIS eloszlásból (N(0,1)) jönnek, azaz a jelenlegi, bevált
        viselkedés. Más eloszlásokhoz lásd a `graph_to_pyg_data_r_*`
        wrappereket (uniform, exponenciális, power-law stb.), amelyek
        ugyanezt a konvertert hívják más `score_fn`-nel. Így az ÚJ
        gráfok a mostani holtversenyes szobatárs-problémához NAGYON
        HASONLÓK, csak a preferenciák eloszlása tér el.

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

    # Preferenciák normális eloszlású pontszámok alapján, RÉTEGELT
    # formában. prefs[node] = rétegek listája, ahol egy réteg minden tagja
    # azonos rangú.
    #   - allow_ties=False (alapértelmezett): HOLTVERSENY NÉLKÜLI, minden
    #     partner saját egyelemű rétegben -> numerikusan a régi, szigorú
    #     adattal egyezik (csak a reprezentáció réteges).
    #   - allow_ties=True: a közeli pontszámú partnerek közös rétegbe
    #     (holtversenybe) kerülnek -> több partner kaphat azonos rangot.
    prefs = {}
    # A pontszám-eloszlás függvénye: ha nincs megadva, standard normális.
    if score_fn is None:
        score_fn = lambda size: np.random.normal(0.0, 1.0, size=size)

    for node in nodes:
        candidates = [other for other in nodes if other != node]
        scores = score_fn(len(candidates))
        if allow_ties:
            prefs[node] = _tiered_ranking_from_scores(
                candidates, list(scores)
            )
        else:
            prefs[node] = _strict_tiers_from_scores(
                candidates, list(scores)
            )

    # rank_of[u][v] = u mennyire rangsorolja v-t (0 = legjobb). Réteges
    # formánál egy réteg minden tagja AZONOS rangot kap (holtverseny).
    rank_of = {
        u: _tier_rank_of(pref_list)
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
        # A rang a RÉTEG indexe, így a holtversenyben lévő partnerek
        # azonos minőséget (quality) kapnak.
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
    # A holtversenyes (réteges) preferenciákkal a "legjobb"/"legrosszabb"
    # réteg TÖBB tagot is tartalmazhat, a rang pedig a réteg indexe.
    who_am_i = 0
    data_x = []
    for node in nodes:
        data_x.append([])
        for i in range(4):
            data_x[who_am_i].append(0)

        ranks = []
        others_count = 0  # hány másik csomópont rangsorolja `node`-ot
        for node_ in nodes:
            if node_ == node:
                continue
            other_pref = prefs[node_]
            other_rank_of = rank_of[node_]
            best_rank = 0
            worst_rank = len(other_pref) - 1  # utolsó réteg indexe
            r = other_rank_of[node]
            others_count += 1
            # Hányan tartják `node`-ot a LEGJOBB rétegükben (nyers szám,
            # a ciklus után százalékra alakítjuk).
            if r == best_rank:
                data_x[who_am_i][0] += 1
            # Hányan tartják `node`-ot a LEGROSSZABB rétegükben (nyers
            # szám, a ciklus után százalékra alakítjuk).
            if r == worst_rank:
                data_x[who_am_i][2] += 1
            # Rang-összeg SKÁLÁZOTT rangokból: az adott rangsoroló
            # csomópont LEGJOBB rangja -> 0, LEGROSSZABB rangja -> 1,
            # a köztes rangokat lineárisan skálázzuk (r / worst_rank),
            # majd ezeket a 0..1 közötti értékeket összegezzük. Ha csak
            # egyetlen lehetséges rang van (worst_rank == 0, azaz egyetlen
            # réteg), akkor a skálázott rang 0.
            if worst_rank > best_rank:
                scaled_rank = (r - best_rank) / (worst_rank - best_rank)
            else:
                scaled_rank = 0.0
            data_x[who_am_i][1] += scaled_rank
            # A MEDIÁN embedding is a SKÁLÁZOTT rangokból számol (best=0,
            # worst=1), ezért a skálázott értéket gyűjtjük.
            ranks.append(scaled_rank)
        # A LEGJOBB helyre sorolás embeddingje SZÁZALÉK: a csomópontot az
        # első (legjobb) helyre tevő csomópontok ARÁNYA az összes másik
        # csomóponthoz képest (0..100).
        if others_count > 0:
            data_x[who_am_i][0] = 100.0 * data_x[who_am_i][0] / others_count
        else:
            data_x[who_am_i][0] = 0.0
        # A LEGROSSZABB helyre sorolás embeddingje NEGATÍV SZÁZALÉK: a
        # csomópontot az utolsó (legrosszabb) rétegbe tevő csomópontok
        # ARÁNYA az összes másik csomóponthoz képest, -1-gyel szorozva
        # (-100..0).
        if others_count > 0:
            data_x[who_am_i][2] = -100.0 * data_x[who_am_i][2] / others_count
        else:
            data_x[who_am_i][2] = 0.0
        ranks.sort()
        data_x[who_am_i][3] = statistics.median(ranks)
        who_am_i += 1
    data_x = torch.tensor(data_x, dtype=torch.float)

    # Szobatárs esetben nincs proposer/proposee felosztás: a teljes
    # preferencia a proposer_pref-be kerül, a proposee_pref üres.
    # A preferencia RÉTEGELT formában marad (listák listája), a kulcsokat
    # és a tagokat a leképezett (mapped) node-id-kkel adjuk meg, hogy
    # egyezzenek az edge_index indexeivel. A
    # preference_lists_to_edge_ranks felismeri a réteges formát és a
    # holtversenyben lévő partnereknek azonos rangot ad.
    proposer_pref = {
        node_id_map[u]: [
            [node_id_map[v] for v in tier]
            for tier in pref_list
        ]
        for u, pref_list in prefs.items()
    }
    proposee_pref = {}

    # Szomszédos preferencia-párok irányított él-indexei (ranking-tanítójel).
    # Holtversenynél CSAK a rétegHATÁROKON túli párokat tekintjük
    # jobb/rosszabb viszonynak: egy rétegen belüli tagok azonos rangúak,
    # így NEM alkotnak better/worse párt. Minden réteg legutolsó tagját a
    # következő réteg első tagjával kötjük össze.
    edge_id_of = {}
    for eid, (a, b) in enumerate(edges):
        edge_id_of[(a, b)] = eid

    better_indices = []
    worse_indices = []
    for u, pref_list in prefs.items():
        ui = node_id_map[u]

        # Rétegenként az él-id-k, a rétegsorrend megtartásával.
        tier_edge_ids = []  # list[list[int]] rétegenként
        for tier in pref_list:
            ids = []
            for v in tier:
                vi = node_id_map[v]
                if (ui, vi) in edge_id_of:
                    ids.append(edge_id_of[(ui, vi)])
            tier_edge_ids.append(ids)

        # Egymást követő, NEM üres rétegek között húzunk better->worse élt
        # (az előző réteg bármely tagja jobb a következő réteg bármely
        # tagjánál). A párosítási tanítójelhez elég rétegenként egy
        # reprezentatív pár (előző réteg utolsó -> következő réteg első).
        prev_ids = None
        for ids in tier_edge_ids:
            if not ids:
                continue
            if prev_ids is not None:
                better_indices.append(prev_ids[-1])
                worse_indices.append(ids[0])
            prev_ids = ids

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


# ---------------------------------------------------------------------------
# Szobatárs (roommate) gráfok MÁS preferencia-eloszlásokkal.
#
# Mindegyik converter TELJESEN AZONOS a `graph_to_pyg_data_r_normal`-lal
# (ugyanaz a réteges/holtversenyes reprezentáció, edge_attr, data_x,
# rank_better/worse_idx, proposer_pref stb.), KIZÁRÓLAG a preferencia-
# pontszámok ELOSZLÁSA tér el. Így közvetlenül összehasonlítható, hogy a
# modell mennyire általánosít más eloszlású preferenciákra.
#
# A pontszámokat minden eloszlásnál ~egységnyi szórásúra normalizáljuk,
# hogy a `_tiered_ranking_from_scores` 0.6-os bucketje hasonló
# holtverseny-gyakoriságot adjon, mint a normális esetben (különben a
# holtversenyek sűrűsége erősen eltérne eloszlásonként).
# A pontszámok ELŐJELE/skálája közömbös a rangsor szempontjából (csak a
# relatív sorrend számít), de a bucketeléshez a skála lényeges.
# ---------------------------------------------------------------------------


def graph_to_pyg_data_r_uniform(G, group_size, allow_ties=False):
    """Szobatárs gráf EGYENLETES (uniform) eloszlású preferencia-pontszámokkal.

    A pontszámok U(0,1)-ből jönnek, ~egységnyi szórásúra skálázva. A
    többi (reprezentáció, mezők) megegyezik `graph_to_pyg_data_r_normal`-lal.
    """
    def score_fn(size):
        # U(0,1) szórása ~0.2887; /0.2887 ~ egységnyi szórás.
        return np.random.uniform(0.0, 1.0, size=size) / 0.2887

    return graph_to_pyg_data_r_normal(
        G, group_size, allow_ties=allow_ties, score_fn=score_fn
    )


def graph_to_pyg_data_r_exponential(G, group_size, allow_ties=False):
    """Szobatárs gráf EXPONENCIÁLIS eloszlású preferencia-pontszámokkal.

    Erősen jobbra ferde eloszlás: sok közeli (jó) és néhány kiugró (rossz)
    pontszám. Exp(1) szórása 1, így skálázás nem szükséges. A többi
    megegyezik `graph_to_pyg_data_r_normal`-lal.
    """
    def score_fn(size):
        return np.random.exponential(1.0, size=size)

    return graph_to_pyg_data_r_normal(
        G, group_size, allow_ties=allow_ties, score_fn=score_fn
    )


def graph_to_pyg_data_r_powerlaw(G, group_size, allow_ties=False):
    """Szobatárs gráf POWER-LAW (Pareto) eloszlású preferencia-pontszámokkal.

    Nehéz farkú eloszlás: néhány csomópont sokkal "jobb"/"rosszabb"
    pontszámot kaphat. Pareto(a=3)-ból mintázunk, majd ~egységnyi szórásúra
    skálázunk. A többi megegyezik `graph_to_pyg_data_r_normal`-lal.
    """
    a = 3.0
    # Pareto(a) szórása (a>2): sqrt(a / ((a-1)^2 (a-2))).
    std = (a / (((a - 1.0) ** 2) * (a - 2.0))) ** 0.5

    def score_fn(size):
        return np.random.pareto(a, size=size) / std

    return graph_to_pyg_data_r_normal(
        G, group_size, allow_ties=allow_ties, score_fn=score_fn
    )


def graph_to_pyg_data_r_bimodal(G, group_size, allow_ties=False):
    """Szobatárs gráf BIMODÁLIS (két csúcsú) preferencia-pontszámokkal.

    Két normális komponens keveréke (az egyik ~ -2, a másik ~ +2 körül),
    ami "polarizált" preferenciákat ad: a partnerek jellemzően vagy
    "jók", vagy "rosszak", kevés köztes. ~egységnyi szórásúra skálázva.
    A többi megegyezik `graph_to_pyg_data_r_normal`-lal.
    """
    def score_fn(size):
        # Komponensválasztás 50/50, középpontok -2 és +2, szórás 0.5.
        comp = np.random.randint(0, 2, size=size)
        centers = np.where(comp == 0, -2.0, 2.0)
        raw = centers + np.random.normal(0.0, 0.5, size=size)
        # A keverék szórása ~ sqrt(2^2 + 0.5^2) ~ 2.06; egységnyire skáláz.
        return raw / 2.06

    return graph_to_pyg_data_r_normal(
        G, group_size, allow_ties=allow_ties, score_fn=score_fn
    )


def graph_to_pyg_data_r_top_ties(G, group_size, allow_ties=True):
    """Szobatárs gráf, ahol az ELSŐ (legjobb) helyen NAGY VALÓSZÍNŰSÉGGEL SOK a tie.

    Célzottan olyan preferencia-pontszámokat gyárt, hogy a `_tiered_ranking_
    from_scores` kvantálása után a LEGJOBB réteg (tier 0) tipikusan több
    tagot tartalmazzon -> sok holtverseny az első helyen.

    Hogyan: a jelöltek egy VÉLETLEN hányada egy szűk "top platóra" kerül a
    pontszám-minimum közelében (a kvantáló `bucket`-jénél jóval kisebb
    szóráson belül), így ezek AZONOS kvantált kulcsot kapnak -> közös,
    legjobb rétegbe (holtverseny). A maradék jelölt pontszáma jóval nagyobb
    (rosszabb), és szétszórt, így külön, rosszabb rétegekbe esik.

    A tie-k az `allow_ties=True` esetén jelennek meg (a kvantálás ekkor
    aktív); `allow_ties=False`-nál a reprezentáció szigorú marad, mint a
    többi konverternél. A platóméretet mintánként véletlenül választjuk,
    hogy ne legyen determinisztikus a tie-k száma.

    A `_tiered_ranking_from_scores` alap bucket-je 0.6. A plató szórását
    ennél jóval kisebbre (~0.05) vesszük, hogy a platótagok megbízhatóan
    egy rétegbe essenek; a nem-platós jelölteket pedig legalább néhány
    bucket-nyivel feljebb (rosszabbra) toljuk.
    """
    def score_fn(size):
        if size <= 0:
            return np.array([], dtype=float)
        # A plató (top, holtversenyes) jelöltek aránya: 40%..90% a
        # jelöltek közül, de legalább 2 (hogy legyen tényleges tie), ha a
        # méret engedi.
        frac = np.random.uniform(0.4, 0.9)
        n_top = int(round(frac * size))
        n_top = max(min(n_top, size), min(2, size))

        scores = np.empty(size, dtype=float)
        # Top plató: szűk szórású a 0 körül (<< bucket=0.6), így azonos
        # kvantált kulcs -> közös legjobb réteg.
        scores[:n_top] = np.random.normal(0.0, 0.05, size=n_top)
        # A maradék (rosszabb) jelölt: legalább ~2 bucketnyivel feljebb
        # (>= 1.2), szétszórva, hogy több rosszabb rétegbe essenek.
        n_rest = size - n_top
        if n_rest > 0:
            scores[n_top:] = 1.2 + np.random.exponential(1.0, size=n_rest)
        # Keverjük, hogy a plató ne a lista elején legyen (a rangsorolást
        # a pontszám dönti el, a sorrend csak a holtversenyen belül számít).
        np.random.shuffle(scores)
        return scores

    return graph_to_pyg_data_r_normal(
        G, group_size, allow_ties=allow_ties, score_fn=score_fn
    )


def graph_to_pyg_data_r_equal_tiers(G, group_size, allow_ties=True):
    """Szobatárs gráf, ahol MINDEN tier (réteg) AZONOS méretű (ugyanannyi tie).

    Ugyanúgy épül fel, mint `graph_to_pyg_data_r_top_ties`, de a cél itt az,
    hogy a `_tiered_ranking_from_scores` kvantálása után MINDEN réteg
    PONTOSAN ugyanannyi tagot (holtversenyzőt) tartalmazzon.

    Hogyan: a jelölteket EGYENLŐ méretű csoportokra osztjuk, és minden
    csoportnak egy jól elkülönülő, KÖZÖS pontszám-sávot adunk (szűk
    szóráson belül, a sávok között a `bucket`-nél jóval nagyobb
    távolsággal). Így minden csoport külön, egyedi kvantált kulcsot kap ->
    annyi réteg lesz, ahány csoport, és mindegyik mérete = a csoportméret
    (tie_size).

    A tier-méret csak akkor lehet MINDENHOL azonos, ha osztja a jelöltek
    számát. Ezért a kívánt tie-méretet a jelöltszám legközelebbi OSZTÓJÁRA
    igazítjuk (így garantáltan egyenlő tier-ek jönnek ki). A tie-méretet
    mintánként véletlenül választjuk az osztók közül (de >= 2, ha lehet),
    hogy a tie-k száma ne legyen determinisztikus.

    A tie-k az `allow_ties=True` esetén jelennek meg (a kvantálás ekkor
    aktív); `allow_ties=False`-nál a reprezentáció szigorú marad, mint a
    többi konverternél.

    A `_tiered_ranking_from_scores` alap bucket-je 0.6. A sávon belüli
    szórást ennél jóval kisebbre (~0.05) vesszük, a sávközt pedig
    bucket * 3-ra (= 1.8), hogy a csoportok megbízhatóan külön kulcsokba
    (külön rétegekbe) essenek.
    """
    def _divisors(n):
        """n osztói növekvő sorrendben (1 és n is benne)."""
        ds = []
        for d in range(1, n + 1):
            if n % d == 0:
                ds.append(d)
        return ds

    def score_fn(size):
        if size <= 0:
            return np.array([], dtype=float)
        if size == 1:
            return np.random.normal(0.0, 0.05, size=1)

        # Lehetséges tie-méretek: a jelöltszám osztói. A triviális size
        # (minden egy rétegben) kizárható, hogy legyen >1 réteg; a >=2
        # tie-t preferáljuk, hogy legyen tényleges holtverseny.
        divisors = [d for d in _divisors(size) if d < size]
        # Előnyben a >= 2 tie-méret (valódi holtverseny), ha van ilyen.
        candidates_ts = [d for d in divisors if d >= 2] or divisors
        tie_size = int(np.random.choice(candidates_ts))

        n_groups = size // tie_size
        bucket = 0.6
        band_gap = bucket * 3.0  # jól elkülönülő sávok (külön kulcsok)

        scores = np.empty(size, dtype=float)
        idx = 0
        for g in range(n_groups):
            center = g * band_gap
            scores[idx:idx + tie_size] = center + np.random.normal(
                0.0, 0.05, size=tie_size
            )
            idx += tie_size
        # (size osztható tie_size-zal, így idx == size itt.)

        # Keverés: a rangsorolást a pontszám dönti el, a sorrend csak a
        # holtversenyen belül számít; a sávok így is külön rétegekbe esnek.
        np.random.shuffle(scores)
        return scores

    return graph_to_pyg_data_r_normal(
        G, group_size, allow_ties=allow_ties, score_fn=score_fn
    )


# A szobatárs (roommate) PyG-konverterek halmaza. Mindegyik ugyanazt a
# réteges reprezentációt és mezőkészletet adja, és támogatja az
# `allow_ties` paramétert; kizárólag a preferencia-pontszámok eloszlása
# tér el. A `get_or_generate_dataset` ez alapján dönti el, hogy át kell-e
# adni az `allow_ties` paramétert a konverternek.
ROOMMATE_CONVERTERS = (
    graph_to_pyg_data_r_normal,
    graph_to_pyg_data_r_uniform,
    graph_to_pyg_data_r_exponential,
    graph_to_pyg_data_r_powerlaw,
    graph_to_pyg_data_r_bimodal,
    graph_to_pyg_data_r_top_ties,
    graph_to_pyg_data_r_equal_tiers,
)


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
