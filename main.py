import ast
import os
import copy
import data_generator
import random
import encoder
import decoder
from collections import Counter, deque, defaultdict
from dataclasses import dataclass
from itertools import permutations
import torch.nn.functional as F
import torch
import numpy as np
from typing import Dict
from typing import List, Tuple
import random

# A preferencia-reprezentáció verziója. A szobatárs (roommate) gráfok
# mostantól HOLTVERSENYES (réteges) preferenciákat használnak, ezért a
# régi (holtverseny nélküli) cache-eket érvénytelenítjük ezzel a kulccsal.
#
# v2: a "legjobb helyre sorolás" node-embedding (feature index 0) mostantól
# SZÁZALÉK (hány % másik csomópont teszi az adott csomópontot az első helyre),
# a korábbi NYERS DARABSZÁM helyett. A régi (összegző) cache-eket ezzel a
# verzióemeléssel érvénytelenítjük, hogy a roommate adathalmazok (tied és
# nontied is) újragenerálódjanak az új százalékos embeddinggel.
#
# v3: a "rang-összeg" node-embedding (feature index 1) mostantól SKÁLÁZOTT
# rangok összege: a rangsoroló csomópont legjobb rangja -> 0, legrosszabb
# rangja -> 1, a köztes rangok lineárisan skálázva; e 0..1 értékek összege
# az embedding. Korábban a NYERS (réteg index + 1) rangokat összegeztük.
#
# v4: a "legrosszabb helyre sorolás" node-embedding (feature index 2)
# mostantól NEGATÍV SZÁZALÉK (hány % másik csomópont teszi az adott
# csomópontot az utolsó/legrosszabb helyre, -1-gyel szorozva), a korábbi
# NYERS NEGATÍV DARABSZÁM helyett.
#
# v5: a "medián rang" node-embedding (feature index 3) mostantól a
# SKÁLÁZOTT rangok (best=0, worst=1, lineárisan skálázva) MEDIÁNJA, a
# korábbi NYERS rang-medián helyett.
PREF_FORMAT_VERSION = "roommate_ties_v6"


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

# Legfeljebb ennyi teljes matchinget állítunk elő gráfonként. Ha a gráfnak
# ennél több teljes matchingje van, akkor nem soroljuk fel mindet,
# hanem egy stabil matchingből kiindulva random
# perturbációval mintázunk összesen MAX_SAMPLED_MATCHINGS darabot.
MAX_SAMPLED_MATCHINGS = 10


def _enumerate_matchings_from_pairs(nodes, allowed_pairs):
    """
    Felsorolja a `nodes` csomóponthalmaz azon tökéletes párosításait,
    amelyek kizárólag `allowed_pairs`-beli (kanonikus (min,max)) párokat
    használnak.

    A legkisebb indexű szabad node-ot párosítja minden olyan szabad
    node-dal, amellyel megengedett pár alkotható, rekurzívan. Minden
    matching kanonikus párok rendezett listája.
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
        if pair not in allowed_pairs:
            continue
        remaining = rest[:i] + rest[i + 1:]
        for sub in _enumerate_matchings_from_pairs(remaining, allowed_pairs):
            yield [pair] + sub


def _enumerate_matchings_with_cap(nodes, allowed_pairs, cap):
    """
    Lusta felsorolás korai leállással.

    Visszatér: (matchings, exceeded)
      matchings: legfeljebb `cap` darab teljes matching (kanonikus
                 párlisták) listája.
      exceeded:  True, ha a gráfnak `cap`-nél TÖBB teljes matchingje van
                 (ekkor a matchings pontosan `cap` elemet tartalmaz).

    Így a "<= cap vagy > cap" döntés pontos anélkül, hogy az összes
    (akár faktoriális számú) matchinget materializálnánk.
    """
    matchings = []
    for matching in _enumerate_matchings_from_pairs(nodes, allowed_pairs):
        matchings.append(matching)
        if len(matchings) > cap:
            return matchings[:cap], True
    return matchings, False


def _matching_pairs_to_partner(matching, num_nodes):
    """Kanonikus párlistából [num_nodes] partner-tensor (-1 = unmatched)."""
    partner = torch.full((num_nodes,), -1, dtype=torch.long)
    for a, b in matching:
        partner[a] = b
        partner[b] = a
    return partner


def _canonical_matching(partner_pairs):
    """Párlistát kanonikus, rendezett alakra hoz (összehasonlíthatóság)."""
    canon = sorted((min(a, b), max(a, b)) for a, b in partner_pairs)
    return canon


def _gale_shapley_matching(left_nodes, right_nodes, prefs, allowed_pairs):
    """
    Gale-Shapley stabil párosítás házassági (bipartit) esetre, kizárólag
    preferencialistákból és a megengedett (kanonikus) párok halmazából.

    left_nodes javasolnak. `prefs[node]` a node preferencialistája
    (legjobb elöl). Csak azokat a jelölteket vesszük figyelembe, amelyek
    `allowed_pairs`-beli párt alkotnak (azaz kölcsönösen elfogadható él).

    Visszatér: kanonikus párlista. Lehet részleges, ha nincs teljes
    stabil párosítás, de bipartit esetben az allowed_pairs definíció
    szerint kölcsönös, így általában teljes.
    """
    right_set = set(right_nodes)

    def pair_ok(u, v):
        return (min(u, v), max(u, v)) in allowed_pairs

    # Szűrt javaslati listák (sorrend megtartva).
    proposer_list = {
        u: [v for v in prefs.get(u, []) if v in right_set and pair_ok(u, v)]
        for u in left_nodes
    }

    rank_right = {}
    for v in right_nodes:
        rank_right[v] = {
            u: i
            for i, u in enumerate(
                [u for u in prefs.get(v, []) if pair_ok(u, v)]
            )
        }

    match_left = {u: None for u in left_nodes}
    match_right = {v: None for v in right_nodes}
    next_idx = {u: 0 for u in left_nodes}

    free = deque(left_nodes)

    while free:
        u = free.popleft()
        if next_idx[u] >= len(proposer_list[u]):
            continue
        v = proposer_list[u][next_idx[u]]
        next_idx[u] += 1

        if v not in rank_right or u not in rank_right[v]:
            # v nem fogadja el u-t: u marad szabad, próbál tovább.
            free.append(u)
            continue

        current = match_right[v]
        if current is None:
            match_left[u] = v
            match_right[v] = u
        elif rank_right[v][u] < rank_right[v][current]:
            match_left[current] = None
            free.append(current)
            match_left[u] = v
            match_right[v] = u
        else:
            free.append(u)

    pairs = [
        (u, v) for u, v in match_left.items() if v is not None
    ]
    return _canonical_matching(pairs)


def _irving_stable_roommate(nodes, prefs, allowed_pairs):
    """
    Irving algoritmusa a stabil szobatárs-problémára.

    `prefs[x]` x preferencialistája (legjobb elöl). Lehet SIMA lista vagy
    RÉTEGELT / holtversenyes (listák listája) formátum. Holtverseny esetén
    a rétegeket tetszőleges, de rögzített sorrendben "szigorúsítjuk"
    (tie-break), majd a klasszikus (szigorú preferenciákat feltételező)
    Irving-algoritmust futtatjuk. Egy tetszőleges tie-break szerinti
    stabil párosítás az eredeti (holtversenyes) instanciában GYENGÉN
    stabil, így ez korrekt (bár nem teljes) megoldhatósági próba a gyenge
    stabilitásra.

    Csak az allowed_pairs-beli (kölcsönösen elfogadható) jelölteket
    vesszük figyelembe.

    Visszatér: kanonikus párlista, ha a tie-break szerint létezik stabil
    párosítás; egyébként None.

    Megvalósítás a klasszikus két fázis szerint:
      1. fázis: javaslatok (proposal), amíg mindenki tart egy ajánlatot.
      2. fázis: rotációk kiküszöbölése (rotation elimination).
    """
    def pair_ok(a, b):
        return (min(a, b), max(a, b)) in allowed_pairs

    # Réteges (holtversenyes) preferenciát sima listává lapítunk
    # (tie-break): a réteg indexe szerint, azon belül a megadott
    # sorrendben. Így a szigorú Irving futtatható.
    flat_prefs = {
        x: data_generator._flatten_tiers(prefs.get(x, []))
        for x in nodes
    }

    # Szűrt preferencialisták (sorrend megtartva, kölcsönös párok).
    pref_list = {
        x: [y for y in flat_prefs.get(x, []) if y != x and pair_ok(x, y)]
        for x in nodes
    }

    # Statikus rangok az EREDETI (szűrt) sorrend szerint, a redukció
    # alatt is érvényesek maradnak az "x jobban preferálja a-t, mint b-t"
    # összehasonlításokhoz.
    rank = {
        x: {y: i for i, y in enumerate(pref_list[x])}
        for x in nodes
    }

    def prefers(x, a, b):
        """x szigorúan jobban preferálja a-t, mint b-t."""
        return rank[x][a] < rank[x][b]

    def remove(a, b):
        """Törli (a,b)-t mindkét fél listájából (szimmetrikusan)."""
        if b in pref_list[a]:
            pref_list[a].remove(b)
        if a in pref_list[b]:
            pref_list[b].remove(a)

    # ---------- 1. fázis: javaslatok ----------
    # proposed_from[y] = kitől tartja y a jelenlegi (legjobb) ajánlatot.
    proposed_from = {x: None for x in nodes}

    free = deque(nodes)
    while free:
        x = free.popleft()
        # x csak akkor javasol, ha még nem tartja senki az ajánlatát.
        if proposed_from_is_held(proposed_from, x):
            continue
        if not pref_list[x]:
            return None  # x listája kiürült -> nincs stabil párosítás.

        y = pref_list[x][0]  # x legjobb még megmaradt jelöltje
        current = proposed_from[y]

        if current is None:
            proposed_from[y] = x
        elif prefers(y, x, current):
            # y jobban preferálja x-et: a korábbi javasló (current) kiesik.
            proposed_from[y] = x
            # current elveszti y-t: töröljük (current, y)-t és current újra
            # szabad lesz.
            remove(current, y)
            free.append(current)
        else:
            # y elutasítja x-et: töröljük (x, y)-t, x próbál tovább.
            remove(x, y)
            free.append(x)

    # Minden node tart valakit? Ha nem, nincs stabil párosítás.
    if any(proposed_from[y] is None for y in nodes):
        return None

    # 1. fázis utáni listaszűkítés: ha y tartja x ajánlatát, akkor
    #   - y listájából töröljük mindazokat, akiket y rosszabbnak tart x-nél,
    #   - szimmetrikusan y-t is töröljük azok listájából.
    for y in nodes:
        x = proposed_from[y]
        worse = [z for z in list(pref_list[y]) if z != x and prefers(y, x, z)]
        for z in worse:
            remove(y, z)

    # ---------- 2. fázis: rotációk kiküszöbölése ----------
    def all_singletons():
        return all(len(pref_list[x]) == 1 for x in nodes)

    def any_empty():
        return any(len(pref_list[x]) == 0 for x in nodes)

    max_iter = len(nodes) * len(nodes) + 10
    it = 0
    while not all_singletons() and it < max_iter:
        it += 1
        if any_empty():
            return None

        # Rotáció keresése: induljunk egy >= 2 hosszú listájú node-ból.
        start = next((x for x in nodes if len(pref_list[x]) >= 2), None)
        if start is None:
            break

        seq = []            # (x_i, y_i) párok
        seen_index = {}
        p = start
        while p not in seen_index:
            seen_index[p] = len(seq)
            if len(pref_list[p]) < 2:
                return None
            # y_i = p második jelöltje
            yi = pref_list[p][1]
            seq.append((p, yi))
            # következő p = y_i UTOLSÓ jelöltje
            if not pref_list[yi]:
                return None
            p = pref_list[yi][-1]

        cycle_start = seen_index[p]
        rotation = seq[cycle_start:]

        # Rotáció eliminálása: minden (x_i, y_i)-re y_i elutasítja az
        # előző rotációs x-et. A standard lépés: y_i törli a párját a
        # következő x-szel (ciklikusan), de a biztos megvalósításhoz
        # minden y_i-ből töröljük azokat, akiket y_i x_{i+1}-nél (a
        # következő javaslónál) rosszabbnak tart. Egyszerűbb, ekvivalens
        # forma kis gráfokra: a rotációban szereplő (y_i, x_{i+1}) párokat
        # töröljük.
        m = len(rotation)
        for i in range(m):
            yi = rotation[i][1]
            x_next = rotation[(i + 1) % m][0]
            remove(yi, x_next)

        if any_empty():
            return None

    if any_empty() or not all_singletons():
        return None

    pairs = []
    for x in nodes:
        y = pref_list[x][0]
        if x < y:
            pairs.append((x, y))

    # Valódi teljes párosítás ellenőrzése.
    matched = set()
    for a, b in pairs:
        matched.add(a)
        matched.add(b)
    if matched != set(nodes):
        return None

    return _canonical_matching(pairs)


def proposed_from_is_held(proposed_from, x):
    """True, ha valaki tartja x ajánlatát (azaz proposed_from[y] == x)."""
    for y, who in proposed_from.items():
        if who == x:
            return True
    return False


def _build_directed_score_map(
    edge_index: torch.Tensor,
    directed_logits: torch.Tensor,
) -> Dict[Tuple[int, int], float]:
    """
    Irányított él-score térkép a decoder nyers logitjaiból.

    edge_index: [2, E], edge_index[:, e] = [u, v] az u -> v él.
    directed_logits: [E], a decoder nyers logitja az adott irányított élre.

    Visszatér: {(u, v): score} dict. Duplikált él esetén a NAGYOBB score-t
    tartjuk meg (ugyanaz a konvenció, mint a greedy dekódernél).
    """
    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()
    logits = directed_logits.detach().cpu().view(-1).tolist()

    score: Dict[Tuple[int, int], float] = {}
    for u, v, s in zip(src, dst, logits):
        if u == v:
            continue
        s = float(s)
        if (u, v) not in score or s > score[(u, v)]:
            score[(u, v)] = s
    return score


def _model_broken_prefs(
    nodes,
    tiered_prefs,
    directed_score: Dict[Tuple[int, int], float],
    attempt: int = 0,
):
    """
    Réteges (holtversenyes) preferenciákat SZIGORÚ (holtverseny nélküli)
    listává alakít úgy, hogy a HOLTVERSENYEKET A MODELL PREDIKCIÓJA bontja.

    Minden `x` node-ra:
      - a rétegeket a legjobbtól a legrosszabbig vesszük (a réteg-sorrend a
        valódi preferencia, ezt NEM írjuk felül);
      - EGY rétegen belül (ahol holtverseny van) a partnereket az
        x -> partner irányított él modell-score-ja szerint CSÖKKENŐ
        sorrendbe rakjuk: nagyobb score = x jobban preferálja.
        Így a tie-break a modell tanult preferenciáját követi.

    attempt:
        A tie-break VARIÁNS indexe (0 = a legjobb, score szerint csökkenő
        alap-sorrend). `attempt > 0` esetén UGYANCSAK score-alapú, de MÁS
        sorrendet ad: a holtversenyes rétegen belül a score szerinti
        `attempt`-edik (0-indexelt) jelöltet emeli a réteg ÉLÉRE, a többit
        a score-sorrendben hagyva. Így `attempt=1` a "második legjobb"-at
        választja elöl, `attempt=2` a harmadikat, stb. Ez determinisztikus,
        kizárólag a modell score-jaiból számított, az alaptól ELTÉRŐ
        megközelítés (nem véletlen). Ha egy réteg rövidebb, mint az
        `attempt+1`, a réteg változatlan marad (nincs mit előrébb hozni).

    Ha egy listához nincs réteges forma (sima lista), érintetlenül
    lapítjuk (nincs holtverseny, nincs mit bontani).

    Visszatér: {x: [partner, ...]} szigorú (best->worst) listák.
    """
    strict = {}
    for x in nodes:
        pref_list = tiered_prefs.get(x, [])

        if not data_generator._is_tiered(pref_list):
            # Sima lista: nincs holtverseny, csak másolat.
            strict[x] = list(pref_list)
            continue

        flat = []
        for tier in pref_list:
            # Egy réteg tagjait a modell x -> tag score-ja szerint
            # csökkenő sorrendbe rakjuk. Hiányzó score esetén -inf
            # (a réteg végére kerül), stabil másodlagos kulcs a node-id.
            ordered = sorted(
                tier,
                key=lambda partner: (
                    directed_score.get((x, partner), float("-inf")),
                    -partner,
                ),
                reverse=True,
            )

            # attempt > 0: a score szerinti `attempt`-edik jelöltet a réteg
            # élére emeljük (score-alapú "második/harmadik ... legjobb"
            # megközelítés), a többi score-sorrend megmarad. Ha a réteg
            # túl rövid, nincs változás.
            if attempt > 0 and len(ordered) > attempt:
                promoted = ordered.pop(attempt)
                ordered.insert(0, promoted)

            flat.extend(ordered)
        strict[x] = flat
    return strict


def irving_srt_decode_pairs(
    edge_index: torch.Tensor,
    directed_logits: torch.Tensor,
    num_nodes: int,
    preferences: Dict[int, list],
    max_attempts: int = 10,
    debug: bool = False,
) -> torch.Tensor:
    """
    SRT (holtversenyes stabil szobatárs) dekóder IRVING algoritmussal,
    ahol a HOLTVERSENYEKET A MODELL PREDIKCIÓJA bontja.

    A klasszikus Irving-algoritmus SZIGORÚ preferenciákat vár. Az SRT
    (Stable Roommates with Ties) esetben a preferenciák rétegesek
    (holtversenyesek). A szokásos megközelítés a holtversenyek tetszőleges
    "szigorúsítása" (tie-break), majd a klasszikus Irving futtatása: egy
    így kapott stabil párosítás az eredeti holtversenyes instanciában
    GYENGÉN stabil.

    Itt a tetszőleges tie-break HELYETT a modell által jósolt él-score-okat
    használjuk: egy holtversenyes rétegen belül azt a partnert soroljuk
    előrébb, amelyikre a modell nagyobb score-t ad. Így a tie-break a
    tanult preferenciát követi, nem véletlen/fix sorrendet.

    FALLBACK LOGIKA (két szint):
      1) Ha az alap (attempt=0) tie-break szerint NEM kapunk GYENGÉN STABIL
         párosítást (az Irving None-t ad, VAGY a kapott párosítás nem
         gyengén stabil), akkor ÚJRA próbáljuk MÁS, de SZINTÉN
         score-alapú megközelítéssel (lásd `_model_broken_prefs` `attempt`
         paramétere: a score szerinti második/harmadik/... legjobb jelöltet
         emeljük a réteg élére). Összesen `max_attempts` (alapból 10)
         különböző megközelítést próbálunk; amint gyengén stabilat találunk,
         MEGÁLLUNK és azt adjuk vissza.
      2) Ha `max_attempts` próbálkozás UTÁN sincs gyengén stabil megoldás,
         a GREEDY dekóderre (`greedy_decode_pairs`) esünk vissza.

    A "gyengén stabil" ellenőrzés a `classify_stability_level(...) !=
    "unstable"` feltétellel történik (nincs weak-blokkoló pár), ugyanúgy,
    ahogy a kiértékelés is méri.

    Input
    -----
    edge_index:
        [2, E], edge_index[:, e] = [u, v] az u -> v él.
    directed_logits:
        [E], a decoder nyers logitja (a tie-break-hez és a megengedett
        párok meghatározásához).
    num_nodes:
        Node-ok száma.
    preferences:
        {node: réteges_vagy_sima_pref_lista}. Jellemzően
        `graph.proposee_pref | graph.proposer_pref`.
    max_attempts:
        Hány KÜLÖNBÖZŐ score-alapú tie-break megközelítést próbáljunk, mielőtt
        a greedy dekóderre esnénk vissza (alapból 10).

    Output
    ------
    partner:
        [num_nodes], partner[u] = v vagy -1.
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

    directed_score = _build_directed_score_map(edge_index, directed_logits)

    # Megengedett (kölcsönösen elfogadható) párok: azok a canonical {u,v}
    # párok, amelyekhez MINDKÉT irányított él létezik. Ez megegyezik a
    # greedy dekóder kölcsönösségi feltételével.
    allowed_pairs = set()
    for (u, v) in directed_score.keys():
        if u < v and (v, u) in directed_score:
            allowed_pairs.add((u, v))

    nodes = list(range(num_nodes))

    # Él-rangok a GYENGE STABILITÁS ellenőrzéséhez (egyszer számoljuk ki).
    edge_ranks = preference_lists_to_edge_ranks(preferences, edge_index)

    def to_partner(matching):
        p = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        if matching is None:
            return p
        for a, b in matching:
            p[a] = b
            p[b] = a
        return p

    def is_weakly_stable(partner_tensor):
        # Minden node párt kapott-e (teljes párosítás) ÉS nincs weak-blokk.
        if not bool((partner_tensor != -1).all().item()):
            return False
        level = classify_stability_level(
            edge_index=edge_index,
            edge_ranks=edge_ranks,
            partner=partner_tensor,
            debug=False,
        )
        return level != "unstable"

    # --- Több score-alapú tie-break megközelítés kipróbálása ---
    for attempt in range(max_attempts):
        strict_prefs = _model_broken_prefs(
            nodes, preferences, directed_score, attempt=attempt,
        )

        if debug:
            tag = "alap" if attempt == 0 else f"{attempt+1}. legjobb elöl"
            print(f"\n--- Model-guided tie-break próbálkozás "
                  f"#{attempt} ({tag}) ---")
            for x in nodes:
                print(f"  {x}: {strict_prefs.get(x, [])}")

        matching = _irving_stable_roommate(
            nodes=nodes,
            prefs=strict_prefs,
            allowed_pairs=allowed_pairs,
        )
        partner = to_partner(matching)

        if is_weakly_stable(partner):
            if debug:
                print(f"Gyengén stabil párosítás a #{attempt} "
                      f"próbálkozással: {matching}")
            return partner
        elif debug:
            reason = ("Irving None" if matching is None
                      else "nem gyengén stabil")
            print(f"  #{attempt} próbálkozás SIKERTELEN ({reason}).")

    # --- Fallback: greedy dekóder ---
    if debug:
        print(f"{max_attempts} score-alapú próbálkozás UTÁN sincs gyengén "
              f"stabil megoldás -> GREEDY fallback.")
    return greedy_decode_pairs(
        edge_index=edge_index,
        directed_logits=directed_logits,
        num_nodes=num_nodes,
        debug=False,
    )


def _random_broken_prefs(nodes, tiered_prefs, rng=None):
    """
    Réteges (holtversenyes) preferenciákat SZIGORÚ (holtverseny nélküli)
    listává alakít úgy, hogy a HOLTVERSENYEKET VÉLETLENSZERŰEN bontja.

    Minden `x` node-ra:
      - a rétegeket a legjobbtól a legrosszabbig vesszük (a réteg-sorrend a
        valódi preferencia, ezt NEM írjuk felül);
      - EGY rétegen belül (ahol holtverseny van) a partnereket VÉLETLEN
        sorrendbe keverjük (uniform random permutáció).

    Ez a `_model_broken_prefs` baseline párja: minden ugyanaz, KIZÁRÓLAG a
    tie-break stratégia tér el (modell-score helyett véletlen).

    rng:
        Opcionális `random.Random` példány a reprodukálható
        keveréshez. Ha None, a globális `random` modult használjuk.

    Visszatér: {x: [partner, ...]} szigorú (best->worst) listák.
    """
    shuffle = (rng.shuffle if rng is not None else random.shuffle)

    strict = {}
    for x in nodes:
        pref_list = tiered_prefs.get(x, [])

        if not data_generator._is_tiered(pref_list):
            # Sima lista: nincs holtverseny, csak másolat.
            strict[x] = list(pref_list)
            continue

        flat = []
        for tier in pref_list:
            members = list(tier)
            shuffle(members)
            flat.extend(members)
        strict[x] = flat
    return strict


def irving_srt_decode_pairs_random(
    edge_index: torch.Tensor,
    directed_logits: torch.Tensor,
    num_nodes: int,
    preferences: Dict[int, list],
    rng=None,
    max_attempts: int = 10,
    debug: bool = False,
) -> torch.Tensor:
    """
    SRT dekóder IRVING algoritmussal, ahol a HOLTVERSENYEKET VÉLETLENSZERŰEN
    bontjuk (baseline az `irving_srt_decode_pairs` modell-vezérelt tie-break
    ellenében).

    MINDEN ugyanaz, mint az `irving_srt_decode_pairs`-ben (ugyanaz az Irving
    algoritmus, ugyanaz az `allowed_pairs` kölcsönösségi szűrés a
    `directed_logits`-ből), KIZÁRÓLAG a tie-break tér el: a holtversenyes
    rétegeket nem a modell score-ja, hanem egy véletlen permutáció
    szigorúsítja.

    A `directed_logits` itt is kell: az él létezéséből (mindkét irány) adódó
    kölcsönös (allowed) párok halmazát innen határozzuk meg, PONTOSAN úgy,
    ahogy a modell-vezérelt és a greedy dekóder is. Így a két Irving-teszt
    KIZÁRÓLAG a tie-break-ben különbözik.

    RETRY LOGIKA:
      Ha az aktuális VÉLETLEN tie-break szerint NEM kapunk GYENGÉN STABIL
      párosítást, ÚJRA próbáljuk egy MÁSIK VÉLETLEN tie-break-kel. Összesen
      `max_attempts` (alapból 10) független véletlen próbálkozást teszünk;
      amint gyengén stabilat találunk, MEGÁLLUNK és azt adjuk vissza.

      A model-vezérelt dekóderrel ELLENTÉTBEN:
        - a próbálkozások VÉLETLEN alapúak (nem score-alapúak),
        - a modell score-ját a tie-break-hez NEM használjuk (csak az
          allowed_pairs meghatározásához, mint eddig),
        - NINCS greedy fallback: ha `max_attempts` próbálkozás után sincs
          gyengén stabil megoldás, az UTOLSÓ próbálkozás eredményét adjuk
          vissza (jellemzően csupa -1 vagy nem gyengén stabil párosítás).

    rng:
        Opcionális `random.Random` a reprodukálható véletlen tie-break-hez.
    max_attempts:
        Hány független VÉLETLEN tie-break próbálkozást teszünk (alapból 10).

    Visszatér: partner tensor [num_nodes] (-1, ahol nincs pár).
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

    directed_score = _build_directed_score_map(edge_index, directed_logits)

    # Megengedett (kölcsönösen elfogadható) párok: azok a canonical {u,v}
    # párok, amelyekhez MINDKÉT irányított él létezik. (Ugyanaz, mint a
    # modell-vezérelt és a greedy dekóderben.)
    allowed_pairs = set()
    for (u, v) in directed_score.keys():
        if u < v and (v, u) in directed_score:
            allowed_pairs.add((u, v))

    nodes = list(range(num_nodes))

    # Él-rangok a GYENGE STABILITÁS ellenőrzéséhez (egyszer számoljuk ki).
    edge_ranks = preference_lists_to_edge_ranks(preferences, edge_index)

    def to_partner(matching):
        p = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        if matching is None:
            return p
        for a, b in matching:
            p[a] = b
            p[b] = a
        return p

    def is_weakly_stable(partner_tensor):
        # Minden node párt kapott-e (teljes párosítás) ÉS nincs weak-blokk.
        if not bool((partner_tensor != -1).all().item()):
            return False
        level = classify_stability_level(
            edge_index=edge_index,
            edge_ranks=edge_ranks,
            partner=partner_tensor,
            debug=False,
        )
        return level != "unstable"

    # --- Több FÜGGETLEN VÉLETLEN tie-break próbálkozás ---
    last_partner = torch.full(
        (num_nodes,), -1, dtype=torch.long, device=device,
    )
    for attempt in range(max_attempts):
        # Minden próbálkozás ÚJ véletlen tie-break (ugyanazt az rng-t
        # használjuk, így a sorozat reprodukálható, de próbálkozásonként
        # eltérő permutáció).
        strict_prefs = _random_broken_prefs(nodes, preferences, rng=rng)

        if debug:
            print(f"\n--- Random tie-break próbálkozás #{attempt} ---")
            for x in nodes:
                print(f"  {x}: {strict_prefs.get(x, [])}")

        matching = _irving_stable_roommate(
            nodes=nodes,
            prefs=strict_prefs,
            allowed_pairs=allowed_pairs,
        )
        partner = to_partner(matching)
        last_partner = partner

        if is_weakly_stable(partner):
            if debug:
                print(f"Gyengén stabil párosítás a #{attempt} véletlen "
                      f"próbálkozással: {matching}")
            return partner
        elif debug:
            reason = ("Irving None" if matching is None
                      else "nem gyengén stabil")
            print(f"  #{attempt} próbálkozás SIKERTELEN ({reason}).")

    # NINCS greedy fallback: az utolsó próbálkozás eredményét adjuk vissza.
    if debug:
        print(f"{max_attempts} véletlen próbálkozás UTÁN sincs gyengén "
              f"stabil megoldás -> az utolsó eredmény (nincs greedy "
              f"fallback).")
    return last_partner


def _random_perfect_matching(nodes, allowed_pairs, max_tries=200):
    """
    Véletlen teljes párosítás kizárólag `allowed_pairs`-beli párokból.

    Backtracking + randomizált választás. Ha `max_tries` próbálkozásból
    sem sikerül teljes párosítást találni, None-t ad vissza.
    """
    # Szomszédsági lista a megengedett párokból.
    neighbors = {x: [] for x in nodes}
    for (a, b) in allowed_pairs:
        if a in neighbors and b in neighbors:
            neighbors[a].append(b)
            neighbors[b].append(a)

    node_list = list(nodes)

    def backtrack(free_set):
        if not free_set:
            return []
        # A legkisebb indexű szabad node-ot párosítjuk.
        x = min(free_set)
        candidates = [y for y in neighbors[x] if y in free_set and y != x]
        random.shuffle(candidates)
        for y in candidates:
            rest = free_set - {x, y}
            sub = backtrack(rest)
            if sub is not None:
                return [(min(x, y), max(x, y))] + sub
        return None

    for _ in range(max_tries):
        result = backtrack(set(node_list))
        if result is not None:
            return _canonical_matching(result)
    return None


def _perturb_matching(matching, nodes, allowed_pairs, max_tries=300):
    """
    Egy meglévő teljes matchingből random perturbációval ÚJ, érvényes
    teljes matchinget állít elő.

    Stratégia: véletlenszerűen kiválaszt két párt (a,b) és (c,d), majd
    megpróbálja átcserélni őket úgy, hogy az új párok ((a,c),(b,d) vagy
    (a,d),(b,c)) is megengedettek legyenek. Ha egyetlen swap sem ad új
    érvényes matchinget, teljes random matchinggel próbálkozik.
    """
    def pair_ok(a, b):
        return (min(a, b), max(a, b)) in allowed_pairs

    base = {}
    for a, b in matching:
        base[a] = b
        base[b] = a

    pairs = list(matching)

    for _ in range(max_tries):
        if len(pairs) >= 2:
            i, j = random.sample(range(len(pairs)), 2)
            a, b = pairs[i]
            c, d = pairs[j]
            # Két lehetséges átkötés.
            options = [
                ((a, c), (b, d)),
                ((a, d), (b, c)),
            ]
            random.shuffle(options)
            for (p1, p2) in options:
                if pair_ok(*p1) and pair_ok(*p2):
                    new_pairs = list(pairs)
                    new_pairs[i] = (min(p1), max(p1))
                    new_pairs[j] = (min(p2), max(p2))
                    canon = _canonical_matching(new_pairs)
                    if canon != _canonical_matching(matching):
                        return canon
        # Fallback: teljesen random matching.
        rnd = _random_perfect_matching(nodes, allowed_pairs, max_tries=50)
        if rnd is not None and rnd != _canonical_matching(matching):
            return rnd

    return None


def _sample_matchings(
    nodes,
    allowed_pairs,
    roommate,
    prefs,
    left_nodes=None,
    right_nodes=None,
    target=MAX_SAMPLED_MATCHINGS,
):
    """
    Egy stabil matchingből kiindulva (házasság: Gale-Shapley; szobatárs:
    Irving, ha létezik stabil, különben random) random perturbációval
    `target` darab KÜLÖNBÖZŐ teljes matchinget állít elő.

    Visszatér: kanonikus párlisták listája (legfeljebb `target` elem).
    """
    collected = []
    seen = set()

    def add(m):
        if m is None:
            return False
        key = tuple(m)
        if key in seen:
            return False
        seen.add(key)
        collected.append(m)
        return True

    # Kiindulási (stabil vagy random) matching.
    if roommate:
        base = _irving_stable_roommate(nodes, prefs, allowed_pairs)
        if base is None:
            base = _random_perfect_matching(nodes, allowed_pairs)
    else:
        base = _gale_shapley_matching(
            left_nodes, right_nodes, prefs, allowed_pairs
        )
        # Ha a Gale-Shapley valamiért nem teljes, essünk vissza randomra.
        if base is None or len(base) * 2 != len(nodes):
            base = _random_perfect_matching(nodes, allowed_pairs)

    if base is None:
        return []

    add(base)

    # Perturbációk, amíg el nem érjük a target-et.
    stale = 0
    max_stale = target * 50
    current = base
    while len(collected) < target and stale < max_stale:
        candidate = _perturb_matching(current, nodes, allowed_pairs)
        if add(candidate):
            stale = 0
            current = candidate
        else:
            stale += 1
            # Néha a bázisból indulva perturbáljunk újra a diverzitásért.
            if stale % 10 == 0:
                current = random.choice(collected)

    return collected




def prepare_matching_ranking_data(
    edge_index: torch.Tensor,
    edge_ranks: torch.Tensor,
    num_nodes: int,
    left_nodes: list[int],
    right_nodes: list[int],
    preferences: dict[int, list[int]] | None = None,
) -> MatchingRankingData:
    """
    Előkészítés kis bipartit gráfokra.

    edge_ranks:
        [E], eredeti preferenciarangok.
        Kisebb érték = jobb partner.

    preferences:
        {node: [legjobb, ..., legrosszabb]} preferencialisták. Akkor kell,
        ha a gráfnak MAX_SAMPLED_MATCHINGS-nél több teljes matchingje van:
        ekkor nem soroljuk fel mindet, hanem egy Gale-Shapley stabil
        matchingből kiindulva random perturbációval mintázunk
        MAX_SAMPLED_MATCHINGS darabot.

    K:
        A felhasznált teljes matchingek száma (<= MAX_SAMPLED_MATCHINGS, ha
        a gráfnak több matchingje van, különben az összes).

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

    # Kanonikus (min,max) párhalmaz a mintavételi segédekhez, és egy
    # visszaképezés a kanonikus párból a bal->jobb orientált párra
    # (pair_id kulcsa).
    allowed_pairs = set()
    canon_to_oriented = {}
    for (u, v) in candidate_pairs:
        canon = (min(u, v), max(u, v))
        allowed_pairs.add(canon)
        canon_to_oriented[canon] = (u, v)

    # Döntés: legfeljebb MAX_SAMPLED_MATCHINGS teljes matchinget sorolunk
    # fel korai leállással. Ha ennél több van, mintavételezünk.
    enumerated, exceeded = _enumerate_matchings_with_cap(
        all_nodes, allowed_pairs, MAX_SAMPLED_MATCHINGS
    )

    if exceeded:
        if preferences is None:
            raise ValueError(
                "A gráfnak több teljes matchingje van, mint "
                f"{MAX_SAMPLED_MATCHINGS}; a mintavételhez preferences kell."
            )
        canonical_matchings = _sample_matchings(
            nodes=all_nodes,
            allowed_pairs=allowed_pairs,
            roommate=False,
            prefs=preferences,
            left_nodes=left_nodes,
            right_nodes=right_nodes,
            target=MAX_SAMPLED_MATCHINGS,
        )
    else:
        canonical_matchings = enumerated

    matching_pair_ids = []
    blocking_counts = []

    for matching in canonical_matchings:
        # A kanonikus párokat visszaképezzük a bal->jobb orientált
        # párokra, amelyekre a pair_id kulcsolva van.
        try:
            oriented = [canon_to_oriented[c] for c in matching]
        except KeyError:
            # Egy nem megengedett pár került bele: kihagyjuk.
            continue

        if any(pair not in pair_id for pair in oriented):
            continue

        partner = _matching_pairs_to_partner(matching, num_nodes)

        blocking = find_blocking_pairs(
            edge_index=ei,
            edge_attr=ranks,
            partner=partner,
        )

        matching_pair_ids.append(
            [pair_id[pair] for pair in oriented]
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
    preferences: dict[int, list[int]] | None = None,
) -> MatchingRankingData:
    """
    A `prepare_matching_ranking_data` stabil-szobatárs (roommate)
    változata: nincs bal/jobb oldal.

    Minden kölcsönös (u,v) & (v,u) él párt lehetséges párosítás-jelöltnek
    tekint. Ha a gráfnak legfeljebb MAX_SAMPLED_MATCHINGS teljes
    párosítása van, mindet felsoroljuk; ha több, akkor egy stabil
    párosításból (Irving algoritmusa; ha nincs stabil, random matching)
    kiindulva random perturbációval mintázunk MAX_SAMPLED_MATCHINGS
    darabot. A blocking-count-ot a `find_blocking_pairs` adja, a
    visszatérési struktúra megegyezik a páros esetben használttal, így a
    `matching_ranking_loss` változatlanul működik.

    preferences:
        {node: [legjobb, ..., legrosszabb]} preferencialisták; a stabil
        párosítás (Irving) kiszámításához kell, ha a gráfnak
        MAX_SAMPLED_MATCHINGS-nél több teljes párosítása van.
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

    all_nodes = list(range(num_nodes))
    allowed_pairs = set(candidate_pairs)  # már kanonikus (min,max) alak

    # Döntés: legfeljebb MAX_SAMPLED_MATCHINGS teljes párosítást sorolunk
    # fel korai leállással. Ha ennél több van, mintavételezünk.
    enumerated, exceeded = _enumerate_matchings_with_cap(
        all_nodes, allowed_pairs, MAX_SAMPLED_MATCHINGS
    )

    if exceeded:
        if preferences is None:
            raise ValueError(
                "A gráfnak több teljes párosítása van, mint "
                f"{MAX_SAMPLED_MATCHINGS}; a mintavételhez preferences kell."
            )
        canonical_matchings = _sample_matchings(
            nodes=all_nodes,
            allowed_pairs=allowed_pairs,
            roommate=True,
            prefs=preferences,
            target=MAX_SAMPLED_MATCHINGS,
        )
    else:
        canonical_matchings = enumerated

    matching_pair_ids = []
    blocking_counts = []

    for matching in canonical_matchings:
        if any(pair not in pair_id for pair in matching):
            continue

        partner = _matching_pairs_to_partner(matching, num_nodes)

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

    Két formátumot kezel:
      - SIMA (házassági): preferences[u] = [legjobb, ..., legrosszabb];
        minden partner KÜLÖN rangot kap (az index).
      - RÉTEGELT / holtversenyes (szobatárs): preferences[u] =
        [[legjobb réteg], ..., [legrosszabb réteg]]; egy réteg minden
        tagja AZONOS rangot kap (a réteg indexét) -> holtverseny.

    A formátumot per-node automatikusan felismeri (`_is_tiered`): ha az
    első elem lista, réteges, különben sima. Így a házassági útvonal
    változatlan marad, a szobatárs út pedig a holtversenyes ranggal fut.

    output[e] = rankja annak a partnernek, amelyhez
    edge_index[:, e] = [u, v] tartozik. 0 = legjobb partner.
    """
    rank_dict = {
        u: data_generator._tier_rank_of(pref_list)
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


def classify_stability_level(
    edge_index: torch.Tensor,
    edge_ranks: torch.Tensor,
    partner: torch.Tensor,
    debug: bool = False,
) -> str:
    """
    Egy párosítás LEGERŐSEBB teljesülő SRT (holtversenyes) stabilitási
    szintjét adja vissza: "super", "strong", "weak" vagy "unstable".

    A rangok (edge_ranks) a `preference_lists_to_edge_ranks` kimenete,
    ahol a holtversenyben lévő partnerek AZONOS rangot kapnak. Kisebb rang
    = jobb partner. A párt csak kölcsönös elfogadhatóság esetén nézzük
    (u->v és v->u is létezik).

    Egy (u,v) pár (nem egymás párja) blokkolása a három definíció szerint:
      - super : MINDKÉT fél GYENGÉN (<=) preferálja a másikat a jelenlegi
                partneréhez képest (azaz legalább annyira szereti).
      - strong: az egyik fél SZIGORÚAN (<), a másik GYENGÉN (<=) preferál.
      - weak  : MINDKÉT fél SZIGORÚAN (<) preferál.

    Egy párosítás:
      - super-stabil, ha nincs super-blokkoló pár,
      - strong-stabil, ha nincs strong-blokkoló pár,
      - weak-stabil, ha nincs weak-blokkoló pár.
    A hierarchia: super => strong => weak. A visszaadott szint a
    legerősebb, amelyet a párosítás teljesít; ha a weak sem teljesül,
    "unstable".
    """
    if edge_ranks.ndim == 2:
        rvals = edge_ranks[:, 0]
    elif edge_ranks.ndim == 1:
        rvals = edge_ranks
    else:
        raise ValueError(
            f"edge_ranks shape-ja [E] vagy [E,F] legyen, "
            f"de ez: {tuple(edge_ranks.shape)}"
        )

    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()
    rvals = rvals.detach().cpu().tolist()
    partner = partner.detach().cpu().tolist()

    # rank_of[u][v] = u milyen rangra teszi v-t (kisebb = jobb).
    rank_of = {}
    for u, v, r in zip(src, dst, rvals):
        rank_of.setdefault(u, {})[v] = float(r)

    if debug:
        print("\n==== classify_stability_level DEBUG ====")
        print(f"Node-ok száma (partner tömb): {len(partner)}")
        print("Partnerek (partner[u] = v, -1 = párosítatlan):")
        for a in range(len(partner)):
            p = partner[a]
            print(f"  node {a:>3} -> partner {p}" + ("  [PÁROSÍTATLAN]" if p == -1 else ""))
        print("Rang-térkép (rank_of[u][v], kisebb = jobb; holtverseny = azonos rang):")
        for u in sorted(rank_of):
            items = ", ".join(
                f"{v}:{rank_of[u][v]:g}" for v in sorted(rank_of[u])
            )
            print(f"  node {u:>3}: {{{items}}}")

    def rank_to_partner(agent: int) -> float:
        """Az agent jelenlegi partnerének rangja; unmatched/ismeretlen = +inf
        (azaz bárki elfogadható partner jobb nála)."""
        cur = partner[agent]
        if cur == -1:
            return float("inf")
        return rank_of.get(agent, {}).get(cur, float("inf"))

    # A legenyhébb (super) definíció sérülése a "legrosszabb" eset.
    # Egyszerre követjük, hogy van-e super- és strong-blokkoló pár.
    has_super_block = False
    has_strong_block = False
    has_weak_block = False

    if debug:
        print("\n-- Kanonikus (u<v, kölcsönösen elfogadható, nem egymás párja) párok vizsgálata --")

    for u, neighbors in rank_of.items():
        cur_u = rank_to_partner(u)
        for v, r_uv in neighbors.items():
            if u == v:
                continue
            # Kölcsönös elfogadhatóság.
            if u not in rank_of.get(v, {}):
                if debug and u < v:
                    print(f"  ({u},{v}): kihagyva - NEM kölcsönösen elfogadható (v nem fogadja el u-t)")
                continue
            # Egymás párja: nem blokkol.
            if partner[u] == v:
                if debug and u < v:
                    print(f"  ({u},{v}): kihagyva - EGYMÁS PÁRJA (nem blokkol)")
                continue
            # Csak egy irányból dolgozzuk fel a kanonikus párt.
            if u > v:
                continue

            r_vu = rank_of[v][u]
            cur_v = rank_to_partner(v)

            # u (ill. v) viszonya a jelenlegi partneréhez.
            u_strict = r_uv < cur_u
            u_weak = r_uv <= cur_u
            v_strict = r_vu < cur_v
            v_weak = r_vu <= cur_v

            # super-blokk: mindkettő gyengén preferál.
            is_super = u_weak and v_weak
            # strong-blokk: egyik szigorú, másik gyenge.
            is_strong = (u_strict and v_weak) or (u_weak and v_strict)
            # weak-blokk: mindkettő szigorú.
            is_weak = u_strict and v_strict

            if is_super:
                has_super_block = True
            if is_strong:
                has_strong_block = True
            if is_weak:
                has_weak_block = True

            if debug:
                def fmt(x):
                    return "+inf" if x == float("inf") else f"{x:g}"
                tags = []
                if is_weak:
                    tags.append("WEAK-BLOKK")
                if is_strong:
                    tags.append("STRONG-BLOKK")
                if is_super:
                    tags.append("SUPER-BLOKK")
                tag_str = ", ".join(tags) if tags else "nem blokkol"
                print(
                    f"  ({u},{v}): "
                    f"u={u} r(u->v)={fmt(r_uv)} vs partnere r={fmt(cur_u)} "
                    f"[{'szig.' if u_strict else ('gyenge' if u_weak else 'rosszabb')}] | "
                    f"v={v} r(v->u)={fmt(r_vu)} vs partnere r={fmt(cur_v)} "
                    f"[{'szig.' if v_strict else ('gyenge' if v_weak else 'rosszabb')}] "
                    f"=> {tag_str}"
                )

            # Ha már weak-blokk is van, erősebbet nem tudunk teljesíteni.
            if has_weak_block:
                # Nem lépünk ki azonnal: a hierarchia miatt a weak-blokk
                # jelenléte eldönti a végeredményt (unstable), de a
                # super/strong flagek már True-k lennének, így korrekt.
                pass

    if debug:
        print("\n-- Összegzés (van-e blokkoló pár az adott szinten) --")
        print(f"  super-blokk  van? {has_super_block}")
        print(f"  strong-blokk van? {has_strong_block}")
        print(f"  weak-blokk   van? {has_weak_block}")

    if not has_super_block:
        if debug:
            print("  => Nincs super-blokk  => SZINT: super")
            print("==== /classify_stability_level DEBUG ====\n")
        return "super"
    if not has_strong_block:
        if debug:
            print("  => Nincs strong-blokk => SZINT: strong")
            print("==== /classify_stability_level DEBUG ====\n")
        return "strong"
    if not has_weak_block:
        if debug:
            print("  => Nincs weak-blokk   => SZINT: weak")
            print("==== /classify_stability_level DEBUG ====\n")
        return "weak"
    if debug:
        print("  => Van weak-blokk     => SZINT: unstable")
        print("==== /classify_stability_level DEBUG ====\n")
    return "unstable"


def get_or_generate_dataset(
    group_size,
    gen_fn,
    n_train=1000,
    n_val=500,
    cache_path="dataset.pt",
    seed=None,
    graph_fn=None,
    group_sizes=None,
    allow_ties=False,
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

    group_sizes: ha None, minden mintát a fix `group_size`-zal generál.
            Ha lista (pl. [6, 8, 10, 12] szobatárs node-számokra, vagy
            [3, 4, 5, 6] páros oldalankénti létszámokra), akkor mintánként
            véletlenszerűen választ belőle egy méretet, így a train és a
            val split is vegyes méretű gráfokat tartalmaz. Ezáltal a
            kiértékelés méret szerint is informatív.
    """
    if graph_fn is None:
        graph_fn = data_generator.generate_graph_m

    gen_name = getattr(gen_fn, "__name__", str(gen_fn))
    graph_name = getattr(graph_fn, "__name__", str(graph_fn))

    if os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False)
        if (
            blob.get("group_size") == group_size
            and blob.get("group_sizes") == group_sizes
            and blob.get("gen_name") == gen_name
            and blob.get("graph_name") == graph_name
            and blob.get("pref_format") == PREF_FORMAT_VERSION
            and blob.get("allow_ties") == allow_ties
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

    def pick_size():
        if group_sizes is None:
            return group_size
        return random.choice(group_sizes)

    def make_sample():
        gs = pick_size()
        G = graph_fn(gs)
        # A szobatárs (roommate) konverterek támogatják a holtversenyt
        # (allow_ties); a páros (marriage) konverter nem, ezért oda nem
        # adjuk át a paramétert. Bármely regisztrált roommate converter
        # (különböző preferencia-eloszlások) megkapja az allow_ties-t.
        if gen_fn in data_generator.ROOMMATE_CONVERTERS:
            return gen_fn(G, gs, allow_ties=allow_ties)
        return gen_fn(G, gs)

    train_data = [make_sample() for _ in range(n_train)]
    val_data = [make_sample() for _ in range(n_val)]

    torch.save(
        {
            "train_data": train_data,
            "val_data": val_data,
            "group_size": group_size,
            "group_sizes": group_sizes,
            "gen_name": gen_name,
            "graph_name": graph_name,
            "pref_format": PREF_FORMAT_VERSION,
            "allow_ties": allow_ties,
        },
        cache_path,
    )
    print("Dataset legenerálva és elmentve ide: " + cache_path)
    return train_data, val_data


# A különböző preferencia-eloszlású szobatárs (roommate) konverterek
# név -> converter leképezése, hogy teszteléshez könnyen választható
# legyen az eloszlás.
ROOMMATE_DISTRIBUTIONS = {
    "normal": data_generator.graph_to_pyg_data_r_normal,
    "uniform": data_generator.graph_to_pyg_data_r_uniform,
    "exponential": data_generator.graph_to_pyg_data_r_exponential,
    "powerlaw": data_generator.graph_to_pyg_data_r_powerlaw,
    "bimodal": data_generator.graph_to_pyg_data_r_bimodal,
    # Olyan eloszlás, ahol az ELSŐ (legjobb) helyen nagy valószínűséggel
    # sok a holtverseny (tie). Csak allow_ties=True mellett ad tényleges
    # tie-kat.
    "top_ties": data_generator.graph_to_pyg_data_r_top_ties,
    # Olyan eloszlás, ahol MINDEN tier (réteg) azonos méretű, azaz minden
    # rétegben ugyanannyi a holtverseny. Csak allow_ties=True mellett ad
    # tényleges tie-kat.
    "equal_tiers": data_generator.graph_to_pyg_data_r_equal_tiers,
}


def get_or_generate_roommate_test_dataset(
    distribution,
    group_sizes=(6, 8, 10, 12,14,16,18,20,22,24,26,28,30),
    n_train=1000,
    n_val=500,
    allow_ties=True,
    seed=42,
    cache_path=None,
):
    """
    Szobatárs (roommate) TESZT-adathalmaz a megadott preferencia-
    eloszlással. A `graph_to_pyg_data_r_normal`-lal (jelenlegi holtversenyes
    szobatárs-probléma) MINDEN szempontból megegyezik, KIZÁRÓLAG a
    preferencia-pontszámok eloszlása más.

    distribution:
        "normal" | "uniform" | "exponential" | "powerlaw" | "bimodal"
        (lásd ROOMMATE_DISTRIBUTIONS).

    A visszatérés és a cache-séma megegyezik a `get_or_generate_dataset`-tel.
    """
    if distribution not in ROOMMATE_DISTRIBUTIONS:
        raise ValueError(
            f"Ismeretlen eloszlás: {distribution!r}. "
            f"Választható: {sorted(ROOMMATE_DISTRIBUTIONS)}"
        )

    gen_fn = ROOMMATE_DISTRIBUTIONS[distribution]

    if cache_path is None:
        cache_path = f"dataset_roommate_{distribution}.pt"

    return get_or_generate_dataset(
        # A szobatárs esetben a méret = node-szám; a group_sizes már
        # node-számokban (6,8,10,12) van megadva, a group_size csak
        # cache-kulcs/fallback.
        group_size=group_sizes[0],
        gen_fn=gen_fn,
        graph_fn=data_generator.generate_graph_r,
        n_train=n_train,
        n_val=n_val,
        cache_path=cache_path,
        seed=seed,
        group_sizes=list(group_sizes),
        allow_ties=allow_ties,
    )


def get_or_generate_mixed_dataset(
    group_size,
    n_train=1000,
    n_val=500,
    cache_path="dataset_mixed.pt",
    seed=None,
    allow_ties=False,
):
    """
    50/50 arányban kevert train/val adathalmazt készít szobatárs
    (roommate) és házassági (marriage/bipartite) gráfokból.

    Mindkét gráftípus azonos csomópontszámmal (2*group_size) jön létre,
    és mindkét PyG-konverter (`graph_to_pyg_data_r_normal`,
    `graph_to_pyg_data_normal`) azonos mezőkészletű `Data` objektumot ad
    vissza (x, edge_index, edge_attr, edge_y, proposee_pref,
    proposer_pref, rank_better_idx, rank_worse_idx), így a két típus egy
    adathalmazban keverhető.

    A minták véletlenszerűen keverednek (nem blokkokban), hogy a két típus
    egyenletesen oszoljon el a tanítás során. Páratlan elemszámnál a
    szobatárs-oldal kap eggyel többet.

    Az adat NYERS (normalizálatlan) formában kerül mentésre, a
    `get_or_generate_dataset`-tel megegyező cache-sémát követve.
    """
    # A train ÉS a val split is mintánként véletlen csoportméretet
    # [3, 4, 5, 6] használ, így a kiértékelésnél méret szerint is látható
    # a pontosság (lásd a build_split hívásokat lentebb).

    gen_name = "mixed_roommate_marriage_50_50_valsizes_3456"
    graph_name = "mixed(generate_graph_r+generate_graph_m)"

    if os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False)
        if (
            blob.get("group_size") == group_size
            and blob.get("gen_name") == gen_name
            and blob.get("graph_name") == graph_name
            and blob.get("pref_format") == PREF_FORMAT_VERSION
            and blob.get("allow_ties") == allow_ties
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

    def build_split(n, group_sizes=None):
        """
        n mintából álló kevert (szobatárs + házassági) adathalmazt épít.

        group_sizes: ha None, akkor a külső `group_size`-t használja minden
            mintához (fix méret). Ha lista (pl. [3, 4, 5, 6]), akkor
            mintánként véletlenszerűen választ belőle egy csoportméretet,
            így a split vegyes méretű gráfokat tartalmaz.
        """
        # Páratlan n esetén a szobatárs kap eggyel többet.
        n_roommate = (n + 1) // 2
        n_marriage = n - n_roommate

        def pick_group_size():
            if group_sizes is None:
                return group_size
            return random.choice(group_sizes)

        samples = []

        for _ in range(n_roommate):
            gs = pick_group_size()
            r_nodes = 2 * gs
            G = data_generator.generate_graph_r(r_nodes)
            samples.append(
                data_generator.graph_to_pyg_data_r_normal(
                    G, r_nodes, allow_ties=allow_ties
                )
            )

        for _ in range(n_marriage):
            gs = pick_group_size()
            G = data_generator.generate_graph_m(gs)
            samples.append(
                data_generator.graph_to_pyg_data_normal(
                    G, gs
                )
            )

        # Véletlenszerű keverés, hogy a két típus ne blokkokban álljon.
        random.shuffle(samples)
        return samples

    # Train és Val: vegyes csoportméretek (3, 4, 5, 6) mintánként
    # véletlenszerűen, hogy a kiértékelés méret szerint is informatív
    # legyen (lásd a méret szerinti metrikákat a kiértékelésben).
    train_data = build_split(n_train, group_sizes=[3, 4, 5, 6])
    val_data = build_split(n_val, group_sizes=[3, 4, 5, 6])

    torch.save(
        {
            "train_data": train_data,
            "val_data": val_data,
            "group_size": group_size,
            "gen_name": gen_name,
            "graph_name": graph_name,
            "pref_format": PREF_FORMAT_VERSION,
            "allow_ties": allow_ties,
        },
        cache_path,
    )
    print("Dataset legenerálva és elmentve ide: " + cache_path)
    return train_data, val_data


def build_decoder_input(out):
    return torch.cat(
            [
            out.edge_embedding,  # [E, 128]
            out.auxiliary["mutual_quality"].unsqueeze(-1),  # [E, 1]
            out.auxiliary["rank_score"].unsqueeze(-1),  # [E, 1]
        ],
        dim=-1,
    )


def is_roommate_graph(graph):
    """
    Egy gráf szobatárs-típusú-e. A `graph_to_pyg_data_r_normal` üres
    proposee_pref-et ad (nincs oldal-felosztás), míg a házassági
    (bipartite) konverterek kitöltik mindkét oldalt. Ez alapján
    dönthető el per-gráf a típus a kevert adathalmazban.
    """
    proposee = getattr(graph, "proposee_pref", None)
    return not proposee



def prepare_dataset_ranking(
        dataset,
        left_nodes,
        right_nodes,
        roommate=False,
        mixed=False,
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

        # Kevert adathalmaznál per-gráf dönt a típus; egyébként a
        # globális `roommate` flag.
        use_roommate = (
            is_roommate_graph(graph) if mixed else roommate
        )

        if use_roommate:
            ranking_data = prepare_matching_ranking_data_roommate(
                edge_index=graph.edge_index,
                edge_ranks=edge_ranks,
                num_nodes=graph.num_nodes,
                preferences=preferences,
            )
        else:
            # Páros (marriage) gráfnál a bal/jobb oldali node-okat
            # per-gráf a saját méretéből származtatjuk, hogy a vegyes
            # méretű train adathalmaz (group_size 3..6) is helyes
            # oldal-felosztást kapjon. A konverter (generate_graph_m +
            # graph_to_pyg_data_normal) a 0..n-1 node-okat teszi a bal,
            # az n..2n-1 node-okat a jobb oldalra, ahol n = num_nodes/2.
            if mixed:
                half = graph.num_nodes // 2
                g_left_nodes = list(range(half))
                g_right_nodes = list(range(half, graph.num_nodes))
            else:
                g_left_nodes = left_nodes
                g_right_nodes = right_nodes

            ranking_data = prepare_matching_ranking_data(
                edge_index=graph.edge_index,
                edge_ranks=edge_ranks,
                num_nodes=graph.num_nodes,
                left_nodes=g_left_nodes,
                right_nodes=g_right_nodes,
                preferences=preferences,
            )

        ranking_dataset.append(ranking_data)

    return ranking_dataset





def main(training=False, roommate=True, mixed=False, allow_ties=False):

    group_size=3

    if mixed:
        # 50/50 keverék: szobatárs (roommate) és házassági (marriage)
        # gráfok egy adathalmazban. A ranking-prep és a kiértékelés
        # per-gráf dönti el a típust (lásd lentebb a `is_roommate_graph`
        # detektálást a proposee_pref üres volta alapján).
        train_data, val_data = get_or_generate_mixed_dataset(
            group_size,
            n_train=1000,
            n_val=500,
            cache_path="dataset_mixed.pt",
            seed=42,
            allow_ties=allow_ties,
        )
    elif roommate:
        # Stabil szobatárs (roommate) eset: nem-páros bázisgráf
        # (generate_graph_r) és normális eloszlású preferenciák.
        # A tökéletes szobatárs-párosításhoz PÁROS számú node kell, ezért
        # 2*group_size node-ot használunk (a páros eset összes node-jával
        # megegyező méret).
        #
        # Vegyes méretek: mintánként véletlenszerűen választunk a
        # [6, 8, 10, 12] node-számok közül (ezek a 3, 4, 5, 6 oldalankénti
        # létszámok kétszeresei), így a kiértékelés méret szerint is
        # informatív (lásd a méret szerinti metrikákat a kiértékelésben).
        # A `group_size` csak a cache-kulcs része / fallback marad.
        roommate_nodes = 2 * group_size
        """
        train_data, val_data = get_or_generate_dataset(
            roommate_nodes,
            gen_fn=data_generator.graph_to_pyg_data_r_normal,
            graph_fn=data_generator.generate_graph_r,
            n_train=1000,
            n_val=500,
            cache_path="dataset_roommate.pt",
            seed=42,
            group_sizes=[2 * gs for gs in (3, 4, 5, 6,7,8)],
            allow_ties=allow_ties,
        )
        """
        train_data, val_data = get_or_generate_roommate_test_dataset(
            "normal", allow_ties=True, n_val=1000,
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
        mixed=mixed,
    )

    # A validációs (korai leállításhoz / modellválasztáshoz használt)
    # halmaz ranking-előkészítése, ugyanazzal a konfigurációval, mint a
    # tanító halmaz.
    val_ranking = prepare_dataset_ranking(
        val_data,
        left_nodes,
        right_nodes,
        roommate=roommate,
        mixed=mixed,
    )

    encoder_weight = 0.2
    decoder_weight = 1.0

    node_feature_dim = train_data[0].x.shape[1]

    def build_models():
        """Friss (véletlen inicializált) encoder+decoder+optimizer egy
        tanítási futáshoz."""
        enc = encoder.StableMatchingEdgeEncoder(
            node_feature_dim=node_feature_dim,
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
        return enc, dec, optimizer_joint

    def evaluate_val_loss(enc, dec, temperature):
        """A validációs össz-loss (gradiens nélkül, eval módban) az adott
        hőmérséklettel. Ugyanazt a súlyozott loss-t használja, mint a
        tanítás, így a korai leállítás és a modellválasztás a tanítási
        célt tükröző metrikán alapul."""
        enc.eval()
        dec.eval()
        total = 0.0
        with torch.no_grad():
            for gid in range(len(val_data)):
                graph = val_data[gid]
                ranking_data = val_ranking[gid]

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
                    temperature=temperature,
                    score_reg_weight=1e-4,
                )
                loss = encoder_weight * loss_enc + decoder_weight * loss_dec
                total += loss.item()
        n = max(len(val_data), 1)
        return total / n

    def train_one_run(temperature, max_epochs=100, patience=5, min_delta=1e-3):
        """EGY teljes tanítási futás KORAI LEÁLLÍTÁSSAL az adott
        hőmérséklettel. Friss modellt épít, a validációs loss alapján
        early-stoppol, és a futás LEGJOBB (legkisebb val-loss-ú) modell-
        állapotát adja vissza.

        Visszatérés: (best_val_loss, best_enc_state, best_dec_state,
        best_epoch).
        """
        enc, dec, optimizer_joint = build_models()

        best_val_loss = float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        best_enc_state = copy.deepcopy(enc.state_dict())
        best_dec_state = copy.deepcopy(dec.state_dict())

        for epoch in range(max_epochs):
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
                    temperature=temperature,
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

            # Validációs loss az epoch végén (korai leállításhoz) UGYANAZZAL
            # a hőmérséklettel, amivel a futás tanul.
            val_loss = evaluate_val_loss(enc, dec, temperature)

            print(
                f"temp={temperature:.1f} | epoch={epoch} | "
                f"train_total={total_loss / count:.6f} | "
                f"train_enc={total_encoder_loss / count:.6f} | "
                f"train_rank={total_rank_loss / count:.6f} | "
                f"val_total={val_loss:.6f}"
            )

            # Javulás-e a legjobb eddigi val-loss-hoz képest?
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                best_enc_state = copy.deepcopy(enc.state_dict())
                best_dec_state = copy.deepcopy(dec.state_dict())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(
                        f"temp={temperature:.1f} | Korai leállítás: "
                        f"{patience} epoch óta nincs javulás a validációs "
                        f"loss-ban. Legjobb epoch={best_epoch}, "
                        f"val_total={best_val_loss:.6f}."
                    )
                    break

        return best_val_loss, best_enc_state, best_dec_state, best_epoch

    if training:
        # Egyetlen TELJES tanítási futás KORAI LEÁLLÍTÁSSAL (temperature=1.0).
        # A futás a validációs loss alapján early-stoppol, és a LEGJOBB
        # (legkisebb val-loss-ú) modell-állapotot tartjuk meg.
        temperature = 1.0

        best_val_loss, best_enc_state, best_dec_state, best_epoch = (
            train_one_run(temperature)
        )

        # A LEGJOBB (validáción legkisebb loss-ú) súlyok visszatöltése,
        # hogy az elmentett modell a korai leállítás szerinti legjobb
        # állapot legyen.
        enc, dec, optimizer_joint = build_models()
        enc.load_state_dict(best_enc_state)
        dec.load_state_dict(best_dec_state)
        print(
            f"Legjobb modell visszatöltve (epoch={best_epoch}, "
            f"val_total={best_val_loss:.6f})."
        )

        # A legjobb encoder és decoder súlyok mentése, hogy később
        # visszatölthetők legyenek.
        torch.save(enc.state_dict(), "enc.pt")
        torch.save(dec.state_dict(), "dec.pt")
        print("Encoder és decoder modellek elmentve: enc.pt, dec.pt")
    else:
        # Nem-tanító mód: üres modellek, amelyeket a kiértékeléshez
        # lemezről töltünk be (lásd lentebb).
        enc, dec, optimizer_joint = build_models()



    # A legjobb (early-stopped) közös súlyok visszatöltése az értékeléshez.
    if not training:
        enc.load_state_dict(torch.load("enc.pt", weights_only=True))
        dec.load_state_dict(torch.load("dec.pt", weights_only=True))
    counts = []
    good = 0
    bad = 0

    # Típusonkénti (szobatárs vs. házassági) számlálás és pontosság.
    # good = stabil párosítás (nincs blokkoló pár), bad = instabil.
    roommate_good = 0
    roommate_bad = 0
    marriage_good = 0
    marriage_bad = 0

    # Teljes (complete) párosítás számlálás: a dekódolt megoldásban minden
    # node párosítva van-e (nincs -1 partner). Típusonként külön is.
    complete_count = 0           # ebből teljes párosítású
    complete_total = 0           # összes kiértékelt gráf
    roommate_complete = 0
    roommate_complete_total = 0
    marriage_complete = 0
    marriage_complete_total = 0

    # Méret szerinti bontás. Kulcs: (típus, agent_count), érték:
    # [stabil, összes]. A "méret" a résztvevők (agent) száma:
    #   - házasság (marriage): num_nodes // 2 fő oldalanként,
    #   - szobatárs (roommate): num_nodes fő összesen.
    # Így per-méret látható, hogy a modell mekkora problémákon mennyire
    # pontos (stabilitási arány).
    size_stats = defaultdict(lambda: [0, 0])      # (type, size) -> [stable, total]
    size_stats_overall = defaultdict(lambda: [0, 0])  # size -> [stable, total]

    # Stabilitási szintek szerinti számlálás (SRT holtversenyes esetben).
    # Minden kiértékelt párosítást a LEGERŐSEBB teljesülő szint szerint
    # sorolunk be: super > strong > weak > unstable (a hierarchia miatt a
    # super egyben strong és weak is, stb.). Kulcs: típus -> szint -> db,
    # plusz egy összesített (overall) számláló.
    stability_levels = ["super", "strong", "weak", "unstable"]
    stability_stats = {
        "overall": {lvl: 0 for lvl in stability_levels},
        "roommate": {lvl: 0 for lvl in stability_levels},
        "marriage": {lvl: 0 for lvl in stability_levels},
    }
    # Méret szerinti stabilitási szint bontás: (típus, size) -> {szint: db}.
    stability_by_size = defaultdict(
        lambda: {lvl: 0 for lvl in stability_levels}
    )

    # Hány gráfra írjunk RÉSZLETES stabilitás-minősítő debug printet.
    # (Az egész val-setre túl sok lenne; az első pár gráf elég az
    # ellenőrzéshez.) Állítsd 0-ra a kikapcsoláshoz, nagyobbra többért.
    STABILITY_DEBUG_GRAPHS = 0
    stability_debug_done = 0

    for graph in val_data:
        enc.eval()
        dec.eval()

        graph_is_roommate = is_roommate_graph(graph)

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

        # Teljes párosítás ellenőrzése: a dekódolt megoldás AKKOR teljes,
        # ha minden node párt kapott (nincs -1 partner). A greedy dekóder
        # feasibilitást garantál, de teljességet nem, ezért külön nézzük.
        is_complete = bool((partner != -1).all().item())
        complete_total += 1
        if is_complete:
            complete_count += 1
        if graph_is_roommate:
            roommate_complete_total += 1
            if is_complete:
                roommate_complete += 1
        else:
            marriage_complete_total += 1
            if is_complete:
                marriage_complete += 1
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

        # A párosítás legerősebb teljesülő stabilitási szintje
        # (super > strong > weak > unstable) az SRT holtversenyes
        # definíciók szerint. A "weak" megegyezik azzal, hogy nincs
        # (gyenge) blokkoló pár, azaz is_stable igaz.
        do_stability_debug = stability_debug_done < STABILITY_DEBUG_GRAPHS
        if do_stability_debug:
            print(
                f"\n######## Stabilitás-minősítés debug "
                f"(gráf #{stability_debug_done + 1}/"
                f"{STABILITY_DEBUG_GRAPHS}, "
                f"típus={'roommate' if graph_is_roommate else 'marriage'}, "
                f"node-ok={int(graph.num_nodes)}) ########"
            )
        level = classify_stability_level(
            edge_index=graph.edge_index,
            edge_ranks=ranks,
            partner=partner,
            debug=do_stability_debug,
        )
        if do_stability_debug:
            stability_debug_done += 1
        is_stable = level != "unstable"  # = gyengén stabil (nincs weak-blokk)

        if is_stable:
            good += 1
        else:
            bad += 1

        if graph_is_roommate:
            if is_stable:
                roommate_good += 1
            else:
                roommate_bad += 1
        else:
            if is_stable:
                marriage_good += 1
            else:
                marriage_bad += 1

        # Méret szerinti akkumuláció.
        #   - Típusonként a "size" a résztvevők (agent) száma a típus
        #     természetes egységében: házasságnál az oldalankénti létszám
        #     (num_nodes // 2), szobatársnál a teljes létszám (num_nodes).
        #   - Az ÖSSZESÍTETT bontás a node-számot (num_nodes) használja
        #     kulcsként, hogy a két típus mérete egyértelmű, közös skálán
        #     legyen összehasonlítható.
        num_nodes = int(graph.num_nodes)
        if graph_is_roommate:
            size = num_nodes
            type_name = "roommate"
        else:
            size = num_nodes // 2
            type_name = "marriage"

        size_stats[(type_name, size)][1] += 1
        size_stats_overall[num_nodes][1] += 1
        if is_stable:
            size_stats[(type_name, size)][0] += 1
            size_stats_overall[num_nodes][0] += 1

        # Stabilitási szint szerinti számlálás.
        stability_stats["overall"][level] += 1
        stability_stats[type_name][level] += 1
        stability_by_size[(type_name, size)][level] += 1

    print("Blocking-count eloszlás:", dict(sorted(Counter(counts).items())))
    if counts:
        print("Átlag:", sum(counts) / len(counts))
        print("Stabilitási arány:",
              sum(c == 0 for c in counts) / len(counts))
    else:
        print("Nincs kiértékelhető gráf (minden gráf kihagyva).")

    # --- Teljes párosítás (complete matching) arány ---
    # Hány dekódolt megoldás volt teljes (minden node párt kapott),
    # összességében és típusonként.
    print("--- Teljes párosítás (complete matching) ---")
    if complete_total > 0:
        print(
            f"  Összesített: {complete_count / complete_total:.4f} "
            f"({complete_count}/{complete_total})"
        )
    else:
        print("  Nincs kiértékelt gráf.")
    if roommate_complete_total > 0:
        print(
            f"  Szobatárs (roommate): "
            f"{roommate_complete / roommate_complete_total:.4f} "
            f"({roommate_complete}/{roommate_complete_total})"
        )
    if marriage_complete_total > 0:
        print(
            f"  Házassági (marriage): "
            f"{marriage_complete / marriage_complete_total:.4f} "
            f"({marriage_complete}/{marriage_complete_total})"
        )

    # Típusonkénti bontás: hány szobatárs / házassági problémát néztünk,
    # és mekkora volt a stabilitási arány (pontosság) külön-külön.
    roommate_total = roommate_good + roommate_bad
    marriage_total = marriage_good + marriage_bad

    # --- Stabilitási szintek (SRT holtversenyes) ---
    # Minden kiértékelt párosítást a LEGERŐSEBB teljesülő szintje szerint
    # soroltunk be (super > strong > weak > unstable). A hierarchia miatt:
    #   super-stabil arány  = super
    #   strong-stabil arány = super + strong
    #   weak-stabil arány   = super + strong + weak
    # (azaz a magasabb szint mindig teljesíti az alacsonyabbat is).
    def print_stability_block(label, levels_counts):
        total = sum(levels_counts.values())
        print(f"{label} (összes: {total})")
        if total == 0:
            return
        n_super = levels_counts["super"]
        n_strong = levels_counts["strong"]
        n_weak = levels_counts["weak"]
        n_unstable = levels_counts["unstable"]
        # Kumulatív (hierarchikus) arányok.
        super_ratio = n_super / total
        strong_ratio = (n_super + n_strong) / total
        weak_ratio = (n_super + n_strong + n_weak) / total
        print(
            f"  super : {super_ratio:.4f} ({n_super}/{total})"
        )
        print(
            f"  strong: {strong_ratio:.4f} "
            f"({n_super + n_strong}/{total})  [super+strong]"
        )
        print(
            f"  weak  : {weak_ratio:.4f} "
            f"({n_super + n_strong + n_weak}/{total})  [super+strong+weak]"
        )
        print(
            f"  instabil (egyik szint sem): "
            f"{n_unstable / total:.4f} ({n_unstable}/{total})"
        )
        # Nyers, egymást kizáró szint-eloszlás is.
        print(
            f"  nyers eloszlás: super={n_super}, strong={n_strong}, "
            f"weak={n_weak}, unstable={n_unstable}"
        )

    print("--- Stabilitási szintek (super / strong / weak) ---")
    print_stability_block("Összesített", stability_stats["overall"])
    print_stability_block("Szobatárs (roommate)", stability_stats["roommate"])
    print_stability_block("Házassági (marriage)", stability_stats["marriage"])

    print("--- Típusonkénti kiértékelés ---")
    print(f"Szobatárs (roommate) problémák: {roommate_total}")
    if roommate_total > 0:
        print(
            f"  Szobatárs stabilitási arány: "
            f"{roommate_good / roommate_total:.4f} "
            f"({roommate_good}/{roommate_total})"
        )
    print(f"Házassági (marriage) problémák: {marriage_total}")
    if marriage_total > 0:
        print(
            f"  Házassági stabilitási arány: "
            f"{marriage_good / marriage_total:.4f} "
            f"({marriage_good}/{marriage_total})"
        )


    # --- Stabilitási szintek méret szerint ---
    # Kumulatív arányok (super <= strong <= weak) méretenként, hogy
    # látható legyen, a nagyobb problémákon melyik szint mennyire tartható.
    print("--- Stabilitási szintek méret szerint (kumulatív arányok) ---")
    for type_name, type_label, unit in (
        ("marriage", "Házassági (marriage)", "oldalankénti létszám"),
        ("roommate", "Szobatárs (roommate)", "teljes létszám"),
    ):
        sizes = sorted(
            s for (t, s) in stability_by_size if t == type_name
        )
        if not sizes:
            continue
        print(f"{type_label} ({unit}):")
        for size in sizes:
            lc = stability_by_size[(type_name, size)]
            total = sum(lc.values())
            if total == 0:
                continue
            n_super = lc["super"]
            n_strong = lc["strong"]
            n_weak = lc["weak"]
            super_r = n_super / total
            strong_r = (n_super + n_strong) / total
            weak_r = (n_super + n_strong + n_weak) / total
            print(
                f"  méret={size:>3} (n={total}): "
                f"super={super_r:.3f} | strong={strong_r:.3f} | "
                f"weak={weak_r:.3f}"
            )
    print("finished")

if __name__ == "__main__":
    training = False
    roommate = True
    mixed = False
    allow_ties =  True
    main(training, roommate, mixed, allow_ties)