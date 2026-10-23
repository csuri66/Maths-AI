"""
Két dekóder összehasonlító tesztje UGYANAZON az adathalmazon.

1) test_greedy_decode:
   A JELENLEGI greedy dekóder (`greedy_decode_pairs`) az SRT (holtversenyes
   stabil szobatárs) párosítások előállítására.

2) test_irving_srt_decode:
   AZ ÚJ Irving-alapú SRT dekóder (`irving_srt_decode_pairs`), amely a
   klasszikus Irving-algoritmust futtatja a szobatárs-problémára, és a
   HOLTVERSENYEKET A MODELL PREDIKCIÓJA bontja (model-guided tie-break).

Mindkét teszt:
  - UGYANAZT a betöltött modellt (enc.pt / dec.pt) és
  - UGYANAZT a validációs adathalmazt használja,
így az eredmények közvetlenül összehasonlíthatók.

A tesztek NEM futnak automatikusan (nincs __main__ hívás a végén az
automatikus futtatáshoz szükséges argumentumok nélkül); a `run_both_tests`
hívásával indíthatók.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict

import torch

import main


# Az enc/dec modell súlyainak elérési útja (ugyanaz, amit a main.py is ment).
ENC_WEIGHTS_PATH = "enc.pt"
DEC_WEIGHTS_PATH = "dec.pt"

# Az összehasonlításhoz használt szobatárs (roommate) eloszlás és méret.
# Ugyanaz, amit a main() is használ a kiértékeléshez.
DISTRIBUTION = "top_ties"
GROUP_SIZE = 3

# CSAK ilyen méretű (ennyi AGENT = ennyi NODE) gráfokon tesztelünk.
# Szobatárs (roommate) esetben nincs oldal-felosztás, így az agentek száma
# = a node-ok száma. A közös adathalmaz KIZÁRÓLAG ekkora gráfokat tartalmaz
# (a generátornak `group_sizes=[AGENT_COUNT]`-ot adunk), és a kiértékelő egy
# biztonsági szűrővel is garantálja, hogy csak ennyi node-ú gráf kerül be.
AGENT_COUNT = 100


def load_models():
    """
    A main.py-beli encoder + decoder felépítése és a lemezre mentett
    (enc.pt / dec.pt) súlyok betöltése. A node-feature dimenziót az első
    betöltött gráfból vesszük, hogy pontosan egyezzen a tanításkor
    használttal.
    """
    # Egy gráf a feature-dimenzió kiolvasásához.
    _, val_data = load_shared_dataset()
    node_feature_dim = val_data[0].x.shape[1]

    enc = main.encoder.StableMatchingEdgeEncoder(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=1,
        hidden_dim=32,
        num_layers=6,
    )
    dec = main.decoder.GreedyMatchingDecoder(
        edge_embedding_dim=34,
        hidden_dim=34,
    )

    enc.load_state_dict(torch.load(ENC_WEIGHTS_PATH, weights_only=True))
    dec.load_state_dict(torch.load(DEC_WEIGHTS_PATH, weights_only=True))

    enc.eval()
    dec.eval()
    return enc, dec


# Egyszeri (process-szintű) cache, hogy MINDKÉT teszt PONTOSAN ugyanazt az
# adathalmazt kapja (ugyanazzal a seeddel generált / betöltött objektumok).
_SHARED_DATASET = None


def load_shared_dataset():
    """
    A KÖZÖS validációs adathalmaz betöltése (vagy legenerálása és
    cache-elése). Mindhárom teszt ezt hívja, így garantáltan UGYANAZON az
    adathalmazon fut.

    A halmaz KIZÁRÓLAG `AGENT_COUNT` (= 100) node-ú szobatárs-gráfokat
    tartalmaz: a generátornak `group_sizes=[AGENT_COUNT]`-ot adunk, így
    minden minta pontosan ennyi agentből áll. Külön cache-fájlt használunk
    (az agentszám a fájlnévben), hogy ne ütközzön a vegyes méretű
    alap-adathalmaz cache-ével.
    """
    global _SHARED_DATASET
    if _SHARED_DATASET is None:
        cache_path = (
            f"dataset_roommate_{DISTRIBUTION}_agents{AGENT_COUNT}.pt"
        )
        _SHARED_DATASET = main.get_or_generate_roommate_test_dataset(
            DISTRIBUTION,
            allow_ties=True,
            n_train=0,
            n_val=500,
            group_sizes=[AGENT_COUNT],
            cache_path=cache_path,
        )
    return _SHARED_DATASET


def _forward_pair_logits(enc, dec, graph):
    """
    Egy gráfra lefuttatja az encodert + decodert, és visszaadja az
    irányított él-logitokat ([E]) UGYANÚGY, ahogy a main() kiértékelő
    hurka teszi (mutual_quality és rank_score feature-ök hozzáfűzve az
    edge_embeddinghez).
    """
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
    return dec(out.edge_embedding)


def evaluate_decoder(enc, dec, val_data, decode_fn, label):
    """
    Egy tetszőleges dekóló stratégiát (`decode_fn`) kiértékel a megadott
    adathalmazon, és kiírja a stabilitási / teljességi metrikákat.

    decode_fn(graph, pair_logits) -> partner tensor [num_nodes]
        A dekóder absztrakciója: a két tesztben más-más dekódert adunk át,
        de MINDEN egyéb (modell, adat, metrika-számítás) megegyezik.

    Visszatér: a metrikák dict-je (programozott összehasonlításhoz is).
    """
    print(f"\n==================== {label} ====================")

    counts = []
    good = 0
    bad = 0

    complete_count = 0
    complete_total = 0

    stability_levels = ["super", "strong", "weak", "unstable"]
    stability_stats = {lvl: 0 for lvl in stability_levels}
    size_stats = defaultdict(lambda: [0, 0])  # size -> [stable, total]

    skipped = 0

    # A gráfonkénti (dekódolási) megoldási idők gyűjtése.
    solve_times = []

    for graph in val_data:
        # BIZTONSÁGI szűrő: csak AGENT_COUNT (=100) node-ú gráfokat
        # értékelünk. Szobatárs esetben az agentek száma = node-ok száma.
        # A közös adathalmaz már eleve csak ekkora gráfokat tartalmaz, de a
        # szűrő garantálja ezt akkor is, ha más adaton futna.
        if int(graph.num_nodes) != AGENT_COUNT:
            skipped += 1
            continue

        pair_logits = _forward_pair_logits(enc, dec, graph)

        # A dekódolás (megoldás) idejének mérése gráfonként.
        _t0 = time.perf_counter()
        partner = decode_fn(graph, pair_logits)
        solve_times.append(time.perf_counter() - _t0)

        # Teljes párosítás ellenőrzése (minden node párt kapott-e).
        is_complete = bool((partner != -1).all().item())
        complete_total += 1
        if is_complete:
            complete_count += 1

        ranks = main.preference_lists_to_edge_ranks(
            graph.proposee_pref | graph.proposer_pref,
            graph.edge_index,
        )
        blocking_pairs = main.find_blocking_pairs(
            edge_index=graph.edge_index,
            edge_attr=ranks,
            partner=partner,
        )
        counts.append(len(blocking_pairs))

        level = main.classify_stability_level(
            edge_index=graph.edge_index,
            edge_ranks=ranks,
            partner=partner,
            debug=False,
        )
        is_stable = level != "unstable"
        if is_stable:
            good += 1
        else:
            bad += 1

        stability_stats[level] += 1

        num_nodes = int(graph.num_nodes)
        size_stats[num_nodes][1] += 1
        if is_stable:
            size_stats[num_nodes][0] += 1

    # --- Összegzés ---
    print(f"Kiértékelt gráfok (agent={AGENT_COUNT}): {len(counts)} "
          f"(kihagyva más méret miatt: {skipped})")
    print("Blocking-count eloszlás:", dict(sorted(Counter(counts).items())))
    total = max(len(counts), 1)
    avg_blocking = sum(counts) / total
    stability_ratio = sum(c == 0 for c in counts) / total
    print(f"Átlagos blokkoló-pár szám: {avg_blocking:.4f}")
    print(f"(Blocking==0) arány: {stability_ratio:.4f}")

    # Weak-stabilitási arány (nincs weak-blokkoló pár) a szint-besorolásból.
    weak_stable = good / total
    print(f"Weak-stabil (level != unstable) arány: "
          f"{weak_stable:.4f} ({good}/{total})")

    print(f"Teljes párosítás arány: "
          f"{complete_count / max(complete_total, 1):.4f} "
          f"({complete_count}/{complete_total})")

    # Stabilitási szintek (kumulatív, super <= strong <= weak).
    n_super = stability_stats["super"]
    n_strong = stability_stats["strong"]
    n_weak = stability_stats["weak"]
    n_unstable = stability_stats["unstable"]
    print("Stabilitási szintek (kumulatív arányok):")
    print(f"  super : {n_super / total:.4f}")
    print(f"  strong: {(n_super + n_strong) / total:.4f}  [super+strong]")
    print(f"  weak  : {(n_super + n_strong + n_weak) / total:.4f}  "
          f"[super+strong+weak]")
    print(f"  instabil: {n_unstable / total:.4f}")
    print(f"  nyers eloszlás: super={n_super}, strong={n_strong}, "
          f"weak={n_weak}, unstable={n_unstable}")

    # Méret szerinti weak-stabilitási arány.
    print("Weak-stabil arány méret (node-szám) szerint:")
    for size in sorted(size_stats):
        stable, tot = size_stats[size]
        if tot == 0:
            continue
        print(f"  méret={size:>3} (n={tot}): {stable / tot:.4f}")

    # Átlagos (gráfonkénti) megoldási idő.
    avg_solve_time = (sum(solve_times) / len(solve_times)) if solve_times else 0.0
    print(f"Átlagos megoldási idő gráfonként: "
          f"{avg_solve_time * 1000:.3f} ms  "
          f"(összes: {sum(solve_times):.3f} s, n={len(solve_times)})")

    return {
        "avg_blocking": avg_blocking,
        "stability_ratio": stability_ratio,
        "weak_stable": weak_stable,
        "complete_ratio": complete_count / max(complete_total, 1),
        "levels": dict(stability_stats),
        "avg_solve_time": avg_solve_time,
    }


def test_greedy_decode(enc=None, dec=None, val_data=None):
    """
    TESZT 1: a JELENLEGI greedy dekóder (`greedy_decode_pairs`).

    A greedy dekóder az irányított él-logitokból kölcsönös párokat képez,
    és score szerint csökkenő sorrendben, mohón választja ki a párokat
    (hard mask garantálja a matching-feasibilityt). Stabilitást nem
    garantál.
    """
    if enc is None or dec is None:
        enc, dec = load_models()
    if val_data is None:
        _, val_data = load_shared_dataset()

    def greedy_decode_fn(graph, pair_logits):
        return main.greedy_decode_pairs(
            edge_index=graph.edge_index,
            directed_logits=pair_logits,
            num_nodes=graph.x.size(0),
            debug=False,
        )

    return evaluate_decoder(
        enc=enc,
        dec=dec,
        val_data=val_data,
        decode_fn=greedy_decode_fn,
        label="TESZT 1: GREEDY DEKÓDER",
    )


def test_irving_srt_decode(enc=None, dec=None, val_data=None):
    """
    TESZT 2: AZ ÚJ Irving-alapú SRT dekóder (`irving_srt_decode_pairs`).

    A klasszikus Irving-algoritmust futtatja a szobatárs-problémára. A
    holtversenyeket (ties) a MODELL PREDIKCIÓJA bontja: egy holtversenyes
    rétegen belül a nagyobb él-logitú partner kerül előrébb a
    preferencialistában. Így az eredmény az eredeti holtversenyes
    instanciában gyengén stabil (ha a tie-break szerint létezik stabil
    párosítás).
    """
    if enc is None or dec is None:
        enc, dec = load_models()
    if val_data is None:
        _, val_data = load_shared_dataset()

    def irving_decode_fn(graph, pair_logits):
        return main.irving_srt_decode_pairs(
            edge_index=graph.edge_index,
            directed_logits=pair_logits,
            num_nodes=graph.x.size(0),
            preferences=graph.proposee_pref | graph.proposer_pref,
            debug=False,
        )

    return evaluate_decoder(
        enc=enc,
        dec=dec,
        val_data=val_data,
        decode_fn=irving_decode_fn,
        label="TESZT 2: IRVING SRT DEKÓDER (model-guided tie-break)",
    )


def test_irving_srt_decode_random(enc=None, dec=None, val_data=None, seed=42):
    """
    TESZT 3: AZ Irving-alapú SRT dekóder VÉLETLEN tie-break-kel
    (`irving_srt_decode_pairs_random`).

    MINDEN megegyezik a TESZT 2-vel (ugyanaz az Irving-algoritmus, ugyanaz
    az allowed_pairs kölcsönösségi szűrés), KIZÁRÓLAG a holtverseny-feloldás
    tér el: itt a modell predikciója HELYETT egy VÉLETLEN permutáció
    szigorúsítja a holtversenyes rétegeket. Ez a baseline, amelyhez a
    model-guided tie-break (TESZT 2) hasznosságát mérni lehet.

    seed:
        A véletlen tie-break reprodukálhatóságához. Rögzített `seed` esetén
        a teszt determinisztikus (ugyanaz a permutáció-sorozat).
    """
    if enc is None or dec is None:
        enc, dec = load_models()
    if val_data is None:
        _, val_data = load_shared_dataset()

    # Rögzített RNG a reprodukálhatóságért: minden futás ugyanazt a
    # véletlen tie-break-sorozatot adja.
    import random as _random
    rng = _random.Random(seed)

    def irving_random_decode_fn(graph, pair_logits):
        return main.irving_srt_decode_pairs_random(
            edge_index=graph.edge_index,
            directed_logits=pair_logits,
            num_nodes=graph.x.size(0),
            preferences=graph.proposee_pref | graph.proposer_pref,
            rng=rng,
            debug=False,
        )

    return evaluate_decoder(
        enc=enc,
        dec=dec,
        val_data=val_data,
        decode_fn=irving_random_decode_fn,
        label="TESZT 3: IRVING SRT DEKÓDER (random tie-break)",
    )


def run_both_tests():
    """
    Mindhárom tesztet UGYANAZON a modellen és UGYANAZON az adathalmazon
    futtatja, majd rövid összehasonlítást ír ki.

    A nevében "both" maradt a visszafelé kompatibilitásért, de már három
    dekódert hasonlít össze:
      - greedy,
      - Irving model-guided tie-break,
      - Irving random tie-break.
    """
    enc, dec = load_models()
    _, val_data = load_shared_dataset()

    greedy_metrics = test_greedy_decode(enc, dec, val_data)
    irving_metrics = test_irving_srt_decode(enc, dec, val_data)
    irving_random_metrics = test_irving_srt_decode_random(enc, dec, val_data)

    print("\n==================== ÖSSZEHASONLÍTÁS ====================")
    print(f"{'metrika':<28}{'greedy':>12}{'irving':>12}{'irving_rnd':>12}")
    for key in ("stability_ratio", "weak_stable", "complete_ratio",
                "avg_blocking"):
        g = greedy_metrics[key]
        i = irving_metrics[key]
        r = irving_random_metrics[key]
        print(f"{key:<28}{g:>12.4f}{i:>12.4f}{r:>12.4f}")

    return greedy_metrics, irving_metrics, irving_random_metrics


# A tesztek NEM futnak le automatikusan import-kor. Szándékos futtatáshoz:
#
#   python -c "import test_decoders; test_decoders.test_irving_srt_decode()"
#
# vagy a következő blokk kikommentezésével / külön futtatásával.
#
if __name__ == "__main__":
    # CSAK az Irving + modell-vezérelt holtverseny-feloldás (TESZT 2) fut,
    # és kiírja a gráfonkénti átlagos megoldási időt.
    test_irving_srt_decode()
