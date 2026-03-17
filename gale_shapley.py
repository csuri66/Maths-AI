import networkx as nx
from collections import deque


def stable_matching_with_preferences(G, left_nodes, right_nodes, prefs_left, prefs_right):

    left_nodes = list(left_nodes)
    right_nodes = list(right_nodes)
    right_set = set(right_nodes)
    left_set = set(left_nodes)

    # Csak létező és a gráfban ténylegesen jelen levő szomszédok maradnak
    filtered_left = {}
    for u in left_nodes:
        pref_list = prefs_left.get(u, [])
        filtered_left[u] = [v for v in pref_list if v in right_set and G.has_edge(u, v)]

    filtered_right = {}
    for v in right_nodes:
        pref_list = prefs_right.get(v, [])
        filtered_right[v] = [u for u in pref_list if u in left_set and G.has_edge(u, v)]

    rank_right = {
        v: {u: i for i, u in enumerate(filtered_right[v])}
        for v in right_nodes
    }

    match_left = {u: None for u in left_nodes}
    match_right = {v: None for v in right_nodes}
    next_proposal_idx = {u: 0 for u in left_nodes}

    free_left = deque(left_nodes)

    while free_left:
        u = free_left.popleft()

        if next_proposal_idx[u] >= len(filtered_left[u]):
            continue

        v = filtered_left[u][next_proposal_idx[u]]
        next_proposal_idx[u] += 1

        current = match_right[v]

        if current is None:
            match_left[u] = v
            match_right[v] = u
        else:
            if rank_right[v][u] < rank_right[v][current]:
                match_left[current] = None
                free_left.append(current)

                match_left[u] = v
                match_right[v] = u
            else:
                free_left.append(u)

    return match_left