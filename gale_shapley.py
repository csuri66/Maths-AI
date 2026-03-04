import networkx as nx
from collections import deque


def solve(G, proposers, proposees, proposer_prefs, proposee_prefs):
    # Current proposals index for each proposer
    next_proposal = {p: 0 for p in proposers}

    # Free proposers queue
    free_proposers = deque(proposers)

    # Current matches: proposee -> proposer
    matches = {pe: None for pe in proposees}

    while free_proposers:
        p = free_proposers.popleft()

        prefs = proposer_prefs[p]
        proposed = False

        while next_proposal[p] < len(prefs) and not proposed:
            pe = prefs[next_proposal[p]]

            if pe not in G[p]:  # No edge, skip
                next_proposal[p] += 1
                continue

            next_proposal[p] += 1

            current = matches[pe]

            if current is None:
                # Accept
                matches[pe] = p
                proposed = True
            else:
                # Compare ranks (lower better)
                if proposee_prefs[pe][p] < proposee_prefs[pe][current]:
                    # Accept new, reject old
                    matches[pe] = p
                    free_proposers.append(current)
                    proposed = True


    # Return proposer-centric matching
    result = {}
    for pe, p in matches.items():
        if p is not None:
            result[p] = pe
    return result