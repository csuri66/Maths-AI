import networkx as nx
import torch
from torch_geometric.data import Data
import random
import gale_shapley
from itertools import chain

def generate_graph(group_nodes=10):
    G = nx.Graph()

    G.add_nodes_from(list(range(group_nodes)), bipartite=0)
    G.add_nodes_from(list(range(group_nodes)), bipartite=1)

    for i in range(group_nodes):
        for j in range(group_nodes):
            G.add_edge(i,group_nodes+j)
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

def graph_to_pyg_data_random(G, group_size,verbose=False):

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
        edge_weight += [group_size - x for x in prefs_for_model]
        x.append(prefs_for_model)
        proposer_pref[node] = prefs_for_gale

    temp = []
    for node in set2:
        prefs_for_gale = list(range(group_size))
        random.shuffle(prefs_for_gale)
        prefs_for_model = prefs_for_gale
        temp += [group_size - x for x in prefs_for_model]

        edge_weight= list(chain(*zip(temp, edge_weight)))
        x.append(prefs_for_model)
        proposee_pref[node] = prefs_for_gale
    e_w = []
    for i in range(0,len(edge_weight),2):
        e_w.append(edge_weight[i] + edge_weight[i+1])
    edge_weight=e_w
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    look=group_size
    who_am_i=0
    where_am_i =0
    x_final = []
    for node in set1:
        look = 1 * group_size
        x_final.append([])
        x_final[who_am_i].append(0)
        x_final[who_am_i].append(0)
        x_final[who_am_i].append(0)
        for node2 in set2:
            if x[look][0] == who_am_i:
                x_final[who_am_i][0]  +=10
            if x[look][-1] == who_am_i:
                x_final[who_am_i][2] -=10
            x_final[who_am_i][1]-=x[look].index(who_am_i)+1
            look+=1
        where_am_i+=1
        who_am_i+=1


    look = 0
    who_am_i = 0
    where_am_i = group_size
    for node in set2:
        look = 0
        x_final.append([])
        x_final[where_am_i].append(0)
        x_final[where_am_i].append(0)
        x_final[where_am_i].append(0)
        for node2 in set1:
            if x[look][0] == who_am_i:
                x_final[where_am_i][0] += 10
            if x[look][-1] == who_am_i:
                x_final[where_am_i][2] -=10
            x_final[where_am_i][1] -= x[look].index(who_am_i)+1
            look += 1
        where_am_i+=1
        who_am_i += 1



    edges = []
    for u, v in G.edges():
        ui, vi = node_id_map[u], node_id_map[v]
        edges.append([ui, vi])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()




    matching = gale_shapley.stable_matching_with_preferences(G,set1,set2,proposer_pref,proposee_pref)
    if(verbose):
        for line in x:
            print(line)
        print("Optimal matching")
        print(is_stable_matching(matching, proposer_pref, proposee_pref))
        print(matching)
    e_attr= []

    for u,v in G.edges():
        if u in matching and matching[u] == v:
            e_attr.append(1)
        else:
            e_attr.append(0)
    edge_attr = torch.tensor(e_attr, dtype=torch.float)
    data_x = torch.tensor(x_final, dtype=torch.float)
    data = Data(
        x=data_x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        edge_y=edge_attr,
        proposee_pref = proposee_pref,
        proposer_pref = proposer_pref
    )
    return data

def graph_to_pyg_data_dominant_proposee(G, group_size,verbose=False):

    nodes = list(G.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes)}



    set1, set2 = nx.bipartite.sets(G)

    x = []
    proposer_pref = {}
    proposee_pref = {}

    edge_weight = []

    dominant = 0
    noDominant = True
    for node in set1:
        prefs_for_gale = list(range(group_size,group_size*2))
        random.shuffle(prefs_for_gale)
        if noDominant:
            dominant = prefs_for_gale[0]
            noDominant = False
        else:
            prefs_for_gale.remove(dominant)
            prefs_for_gale.insert(0,dominant)

        prefs_for_model = [x - group_size for x in prefs_for_gale]
        edge_weight += [group_size - x for x in prefs_for_model]
        x.append(prefs_for_model)
        proposer_pref[node] = prefs_for_gale

    temp = []
    for node in set2:
        prefs_for_gale = list(range(group_size))
        random.shuffle(prefs_for_gale)
        prefs_for_model = prefs_for_gale
        temp += [group_size - x for x in prefs_for_model]

        edge_weight= list(chain(*zip(temp, edge_weight)))
        x.append(prefs_for_model)
        proposee_pref[node] = prefs_for_gale
    e_w = []
    for i in range(0,len(edge_weight),2):
        e_w.append(edge_weight[i] + edge_weight[i+1])
    edge_weight=e_w
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    look=group_size
    who_am_i=0
    where_am_i =0
    x_final = []
    for node in set1:
        look = 1 * group_size
        x_final.append([])
        x_final[who_am_i].append(0)
        x_final[who_am_i].append(0)
        x_final[who_am_i].append(0)
        for node2 in set2:
            if x[look][0] == who_am_i:
                x_final[who_am_i][0]  +=10 *(group_size/2)
            if x[look][-1] == who_am_i:
                x_final[who_am_i][2] -=10 *(group_size/2)
            x_final[who_am_i][1]-=x[look].index(who_am_i)+group_size
            look+=1
        where_am_i+=1
        who_am_i+=1


    look = 0
    who_am_i = 0
    where_am_i = group_size
    for node in set2:
        look = 0
        x_final.append([])
        x_final[where_am_i].append(0)
        x_final[where_am_i].append(0)
        x_final[where_am_i].append(0)
        for node2 in set1:
            if x[look][0] == who_am_i:
                x_final[where_am_i][0] += 10
            if x[look][-1] == who_am_i:
                x_final[where_am_i][2] -= 10
            x_final[where_am_i][1] -= x[look].index(who_am_i)+1
            look += 1
        where_am_i+=1
        who_am_i += 1



    edges = []
    for u, v in G.edges():
        ui, vi = node_id_map[u], node_id_map[v]
        edges.append([ui, vi])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()




    matching = gale_shapley.stable_matching_with_preferences(G,set1,set2,proposer_pref,proposee_pref)
    if(verbose):
        for line in x:
            print(line)
        print("Data")
        print(x_final)
    e_attr= []

    for u,v in G.edges():
        if u in matching and matching[u] == v:
            e_attr.append(1)
        else:
            e_attr.append(0)
    edge_attr = torch.tensor(e_attr, dtype=torch.float)
    data_x = torch.tensor(x_final, dtype=torch.float)
    data = Data(
        x=data_x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        edge_y=edge_attr,
        proposee_pref = proposee_pref,
        proposer_pref = proposer_pref
    )
    return data

def graph_to_pyg_data_dominant_proposer(G, group_size,verbose=False):

    nodes = list(G.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes)}



    set1, set2 = nx.bipartite.sets(G)

    x = []
    proposer_pref = {}
    proposee_pref = {}

    edge_weight = []


    noDominant = True
    for node in set1:
        prefs_for_gale = list(range(group_size,group_size*2))
        random.shuffle(prefs_for_gale)

        prefs_for_model = [x - group_size for x in prefs_for_gale]
        edge_weight += [group_size - x for x in prefs_for_model]
        x.append(prefs_for_model)
        proposer_pref[node] = prefs_for_gale

    dominant = 0
    temp = []
    for node in set2:
        prefs_for_gale = list(range(group_size))
        random.shuffle(prefs_for_gale)
        if noDominant:
            dominant = prefs_for_gale[0]
            noDominant = False
        else:
            prefs_for_gale.remove(dominant)
            prefs_for_gale.insert(0,dominant)

        prefs_for_model = prefs_for_gale
        temp += [group_size - x for x in prefs_for_model]

        edge_weight= list(chain(*zip(temp, edge_weight)))
        x.append(prefs_for_model)
        proposee_pref[node] = prefs_for_gale
    e_w = []
    for i in range(0,len(edge_weight),2):
        e_w.append(edge_weight[i] + edge_weight[i+1])
    edge_weight=e_w
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    look=group_size
    who_am_i=0
    where_am_i =0
    x_final = []
    for node in set1:
        look = 1 * group_size
        x_final.append([])
        x_final[who_am_i].append(0)
        x_final[who_am_i].append(0)
        x_final[who_am_i].append(0)
        for node2 in set2:
            if x[look][0] == who_am_i:
                x_final[who_am_i][0]  +=10
            if x[look][-1] == who_am_i:
                x_final[who_am_i][2] -=10
            x_final[who_am_i][1]-=x[look].index(who_am_i)+1
            look+=1
        where_am_i+=1
        who_am_i+=1


    look = 0
    who_am_i = 0
    where_am_i = group_size
    for node in set2:
        look = 0
        x_final.append([])
        x_final[where_am_i].append(0)
        x_final[where_am_i].append(0)
        x_final[where_am_i].append(0)
        for node2 in set1:
            if x[look][0] == who_am_i:
                x_final[where_am_i][0] += 10
            if x[look][-1] == who_am_i:
                x_final[where_am_i][2] -=10
            x_final[where_am_i][1] -= x[look].index(who_am_i)+1
            look += 1
        where_am_i+=1
        who_am_i += 1



    edges = []
    for u, v in G.edges():
        ui, vi = node_id_map[u], node_id_map[v]
        edges.append([ui, vi])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()




    matching = gale_shapley.stable_matching_with_preferences(G,set1,set2,proposer_pref,proposee_pref)
    if(verbose):
        for line in x:
            print(line)
        print("Data")
        print(x_final)
    e_attr= []

    for u,v in G.edges():
        if u in matching and matching[u] == v:
            e_attr.append(1)
        else:
            e_attr.append(0)
    edge_attr = torch.tensor(e_attr, dtype=torch.float)
    data_x = torch.tensor(x_final, dtype=torch.float)
    data = Data(
        x=data_x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        edge_y=edge_attr,
        proposee_pref = proposee_pref,
        proposer_pref = proposer_pref
    )
    return data

def graph_to_pyg_data_trivial(G, group_size,verbose=False):

    nodes = list(G.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes)}



    set1, set2 = nx.bipartite.sets(G)

    x = []
    proposer_pref = {}
    proposee_pref = {}

    edge_weight = []

    for node in set1:
        prefs_for_gale = list(range(group_size,group_size*2))
        prefs_for_model = [x - group_size for x in prefs_for_gale]
        edge_weight += [group_size - x for x in prefs_for_model]
        x.append(prefs_for_model)
        proposer_pref[node] = prefs_for_gale

    temp = []
    for node in set2:
        prefs_for_gale = list(range(group_size))
        prefs_for_model = prefs_for_gale
        temp += [group_size - x for x in prefs_for_model]

        edge_weight= list(chain(*zip(temp, edge_weight)))
        x.append(prefs_for_model)
        proposee_pref[node] = prefs_for_gale
    e_w = []
    for i in range(0,len(edge_weight),2):
        e_w.append(edge_weight[i] + edge_weight[i+1])
    edge_weight=e_w
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    look=group_size
    who_am_i=0
    where_am_i =0
    x_final = []
    for node in set1:
        look = 1 * group_size
        x_final.append([])
        x_final[who_am_i].append(0)
        x_final[who_am_i].append(0)
        for node2 in set2:
            if x[look][0] == who_am_i:
                x_final[who_am_i][0]  +=10
            x_final[who_am_i][1]-=x[look].index(who_am_i)+1
            look+=1
        where_am_i+=1
        who_am_i+=1


    look = 0
    who_am_i = 0
    where_am_i = group_size
    for node in set2:
        look = 0
        x_final.append([])
        x_final[where_am_i].append(0)
        x_final[where_am_i].append(0)
        for node2 in set1:
            if x[look][0] == who_am_i:
                x_final[where_am_i][0] += 10
            x_final[where_am_i][1] -= x[look].index(who_am_i)+1
            look += 1
        where_am_i+=1
        who_am_i += 1



    edges = []
    for u, v in G.edges():
        ui, vi = node_id_map[u], node_id_map[v]
        edges.append([ui, vi])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()




    matching = gale_shapley.stable_matching_with_preferences(G,set1,set2,proposer_pref,proposee_pref)
    if(verbose):
        for line in x:
            print(line)
        print("Optimal matching")
        print(is_stable_matching(matching, proposer_pref, proposee_pref))
        print(matching)
    e_attr= []

    for u,v in G.edges():
        if u in matching and matching[u] == v:
            e_attr.append(1)
        else:
            e_attr.append(0)
    edge_attr = torch.tensor(e_attr, dtype=torch.float)
    data_x = torch.tensor(x_final, dtype=torch.float)
    data = Data(
        x=data_x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        edge_y=edge_attr,
        proposee_pref = proposee_pref,
        proposer_pref = proposer_pref
    )
    return data

def graph_to_pyg_data_low_diff(G, group_size,verbose=False):

    nodes = list(G.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes)}



    set1, set2 = nx.bipartite.sets(G)

    x = []
    proposer_pref = {}
    proposee_pref = {}

    edge_weight = []

    for node in set1:
        prefs_for_gale = list(range(group_size,group_size*2))
        first_half = prefs_for_gale[:len(prefs_for_gale)//2].copy()
        second_half = prefs_for_gale[len(prefs_for_gale)//2:].copy()
        random.shuffle(second_half)
        prefs_for_gale = first_half + second_half

        prefs_for_model = [x - group_size for x in prefs_for_gale]
        edge_weight += [group_size - x for x in prefs_for_model]
        x.append(prefs_for_model)
        proposer_pref[node] = prefs_for_gale

    temp = []
    for node in set2:
        prefs_for_gale = list(range(group_size))

        first_half = prefs_for_gale[:len(prefs_for_gale) // 2].copy()
        second_half = prefs_for_gale[len(prefs_for_gale) // 2:].copy()
        random.shuffle(second_half)
        prefs_for_gale = first_half + second_half

        prefs_for_model = prefs_for_gale
        temp += [group_size - x for x in prefs_for_model]

        edge_weight= list(chain(*zip(temp, edge_weight)))
        x.append(prefs_for_model)
        proposee_pref[node] = prefs_for_gale
    e_w = []
    for i in range(0,len(edge_weight),2):
        e_w.append(edge_weight[i] + edge_weight[i+1])
    edge_weight=e_w
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    look=group_size
    who_am_i=0
    where_am_i =0
    x_final = []
    for node in set1:
        look = 1 * group_size
        x_final.append([])
        x_final[who_am_i].append(0)
        x_final[who_am_i].append(0)
        x_final[who_am_i].append(0)
        for node2 in set2:
            if x[look][0] == who_am_i:
                x_final[who_am_i][0]  +=10
            if x[look][-1] == who_am_i:
                x_final[who_am_i][2] -=10
            x_final[who_am_i][1]-=x[look].index(who_am_i)+1
            look+=1
        where_am_i+=1
        who_am_i+=1


    look = 0
    who_am_i = 0
    where_am_i = group_size
    for node in set2:
        look = 0
        x_final.append([])
        x_final[where_am_i].append(0)
        x_final[where_am_i].append(0)
        x_final[where_am_i].append(0)
        for node2 in set1:
            if x[look][0] == who_am_i:
                x_final[where_am_i][0] += 10
            if x[look][-1] == who_am_i:
                x_final[where_am_i][2] -=10
            x_final[where_am_i][1] -= x[look].index(who_am_i)+1
            look += 1
        where_am_i+=1
        who_am_i += 1


    edge_counter = 0
    edges = []
    for u, v in G.edges():
        ui, vi = node_id_map[u], node_id_map[v]
        edges.append([ui, vi])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()




    matching = gale_shapley.stable_matching_with_preferences(G,set1,set2,proposer_pref,proposee_pref)
    if(verbose):
        for line in x:
            print(line)
        print("Optimal matching")
        print(is_stable_matching(matching, proposer_pref, proposee_pref))
        print(matching)
    e_attr= []

    for u,v in G.edges():
        if u in matching and matching[u] == v:
            e_attr.append(1)
        else:
            e_attr.append(0)
    edge_attr = torch.tensor(e_attr, dtype=torch.float)
    data_x = torch.tensor(x_final, dtype=torch.float)
    data = Data(
        x=data_x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        edge_y=edge_attr,
        proposee_pref = proposee_pref,
        proposer_pref = proposer_pref
    )
    return data

def graph_to_pyg_data_high_diff(G, group_size,verbose=False):

    nodes = list(G.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes)}


    set1, set2 = nx.bipartite.sets(G)

    x = []
    proposer_pref = {}
    proposee_pref = {}

    edge_weight = []

    for node in set1:
        prefs_for_gale = list(range(group_size,group_size*2))
        first_half = prefs_for_gale[:len(prefs_for_gale) // 2].copy()
        second_half = prefs_for_gale[len(prefs_for_gale) // 2:].copy()
        random.shuffle(second_half)
        prefs_for_gale = first_half + second_half
        prefs_for_model = [x - group_size for x in prefs_for_gale]
        edge_weight += [group_size - x for x in prefs_for_model]
        x.append(prefs_for_model)
        proposer_pref[node] = prefs_for_gale

    temp = []
    for node in set2:
        prefs_for_gale = list(range(group_size))
        first_half = prefs_for_gale[:len(prefs_for_gale) // 2].copy()
        second_half = prefs_for_gale[len(prefs_for_gale) // 2:].copy()
        random.shuffle(first_half)
        prefs_for_gale =second_half + first_half
        prefs_for_model = prefs_for_gale
        temp += [group_size - x for x in prefs_for_model]

        edge_weight= list(chain(*zip(temp, edge_weight)))
        x.append(prefs_for_model)
        proposee_pref[node] = prefs_for_gale
    e_w = []
    for i in range(0,len(edge_weight),2):
        e_w.append(edge_weight[i] + edge_weight[i+1])
    edge_weight=e_w
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    look=group_size
    who_am_i=0
    where_am_i =0
    x_final = []
    for node in set1:
        look = 1 * group_size
        x_final.append([])
        x_final[who_am_i].append(0)
        x_final[who_am_i].append(0)
        x_final[who_am_i].append(0)
        for node2 in set2:
            if x[look][0] == who_am_i:
                x_final[who_am_i][0]  +=10
            if x[look][-1] == who_am_i:
                x_final[who_am_i][2] -=10
            x_final[who_am_i][1]-=x[look].index(who_am_i)+1
            look+=1
        where_am_i+=1
        who_am_i+=1


    look = 0
    who_am_i = 0
    where_am_i = group_size
    for node in set2:
        look = 0
        x_final.append([])
        x_final[where_am_i].append(0)
        x_final[where_am_i].append(0)
        x_final[where_am_i].append(0)
        for node2 in set1:
            if x[look][0] == who_am_i:
                x_final[where_am_i][0] += 10
            if x[look][-1] == who_am_i:
                x_final[where_am_i][2] -=10
            x_final[where_am_i][1] -= x[look].index(who_am_i)+1
            look += 1
        where_am_i+=1
        who_am_i += 1



    edges = []
    for u, v in G.edges():
        ui, vi = node_id_map[u], node_id_map[v]
        edges.append([ui, vi])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()




    matching = gale_shapley.stable_matching_with_preferences(G,set1,set2,proposer_pref,proposee_pref)
    if(verbose):
        for line in x:
            print(line)
        print("Optimal matching")
        print(is_stable_matching(matching, proposer_pref, proposee_pref))
        print(matching)
    e_attr= []

    for u,v in G.edges():
        if u in matching and matching[u] == v:
            e_attr.append(1)
        else:
            e_attr.append(0)
    edge_attr = torch.tensor(e_attr, dtype=torch.float)
    data_x = torch.tensor(x_final, dtype=torch.float)
    data = Data(
        x=data_x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        edge_y=edge_attr,
        proposee_pref = proposee_pref,
        proposer_pref = proposer_pref
    )
    return data