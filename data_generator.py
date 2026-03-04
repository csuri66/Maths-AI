import networkx as nx
import torch
from torch_geometric.data import Data
import random
import gale_shapley

def generate_graph(group_nodes=10):
    G = nx.Graph()

    G.add_nodes_from(list(range(group_nodes)), bipartite=0)
    G.add_nodes_from(list(range(group_nodes)), bipartite=1)

    for i in range(group_nodes):
        for j in range(group_nodes):
            G.add_edge(i,group_nodes+j)

    return G

def graph_to_pyg_data(G, group_size):

    nodes = list(G.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes)}

    edges = []
    for u, v in G.edges():
        ui, vi = node_id_map[u], node_id_map[v]
        edges.append([ui, vi])
        edges.append([vi, ui])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    set1, set2 = nx.bipartite.sets(G)

    x = []
    proposer_pref = {}
    proposee_pref = {}

    for node in set1:
        prefs = list(range(group_size,group_size*2))
        random.shuffle(prefs)
        x.append(prefs)
        proposer_pref[node] = prefs

    for node in set2:
        prefs = list(range(group_size))
        random.shuffle(prefs)
        x.append(prefs)
        ranks = {}
        for set1node in set1:
            ranks[set1node] = prefs[set1node]
        proposee_pref[node] = ranks

    data_x= torch.tensor(x, dtype=torch.float)

    matching = gale_shapley.solve(G,set1,set2,proposer_pref,proposee_pref)
    e_attr= []

    for u,v in G.edges():
        if (u in matching and matching[u] == v) or (v in matching and matching[v] == u):
            e_attr.append(1)
        else:
            e_attr.append(0)

    edge_attr = torch.tensor(e_attr, dtype=torch.float)
    print(len(edge_index[0]))
    print(len(edge_index[1]))
    print(len(edge_attr))
    data = Data(
        x=data_x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    return data
