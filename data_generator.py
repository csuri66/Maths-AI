import networkx as nx
import torch
from torch_geometric.data import Data
import random

def generate_graph(group_nodes=10):
    G = nx.Graph()

    for i in range(group_nodes*2):
        G.add_node(i)

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



    x = []
    for i in range(len(nodes)):
        temp = []
        for j in range(group_size):
            temp.append(j)
        random.shuffle(temp)
        x.append(temp)
    print(x)



    data = Data(
        x=x,
        edge_index=edge_index

    )
    return data
