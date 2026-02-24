
import gale_shapley
import GAT

import data_generator

group_size=3

G = data_generator.generate_graph(group_size)
F= data_generator.graph_to_pyg_data(G,group_size)
print("Data")
print(F)


#TODO: megfelelő élek kiválasztása majd él korrektség alapján loss
#How: Gale-Shapley segitségével felszámozom