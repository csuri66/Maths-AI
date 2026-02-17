
import gale_shapley
import GAT

import data_generator

group_size=10

G = data_generator.generate_graph(group_size)
F= data_generator.graph_to_pyg_data(G,group_size)
print(F)

men_preferences = {
    'Man1': ['Woman1', 'Woman2', 'Woman3'],
    'Man2': ['Woman2', 'Woman3', 'Woman1'],
    'Man3': ['Woman3', 'Woman1', 'Woman2'],
}

women_preferences = {
    'Woman1': ['Man2', 'Man1', 'Man3'],
    'Woman2': ['Man1', 'Man2', 'Man3'],
    'Woman3': ['Man3', 'Man1', 'Man2'],
}

result = gale_shapley.solve(men_preferences, women_preferences)
print(result)

#TODO: cimkézés miként történjen
# ötlet: regresszió, a címkék a gale-shapely által megadott pár száma