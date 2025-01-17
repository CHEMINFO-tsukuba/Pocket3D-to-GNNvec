from graphsite import Graphsite

#referred to Graphsite/graphsite.py
def graph_representation(mol_path, profile_path, pop_path):
    gs = Graphsite()

    node_feature, edge_index, edge_attr = gs(mol_path, profile_path, pop_path)
    
    return node_feature, edge_index, edge_attr
