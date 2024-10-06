from script_.graphsite import graph_representation
import sys


#input recognition
args = sys.argv

# path to the .mol2 file of pocket
mol_path        = args[1]
# path to the .profile file of pocket which
# contains the sequence entropy node feature
profile_path    = args[2]
# path to the .popsa file of pocket which contains
# the solvent-accessible surface area node feature
pop_path        = args[3]

node_feature, edge_index, edge_attr = graph_representation(mol_path,profile_path,pop_path)
print(node_feature, edge_index, edge_attr)


