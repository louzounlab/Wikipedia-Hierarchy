import pickle
import itertools
import networkx as nx
import configparser

MOAC = 0



'''
Example of the dictionary representation for a small category tree:

labels_dict = {(0, 0): (0, 0, 0), (0, 1): (0, 1, 0), (0, 2): (0, 1, 0), (0, 3): (0, 2, 0), (0, 4): (0, 2, 0),
               (0, 5): (0, 2, 0), (0, 6): (0, 2, 0), (1, 0): (0, 1, 0), (1, 1): (0, 0, 1), (1, 2): (1, 1, 0),
               (1, 3): (0, 1, 1), (1, 4): (0, 1, 1), (1, 5): (1, 2, 0), (1, 6): (1, 2, 0), (2, 0): (1, 0, 0),
               (2, 1): (1, 1, 0), (2, 2): (0, 0, 1), (2, 3): (1, 2, 0), (2, 4): (1, 2, 0), (2, 5): (0, 1, 1),
               (2, 6): (0, 1, 1), (3, 0): (2, 0, 0), (3, 1): (1, 0, 1), (3, 2): (2, 1, 0), (3, 3): (0, 0, 2),
               (3, 4): (1, 1, 1), (3, 5): (2, 2, 0), (3, 6): (2, 2, 0), (4, 0): (2, 0, 0), (4, 1): (1, 0, 1),
               (4, 2): (2, 1, 0), (4, 3): (1, 1, 1), (4, 4): (0, 0, 2), (4, 5): (2, 2, 0), (4, 6): (2, 2, 0),
               (5, 0): (2, 0, 0), (5, 1): (2, 1, 0), (5, 2): (1, 0, 1), (5, 3): (2, 2, 0), (5, 4): (2, 2, 0),
               (5, 5): (0, 0, 2), (5, 6): (1, 1, 1), (6, 0): (2, 0, 0), (6, 1): (2, 1, 0), (6, 2): (1, 0, 1),
               (6, 3): (2, 2, 0), (6, 4): (2, 2, 0), (6, 5): (1, 1, 1), (6, 6): (0, 0, 2)}

'''
discipline_to_idx = {"Math": 0, "Algebra": 1, "Arithmetics": 2, "AbstractAlgebra": 3, "LinearAlgebra": 4,

                     "BinaryArithmetic": 5, "ComputerArithmetic": 6}

Tree = nx.DiGraph()
# Tree.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
with open('Tree.pkl', 'rb') as t:
    Tree = pickle.load(t)

labels_dict = {}
for element in itertools.product(Tree.nodes, Tree.nodes):
    lowest_common = nx.lowest_common_ancestor(Tree, element[0], element[1])
    dist_first_from_LC = len(nx.shortest_path(Tree, lowest_common, element[0])) - 1
    dist_sec_from_LC = len(nx.shortest_path(Tree, lowest_common, element[1])) - 1
    dist_from_MOAC = len(nx.shortest_path(Tree, MOAC, lowest_common)) - 1
    labels_dict[(element[0], element[1])] = dist_first_from_LC, dist_sec_from_LC, dist_from_MOAC
    labels_dict[(element[1], element[0])] = dist_sec_from_LC, dist_first_from_LC, dist_from_MOAC

pickle.dump(labels_dict, open('labels.pkl', 'wb'))


